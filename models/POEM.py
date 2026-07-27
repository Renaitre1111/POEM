import math

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        x = (x - mean) / stdev
        if self.affine:
            x = x * self.weight + self.bias
        return x, (mean, stdev)

    def denormalize(self, x, stats):
        mean, stdev = stats
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps * self.eps)
        return x * stdev + mean


class PhaseInteraction(nn.Module):
    def __init__(self, num_tokens, d_model, dropout, rank=4):
        super().__init__()
        kernel_size = 3

        self.phase_offset = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            padding_mode="circular",
        )
        self.real_down = nn.Linear(num_tokens, rank, bias=False)
        self.real_up = nn.Linear(rank, num_tokens, bias=False)
        self.imag_down = nn.Linear(num_tokens, rank, bias=False)
        self.imag_up = nn.Linear(rank, num_tokens, bias=False)
        self.branch_gate = nn.Linear(d_model, 2)
        self.layer_scale = nn.Parameter(torch.full((d_model,), 1e-2))
        self.dropout = nn.Dropout(dropout)

        phase = 2 * math.pi * torch.arange(num_tokens) / num_tokens
        self.register_buffer("base_phase", phase.view(1, num_tokens, 1))
        nn.init.zeros_(self.phase_offset.weight)
        nn.init.zeros_(self.phase_offset.bias)
        nn.init.zeros_(self.branch_gate.weight)
        nn.init.zeros_(self.branch_gate.bias)

    @staticmethod
    def _low_rank_mix(x, down, up):
        x = x.transpose(1, 2)
        return up(torch.nn.functional.silu(down(x))).transpose(1, 2)

    def forward(self, x):
        phase_offset = self.phase_offset(x.transpose(1, 2)).transpose(1, 2)
        phase = self.base_phase + math.pi * torch.tanh(phase_offset)

        phase_cos = torch.cos(phase)
        phase_sin = torch.sin(phase)
        real_mix = self._low_rank_mix(x * phase_cos, self.real_down, self.real_up)
        imag_mix = self._low_rank_mix(x * phase_sin, self.imag_down, self.imag_up)
        global_mix = real_mix * phase_cos + imag_mix * phase_sin
        local_mix = 0.5 * (torch.roll(x, 1, dims=1) + torch.roll(x, -1, dims=1)) - x
        branch_weight = torch.softmax(self.branch_gate(x), dim=-1)
        mixed = branch_weight[..., :1] * local_mix
        mixed = mixed + branch_weight[..., 1:] * global_mix
        return self.dropout(mixed) * self.layer_scale


class HarmonicModulation(nn.Module):
    def __init__(self, num_tokens, d_model, dropout, harmonics=2):
        super().__init__()
        hidden_dim = max(1, d_model // 4)
        self.input_projection = nn.Linear(d_model, 2 * hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, d_model)
        self.harmonic_cos = nn.Parameter(torch.zeros(harmonics, hidden_dim))
        self.harmonic_sin = nn.Parameter(torch.zeros(harmonics, hidden_dim))
        self.layer_scale = nn.Parameter(torch.full((d_model,), 1e-2))
        self.dropout = nn.Dropout(dropout)

        phase = 2 * math.pi * torch.arange(num_tokens) / num_tokens
        harmonic_index = torch.arange(1, harmonics + 1)
        angles = phase[:, None] * harmonic_index[None, :]
        self.register_buffer("phase_cos", torch.cos(angles).view(1, num_tokens, harmonics, 1))
        self.register_buffer("phase_sin", torch.sin(angles).view(1, num_tokens, harmonics, 1))

    def forward(self, x):
        value, gate = self.input_projection(x).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        hidden = torch.nn.functional.silu(value) * gate

        phase_cos = self.phase_cos[0, :, :, 0]
        phase_sin = self.phase_sin[0, :, :, 0]
        basis_scale = 2.0 / x.size(1)
        cos_context = basis_scale * torch.einsum("bpd,ph->bhd", gate, phase_cos)
        sin_context = basis_scale * torch.einsum("bpd,ph->bhd", gate, phase_sin)
        harmonic_cos = self.harmonic_cos.unsqueeze(0)
        harmonic_sin = self.harmonic_sin.unsqueeze(0)
        cos_weight = harmonic_cos * (1.0 + cos_context)
        cos_weight = cos_weight - harmonic_sin * sin_context
        sin_weight = harmonic_sin * (1.0 + cos_context)
        sin_weight = sin_weight + harmonic_cos * sin_context
        modulation = 1.0 + torch.einsum("ph,bhd->bpd", phase_cos, cos_weight)
        modulation = modulation + torch.einsum("ph,bhd->bpd", phase_sin, sin_weight)
        output = self.output_projection(hidden * modulation)
        return self.dropout(output) * self.layer_scale


class VanillaTokenMixer(nn.Module):
    def __init__(self, num_tokens, d_model, dropout):
        super().__init__()
        rank = 4
        target_params = 4 * num_tokens * rank + 7 * d_model + 2
        hidden_dim = max(
            1,
            round((target_params - num_tokens - d_model) / (2 * num_tokens + 1)),
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_tokens, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens),
        )
        self.layer_scale = nn.Parameter(torch.full((d_model,), 1e-2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mixed = self.mlp(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(mixed) * self.layer_scale


class VanillaFeatureMixer(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        harmonic_hidden = max(1, d_model // 4)
        target_params = 3 * d_model * harmonic_hidden
        target_params += 6 * harmonic_hidden + 2 * d_model
        hidden_dim = max(
            1,
            round((target_params - 2 * d_model) / (2 * d_model + 1)),
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.layer_scale = nn.Parameter(torch.full((d_model,), 1e-2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.mlp(x)) * self.layer_scale


class POEMBlock(nn.Module):
    def __init__(
        self,
        num_tokens,
        d_model,
        dropout,
        use_phase_interaction,
        use_harmonic_modulation,
        use_vanilla_mixer,
        phase_rank,
        harmonics,
    ):
        super().__init__()
        use_token_mixer = use_phase_interaction or use_vanilla_mixer
        use_feature_mixer = use_harmonic_modulation or use_vanilla_mixer

        self.interaction_norm = nn.LayerNorm(d_model) if use_token_mixer else None
        if use_vanilla_mixer:
            self.phase_interaction = VanillaTokenMixer(num_tokens, d_model, dropout)
        elif use_phase_interaction:
            self.phase_interaction = PhaseInteraction(
                num_tokens=num_tokens,
                d_model=d_model,
                dropout=dropout,
                rank=phase_rank,
            )
        else:
            self.phase_interaction = None

        self.modulation_norm = nn.LayerNorm(d_model) if use_feature_mixer else None
        if use_vanilla_mixer:
            self.harmonic_modulation = VanillaFeatureMixer(d_model, dropout)
        elif use_harmonic_modulation:
            self.harmonic_modulation = HarmonicModulation(
                num_tokens=num_tokens,
                d_model=d_model,
                dropout=dropout,
                harmonics=harmonics,
            )
        else:
            self.harmonic_modulation = None

    def forward(self, x):
        if self.phase_interaction is not None:
            x = x + self.phase_interaction(self.interaction_norm(x))
        if self.harmonic_modulation is not None:
            x = x + self.harmonic_modulation(self.modulation_norm(x))
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        if self.period_len > self.seq_len:
            raise ValueError("period_len must not exceed seq_len")

        self.input_cycle_count = self.seq_len // self.period_len
        self.forecast_cycle_count = math.ceil(self.pred_len / self.period_len)
        self.cycle_trim_len = self.seq_len - self.input_cycle_count * self.period_len

        d_model = configs.d_model
        mixer_layers = getattr(configs, "mixer_layers", 2)
        dropout = getattr(configs, "mixer_dropout", 0.1)
        phase_rank = getattr(configs, "phase_rank", 4)
        harmonics = getattr(configs, "harmonics", 2)
        geometry_type = getattr(configs, "geometry_type", "fixed")
        use_phase_interaction = bool(
            getattr(configs, "use_phase_interaction", 1)
        )
        use_harmonic_modulation = bool(
            getattr(configs, "use_harmonic_modulation", 1)
        )
        use_vanilla_mixer = bool(getattr(configs, "use_vanilla_mixer", 0))
        self.use_global_forecast = bool(
            getattr(configs, "use_global_forecast", 1)
        )
        if use_vanilla_mixer and (
            use_phase_interaction or use_harmonic_modulation
        ):
            raise ValueError(
                "Vanilla mixer cannot be combined with the POEM core modules"
            )

        self.revin = None
        if getattr(configs, "revin", 1):
            self.revin = RevIN(
                num_features=self.enc_in,
                affine=bool(getattr(configs, "affine", 0)),
            )

        self.phase_embedding = nn.Linear(self.input_cycle_count, d_model)
        phase = 2 * math.pi * torch.arange(self.period_len) / self.period_len
        direction = 2 * math.pi * torch.arange(d_model) / d_model
        phase_geometry = torch.cos(
            phase[:, None] - direction[None, :]
        ).unsqueeze(0)
        if geometry_type == "fixed":
            self.register_buffer("phase_geometry", phase_geometry)
        elif geometry_type == "learnable":
            self.phase_geometry = nn.Parameter(torch.empty_like(phase_geometry))
            nn.init.trunc_normal_(self.phase_geometry, std=0.02)
        elif geometry_type == "none":
            self.register_buffer("phase_geometry", None)
        else:
            raise ValueError(f"Unsupported geometry_type: {geometry_type}")
        self.input_norm = nn.LayerNorm(d_model)
        self.poem_blocks = nn.Sequential(*[
            POEMBlock(
                num_tokens=self.period_len,
                d_model=d_model,
                dropout=dropout,
                use_phase_interaction=use_phase_interaction,
                use_harmonic_modulation=use_harmonic_modulation,
                use_vanilla_mixer=use_vanilla_mixer,
                phase_rank=phase_rank,
                harmonics=harmonics,
            )
            for _ in range(mixer_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.cycle_forecast_head = nn.Linear(d_model, self.forecast_cycle_count)
        self.phase_residual_head = nn.Linear(
            self.input_cycle_count,
            self.forecast_cycle_count,
            bias=False,
        )
        if self.use_global_forecast:
            self.global_context_pool = nn.Linear(
                self.input_cycle_count, 1, bias=False
            )
            self.global_forecast_head = nn.Linear(1, self.pred_len, bias=False)
        else:
            self.global_context_pool = None
            self.global_forecast_head = None
        nn.init.zeros_(self.phase_residual_head.weight)
        if self.global_forecast_head is not None:
            nn.init.zeros_(self.global_forecast_head.weight)

    def _align_cycles(self, x):
        if self.cycle_trim_len == 0:
            return x
        return x[:, :, self.cycle_trim_len:]

    def forward(self, x):
        batch_size, _, num_channels = x.shape
        if self.revin is not None:
            x, revin_stats = self.revin.normalize(x)

        x = self._align_cycles(x.transpose(1, 2))
        x = x.reshape(
            batch_size,
            num_channels,
            self.input_cycle_count,
            self.period_len,
        )
        x = x.permute(0, 1, 3, 2).reshape(
            -1, self.period_len, self.input_cycle_count
        )

        phase_series = x
        global_forecast = None
        if self.use_global_forecast:
            cycle_level = phase_series.mean(dim=1).reshape(
                batch_size,
                num_channels,
                self.input_cycle_count,
            )
            global_forecast = self.global_forecast_head(
                self.global_context_pool(cycle_level)
            )
        x = self.phase_embedding(phase_series)
        if self.phase_geometry is not None:
            x = x + self.phase_geometry
        x = self.input_norm(x)
        x = self.poem_blocks(x)
        x = self.cycle_forecast_head(self.output_norm(x))
        x = x + self.phase_residual_head(phase_series)

        x = x.reshape(
            batch_size,
            num_channels,
            self.period_len,
            self.forecast_cycle_count,
        )
        x = x.permute(0, 1, 3, 2).reshape(batch_size, num_channels, -1)
        x = x[:, :, :self.pred_len]
        if global_forecast is not None:
            x = x + global_forecast
        x = x.transpose(1, 2)

        if self.revin is not None:
            x = self.revin.denormalize(x, revin_stats)
        return x
