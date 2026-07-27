from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        _, _, _, embedding_dim = queries.shape
        scale = self.scale or 1.0 / sqrt(embedding_dim)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        values = torch.einsum('bhls,bshd->blhd', attention, values)
        if self.output_attention:
            return values.contiguous(), attention
        return values.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        batch_size, query_len, _ = queries.shape
        _, key_len, _ = keys.shape
        queries = self.query_projection(queries).view(
            batch_size, query_len, self.n_heads, -1
        )
        keys = self.key_projection(keys).view(batch_size, key_len, self.n_heads, -1)
        values = self.value_projection(values).view(
            batch_size, key_len, self.n_heads, -1
        )
        output, attention = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        output = output.view(batch_size, query_len, -1)
        return self.out_projection(output), attention


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        stdev = (variance + self.eps).sqrt()
        x = (x - mean) / stdev
        if self.affine:
            x = x * self.weight + self.bias
        return x, (mean, stdev)

    @staticmethod
    def denormalize(y, stats):
        mean, stdev = stats
        return y * stdev + mean


class CrossPhaseRoutingLayer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_routers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_relpos: bool = True,
        period_len: int = 24,
        window_size: Optional[int] = None,
        attention_dim: Optional[int] = None,
        use_pos_embed: bool = False,
        pos_dropout: float = 0.0,
    ):
        super().__init__()
        self.attention_dim = attention_dim or latent_dim
        assert self.attention_dim % num_heads == 0, (
            'attention_dim must be divisible by num_heads'
        )
        self.latent_dim = latent_dim
        self.num_routers = num_routers
        self.num_heads = num_heads
        self.head_dim = self.attention_dim // num_heads
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.period_len = period_len
        self.use_relpos = use_relpos
        self.window_size = window_size

        self.router = nn.Parameter(torch.randn(num_routers, latent_dim))
        nn.init.trunc_normal_(self.router, std=0.02)

        if self.use_pos_embed:
            self.pos_embedding = nn.Parameter(torch.zeros(period_len, latent_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            self.pos_dropout = nn.Dropout(pos_dropout)

        self.router_sender = AttentionLayer(
            FullAttention(
                False, factor=5, attention_dropout=dropout, output_attention=False
            ),
            latent_dim,
            num_heads,
        )
        self.router_receiver = AttentionLayer(
            FullAttention(
                False, factor=5, attention_dropout=dropout, output_attention=False
            ),
            latent_dim,
            num_heads,
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, phase_tokens):
        batch_size, channels, phase_count, latent_dim = phase_tokens.shape
        x = phase_tokens.view(batch_size * channels, phase_count, latent_dim)
        if self.use_pos_embed:
            if phase_count == self.period_len:
                position = self.pos_embedding.unsqueeze(0).expand(
                    batch_size * channels, -1, -1
                )
            elif phase_count < self.period_len:
                position = self.pos_embedding[:phase_count].unsqueeze(0).expand(
                    batch_size * channels, -1, -1
                )
            else:
                repeat_factor = (
                    phase_count + self.period_len - 1
                ) // self.period_len
                position = self.pos_embedding.repeat(repeat_factor, 1)[:phase_count]
                position = position.unsqueeze(0).expand(batch_size * channels, -1, -1)
            x = self.pos_dropout(x + position)

        routers = self.router.unsqueeze(0).expand(batch_size * channels, -1, -1)
        router_buffer, _ = self.router_sender(routers, x, x, attn_mask=None)
        received, _ = self.router_receiver(
            x, router_buffer, router_buffer, attn_mask=None
        )
        output = self.norm1(x + self.dropout_layer(received))
        output = self.norm2(output + self.dropout_layer(self.mlp(output)))
        return output.view(batch_size, channels, phase_count, latent_dim)


class PhaseEmbedding(nn.Module):
    def __init__(self, p_in, latent_dim, hidden=32, use_mlp=False, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        if use_mlp:
            self.projection = nn.Sequential(
                nn.Linear(p_in, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, latent_dim),
            )
        else:
            self.projection = nn.Linear(p_in, latent_dim)

    def forward(self, phase_series):
        return self.norm(self.projection(phase_series))


class PhasePredictor(nn.Module):
    def __init__(self, p_out, latent_dim, hidden, use_mlp=False, dropout=0.0):
        super().__init__()
        self.use_mlp = use_mlp
        if use_mlp:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden, p_out),
            )
        else:
            self.decoder = nn.Linear(latent_dim, p_out)
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, latent):
        if self.use_mlp:
            return self.decoder(latent)
        return self.decoder(self.dropout(latent))


class CrossPhaseRoutingUnit(nn.Module):
    def __init__(
        self,
        *,
        apply_in_proj,
        apply_out_proj,
        num_periods_input,
        latent_dim,
        phase_attn_heads,
        phase_attn_dropout,
        phase_attn_use_relpos,
        period_len,
        phase_attn_window=None,
        phase_attention_dim=None,
        phase_num_routers=8,
        phase_use_pos_embed=False,
        phase_pos_dropout=0.0,
    ):
        super().__init__()
        self.apply_in_proj = apply_in_proj
        if apply_in_proj:
            self.in_proj = nn.Sequential(
                nn.Linear(num_periods_input, latent_dim),
                nn.LayerNorm(latent_dim),
            )
        else:
            self.in_proj = None
        self.interact = CrossPhaseRoutingLayer(
            latent_dim=latent_dim,
            num_routers=phase_num_routers,
            num_heads=phase_attn_heads,
            dropout=phase_attn_dropout,
            use_relpos=phase_attn_use_relpos,
            period_len=period_len,
            window_size=phase_attn_window,
            attention_dim=phase_attention_dim,
            use_pos_embed=phase_use_pos_embed,
            pos_dropout=phase_pos_dropout,
        )
        self.out_proj = (
            nn.Linear(latent_dim, num_periods_input) if apply_out_proj else None
        )

    def forward(self, phase_series, previous_latent=None):
        if self.apply_in_proj:
            current_latent = self.in_proj(phase_series)
            latent = (
                previous_latent + current_latent
                if previous_latent is not None
                else current_latent
            )
        else:
            assert previous_latent is not None
            latent = previous_latent
        latent = self.interact(latent)
        phase_steps = self.out_proj(latent) if self.out_proj is not None else None
        return latent, phase_steps


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = getattr(configs, 'period_len', 24)
        self.latent_dim = getattr(configs, 'latent_dim', 8)
        self.phase_encoder_hidden = getattr(configs, 'phase_encoder_hidden', 32)
        self.predictor_hidden = getattr(configs, 'predictor_hidden', 64)
        self.phase_attn_heads = getattr(configs, 'phase_attn_heads', 4)
        self.phase_attn_dropout = getattr(configs, 'phase_attn_dropout', 0.0)
        self.phase_attn_use_relpos = bool(
            getattr(configs, 'phase_attn_use_relpos', True)
        )
        self.phase_attn_window = getattr(configs, 'phase_attn_window', None)
        self.phase_attention_dim = getattr(configs, 'phase_attention_dim', None)
        self.phase_num_routers = getattr(configs, 'phase_num_routers', 8)
        self.phase_use_pos_embed = bool(
            getattr(configs, 'phase_use_pos_embed', False)
        )
        self.phase_pos_dropout = getattr(configs, 'phase_pos_dropout', 0.0)

        self.num_periods_input = (
            self.seq_len + self.period_len - 1
        ) // self.period_len
        self.num_periods_output = (
            self.pred_len + self.period_len - 1
        ) // self.period_len
        self.total_len_in = self.num_periods_input * self.period_len
        self.pad_seq_len = self.total_len_in - self.seq_len

        self.use_revin = bool(getattr(configs, 'use_revin', True))
        if self.use_revin:
            self.revin = RevIN(
                num_features=self.enc_in,
                eps=getattr(configs, 'revin_eps', 1e-5),
                affine=bool(getattr(configs, 'revin_affine', False)),
            )
        self.phase_layers = getattr(configs, 'phase_layers', 1)
        self.embedding = PhaseEmbedding(
            p_in=self.num_periods_input,
            latent_dim=self.latent_dim,
            hidden=self.phase_encoder_hidden,
            use_mlp=bool(getattr(configs, 'phase_encoder_use_mlp', False)),
            dropout=getattr(configs, 'phase_encoder_dropout', 0.0),
        )

        routing_units = []
        for layer_index in range(self.phase_layers):
            is_first = layer_index == 0
            is_last = layer_index == self.phase_layers - 1
            routing_units.append(
                CrossPhaseRoutingUnit(
                    apply_in_proj=not is_first,
                    apply_out_proj=not is_last,
                    num_periods_input=self.num_periods_input,
                    latent_dim=self.latent_dim,
                    phase_attn_heads=self.phase_attn_heads,
                    phase_attn_dropout=self.phase_attn_dropout,
                    phase_attn_use_relpos=self.phase_attn_use_relpos,
                    period_len=self.period_len,
                    phase_attn_window=self.phase_attn_window,
                    phase_attention_dim=self.phase_attention_dim,
                    phase_num_routers=self.phase_num_routers,
                    phase_use_pos_embed=self.phase_use_pos_embed,
                    phase_pos_dropout=self.phase_pos_dropout,
                )
            )
        self.routing_layers = nn.ModuleList(routing_units)
        self.predictor = PhasePredictor(
            p_out=self.num_periods_output,
            latent_dim=self.latent_dim,
            hidden=self.predictor_hidden,
            use_mlp=bool(getattr(configs, 'predictor_use_mlp', False)),
            dropout=getattr(configs, 'predictor_dropout', 0.0),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                *args, **kwargs):
        if self.use_revin:
            x, stats = self.revin.normalize(x_enc)
        else:
            x = x_enc.float()

        x = x.permute(0, 2, 1)
        batch_size, channels, _ = x.shape
        if self.pad_seq_len > 0:
            x = F.pad(x, (0, self.pad_seq_len), mode='circular')
        periods = x.view(
            batch_size, channels, self.num_periods_input, self.period_len
        )
        phase_series = periods.permute(0, 1, 3, 2).contiguous()

        latent = self.embedding(phase_series)
        current_phase_series = phase_series
        for layer_index, routing_unit in enumerate(self.routing_layers):
            latent, phase_steps = routing_unit(current_phase_series, latent)
            if layer_index < len(self.routing_layers) - 1:
                current_phase_series = phase_steps

        phase_steps = self.predictor(latent)
        output_periods = phase_steps.permute(0, 1, 3, 2).contiguous()
        output = output_periods.reshape(batch_size, channels, -1)[..., :self.pred_len]
        output = output.permute(0, 2, 1)
        if self.use_revin:
            output = self.revin.denormalize(output, stats)
        return output
