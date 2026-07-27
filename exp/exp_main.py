from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import POEM, PhaseFormer, SparseTSF, TimeBase
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def _dataset_name(self):
        return os.path.splitext(os.path.basename(self.args.data_path))[0]

    def _output_name(self):
        output_family = self.args.output_family or self.args.model_id
        if output_family in {'ablation', 'sensitivity'}:
            return os.path.join(output_family, f'seed_{self.args.seed}', self.args.model_id)
        return os.path.join(output_family, f'seed_{self.args.seed}')

    def _checkpoint_path(self, setting):
        path = os.path.join(self.args.checkpoints, self._output_name(), self._dataset_name())
        if self.args.output_family in {'ablation', 'sensitivity'}:
            return os.path.join(path, f'pred_{self.args.pred_len}')
        return os.path.join(path, setting)

    def _result_path(self, setting):
        path = os.path.join(
            'results', self._output_name(), self._dataset_name(), f'pred_{self.args.pred_len}'
        )
        if self.args.output_family in {'ablation', 'sensitivity'}:
            return path
        return os.path.join(path, setting)

    def _build_model(self):
        model_module = {
            'POEM': POEM,
            'PhaseFormer': PhaseFormer,
            'TimeBase': TimeBase,
            'SparseTSF': SparseTSF,
        }[self.args.model]
        model = model_module.Model(self.args).float()
        print(sum(p.numel() for p in model.parameters()))
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        if self.args.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def _clip_gradients(self):
        if self.args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

    def _select_criterion(self):
        if self.args.loss == 'mae':
            return nn.L1Loss()
        if self.args.loss == 'huber':
            return nn.SmoothL1Loss(beta=self.args.huber_delta)
        return nn.MSELoss()

    def _prepare_batch(self, batch_x, batch_y):
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_x = batch_x.to(self.device, non_blocking=True)
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y.to(self.device, non_blocking=True)
        return batch_x, batch_y

    def _forward_batch(self, batch_x, return_aux=False):
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = self.model(batch_x)
        auxiliary_loss = None
        if isinstance(outputs, tuple):
            outputs, second_output = outputs
            auxiliary_loss = second_output
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        if return_aux:
            return outputs, auxiliary_loss
        return outputs

    def vali(self, vali_loader, criterion):
        total_loss = 0.0
        total_samples = 0
        self.model.eval()
        with torch.inference_mode():
            for batch_x, batch_y in vali_loader:
                batch_x, batch_y = self._prepare_batch(batch_x, batch_y)
                outputs = self._forward_batch(batch_x)
                batch_size = batch_x.size(0)
                total_loss += criterion(outputs, batch_y).item() * batch_size
                total_samples += batch_size
        self.model.train()
        return total_loss / total_samples

    def _epoch_test(self, test_loader):
        total_mae = 0.0
        total_mse = 0.0
        batch_count = 0
        self.model.eval()
        with torch.inference_mode():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = self._prepare_batch(batch_x, batch_y)
                outputs = self._forward_batch(batch_x)
                total_mae += torch.mean(torch.abs(outputs - batch_y)).item()
                total_mse += torch.mean((outputs - batch_y) ** 2).item()
                batch_count += 1
        self.model.train()
        print(
            'Epoch test | MAE: {:.6f}, MSE: {:.6f}'.format(
                total_mae / batch_count, total_mse / batch_count
            )
        )

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        test_loader = None
        if self.args.eval_test_each_epoch:
            _, test_loader = self._get_data(flag='test')

        path = self._checkpoint_path(setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.scheduler_mode in {'timebase', 'sparsetsf'}:
            # The upstream constructor sets the first-epoch LR to max_lr / 25,
            # even when its type3 schedule does not call scheduler.step().
            lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate,
            )

        if self.args.sanity_val_steps > 0:
            self.model.eval()
            with torch.inference_mode():
                for step, (batch_x, batch_y) in enumerate(vali_loader):
                    batch_x, batch_y = self._prepare_batch(batch_x, batch_y)
                    criterion(self._forward_batch(batch_x), batch_y)
                    if step + 1 >= self.args.sanity_val_steps:
                        break
            self.model.train()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = torch.zeros((), device=self.device)
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x, batch_y = self._prepare_batch(batch_x, batch_y)
                outputs, auxiliary_loss = self._forward_batch(batch_x, return_aux=True)
                loss = criterion(outputs, batch_y)
                train_loss += loss.detach()

                back_loss = loss
                if auxiliary_loss is not None and self.args.use_orthogonal:
                    back_loss = back_loss + self.args.orthogonal_weight * auxiliary_loss
                back_loss.backward()
                self._clip_gradients()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print('\titers: {0}, epoch: {1} | loss: {2:.7f}'.format(
                        i + 1, epoch + 1, loss.item()
                    ))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print('Epoch: {} cost time: {}'.format(epoch + 1, time.time() - epoch_time))
            train_loss = (train_loss / train_steps).item()
            vali_loss = self.vali(vali_loader, criterion)
            print('Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}'.format(
                epoch + 1, train_steps, train_loss, vali_loss
            ))

            early_stopping(vali_loss, self.model, path)
            if test_loader is not None:
                self._epoch_test(test_loader)
            if early_stopping.early_stop:
                print('Early stopping')
                break
            if self.args.scheduler_mode in {'poem', 'timebase', 'sparsetsf'}:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        if not self.args.test_last:
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        _, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            checkpoint_path = os.path.join(self._checkpoint_path(setting), 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        preds = []
        trues = []
        self.model.eval()
        with torch.inference_mode():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = self._prepare_batch(batch_x, batch_y)
                outputs = self._forward_batch(batch_x)
                preds.append(outputs.cpu().numpy())
                trues.append(batch_y.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))

        dataset = self._dataset_name()
        folder_path = self._result_path(setting)
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, rse, corr]))

        settings = {
            'setting': setting,
            'dataset': dataset,
            'data': self.args.data,
            'data_path': self.args.data_path,
            'model': self.args.model,
            'model_id': self.args.model_id,
            'output_family': self.args.output_family,
            'features': self.args.features,
            'seq_len': self.args.seq_len,
            'pred_len': self.args.pred_len,
            'enc_in': self.args.enc_in,
            'period_len': self.args.period_len,
            'basis_num': self.args.basis_num,
            'use_period_norm': self.args.use_period_norm,
            'use_orthogonal': self.args.use_orthogonal,
            'orthogonal_weight': self.args.orthogonal_weight,
            'individual': self.args.individual,
            'model_type': self.args.model_type,
            'mixer_layers': self.args.mixer_layers,
            'mixer_dropout': self.args.mixer_dropout,
            'd_model': self.args.d_model,
            'phase_rank': self.args.phase_rank,
            'harmonics': self.args.harmonics,
            'geometry_type': self.args.geometry_type,
            'use_phase_interaction': self.args.use_phase_interaction,
            'use_harmonic_modulation': self.args.use_harmonic_modulation,
            'use_vanilla_mixer': self.args.use_vanilla_mixer,
            'use_global_forecast': self.args.use_global_forecast,
            'latent_dim': self.args.latent_dim,
            'phase_encoder_hidden': self.args.phase_encoder_hidden,
            'predictor_hidden': self.args.predictor_hidden,
            'phase_layers': self.args.phase_layers,
            'phase_attn_heads': self.args.phase_attn_heads,
            'phase_attn_dropout': self.args.phase_attn_dropout,
            'phase_attn_use_relpos': self.args.phase_attn_use_relpos,
            'phase_attn_window': self.args.phase_attn_window,
            'phase_attention_dim': self.args.phase_attention_dim,
            'phase_num_routers': self.args.phase_num_routers,
            'phase_use_pos_embed': self.args.phase_use_pos_embed,
            'phase_pos_dropout': self.args.phase_pos_dropout,
            'use_revin': self.args.use_revin,
            'revin_affine': self.args.revin_affine,
            'revin_eps': self.args.revin_eps,
            'revin': self.args.revin,
            'affine': self.args.affine,
            'learning_rate': self.args.learning_rate,
            'weight_decay': self.args.weight_decay,
            'gradient_clip': self.args.gradient_clip,
            'loss': self.args.loss,
            'huber_delta': self.args.huber_delta,
            'pct_start': self.args.pct_start,
            'optimizer': self.args.optimizer,
            'lradj': self.args.lradj,
            'scheduler_mode': self.args.scheduler_mode,
            'drop_last_train': self.args.drop_last_train,
            'num_workers': self.args.num_workers,
            'test_last': self.args.test_last,
            'sanity_val_steps': self.args.sanity_val_steps,
            'eval_test_each_epoch': self.args.eval_test_each_epoch,
            'deterministic': self.args.deterministic,
            'matmul_precision': self.args.matmul_precision,
            'train_epochs': self.args.train_epochs,
            'patience': self.args.patience,
            'run_tag': self.args.run_tag,
            'batch_size': self.args.batch_size,
            'seed': self.args.seed,
        }
        with open(os.path.join(folder_path, 'settings.json'), 'w') as f:
            json.dump(settings, f, indent=2)
