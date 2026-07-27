import argparse
import os

CPU_THREADS = 16
for thread_env in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[thread_env] = str(CPU_THREADS)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)

from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='POEM long-term forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument(
    '--model', type=str,
    choices=['POEM'],
    default='POEM', help='model',
)
parser.add_argument('--output_family', type=str, default='', help='top-level output directory name')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# model
parser.add_argument('--period_len', type=int, default=24, help='period length')
parser.add_argument('--mixer_layers', type=int, default=1, help='POEM Mixer block count')
parser.add_argument('--mixer_dropout', type=float, default=0.0, help='Mixer dropout')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--enc_in', type=int, default=7, help='input variable count')
parser.add_argument('--d_model', type=int, default=32, help='POEM embedding dimension')
parser.add_argument('--phase_rank', type=int, default=4, help='POEM phase interaction rank')
parser.add_argument('--harmonics', type=int, default=2, help='POEM harmonic count')
parser.add_argument(
    '--geometry_type', choices=['fixed', 'none', 'learnable'], default='fixed',
    help='POEM phase geometry encoding',
)
parser.add_argument(
    '--use_phase_interaction', type=int, choices=[0, 1], default=1,
    help='enable POEM adaptive phase interaction',
)
parser.add_argument(
    '--use_harmonic_modulation', type=int, choices=[0, 1], default=1,
    help='enable POEM harmonic phase modulation',
)
parser.add_argument(
    '--use_vanilla_mixer', type=int, choices=[0, 1], default=0,
    help='replace both POEM core mixers with vanilla MLP-Mixer blocks',
)
parser.add_argument(
    '--use_global_forecast', type=int, choices=[0, 1], default=1,
    help='enable the direct global forecast branch',
)
# optimization
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='AdamW weight decay')
parser.add_argument('--gradient_clip', type=float, default=0.0, help='gradient norm clipping; 0 disables it')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--run_tag', type=str, default='', help='experiment variant tag')
parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw'], default='adamw', help='optimizer')
parser.add_argument('--huber_delta', type=float, default=1.0, help='Huber loss delta')
parser.add_argument('--drop_last_train', type=int, default=0, help='drop incomplete training batch')
parser.add_argument('--num_workers', type=int, default=0, help='data loader workers')
parser.add_argument('--test_last', type=int, default=0, help='test final epoch weights instead of best checkpoint')
parser.add_argument('--sanity_val_steps', type=int, default=0, help='validation batches before epoch one')
parser.add_argument('--eval_test_each_epoch', type=int, default=0, help='run the test loader after each epoch')
parser.add_argument('--deterministic', type=int, default=0, help='enable deterministic PyTorch algorithms')
parser.add_argument('--matmul_precision', choices=['highest', 'high', 'medium'], default='highest')
parser.add_argument(
    '--loss', type=str, choices=['mae', 'mse', 'huber'], default='mae', help='loss function'
)
parser.add_argument(
    '--lradj', type=str, choices=['type1', 'type2', 'type3', 'constant', '3', '4', '5', '6'],
    default='type3', help='learning-rate schedule'
)

# GPU
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_cpu', action='store_true', help='force CPU execution')

args = parser.parse_args()

torch.set_float32_matmul_precision(args.matmul_precision)
if args.deterministic:
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

args.use_gpu = torch.cuda.is_available() and not args.use_cpu

print('Args in experiment:')
print(args)

# set seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
print(f'Using seed: {args.seed}')

# setting record of experiments
dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
setting = '{}_{}_{}_ft{}_sl{}_pl{}_test_seed{}'.format(
    args.model_id,
    args.model,
    dataset_name,
    args.features,
    args.seq_len,
    args.pred_len,
    args.seed)
exp = Exp_Main(args)

if args.is_training:
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
else:
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    if args.use_gpu:
        torch.cuda.empty_cache()
