import main
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--cuda', type=str, default='cuda:0', help='cuda device')
parser.add_argument('--data', type=str, default='./data/weather.csv', help='dataset path')
parser.add_argument('--feature_type', type=str, default='M', choices=['S', 'M', 'MS'],
    help=(
        'forecasting task, options:[M, S, MS]; M:multivariate predict'
        ' multivariate, S:univariate predict univariate, MS:multivariate'
        ' predict univariate'
    ),
)
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                    help='location of model checkpoints',)
parser.add_argument('--name', type=str, default='weather', help='save best model to checkpoints/name',)
# forecasting task
# 96 192 336 512 672 720
parser.add_argument( '--seq_len', type=int, default=512, help='input sequence length')
# 12 for PEMS  , {96, 192, 336, 720} for others
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
# model hyperparameter
# 1 2 3
parser.add_argument('--n_block', type=int, default=1, help='number of block for deep architecture',)
# 0.0  0.5  1.0
parser.add_argument('--alpha', type=float, default=0.0, help='feature feature dimension',)
parser.add_argument('--mix_layer_num', type=int, default=3, help='num of mix layer',)
parser.add_argument('--mix_layer_scale', type=int, default=2, help='scale of mix layer',)
# 4 8 16
parser.add_argument('--patch', type=int, default=16, help='fully-connected history len',)
parser.add_argument('--norm', type=bool, default=True, help='RevIN',)
parser.add_argument('--layernorm', type=bool, default=True, help='layernorm',)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
# optimization
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of input data')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='optimizer learning rate',)
# save results
parser.add_argument('--result_path', default='result.csv', help='path to save result')
args = parser.parse_args()

main.main(args)