import numpy as np
import argparse
from util import data

# ----------------------------------------------------------------------------

def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Commands')

  # train

  train_parser = subparsers.add_parser('train', help='Train model')
  train_parser.set_defaults(func=train)

  train_parser.add_argument('--dataset', default='mnist')
  train_parser.add_argument('--model', default='dcgan')
  train_parser.add_argument('-e', '--epochs', type=int, default=200)
  train_parser.add_argument('-l', '--logdir', default='logs/mnist-run')
  train_parser.add_argument('--lr', type=float, default=5e-5)
  train_parser.add_argument('--c', type=float, default=1e-2)
  train_parser.add_argument('--n-critic', type=int, default=5)
  train_parser.add_argument('--n-batch', type=int, default=128)

  return parser

# ----------------------------------------------------------------------------

def train(args):
  import models
  import numpy as np
  # np.random.seed(1234)

  if args.dataset == 'mnist':
    X_train, y_train, X_val, y_val, _, _ = data.load_mnist()
    _,_,im_rows, im_cols = X_train.shape
    n_dim, n_out, n_channels = im_rows, 10, 1
    
  elif args.dataset == 'random':
    n_dim, n_out, n_channels = 2, 2, 1
    X_train, y_train = data.load_noise(n=1000, d=n_dim)
    X_val, y_val = X_train, y_train
  else:
    raise ValueError('Invalid dataset name: %s' % args.dataset)

  # set up optimization params
  opt_params = { 'lr' : args.lr, 'c' : args.c, 'n_critic' : args.n_critic }
  
  # create model
  if args.model == 'dcgan':
    model = models.DCGAN(n_dim=n_dim, n_chan=n_channels, opt_params=opt_params)
  elif args.model == 'wdcgan':
    model = models.WDCGAN(n_dim=n_dim, n_chan=n_channels, opt_params=opt_params)    
  else:
    raise ValueError('Invalid model')
  
  # train model
  model.fit(X_train, X_val, 
            n_epoch=args.epochs, n_batch=args.n_batch,
            logdir=args.logdir)

def main():
  parser = make_parser()
  args = parser.parse_args()
  args.func(args)

if __name__ == '__main__':
  main()
