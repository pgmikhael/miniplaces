# arguments:


import argparse
import torch
# from onconet.datasets.factory import get_dataset_class

EMPTY_NAME_ERR = 'Name of transformer or one of its arguments cant be empty\n\
  Use "name/arg1=value/arg2=value" format'
BATCH_SIZE_SPLIT_ERR = 'batch_size (={}) should be a multiple of batch_splits (={})'
DATA_AND_MODEL_PARALLEL_ERR = 'data_parallel and model_parallel should not be used in conjunction.'
INVALID_NUM_BLOCKS_ERR = 'Invalid block_layout. Must be length 4. Received {}'
INVALID_BLOCK_SPEC_ERR = 'Invalid block specification. Must be length 2 with "block_name,num_repeats". Received {}'
POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of list must be >=1'
CONFLICTING_WEIGHTED_SAMPLING_ERR = 'Cannot both use class_bal and year_weighted_class_bal at the same time.'
ILLEGAL_OPTIMIZER_ERR = 'Illegal optimizer {}. Training of meta objective requires using a meta optimizer, see all meta optimizers in onconet/optimizers/meta.py'


def parse_args():
	parser = argparse.ArgumentParser(description='Miniplaces')
	# setup
	parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
	parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
	parser.add_argument('--dev', action='store_true', default=False, help='Whether or not to run model on dev set')
	  

	# data
	# parser.add_argument('--img_dir', type=str, default='/home/administrator/Mounts/Isilon/pngs16', help='dir of images. Note, image path in dataset jsons should stem from here')
	  	# model
	parser.add_argument('--models', nargs = '+', default = ['resnet50'], help = 'Torchvision model to use.')

	# learning
	parser.add_argument('--optimizer', nargs = '+', default=["Adam"], help='Optimizer to use. Combination of SGD, Adam, RMSprop.')
	parser.add_argument('--criterion', nargs = '+', default=["cross_entropy"], help='objective function to use [default: cross_entropy]')
	parser.add_argument('--lr', nargs = '+', default= [0.001], help='initial learning rate [default: 0.001]')
	# parser.add_argument('--momentum', type=float, default=0, help='Momentum to use with SGD')
	# parser.add_argument('--lr_decay', type=float, default=0.5, help='initial learning rate [default: 0.5]')


	parser.add_argument('--epochs', nargs = '+', default=[256], help='number of epochs for train [default: 256]')
	parser.add_argument('--batch_size', nargs = '+', default=[32], help='batch size for training [default: 128]')
	parser.add_argument('--dropout', nargs = '+', default=[0.25], help='Amount of dropout to apply on last hidden layer [default: 0.25]')
	 


	# Alternative training/testing schemes

	# device
	parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
	# parser.add_argument('--num_gpus', type=int, default=1, help='Num GPUs to use in data_parallel.')
	# parser.add_argument('--num_shards', type=int, default=1, help='Num GPUs to shard a single model.')
	# parser.add_argument('--data_parallel', action='store_true', default=False, help='spread batch size across all available gpus. Set to false when using model parallelism. The combo of model and data parallelism may result in unexpected behavior')
	# parser.add_argument('--model_parallel', action='store_true', default=False, help='spread single model across num_shards. Note must have num_shards > 1 to take effect and only support in specific models. So far supported in all models that extend Resnet-base, i.e resnet-[n], nonlocal-resnet[n], custom-resnet models')

	args = parser.parse_args()

	# Set args particular to dataset
	# get_dataset_class(args).set_args(args)

	args.cuda = args.cuda and torch.cuda.is_available()


	return args
