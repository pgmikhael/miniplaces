# arguments:

import argparse
import torch

def parse_args():
	parser = argparse.ArgumentParser(description='Miniplaces')
	
	
	  
	# models
	parser.add_argument('--models', nargs = '+', default = 'resnet50', help = 'Torchvision model to use. [default: resnet50]')

	# setup
	parser.add_argument('--optimizer', nargs = '+', default='Adam', help='Optimizer to use. Combination of SGD, Adam, RMSprop. [default: Adam]')
	parser.add_argument('--criterion', nargs = '+', default='cross_entropy', help='objective function to use [default: cross_entropy]')
	parser.add_argument('--lr', nargs = '+', default= [0.001], help='initial learning rate [default: 0.001]')
	parser.add_argument('--epochs', nargs = '+', default=100000, help='number of epochs for train [default: 100000]')
	parser.add_argument('--batch_size', nargs = '+', default=32, help='batch size for training [default: 32]')
	parser.add_argument('--dropout', nargs = '+', default=0.25, help='Amount of dropout to apply on last hidden layer [default: 0.25]')
	 


	# Alternative training/testing schemes

	# device
	parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
	
	args = parser.parse_args()

	args.cuda = args.cuda and torch.cuda.is_available()


	return args
