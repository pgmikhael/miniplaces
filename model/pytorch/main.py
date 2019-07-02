# main: 
import torch
import multiprocessing as mp
import parsing as parsing
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import DataLoaderDisk, TestLoaderDisk, ValLoaderDisk
from setup import name_models, train_model, name_models_process

args = parsing.parse_args()



# dict for training data
# data_mean = [0.45834960097,0.44674252445,0.41352266842]
load_size = 256

data_mean = [0.485, 0.456, 0.406] 
data_sd = [0.229, 0.224, 0.225]

opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data2/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'data_mean': data_mean,
    'data_sd' : data_sd,
    'transform': True
    }

# dict for validation data
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data2/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'data_mean': data_mean,
    'data_sd' : data_sd,
    'transform': False
    }

# dict for validation data, for evaluation of entire dataset
opt_data_eval = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data2/images/val',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'transform': False
    }

# dict for test data
opt_data_test = {
    'data_root': '../../data2/images/test',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'randomize': False
    }




# train models sequentially
models = name_models(args)
num_epochs = args.epochs
batch_size = args.batch_size
criterion = args.criterion

for i in len(models):
	model_name = models[i][0]
	learning_rate = models[i][1]
	optimizer = models[i][2]

	train_model(model_name, criterion, optimizer, num_epochs, batch_size, learning_rate, opt_data_train, opt_data_val, opt_data_eval)


# OR train models in parallel
_, models = name_models_process(args, opt_data_train, opt_data_val, opt_data_eval)

processes = []
for i in range(len(models)): processes.append(mp.Process(target=train_model, args= models[i]))
for p in processes: p.start()
for p in processes: p.join()






# TEST code
# opt_data_train = {'data_mean': 1}
# opt_data_val = {'data_mean': 2}
# opt_data_eval = {'data_mean': 3}
# _, models = name_models_process(args, opt_data_train, opt_data_val, opt_data_eval)

# print(len(models))
# print(models[4])

# processes =[mp.Process(target=train_model, args= tpl) for tpl in models]
# TEST code



