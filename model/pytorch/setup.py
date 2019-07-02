import torch
import torch.nn as nn
import torch.optim 
import numpy as np
import torchvision.models as models
import itertools
from DataLoader import DataLoaderDisk, ValLoaderDisk



def name_models_process(args, *vars):
	# n_models = num_models(args)

	models = {} 
	lst_of_hyperparams = [args.models, args.criterion, args.optimizer,  args.epochs, args.batch_size, args.lr]

	combs_of_hyperparams = list(itertools.product(*lst_of_hyperparams))

	combs_of_hyperparams = [tpl + (vars) for tpl in combs_of_hyperparams]
    
	for i in range(len(combs_of_hyperparams)):
		models['model'+str(i)] = combs_of_hyperparams[i]

	return models, combs_of_hyperparams

def name_models(args):
	# n_models = num_models(args)

	models = {} 
	lst_of_hyperparams = [args.models, args.lrs, args.optimizer]
	combs_of_hyperparams = list(itertools.product(*lst_of_hyperparams))

	for i in len(combs_of_hyperparams):
		models['model'+str(i)] = combs_of_hyperparams[i]

	return models

def train_model(model_name, criterion, optimizer, num_epochs, batch_size, learning_rate, opt_data_train, opt_data_val, opt_data_eval):

	num_epochs = int(num_epochs)
	batch_size = int(batch_size)
	learning_rate = float(learning_rate)

	loader_train = DataLoaderDisk(**opt_data_train)
	loader_val = DataLoaderDisk(**opt_data_val)

	running_loss, acc1_val, acc5_val = 0.0, 0, 0
	iter_loss = []
	iter_acc1 = []
	iter_acc5 = []

	running_loss_val, acc1_val, acc5_val = 0.0, 0, 0
	iter_loss_val = []
	iter_acc1_val = []
	iter_acc5_val = []

	if torch.cuda.is_available():
		DEVICE = torch.device("cuda")

	if model_name == 'resnet18':
		model = models.resnet18()
	elif model_name == 'resnet50':
		model = models.resnet50()
	elif model_name == 'vgg19_bn':
		model = models.vgg19_bn()

	if torch.cuda.is_available():
		model = model.to(DEVICE)

	if criterion == 'cross_entropy':
		crit = nn.CrossEntropyLoss()  

	if optimizer == 'SGD':
		optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
	elif optimizer == 'Adam':
		optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	elif optimizer == 'RMSprop':
		optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

	step_display = 1000
	step_validate = 100

	# TEST
	# print(model.__class__.__name__) 
	# print(criterion) 
	# print(optimizer)
	# print( '-' + str(num_epochs) + '-' + str(batch_size) + '-' + str(learning_rate))
	# TEST

	for epoch in range(num_epochs):
		model.train()

		X, y = loader_train.next_batch(batch_size)  # get the inputs
		if torch.cuda.is_available():
			X, y = X.to(DEVICE), y.to(DEVICE)   
		optim.zero_grad()                       # zero the parameter gradients
		y_ = model(X.float())                         # forward prop
		loss = crit(y_, y.long())              # compute loss
		loss.backward()                             # back prop
		optim.step()                            # update weights

		_, y_label_ = torch.max(y_, 1)                                              # single prediction
		_, y_label_5 = torch.topk(y_, 5, dim = 1, largest = True, sorted = True)    # top 5 predictions 
		acc1 = torch.sum((y_label_ == y.long()).float()).item()/batch_size          # accuracy of single prediction
		acc5 = sum([y.long()[i] in y_label_5[i] for i in range(batch_size)])/batch_size # accuracy of top 5 preds
		running_loss = loss.item()                                                  # store loss
		
		iter_loss.append(running_loss)
		iter_acc1.append(acc1)
		iter_acc5.append(acc5)

		if (epoch+1) % step_validate == 0:
			model.eval()
			with torch.no_grad():
				Xval, yval = loader_val.next_batch(batch_size)
				if torch.cuda.is_available():
					Xval, yval = Xval.to(DEVICE), yval.to(DEVICE)   
				yval_ = model(Xval.float())

				# Statistics
				loss_val = crit(yval_, yval.long())
				_, yval_label_ = torch.max(y_, 1)
				_, yval_label_5 = torch.topk(yval_, 5, dim = 1, largest = True, sorted = True)

				running_loss_val = loss_val.item() 
				acc1_val = torch.sum((yval_label_ == yval.long()).float()).item()/batch_size
				acc5_val = sum([yval.long()[i] in yval_label_5[i] for i in range(batch_size)])/batch_size

				iter_loss_val.append(running_loss_val)
				iter_acc1_val.append(acc1_val)
				iter_acc5_val.append(acc5_val)

		if (epoch+1) % step_display == 0:    # print every 1000 iterations
			print("-Iter " + str(epoch+1) + ": Training Loss= " + \
				"{:.4f}".format(running_loss) + ", Accuracy Top1 = " + \
				"{:.2f}".format(acc1) + ", Top5 = " + \
				"{:.2f}".format(acc5))

	print('Finished Training')

	results_training = np.zeros((num_epochs, 4)) # save loss and accuracies for each iteration
	results_training[:,0] = range(1,num_epochs+1)
	results_training[:,1] = iter_loss
	results_training[:,2] = iter_acc1
	results_training[:,3] = iter_acc5
	np.savetxt('training-metrics_'+ model_name + '-'+optimizer+'-'+str(learning_rate)+'.csv', \
		results_training, delimiter=",")


	results_validation = np.zeros((num_epochs/step_validate, 4)) # save loss and accuracies for each iteration
	results_validation[:,0] = range(1,num_epochs/step_validate+1, step_validate)
	results_validation[:,1] = iter_loss_val
	results_validation[:,2] = iter_acc1_val
	results_validation[:,3] = iter_acc5_val
	np.savetxt('validation-metrics_'+ model_name + '-'+optimizer+'-'+str(learning_rate)+'.csv', \
		results_validation, delimiter=",")

	# Performance on entire validation
	loader_valeval = ValLoaderDisk(**opt_data_eval)
	model.eval()
	with torch.no_grad():
		for b in range(1,loader_test.size()//batch_size+1):
			Xval, valfilenames = loader_valeval.next_batch(batch_size)

			if torch.cuda.is_available():
				Xval = Xval.to(DEVICE)

			yval_ = net(Xval.float())

			_, yval_label_5 = torch.topk(yval_, 5, dim = 1, largest = True, sorted = True)

			with open("val_results.txt","a") as valresults:
				for i in range(ytest_.shape[0]):
					valresults.write("val/"+valfilenames[i]+" "+ str( yval_label_5[i,:].tolist() ).strip('[]').replace(',', "")+"\n" )
	print('Finished Validation')







