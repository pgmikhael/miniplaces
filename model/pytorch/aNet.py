import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import DataLoaderDisk, TestLoaderDisk, ValLoaderDisk


DEVICE = torch.device("cuda")

# Dataset Parameters
batch_size = 200
load_size = 256
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])


# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units

# dict for training data
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data2/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    # 'data_mean': data_mean,
    'randomize': True
    }

# dict for validation data
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data2/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    # 'data_mean': data_mean,
    'randomize': False
    }

# dict for test data
opt_data_test = {
    'data_root': '../../data2/images/test',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'randomize': False
    }

# dict for validation data, for evaluation of entire dataset
opt_data_eval = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data2/images/val',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'randomize': False
    }




# CNN module
class Net(nn.Module):
    def __init__(self, num_classes=100):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x


# load train and val data
loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)

net = Net()
net = net.to(DEVICE)
criterion = nn.CrossEntropyLoss()                                           # loss
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)     # optimization alg
#optimizer = optim.Adam(net.parameters(), lr=learning_rate)                 # maybe consider Adam or RMSProp


# Train Data
training_iters = 10000      # iteration steps, decreased from 100,000
step_display = 100

running_loss, acc1_val, acc5_val = 0.0, 0, 0
iter_loss = []
iter_acc1 = []
iter_acc5 = []

running_loss_val, acc1_val, acc5_val = 0.0, 0, 0
iter_loss_val = []
iter_acc1_val = []
iter_acc5_val = []

for epoch in range(training_iters):  
    net.train()                                 # set training mode 
    X, y = loader_train.next_batch(batch_size)  # get the inputs
    X, y = X.to(DEVICE), y.to(DEVICE)   
    optimizer.zero_grad()                       # zero the parameter gradients
    y_ = net(X.float())                         # forward prop
    loss = criterion(y_, y.long())              # compute loss
    loss.backward()                             # back prop
    optimizer.step()                            # update weights
        
    _, y_label_ = torch.max(y_, 1)                                              # single prediction
    _, y_label_5 = torch.topk(y_, 5, dim = 1, largest = True, sorted = True)    # top 5 predictions 
    acc1 = torch.sum((y_label_ == y.long()).float()).item()/batch_size          # accuracy of single prediction
    acc5 = sum([y.long()[i] in y_label_5[i] for i in range(batch_size)])/batch_size # accuracy of top 5 preds
    running_loss = loss.item()                                                  # store loss
    
    iter_loss.append(running_loss)
    iter_acc1.append(acc1)
    iter_acc5.append(acc5)

    net.eval()                                      # evaluation mode, store performance on validation set
    with torch.no_grad():                                           
        Xval, yval = loader_val.next_batch(batch_size)
        Xval, yval = Xval.to(DEVICE), yval.to(DEVICE)   
        yval_ = net(Xval.float())

        # Statistics
        loss_val = criterion(yval_, yval.long())
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

results_training = np.zeros((training_iters, 4)) # save loss and accuracies for each iteration
results_training[:,0] = range(1,training_iters+1)
results_training[:,1] = iter_loss
results_training[:,2] = iter_acc1
results_training[:,3] = iter_acc5
np.savetxt('training-metrics.csv', results_training, delimiter=",")


results_validation = np.zeros((training_iters, 4)) # save loss and accuracies for each iteration
results_validation[:,0] = range(1,training_iters+1)
results_validation[:,1] = iter_loss_val
results_validation[:,2] = iter_acc1_val
results_validation[:,3] = iter_acc5_val
np.savetxt('validation-metrics.csv', results_validation, delimiter=",")


# Performance on entire validation and test Sets
loader_test = TestLoaderDisk(**opt_data_test)
loader_valeval = ValLoaderDisk(**opt_data_eval)

net.eval()
with torch.no_grad():
    for b in range(1,loader_test.size()//batch_size+1):
        Xval, valfilenames = loader_valeval.next_batch(batch_size)
        Xtest, testfilenames = loader_test.next_batch(batch_size)
        
        print(testfilenames[0])
        
        Xval = Xval.to(DEVICE)
        Xtest = Xtest.to(DEVICE)
        
        yval_ = net(Xval.float())
        ytest_ = net(Xtest.float())
        
        _, yval_label_5 = torch.topk(yval_, 5, dim = 1, largest = True, sorted = True)
        _, ytest_label_5 = torch.topk(ytest_, 5, dim = 1, largest = True, sorted = True)
        
        with open("test_results.txt","a") as testresults:
            for i in range(ytest_.shape[0]):
                testresults.write("test/"+testfilenames[i]+" "+ str( ytest_label_5[i,:].tolist() ).strip('[]').replace(',', "")+"\n" )
        with open("val_results.txt","a") as valresults:
            for i in range(ytest_.shape[0]):
                valresults.write("val/"+valfilenames[i]+" "+ str( yval_label_5[i,:].tolist() ).strip('[]').replace(',', "")+"\n" )

print('Finished Validation & Test')

