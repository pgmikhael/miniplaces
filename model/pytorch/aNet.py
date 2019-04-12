import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import DataLoaderDisk

# Dataset Parameters
batch_size = 200
load_size = 256
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])


# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units


opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data2/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    # 'data_mean': data_mean,
    'randomize': True
    }

opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    # 'data_mean': data_mean,
    'randomize': False
    }






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


# load data
loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)

net = Net()
criterion = nn.CrossEntropyLoss()                                           # define loss
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)     # define optimization alg


# Train Data
training_iters = 100
step_display = training_iters - 1

# set training mode
net.train()  
running_loss, acc1, acc5 = 0.0, 0, 0
    
for epoch in range(training_iters):                       # loop over the dataset multiple times   
    # X, y = X.to(DEVICE), y.to(DEVICE)   # get the inputs
    X, y = loader_train.next_batch(batch_size)
    optimizer.zero_grad()   # zero the parameter gradients
    y_ = net(X.float())            # forward prop
    loss = criterion(y_, y.long()) # compute loss
    loss.backward()         # back prop
    optimizer.step()        # update weights
        
    _, y_label_ = torch.max(y_, 1)
    _, y_label_5 = torch.topk(y_, 5, dim = 1, largest = True, sorted = True)
    acc1 += torch.sum((y_label_ == y.long()).float()).item()/batch_size
    acc5 += sum([y.long()[i] in y_label_5[i] for i in range(batch_size)])/batch_size
    running_loss += loss.item() * X.shape[0]


    if epoch % step_display == 0:    # print every 2000 iterations
    	print("-Iter " + str(epoch+1) + ": Training Loss= " + \
                  "{:.4f}".format(running_loss/10000) + ", Accuracy Top1 = " + \
                  "{:.2f}".format(acc1) + ", Top5 = " + \
                  "{:.2f}".format(acc5))
print('Finished Training')








