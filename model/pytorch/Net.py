import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision.datasets.folder import ImageFolder
# from torch.utils.data import DataLoader
# from torchvision.transforms import transforms
from DataLoader import *
import numpy as np
import matplotlib.pyplot as plt
"""
TODO:
    - load data
    - train net code
    - test net code
    - compute accuracies: acc1_total /= num_batch; acc5_total /= num_batch
"""

# Dataset Parameters
batch_size = 200
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Load Data
# def load_images(image_size=150, batch_size=64, root="../../data2/images"):

    # transform = transforms.Compose([
    #                 transforms.RandomCrop(image_size),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(data_mean)
    # ])

    # train_set = ImageFolder(root=root, transform=transform)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    # return train_loader  
# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data2/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)

# Transform data into tensors

def imshow(img, index):
    img = img.view(img.size(0), img.size(2), img.size(2), 3)
    plt.imshow(img[index])
    plt.show()

x, y = loader_train.next_batch(5)
imshow(x)

z = torch.from_numpy(x)
w = z.view(batch_size,3,128,128)
# batch size, c, h, w

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 100000
step_display = 50
step_save = 10000
path_save = 'alexnet'
start_from = ''


class Net(nn.Module):
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    """
    def __init__(self, num_classes=1000):
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
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()                                           # define loss
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)     # define optimization alg



# Train Data
N_EPOCHS = 2

for epoch in range(N_EPOCHS):                       # loop over the dataset multiple times
    print(f"Epoch {epoch+1}/{N_EPOCHS}")
    
    # Train
    net.train()  # IMPORTANT
    
    running_loss, correct = 0.0, 0
    
        
    # X, y = X.to(DEVICE), y.to(DEVICE)   # get the inputs
    X, y = loader_train.next_batch()
    optimizer.zero_grad()   # zero the parameter gradients
    y_ = net(X)            # forward prop
    loss = criterion(y_, y) # compute loss
    loss.backward()         # back prop
    optimizer.step()        # update weights
        
    # Statistics
    print(f"    batch loss: {loss.item():0.3f}")
    _, y_label_ = torch.max(y_, 1)
    # _, y_label5 = torch.max(y_, 5)
    correct += (y_label_ == y).sum().item()
        
    running_loss += loss.item() * X.shape[0]
    if epoch % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, running_loss / 2000))
    print(f"  Train Loss: {running_loss / len(y)}")
    print(f"  Train Acc:  {correct / len(y)}")
    print('Finished Training')
    
    
    
# Eval
for epoch in range(N_EPOCHS): 
    net.eval()  # IMPORTANT
    
    running_loss, correct = 0.0, 0
    
    with torch.no_grad():  # IMPORTANT
        X, y = loader_train.next_batch()
        # X, y = X.to(DEVICE), y.to(DEVICE)
                    
        y_ = net(X)
        
        # Statistics
        _, y_label_ = torch.max(y_, 1)
        correct += (y_label_ == y).sum().item()
        loss = criterion(y_, y)
        running_loss += loss.item() * X.shape[0]
    print(f"  Valid Loss: {running_loss / len(y)}")
    print(f"  Valid Acc:  {correct / len(y)}")
    print()





# Test the network

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))