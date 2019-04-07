import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



"""
When we call loss.backward(), the whole graph is differentiated w.r.t. 
the loss, and all Tensors in the graph that has requires_grad=True will 
have their .grad Tensor accumulated with the gradient.
"""

net = Net()
params = list(net.parameters())
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

input = torch.randn(1, 1, 32, 32)
output = net(input)

net.zero_grad()
output.backward(torch.randn(1, 10))

optimizer.zero_grad()   # zero the gradient buffers

target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output

loss = criterion(output, target)
loss.backward()
optimizer.step()  











