from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

path = './_data/torch_data/'

train_dataset = MNIST(path, train=True, download=True)
test_dataset = MNIST(path, train=False, download=True)

x_train, y_train = train_dataset.data/255. , train_dataset.targets
x_test, y_test = test_dataset.data/255. , test_dataset.targets

print(x_train.shape, x_test.size())
print(y_train.shape, y_test.size())
# torch.Size([60000, 28, 28]) torch.Size([10000, 28, 28])
# torch.Size([60000]) torch.Size([10000])

print(np.min(x_train.numpy()), np.max(x_train.numpy()))
# 0.0 1.0

x_train, x_test = x_train.view(-1, 28*28), x_test.view(-1, 28*28)
print(x_train.shape, x_test.size())
# torch.Size([60000, 784]) torch.Size([10000, 784])