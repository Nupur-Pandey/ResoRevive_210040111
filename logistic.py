import numpy as np
import torch
import pandas as pd
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


batch_size=256
train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

class Softmax(torch.nn.Module):
  def __init__(self, num_features, num_classes):
    super(Softmax, self).__init__()
    self.Linear1=torch.nn.Linear(num_features, 200)
    self.bn1=torch.nn.BatchNorm1d(200)
    self.droupout1=torch.nn.Dropout(0)
    self.Linear2=torch.nn.Linear(200, 100)
    self.bn2=torch.nn.BatchNorm1d(100)
    self.droupout2=torch.nn.Dropout(0)
    self.Linear3=torch.nn.Linear(100, num_classes)
    
    self.Linear3.weight.detach().zero_()
    self.Linear3.bias.detach().zero_()
    
  def forward(self, x):
      x=self.Linear1(x)
      x=self.bn1(x)
      x=self.droupout1(x)
      x=self.Linear2(x)
      x=self.bn2(x)
      x=self.droupout2(x)
      logits = self.Linear3(x)
      probas = F.softmax(logits, dim=1)
      return logits, probas


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    
    for features, targets in data_loader:
        features = features.view(-1, 28*28)
        targets = targets
        logits,probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        
    return correct_pred.float() / num_examples * 100
  
    
model=Softmax(num_features=784, num_classes=10)
optimizer=torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.001)
torch.manual_seed(123)
epoch_cost=[]

for epoch in range(25):
  avg_cost=0
  for batch_idx, (features, targets) in enumerate(train_loader):
    features=features.view(-1, 784)
    logits,probas=model(features)
    cost=F.cross_entropy(logits,targets)
    
    optimizer.zero_grad()
    cost.backward()
    avg_cost+=cost
    
    optimizer.step()
    
    if not batch_idx % 50:
      print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
               %(epoch+1, 25, batch_idx, 
                len(train_dataset)//batch_size, cost))
            
  with torch.set_grad_enabled(False):
      avg_cost = avg_cost/len(train_dataset)
      epoch_cost.append(avg_cost)
      print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
            epoch+1, 25, 
            compute_accuracy(model, train_loader)))
 
print('\nModel parameters:')
print('  Weights: %s' % model.Linear3.weight)
print('  Bias: %s' % model.Linear3.bias)
print('Test accuracy: %.2f' % (
            compute_accuracy(model, test_loader)))