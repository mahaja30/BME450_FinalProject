# Title

Brain Tumor MRI classification

## Team members

Mihir Mahajan, Fiona Crenshaw

## Project decription

- dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- This dataset has images of brain tumors separated as glioma, meningioma, pituitary, and no tumors
- With this data, the goal will be to train a neural network to distinguish between a brain with or without a tumor
- Data was collected from a combination of three datasets: figshare, SARTAJ dataset, and Br35H
- Proper pre-processing, including resizing images to a consistent size, is recommended to enhance model accuracy

Code for neural network in google colab: https://colab.research.google.com/drive/1Q-dDghdRO6Um-CjTZKxLY031Ndz8Hgk1?usp=sharing 

from google.colab import drive
drive.mount('/content/drive')

import os
root_dir = "/content/drive/MyDrive/images"

os.chdir(root_dir)

import torch
import torchvision

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels = 1),
    transforms.ToTensor()
])
trainset = datasets.ImageFolder(
    root="/content/drive/MyDrive/Tumor Recognition/Training",
    transform = train_transform
)

trainingload = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)

testset = datasets.ImageFolder(
    root="/content/drive/MyDrive/Tumor Recognition/Testing",
    transform = train_transform
)
testingload = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=0)
#x,y = next(iter(testingload))
#print(y.shape)
#print(x.shape)
categories = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
class MRI_Tumor_Classifier_Net(nn.Module):
  def __init__(self):
    super(MRI_Tumor_Classifier_Net, self).__init__()

    self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
    self.conv1_2 = nn.Conv2d(16,16,3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.conv2_2  = nn.Conv2d(32,32,3,padding=1)
    self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv3_2 = nn.Conv2d(64,64,3, padding=1)
    self.fc1 = nn.Linear(50176, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 4)


  def forward(self, x):
    x = F.relu(self.pool(self.conv1(x)))
    x = F.relu(self.conv1_2(x))
    x = F.relu(self.pool(self.conv2(x)))
    x = F.relu(self.conv2_2(x))
    x = F.relu(self.pool(self.conv3(x)))
    x = F.relu(self.conv3_2(x))
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.softmax(self.fc2(x), dim =-1)

    return x

net = MRI_Tumor_Classifier_Net()
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters())
for epoch in range(10):

    running_loss = 0.0
    for i, data in enumerate(trainingload, 0):
        #inputs = inputs.to(device)
        inputs, labels = data
        #labels = labels.to(device)
        #inputs = inputs.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print('Finished Training')
PATH = './MRI_classification_net.pth'
torch.save(net.state_dict(), PATH)
dataiter = iter(testingload)
images, labels = next(dataiter)
net.load_state_dict(torch.load(PATH))
outputs = net(images)
print(outputs.shape)
_, predicted = torch.max(outputs, 1)

correct = 0
total = 0

with torch.no_grad():
    for data in testingload:
        images, labels = data
        ##labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the 400 test images: {100 * correct // total} %')
correct_pred = {classname: 0 for classname in categories}
total_pred = {classname: 0 for classname in categories}

with torch.no_grad():
    for data in testingload:
        images, labels = data
        #images = images.to(device)
        #labels = labels.to(device)
        outputs = net(images)

        _, predictions = torch.max(outputs, 1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[categories[label]] += 1
            total_pred[categories[label]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for: {classname:5s} is {accuracy:.1f} %')
