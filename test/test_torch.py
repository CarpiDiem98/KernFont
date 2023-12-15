#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:49:48 2022

@author: emanuele
"""

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from progettoKerning.library.dataframe import df_dataset_matrix
start_time = time.time()
print(time.asctime(time.localtime(start_time)))


class Font(Dataset):
    def __init__(self, directory):
        df = df_dataset_matrix(directory)
        self.df = df
        self.X = torch.from_numpy(df[['X1', 'X2']].values)
        self.y = torch.from_numpy(df.y.values)

    def __getitem__(self, index):
        return self.X[[index]], self.y[[index]]

    def __len__(self):
        return len(self.df)


class RegressionCNN(nn.Module):
    def __init__(self):
        super(RegressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128)

    def forward(self, x1, x2):
        x1 = self.pool1(nn.relu(self.conv1(x1)))
        x2 = self.pool2(nn.relu(self.conv2(x2)))
        x = torch.cat((x1, x2), dim=1)
        x = x.view(-1, 8*8*32)
        x = self.fc(x)
        return x


# definizione del dataset
dataset = Font(directory='csv_matrix_font/')

# Dividi il dataset in training e test utilizzando random_split
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)

# Definire il modello, la loss e l'ottimizzatore
# Definire il modello, la loss e l'ottimizzatore
model = RegressionCNN()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Definire il numero di epoche
num_epochs = 10

# Addestra il modello
for epoch in range(num_epochs):
    # Addestra il modello sull'intero set di addestramento
    for X, y in train_dataloader:
        # Pulisci i gradienti
        optimizer.zero_grad()
        # Esegui il forward pass
        output = model(X)
        # Calcola la loss
        loss = loss_fn(output, y)
        # Esegui il backward pass
        loss.backward()
        # Aggiorna i parametri del modello
        optimizer.step()

    # Valida il modello sull'intero set di convalida
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in test_dataloader:
            output = model(X)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Validation Accuracy = {val_acc}")

torch.save(model.state_dict(), 'trained_model.pth')

###############################################################################
os.system('play -nq -t alsa synth {} sine {}'.format(1, 440))  # secondi, Hz
print("--- %s secondi ---" % ((time.time() - start_time)))
