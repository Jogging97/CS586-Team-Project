import os
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from datetime import datetime
from sklearn import preprocessing
import torch.nn.functional as func
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

'''data pre-processing'''

df = pd.read_csv('weatherAUS.csv')

df['Temp9am'] = df['Temp9am'].fillna(value=df['Temp9am'].mean())
df['MinTemp'] = df['MinTemp'].fillna(value=df['MinTemp'].mean())
df['MaxTemp'] = df['MaxTemp'].fillna(value=df['MaxTemp'].mean())
df['Rainfall'] = df['Rainfall'].fillna(value=df['Rainfall'].mean())
df['Humidity9am'] = df['Humidity9am'].fillna(value=df['Humidity9am'].mean())
df['WindSpeed9am'] = df['WindSpeed9am'].fillna(value=df['WindSpeed9am'].mean())

df['RainToday'] = df['RainToday'].fillna(value=df['RainToday'].mode()[0])
df['RainTomorrow'] = df['RainTomorrow'].fillna(value=df['RainToday'].mode()[0])

df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

le = preprocessing.LabelEncoder()
df['Location'] = le.fit_transform(df['Location'])

df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

X = df[['Temp9am', 'MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'WindSpeed9am', 'RainToday', 'Location', 'Year',
        'Month', 'Day']]
y = df[['RainTomorrow']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.from_numpy(X_train.to_numpy()).float()
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
X_test = torch.from_numpy(X_test.to_numpy()).float()
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

'''build model'''


# create model
class Model(nn.Module):
    def __init__(self, input_size, output_size, n_layers=2):
        super(Model, self).__init__()
        self.input_size = input_size
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=16, num_layers=self.n_layers, dropout=0.2,
                          batch_first=True)

        self.fc = nn.Sequential(nn.Linear(16, 16),
                                nn.ReLU(),
                                nn.Linear(16, 5),
                                nn.Dropout(0.2), )

    def forward(self, x):
        x, h = self.rnn(x, None)
        x = self.fc(x[:, -1, :])
        return torch.sigmoid(x)


model = Model(X_train.shape[1], output_size=1)
criterion = nn.BCELoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)

model = model.to(device)
criterion = criterion.to(device)


def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)


def round_tensor(t, decimal_places=5):
    return round(t.item(), decimal_places)


'''train the model'''

for epoch in range(1001):
    y_pred = model(X_train)
    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred, y_train)
    if epoch % 100 == 0:
        train_acc = calculate_accuracy(y_train, y_pred)
        y_test_pred = model(X_test)
        y_test_pred = torch.squeeze(y_test_pred)
        test_loss = criterion(y_test_pred, y_test)
        test_acc = calculate_accuracy(y_test, y_test_pred)
        print(str('epoch ') + str(epoch) + str(' Train loss:') + str(round_tensor(train_loss)) + str(', Train accuracy: ') + str(round_tensor(train_acc)) + str(' Test loss:') + str(round_tensor(test_loss)) + str(', Test accuracy:') + str(round_tensor(test_acc)))
    optimiser.zero_grad()
    train_loss.backward()
    optimiser.step()