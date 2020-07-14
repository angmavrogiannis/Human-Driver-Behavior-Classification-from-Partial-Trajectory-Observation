import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import VehicleDataset

class DeepNet(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, dropout, device):
		super(DeepNet, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
		self.relu = torch.nn.ReLU()
		self.tanh = nn.Tanh()
		self.fc2 = torch.nn.Linear(self.hidden_size, 128)
		self.fc3 = torch.nn.Linear(128, self.output_size)

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		return x

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train_dataset = VehicleDataset(mode='train')
test_dataset = VehicleDataset(mode='test')
train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False, num_workers=4)

# hyper-parameters
num_epochs = 10
lr = 0.001

model = DeepNet(
    input_size=30,
    hidden_size=256,
    output_size=2, 
    dropout=0, 
    device=device
).to(device)
print('test')

mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    epoch_loss = 0
    for n_batch, (in_batch, label) in enumerate(train_loader):
        in_batch, label = in_batch.to(device), label.to(device)
        pred = model(in_batch)

        loss = mse_loss(pred, label)
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (n_batch + 1) % 10 == 0:
        #     print("Epoch: [{}/{}], Batch: {}, Loss: {}".format(
        #         epoch, num_epochs, n_batch, loss.item()))
    print('Epoch: [{}/{}], Loss: {}'.format(epoch + 1, num_epochs, epoch_loss))


l1_err, l2_err = 0, 0
l1_loss = nn.L1Loss()
model.eval()
gt = []
l = []
with torch.no_grad():
    for n_batch, (in_batch, label) in enumerate(test_loader):
        in_batch, label = in_batch.to(device), label.to(device)
        pred = model(in_batch)
        if n_batch == 0:
            print(label)
            print(pred)
            for l1, l2 in zip(label, pred):
                for item1, item2 in zip(l1, l2):
                    gt.append(item1[-1])
                    l.append(item2[-1])
        l1_err += l1_loss(pred, label).item()
        l2_err += mse_loss(pred, label).item()

print("Test L1 error:", l1_err)
print("Test L2 error:", l2_err)

# def mse(gt, l):
#     rmse = 0
#     for i in range(len(gt)):
#         rmse += np.power(gt[i] - l[i], 2)
#     rmse = np.sqrt(rmse / len(gt))

# print('RMSE: ', mse(gt, l))