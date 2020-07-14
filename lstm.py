import torch
import torch.nn as nn
from torch.autograd import Variable


class VehicleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, device):
        super(VehicleLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        
        if self.num_layers > 1:
            self.dropout = dropout
        else:
            self.dropout = 0.0
        
        self.lstm = nn.LSTMCell(
            input_size = self.input_size,
            hidden_size = self.hidden_size
        )
        
        # define the output layer
        self.fc1 = nn.Linear(self.output_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 128)
        self.fc3 = nn.Linear(20, 2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dense = nn.Linear(self.hidden_size, 128)
        self.dense2 = nn.Linear(128, self.output_size)
        self.dense3 = nn.Linear(64, self.output_size)
        
    # initialize hidden state as
    def initial_hidden_state(self, batch):
        return Variable(torch.zeros(batch,self.hidden_size).to(self.device))

    # forward pass through LSTM layer
    def forward(self, x):       
        batch, seq_len, num_points = x.shape
        h = self.initial_hidden_state(batch)
        c = self.initial_hidden_state(batch)
        out = torch.zeros(batch, seq_len, self.output_size).to(self.device)
        for i in range(seq_len):
            # print(x.shape)
            h, c = self.lstm(x[:,i,:], (h, c))
            pred = self.dense(h)
            pred = self.relu(pred)
            pred = self.dense2(pred)
            # pred = self.relu(pred)
            # pred = self.dense3(pred)
            out[:,i,:] = pred
        return out

    # forward pass for test LSTM
    def test(self, x, seq_len=20):
        batch, _, _ = x.shape
        h = self.initial_hidden_state(batch)
        c = self.initial_hidden_state(batch)
        out = torch.zeros(batch, seq_len, self.output_size).to(self.device)
        for i in range(seq_len):
            if i == 0:
                h, c = self.lstm(x[:, 0, :], (h, c))
            else:
                x[:, i, 0] = pred[i, 0]
                print(x[:,i,1].shape)
                print(pred.shape)
                print(pred[i,1].shape)
                x[:, i, 1] = pred[i, 1]
                h, c = self.lstm(x[:, i, :], (h, c))
            pred = self.dense(h)
            pred = self.relu(pred)
            pred = self.dense2(pred)
            # pred = self.relu(pred)
            # pred = self.dense3(pred)
            out[:,i,:] = pred
        return out