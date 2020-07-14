import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle

class VehicleDataset(Dataset):
	def __init__(self, input_file='lstm_inputs.npy', label_file='lstm_labels.npy', ids='ids.npy', mode='train'):
		self.mode = mode
		ids = np.load(ids)
		inputs = np.load(input_file)
		labels = np.load(label_file)

		num_data = len(inputs)
		train_test_split = int(0.8 * num_data)

		if mode is 'train':
			self.data = inputs[:train_test_split, :, :].astype(np.float32)
			self.labels = labels[:train_test_split, :, :].astype(np.float32)
			self.ids = ids[:train_test_split]
		elif mode is 'test':
			self.data = inputs[train_test_split:, :, :].astype(np.float32)
			self.labels = labels[train_test_split:, :, :].astype(np.float32)
			self.ids = ids[train_test_split:]

	def __getitem__(self, index):
		if self.mode is 'train':
			return self.data[index, :, :], self.labels[index, :, :]
		elif self.mode is 'test':
			return self.data[index, :, :], self.labels[index, :, :]

	def __len__(self):
		return len(self.data)

if __name__ == "__main__":
    dataset = VehicleDataset()