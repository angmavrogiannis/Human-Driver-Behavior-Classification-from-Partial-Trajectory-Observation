import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class Vehicle:
	def __init__(self,v_id,num_frames,time,local_x,local_y,v_length,v_width,v_class,v_vel,v_acc,lane_id,preceding,following,space_headway,time_headway):
		self.id = v_id
		self.frames = num_frames
		self.t = time
		self.x = local_x
		self.y = local_y
		self.length = v_length
		self.width = v_width
		self.vtype = v_class
		self.vel = v_vel
		self.acc = v_acc
		self.lane = lane_id
		self.prec = preceding
		self.foll = following
		self.shead = space_headway
		self.thead = time_headway
		self.vrel_avg = []
		self.lat_dev = []
		self.lat_vel = None
		self.lat_acc = None
		self.lat_jerk = None
		self.long_jerk = None
		self.t2seqt = {}
		self.vrel2front = []

class GridVehicle:
	def __init__(self, ff, f, fr, fl, l, r, b, br, bl):
		self.ff = ff
		self.f = f
		self.fr = fr
		self.fl = fl
		self.l = l
		self.r = r
		self.b = b
		self.br = br
		self.bl = bl

#loading vehicle object
filehandler = open('vehicle.obj', 'rb')
vehicle = pickle.load(filehandler)

filehandler = open('gridVehicle.obj', 'rb')
g = pickle.load(filehandler)

filehandler = open('behavior.pkl', 'rb')
behavior = pickle.load(filehandler)
behavior[-1] = 0

ref_feat = 3
num_output = 2
dataset = []
targets = []
t_data = []
batch_f = []
batch_l = []
batch_t = []
seq_len = 20
freq = 10
include_behavior = True
ids = []

c = 0
for i in range(len(vehicle)):
	#print('Vehicle: ', i)
	size = len(vehicle[i].x)
	if size % seq_len != 0:
		size = size - (size % seq_len)
	else:
		size -= seq_len
	for j, t in enumerate(vehicle[i].t):
		if j % freq == 0 and j < size:
			if len(batch_f) == seq_len:
				c += 1
				batch_f = []
				batch_l = []
				batch_t = []
			data = []
			batch_t.append(t)
			data.append(vehicle[i].x[j])
			data.append(vehicle[i].y[j])
			if include_behavior:
				data.append(behavior[i])
			# data.append(vehicle[i].lat_vel[j])
			# data.append(vehicle[i].vel[j])
			# data.append(vehicle[i].vtype)
			labels = [vehicle[i].x[j+freq], vehicle[i].y[j+freq]]
			g[i].ff[j] = int(g[i].ff[j])
			g[i].f[j] = int(g[i].f[j])
			g[i].fr[j] = int(g[i].fr[j])
			g[i].fl[j] = int(g[i].fl[j])
			g[i].l[j] = int(g[i].l[j])
			g[i].r[j] = int(g[i].r[j])
			g[i].b[j] = int(g[i].b[j])
			g[i].br[j] = int(g[i].br[j])
			g[i].bl[j] = int(g[i].bl[j])

			if g[i].ff[j] != -1 and g[i].ff[j] < len(vehicle[i].x):
				if t in vehicle[g[i].ff[j]].t:
					data.append(vehicle[g[i].ff[j]].x[vehicle[g[i].ff[j]].t2seqt[t]] - vehicle[i].x[j])
					data.append(vehicle[g[i].ff[j]].y[vehicle[g[i].ff[j]].t2seqt[t]] - vehicle[i].y[j])
					# labels.append(vehicle[g[i].ff[j]].x[vehicle[g[i].ff[j]].t2seqt[t]+freq])
					# labels.append(vehicle[g[i].ff[j]].y[vehicle[g[i].ff[j]].t2seqt[t]+freq])
					if include_behavior:
						data.append(behavior[g[i].ff[j]])
					# data.append(vehicle[g[i].ff[j]].lat_vel[vehicle[g[i].ff[j]].t2seqt[t]])
					# data.append(vehicle[i].vel[j] - vehicle[g[i].ff[j]].vel[vehicle[g[i].ff[j]].t2seqt[t]])
					# if vehicle[i].vel[j] - vehicle[g[i].ff[j]].vel[vehicle[g[i].ff[j]].t2seqt[t]] != 0:
					# 	data.append((vehicle[g[i].ff[j]].y[vehicle[g[i].ff[j]].t2seqt[t]] - vehicle[i].y[j]) / (vehicle[i].vel[j] - vehicle[g[i].ff[j]].vel[vehicle[g[i].ff[j]].t2seqt[t]]))
					# else:
					# 	data.append(0)
					# data.append(vehicle[g[i].ff[j]].vtype)
				else:
					for k in range(ref_feat):
						data.append(0)
			else:
				for k in range(ref_feat):
					data.append(0)

			if g[i].f[j] != -1 and g[i].f[j] < len(vehicle[i].x):
				if t in vehicle[g[i].f[j]].t:
					data.append(vehicle[g[i].f[j]].x[vehicle[g[i].f[j]].t2seqt[t]] - vehicle[i].x[j])
					data.append(vehicle[g[i].f[j]].y[vehicle[g[i].f[j]].t2seqt[t]] - vehicle[i].y[j])
					if include_behavior:
						data.append(behavior[g[i].f[j]])
					# data.append(vehicle[g[i].f[j]].lat_vel[vehicle[g[i].f[j]].t2seqt[t]])
					# data.append(vehicle[i].vel[j] - vehicle[g[i].f[j]].vel[vehicle[g[i].f[j]].t2seqt[t]])
					# if vehicle[i].vel[j] - vehicle[g[i].f[j]].vel[vehicle[g[i].f[j]].t2seqt[t]] != 0:
					# 	data.append((vehicle[g[i].f[j]].y[vehicle[g[i].f[j]].t2seqt[t]] - vehicle[i].y[j]) / (vehicle[i].vel[j] - vehicle[g[i].f[j]].vel[vehicle[g[i].f[j]].t2seqt[t]]))
					# else:
					# 	data.append(0)
					# data.append(vehicle[g[i].f[j]].vtype)
				else:
					for k in range(ref_feat):
						data.append(0)
			else:
				for k in range(ref_feat):
					data.append(0)

			if g[i].fr[j] != -1 and g[i].fr[j] < len(vehicle[i].x):
				if t in vehicle[g[i].fr[j]].t:
					data.append(vehicle[g[i].fr[j]].x[vehicle[g[i].fr[j]].t2seqt[t]] - vehicle[i].x[j])
					data.append(vehicle[g[i].fr[j]].y[vehicle[g[i].fr[j]].t2seqt[t]] - vehicle[i].y[j])
					if include_behavior:
						data.append(behavior[g[i].fr[j]])
					# data.append(vehicle[g[i].fr[j]].lat_vel[vehicle[g[i].fr[j]].t2seqt[t]])
					# data.append(vehicle[i].vel[j] - vehicle[g[i].fr[j]].vel[vehicle[g[i].fr[j]].t2seqt[t]])
					# if vehicle[i].vel[j] - vehicle[g[i].fr[j]].vel[vehicle[g[i].fr[j]].t2seqt[t]] != 0:
					# 	data.append((vehicle[g[i].fr[j]].y[vehicle[g[i].fr[j]].t2seqt[t]] - vehicle[i].y[j]) / (vehicle[i].vel[j] - vehicle[g[i].fr[j]].vel[vehicle[g[i].fr[j]].t2seqt[t]]))
					# else:
					# 	data.append(0)
					# data.append(vehicle[g[i].fr[j]].vtype)
				else:
					for k in range(ref_feat):
						data.append(0)
			else:
				for k in range(ref_feat):
					data.append(0)

			if g[i].fl[j] != -1 and g[i].fl[j] < len(vehicle[i].x):
				if t in vehicle[g[i].fl[j]].t:
					data.append(vehicle[g[i].fl[j]].x[vehicle[g[i].fl[j]].t2seqt[t]] - vehicle[i].x[j])
					data.append(vehicle[g[i].fl[j]].y[vehicle[g[i].fl[j]].t2seqt[t]] - vehicle[i].y[j])
					if include_behavior:
						data.append(behavior[g[i].fl[j]])
					# data.append(vehicle[g[i].fl[j]].lat_vel[vehicle[g[i].fl[j]].t2seqt[t]])
					# data.append(vehicle[i].vel[j] - vehicle[g[i].fl[j]].vel[vehicle[g[i].fl[j]].t2seqt[t]])
					# if vehicle[i].vel[j] - vehicle[g[i].fl[j]].vel[vehicle[g[i].fl[j]].t2seqt[t]] != 0:
					# 	data.append((vehicle[g[i].fl[j]].y[vehicle[g[i].fl[j]].t2seqt[t]] - vehicle[i].y[j]) / (vehicle[i].vel[j] - vehicle[g[i].fl[j]].vel[vehicle[g[i].fl[j]].t2seqt[t]]))
					# else:
					# 	data.append(0)
					# data.append(vehicle[g[i].fl[j]].vtype)
				else:
					for k in range(ref_feat):
						data.append(0)
			else:
				for k in range(ref_feat):
					data.append(0)

			if g[i].l[j] != -1 and g[i].l[j] < len(vehicle[i].x):
				if t in vehicle[g[i].l[j]].t:
					data.append(vehicle[g[i].l[j]].x[vehicle[g[i].l[j]].t2seqt[t]] - vehicle[i].x[j])
					data.append(vehicle[g[i].l[j]].y[vehicle[g[i].l[j]].t2seqt[t]] - vehicle[i].y[j])
					if include_behavior:
						data.append(behavior[g[i].l[j]])
					# data.append(vehicle[g[i].l[j]].lat_vel[vehicle[g[i].l[j]].t2seqt[t]])
					# data.append(vehicle[i].vel[j] - vehicle[g[i].l[j]].vel[vehicle[g[i].l[j]].t2seqt[t]])
					# if vehicle[i].vel[j] - vehicle[g[i].l[j]].vel[vehicle[g[i].l[j]].t2seqt[t]] != 0:
					# 	data.append((vehicle[g[i].l[j]].y[vehicle[g[i].l[j]].t2seqt[t]] - vehicle[i].y[j]) / (vehicle[i].vel[j] - vehicle[g[i].l[j]].vel[vehicle[g[i].l[j]].t2seqt[t]]))
					# else:
					# 	data.append(0)
					# data.append(vehicle[g[i].l[j]].vtype)
				else:
					for k in range(ref_feat):
						data.append(0)
			else:
				for k in range(ref_feat):
					data.append(0)

			if g[i].r[j] != -1 and g[i].r[j] < len(vehicle[i].x):
				if t in vehicle[g[i].r[j]].t:
					data.append(vehicle[g[i].r[j]].x[vehicle[g[i].r[j]].t2seqt[t]] - vehicle[i].x[j])
					data.append(vehicle[g[i].r[j]].y[vehicle[g[i].r[j]].t2seqt[t]] - vehicle[i].y[j])
					if include_behavior:
						data.append(behavior[g[i].r[j]])
					# data.append(vehicle[g[i].r[j]].lat_vel[vehicle[g[i].r[j]].t2seqt[t]])
					# data.append(vehicle[i].vel[j] - vehicle[g[i].r[j]].vel[vehicle[g[i].r[j]].t2seqt[t]])
					# if vehicle[i].vel[j] - vehicle[g[i].r[j]].vel[vehicle[g[i].r[j]].t2seqt[t]] != 0:
					# 	data.append((vehicle[g[i].r[j]].y[vehicle[g[i].r[j]].t2seqt[t]] - vehicle[i].y[j]) / (vehicle[i].vel[j] - vehicle[g[i].r[j]].vel[vehicle[g[i].r[j]].t2seqt[t]]))
					# else:
					# 	data.append(0)
					# data.append(vehicle[g[i].r[j]].vtype)
				else:
					for k in range(ref_feat):
						data.append(0)
			else:
				for k in range(ref_feat):
					data.append(0)

			if g[i].b[j] != -1 and g[i].b[j] < len(vehicle[i].x):
				if t in vehicle[g[i].b[j]].t:
					data.append(vehicle[g[i].b[j]].x[vehicle[g[i].b[j]].t2seqt[t]] - vehicle[i].x[j])
					data.append(vehicle[g[i].b[j]].y[vehicle[g[i].b[j]].t2seqt[t]] - vehicle[i].y[j])
					if include_behavior:
						data.append(behavior[g[i].b[j]])
					# data.append(vehicle[g[i].b[j]].lat_vel[vehicle[g[i].b[j]].t2seqt[t]])
					# data.append(vehicle[i].vel[j] - vehicle[g[i].b[j]].vel[vehicle[g[i].b[j]].t2seqt[t]])
					# if vehicle[i].vel[j] - vehicle[g[i].b[j]].vel[vehicle[g[i].b[j]].t2seqt[t]] != 0:
					# 	data.append((vehicle[g[i].b[j]].y[vehicle[g[i].b[j]].t2seqt[t]] - vehicle[i].y[j]) / (vehicle[i].vel[j] - vehicle[g[i].b[j]].vel[vehicle[g[i].b[j]].t2seqt[t]]))
					# else:
					# 	data.append(0)
					# data.append(vehicle[g[i].b[j]].vtype)
				else:
					for k in range(ref_feat):
						data.append(0)
			else:
				for k in range(ref_feat):
					data.append(0)

			if g[i].br[j] != -1 and g[i].br[j] < len(vehicle[i].x):
				if t in vehicle[g[i].br[j]].t:
					data.append(vehicle[g[i].br[j]].x[vehicle[g[i].br[j]].t2seqt[t]] - vehicle[i].x[j])
					data.append(vehicle[g[i].br[j]].y[vehicle[g[i].br[j]].t2seqt[t]] - vehicle[i].y[j])
					if include_behavior:
						data.append(behavior[g[i].br[j]])
					# data.append(vehicle[g[i].br[j]].lat_vel[vehicle[g[i].br[j]].t2seqt[t]])
					# data.append(vehicle[i].vel[j] - vehicle[g[i].br[j]].vel[vehicle[g[i].br[j]].t2seqt[t]])
					# if vehicle[i].vel[j] - vehicle[g[i].br[j]].vel[vehicle[g[i].br[j]].t2seqt[t]] != 0:
					# 	data.append((vehicle[g[i].br[j]].y[vehicle[g[i].br[j]].t2seqt[t]] - vehicle[i].y[j]) / (vehicle[i].vel[j] - vehicle[g[i].br[j]].vel[vehicle[g[i].br[j]].t2seqt[t]]))
					# else:
					# 	data.append(0)
					# data.append(vehicle[g[i].br[j]].vtype)
				else:
					for k in range(ref_feat):
						data.append(0)
			else:
				for k in range(ref_feat):
					data.append(0)

			if g[i].bl[j] != -1 and g[i].bl[j] < len(vehicle[i].x):
				if t in vehicle[g[i].bl[j]].t:
					data.append(vehicle[g[i].bl[j]].x[vehicle[g[i].bl[j]].t2seqt[t]] - vehicle[i].x[j])
					data.append(vehicle[g[i].bl[j]].y[vehicle[g[i].bl[j]].t2seqt[t]] - vehicle[i].y[j])
					if include_behavior:
						data.append(behavior[g[i].bl[j]])
					# data.append(vehicle[g[i].bl[j]].lat_vel[vehicle[g[i].bl[j]].t2seqt[t]])
					# data.append(vehicle[i].vel[j] - vehicle[g[i].bl[j]].vel[vehicle[g[i].bl[j]].t2seqt[t]])
					# if vehicle[i].vel[j] - vehicle[g[i].bl[j]].vel[vehicle[g[i].bl[j]].t2seqt[t]] != 0:
					# 	data.append((vehicle[g[i].bl[j]].y[vehicle[g[i].bl[j]].t2seqt[t]] - vehicle[i].y[j]) / (vehicle[i].vel[j] - vehicle[g[i].bl[j]].vel[vehicle[g[i].bl[j]].t2seqt[t]]))
					# else:
					# 	data.append(0)
					# data.append(vehicle[g[i].bl[j]].vtype)
				else:
					for k in range(ref_feat):
						data.append(0)
			else:
				for k in range(ref_feat):
					data.append(0)
			batch_f.append(data)
			batch_l.append(labels)
			if len(batch_f) == seq_len:
				ids.append(i)
				dataset.append(batch_f)
				targets.append(batch_l)
				t_data.append(batch_t)

dataset = np.asarray(dataset)
targets = np.asarray(targets)
t_data = np.asarray(t_data)
ids = np.asarray(ids)

print(dataset.shape)
print(targets.shape)
print(t_data.shape)
print(len(ids))

np.save('lstm_inputs.npy', dataset)
np.save('lstm_labels.npy', targets)
# np.save('lstm_timesteps.npy', t_data)
# np.save('ids.npy', ids)

print('Done!\n')