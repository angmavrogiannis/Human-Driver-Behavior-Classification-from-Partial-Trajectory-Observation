import pickle
import numpy as np
import pandas as pd
from preprocessing import Vehicle

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
print(len(vehicle))

input_file1 = 'trajectories-0400-0415.csv'
input_file2 = 'trajectories-0500-0515.csv'
input_file3 = 'trajectories-0515-0530.csv'

df1 = pd.read_csv(input_file1)
df2 = pd.read_csv(input_file2)
df3 = pd.read_csv(input_file3)
frames = [df1, df2, df3]
df = pd.concat(frames)
data = np.array(df)

#identify unique vehicle ID's from given data
vehicleIDs = np.asarray(np.unique(data[:,0]),dtype=int)
timesteps = np.asarray(np.unique(data[:,3]),dtype=int)

print('Mapping timesteps to individual vehicle time sequences...\n')
for v in vehicle:
	c = 0
	for t in v.t:
		v.t2seqt[t] = c
		c += 1

id2obj = {0:-1}
c = 0
for v in vehicleIDs:
	id2obj[v] = c
	c += 1

# Finding active ID's
timemap = {}
for t in timesteps:
	activeIDs = np.asarray(np.unique(df.loc[df['Global_Time'] == t].values[:,0]),dtype=int)
	timemap[t] = activeIDs

gridVehicle = []

for i in range(len(vehicle)):
	print('Vehicle: ', i)
	ff = []
	f = []
	fr = []
	fl = []
	l = []
	r = []
	b = []
	br = []
	bl = []
	for t in vehicle[i].t:
		l_cand = {}
		r_cand = {}
		ego_t = vehicle[i].t2seqt[t]
		for ref_id in timemap[t]:
			ref = id2obj[ref_id]
			ref_t = vehicle[ref].t2seqt[t]
			if vehicle[ref].lane[ref_t] + 1 == vehicle[i].lane[ego_t]:
				l_cand[ref] = np.abs(vehicle[i].y[ego_t] - vehicle[ref].y[ref_t])
			elif vehicle[ref].lane[ref_t] - 1 == vehicle[i].lane[ego_t]:
				r_cand[ref] = np.abs(vehicle[i].y[ego_t] - vehicle[ref].y[ref_t])
			if len(l_cand) == 0:
				l_cand[-1] = -1
			if len(r_cand) == 0:
				r_cand[-1] = -1
		l_id = min(l_cand, key=l_cand.get)
		r_id = min(r_cand, key=r_cand.get)
		l.append(l_id)
		r.append(r_id)
		if vehicle[i].prec[ego_t] in id2obj:
			f_id = id2obj[vehicle[i].prec[ego_t]]
		else:
			f_id = -1
		f.append(f_id)
		if vehicle[i].foll[ego_t] in id2obj:
			b.append(id2obj[vehicle[i].foll[ego_t]])
		else:
			b.append(-1)
		if t in vehicle[f_id].t:
			ff.append(vehicle[f_id].prec[vehicle[f_id].t2seqt[t]])
		else:
			ff.append(-1)
		if t in vehicle[r_id].t:
			fr.append(vehicle[r_id].prec[vehicle[r_id].t2seqt[t]])
			br.append(vehicle[r_id].foll[vehicle[r_id].t2seqt[t]])
		else:
			fr.append(-1)
			br.append(-1)
		if t in vehicle[l_id].t:
			fl.append(vehicle[l_id].prec[vehicle[l_id].t2seqt[t]])
			bl.append(vehicle[l_id].foll[vehicle[l_id].t2seqt[t]])
		else:
			fl.append(-1)
			bl.append(-1)
	gridVehicle.append(GridVehicle(ff, f, fr, fl, l, r, b, br, bl))


# # Additions for prediction
# print('Extracting features for prediction...\n')
# c = 0
# for obj in vehicle:
# 	print('Vehicle: ', c)
# 	unq_c = 0
# 	xrel = []
# 	yrel = []
# 	vx = []
# 	vyrel = []
# 	ttc = []
# 	vtype = []
# 	ref_cars = {}
# 	for t, y, x, vel, xvel, lane in zip(obj.t, obj.y, obj.x, obj.vel, obj.lat_vel, obj.lane):
# 		for ref_id in timemap[t]:
# 			if ref_id != obj.id and t in vehicle[id2obj[ref_id]].t:
# 				ref = id2obj[ref_id]
# 				if np.abs(vehicle[ref].lane[vehicle[ref].t2seqt[t]] - lane) <= 1 and np.abs(vehicle[ref].y[vehicle[ref].t2seqt[t]] - y) <= near:
# 					if ref_id not in ref_cars:
# 						ref_cars[ref_id] = unq_c
# 						vtype.append(vehicle[ref].vtype)
# 						xrel.append({})
# 						yrel.append({})
# 						vx.append({})
# 						vyrel.append({})
# 						ttc.append({})
# 						unq_c += 1
# 					xrel[ref_cars[ref_id]][obj.t2seqt[t]] = vehicle[ref].x[vehicle[ref].t2seqt[t]] - x
# 					yrel[ref_cars[ref_id]][obj.t2seqt[t]] = vehicle[ref].y[vehicle[ref].t2seqt[t]] - y
# 					vx[ref_cars[ref_id]][obj.t2seqt[t]] = vehicle[ref].lat_vel[vehicle[ref].t2seqt[t]]
# 					vyrel[ref_cars[ref_id]][obj.t2seqt[t]] = vel - vehicle[ref].vel[vehicle[ref].t2seqt[t]]
# 					if vyrel[ref_cars[ref_id]][obj.t2seqt[t]] != 0:
# 						ttc[ref_cars[ref_id]][obj.t2seqt[t]] = yrel[ref_cars[ref_id]][obj.t2seqt[t]] / vyrel[ref_cars[ref_id]][obj.t2seqt[t]]
# 	gridVehicle[c].xrel = xrel
# 	gridVehicle[c].yrel = yrel
# 	gridVehicle[c].vx = vx
# 	gridVehicle[c].vyrel = vyrel
# 	gridVehicle[c].ttc = ttc
# 	gridVehicle[c].types = vtype
# 	gridVehicle[c].ref_cars = ref_cars
# 	c += 1

gridVehicleClass = open('gridVehicle.obj', 'wb')
pickle.dump(gridVehicle, gridVehicleClass)