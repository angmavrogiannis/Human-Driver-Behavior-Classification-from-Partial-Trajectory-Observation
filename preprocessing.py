import numpy as np
import pandas as pd
import pickle

def calcLongLateral(x,acc,dt):
	span = len(x)
	lat_vel = np.zeros(span)
	lat_acc = np.zeros(span)
	lat_jerk = np.zeros(span)
	long_jerk = np.zeros(span)
	for i in range(2, len(x)):
		lat_vel[i] = (x[i] - x[i-1]) / dt
		lat_acc[i] = (lat_vel[i] - lat_vel[i-1]) / dt
		lat_jerk[i] = (lat_acc[i] - lat_acc[i-1]) / dt
		long_jerk[i] = (acc[i] - acc[i-1]) / dt
	return lat_vel, lat_acc, lat_jerk, long_jerk

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

#framerate in seconds
dt = 0.1

#feet to meter conversion parameter
ft2m = 0.3048

#vehicle ID to vehicle class object mapping
id2obj = {}

#time stamp to sequential timestep mapping
t2seqt = {}

#lane width in meters (converted from 12 feet)
lane_width = 3.6576
lane_centers = np.multiply(lane_width / 2, [1,3,5,7,9,11,13])

#vicinity parameter in meters
near = 20

#velocity threshold for space headway (m/s)
vthresh = 11

vehicle = []

#arrange vehicle data in vehicle class
print('\nArranging data into Vehicle class...\n')
c = 0
for v in vehicleIDs:
	id2obj[v] = c
	num_frames = df.loc[df['Vehicle_ID'] == v].values[0,2]
	time = df.loc[df['Vehicle_ID'] == v].values[:,3]
	local_x = df.loc[df['Vehicle_ID'] == v].values[:,4] * ft2m
	local_y = df.loc[df['Vehicle_ID'] == v].values[:,5] * ft2m
	v_length = df.loc[df['Vehicle_ID'] == v].values[0,8] * ft2m
	v_width = df.loc[df['Vehicle_ID'] == v].values[0,9] * ft2m
	v_class = df.loc[df['Vehicle_ID'] == v].values[0,10]
	v_vel = df.loc[df['Vehicle_ID'] == v].values[:,11] * ft2m
	v_acc = df.loc[df['Vehicle_ID'] == v].values[:,12] * ft2m
	lane_id = df.loc[df['Vehicle_ID'] == v].values[:,13]
	preceding = df.loc[df['Vehicle_ID'] == v].values[:,14]
	following = df.loc[df['Vehicle_ID'] == v].values[:,15]
	space_headway = df.loc[df['Vehicle_ID'] == v].values[:,16] * ft2m
	time_headway = df.loc[df['Vehicle_ID'] == v].values[:,17]
	vehicle.append(Vehicle(v,num_frames,time,local_x,local_y,v_length,v_width,v_class,v_vel,v_acc,lane_id,preceding,following,space_headway,time_headway))
	c += 1

print('Mapping timesteps to individual vehicle time sequences...\n')
for v in vehicle:
	c = 0
	for t in v.t:
		v.t2seqt[t] = c
		c += 1

timemap = {}
#find active vehicles for every timestep, identify neighbors using vicinity parameter, and calculate average velocity to neighbors
print('Finding active vehicles for every timestep, identifying neighbors and calculating average velocity relative to neighbors...\n')
vrel_avg = []
c = 0
for t in timesteps:
	activeIDs = np.asarray(np.unique(df.loc[df['Global_Time'] == t].values[:,0]),dtype=int)
	timemap[t] = activeIDs
	for v in activeIDs:
		vrel = []
		ego_id = id2obj[v]
		for v2 in activeIDs:
			ref_id = id2obj[v2] #added np.abs on next line
			if v != v2 and np.abs(vehicle[ref_id].y[vehicle[ref_id].t2seqt[t]] - vehicle[ego_id].y[vehicle[ego_id].t2seqt[t]]) <= near:
				vrel.append(vehicle[ego_id].vel[vehicle[ego_id].t2seqt[t]] - vehicle[ref_id].vel[vehicle[ref_id].t2seqt[t]])
		if len(vrel) != 0:
			vehicle[ego_id].vrel_avg.append(np.sum(vrel) / len(vrel))

#calculate lateral velocity, acceleration and jerk, and longitudinal jerk and assign them to the vehicle objects
#calculate relative velocity of a vehicle to its leading vehicle

print('Calculating lateral velocity, acceleration and jerk, longitudinal jerk and relative velocity to leading vehicle...\n')
for obj in vehicle:
	vrel2front = []
	shead = []
	obj.lat_vel, obj.lat_acc, obj.lat_jerk, obj.long_jerk = calcLongLateral(obj.x,obj.acc,dt)
	for precID,vel,t,x,l_id,y,sh in zip(obj.prec,obj.vel,obj.t,obj.x,obj.lane,obj.y,obj.shead):
		if precID != 0 and precID in vehicleIDs and t in vehicle[id2obj[precID]].t2seqt:
			if np.abs(y - vehicle[id2obj[precID]].y[vehicle[id2obj[precID]].t2seqt[t]]) <= near:
				obj.vrel2front.append(vel - vehicle[id2obj[precID]].vel[vehicle[id2obj[precID]].t2seqt[t]])
				if vel > vthresh:
					shead.append(sh)
		obj.lat_dev.append(x - lane_centers[int(l_id)-1])
	obj.shead = shead

vehicleClass = open('vehicle.obj', 'wb')
pickle.dump(vehicle, vehicleClass)
print('Done!\n')