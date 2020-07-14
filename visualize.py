import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from IPython.display import display, clear_output, HTML, Image

animation.rcParams['animation.writer'] = 'ffmpeg'

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

filehandler = open('vehicle.obj', 'rb')
vehicle = pickle.load(filehandler)

plt.style.use('seaborn-pastel')

fig = plt.figure()
ax = plt.axes(xlim=(0, 20), ylim=(0, 400))
num_lanes = 7
lane_width = 3.7
map_l = 50
cars = []
lines = []

ref_cars = []
ego = 1500
ego_s = vehicle[ego].t[0]
ego_f = vehicle[ego].t[-1]
ego_t = vehicle[ego].t

x_ref = []
y_ref = []

c = 0
for i in range(len(vehicle)):
	if i != ego:
		ref_s = vehicle[i].t[0]
		ref_f = vehicle[i].t[-1]
		if (ref_s >= ego_s and ref_f <= ego_f or ref_s <= ego_s and ref_f >= ego_f
		or ref_s <= ego_s and ref_f >= ego_s or ref_s <= ego_f and ref_f >= ego_f):
			ref_cars.append(i)
			c += 1
			x_ref.append([])
			y_ref.append([])
			for j in range(len(ego_t)):
				if ego_t[j] in vehicle[i].t:
					curr_t = vehicle[i].t2seqt[ego_t[j]]
					if np.abs(vehicle[ego].y[j] - vehicle[i].y[curr_t]) < map_l:
						x_ref[c-1].append(vehicle[i].x[curr_t])
						y_ref[c-1].append(vehicle[i].y[curr_t])
					else:
						x_ref[c-1].append(0)
						y_ref[c-1].append(0)
				else:
					x_ref[c-1].append(0)
					y_ref[c-1].append(0)

size = len(x_ref)
i = 0
while i < size:
	if all(val == 0 for val in x_ref[i]):
		x_ref.pop(i)
		y_ref.pop(i)
		ref_cars.pop(i)
		size -= 1
	else:
		i += 1

num_vehicles = len(ref_cars)
cars.append(ax.plot([], [], 'ro'))
for i in range(num_vehicles):
	cars.append(ax.plot([], [], 'ko'))
for i in range(num_lanes):
	lines.append(ax.plot([], [], 'black'))

x = [[], []]
y = [[], []]
x[0] = vehicle[0].x
x[1] = vehicle[1].x
y[0] = vehicle[0].y
y[1] = vehicle[1].y

def init():
	for i, car in enumerate(cars):
		if i == 0:
			car[0].set_data([vehicle[ego].x[0]], [vehicle[ego].y[0]])
		else:
			if x_ref[i-1][0] != 0 and y_ref[i-1][0] != 0:
				car[0].set_data(x_ref[i-1][0], y_ref[i-1][0])
			else:
				car[0].set_data([], [])
	for i, line in enumerate(lines):
		line[0].set_data([i * lane_width, i * lane_width], [-map_l, map_l])
	return cars, lines

def animate(i):
    for j, car in enumerate(cars):
    	if j == 0:
	    	car[0].set_data(vehicle[ego].x[i], vehicle[ego].y[i])
    	else:
    		if x_ref[j-1][0] != 0 and y_ref[j-1][0] != 0:
    			car[0].set_data(x_ref[j-1][i], y_ref[j-1][i])
    		else:
    			car[0].set_data([], [])
    for j, line in enumerate(lines):
    	line[0].set_data([j * lane_width, j * lane_width], [vehicle[ego].y[i] - map_l, vehicle[ego].y[i] + map_l])
    ax.set_xlim(0, num_lanes * lane_width)
    ax.set_ylim(vehicle[ego].y[i] - map_l, vehicle[ego].y[i] + map_l)
    ax.set_xlabel('Lateral Position (m)')
    ax.set_ylabel('Longitudinal Position (m)')
    cars[0][0].set_label('ego vehicle')
    cars[1][0].set_label('reference vehicles')
    ax.legend()
    return cars, lines

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=int(vehicle[ego].frames), interval=10, blit=False)
#anim.save('aggro.mp4', writer='ffmpeg', fps=10, dpi=100)
plt.show()