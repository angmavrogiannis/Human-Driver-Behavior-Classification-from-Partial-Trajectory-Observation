import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from itertools import combinations
import seaborn as sns
import pickle
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

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

def plot(y, xaxis, yaxis):
	size = len(y)
	x = np.linspace(1, size, num=size)
	plt.plot(x, y)
	plt.xlabel(xaxis)
	plt.ylabel(yaxis)
	plt.show()

def runPCA(data, num_components):
	pca = PCA(n_components=num_components)
	pca.fit(data)
	print('Principal Components: ' , pca.components_)
	print('Explained Variance: ', pca.explained_variance_)
	print('Explained Variance Ratio: ', pca.explained_variance_ratio_)
	return pca.components_

def scatter(x, y, color, title, xlabel, ylabel):
	plt.scatter(x, y, s=10, c=color, alpha=0.5)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

#loading vehicle object
filehandler = open('vehicle.obj', 'rb')
vehicle = pickle.load(filehandler)

num_features = 10
slice_num = 200
num_drivers = len(vehicle)
num_components = 2

for i in range(num_drivers):
	vehicle[i].lat_vel = np.abs(vehicle[i].lat_vel)
	vehicle[i].lat_acc = np.abs(vehicle[i].lat_acc)
	vehicle[i].lat_dev = np.abs(vehicle[i].lat_dev)
	vehicle[i].acc = np.abs(vehicle[i].acc)
	vehicle[i].shead = list(filter((0.0).__ne__, vehicle[i].shead))
	if all(v == 0 for v in vehicle[i].shead):
		vehicle[i].shead = 10
	if len(vehicle[i].vrel2front) == 0:
		vehicle[i].vrel2front.append(0)
	vehicle[i].lat_vel = vehicle[i].lat_vel[(vehicle[i].lat_vel < 20)]

f = [[] for i in range(num_features)]

for i in range(num_drivers):
	f[0].append(np.mean(vehicle[i].lat_vel))
	f[1].append(np.max(vehicle[i].lat_vel))
	f[2].append(np.var(vehicle[i].lat_vel))
	f[3].append(np.mean(vehicle[i].vel))
	f[4].append(np.max(vehicle[i].vel))
	f[5].append(np.var(vehicle[i].vel))
	f[6].append(np.mean(vehicle[i].lat_acc))
	f[7].append(np.mean(vehicle[i].acc))
	f[8].append(np.mean(vehicle[i].vrel_avg))
	f[9].append(np.mean(vehicle[i].vrel2front))
	# f[8].append(np.mean(vehicle[i].shead))
	# f[9].append(np.min(vehicle[i].shead))
	# f[10].append(np.mean(vehicle[i].lat_dev))
	# f[11].append(np.mean(vehicle[i].vrel_avg))

# plt.bar(np.linspace(1, len(f[1]), num=len(f[1])), f[1])
# plt.show()

f = np.asarray(f)
x = np.transpose(f)

# Scale data to 0 mean
x = StandardScaler().fit_transform(x)

pcomponents = runPCA(x, num_components)

rec = np.matmul(pcomponents, f)
plt.scatter(rec[0,:], rec[1,:], s=10, c='black')
plt.xlabel('Principal Component 1', fontsize=10)
plt.ylabel('Principal Component 2', fontsize=10)
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# fig, ax = plt.subplots(figsize=(10, 5))
# ax.scatter(rec[0,:], rec[1,:])

# #adds a title and axes labels
# ax.set_title('Distance vs Workout Duration')
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')

# #removing top and right borders
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# #adds major gridlines
# ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.show()

# rec = np.matmul(pcomponents, f)
# plt.scatter(rec[0,:], rec[2,:], s=10, c='black', alpha=0.5)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 3')
# plt.show()

# Elbow method
# wcss = []
# for i in range(1, 8):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(np.transpose(rec))
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 8), wcss, linewidth=3)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
# plt.show()

# K-means
X = np.transpose(rec)
num_clusters = 4
c0_coords = [[], []]
c1_coords = [[], []]
c2_coords = [[], []]
c3_coords = [[], []]
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
behavior = {}

for i in range(num_drivers):
	behavior[i] = pred_y[i] + 1
	if pred_y[i] == 0:
		c0_coords[0].append(X[i,0])
		c0_coords[1].append(X[i,1])
	elif pred_y[i] == 1:
		c1_coords[0].append(X[i,0])
		c1_coords[1].append(X[i,1])
	elif pred_y[i] == 2:
		c2_coords[0].append(X[i,0])
		c2_coords[1].append(X[i,1])
	elif pred_y[i] == 3:
		c3_coords[0].append(X[i,0])
		c3_coords[1].append(X[i,1])
a = plt.scatter(c0_coords[0], c0_coords[1], s=10, c='tab:blue')
b = plt.scatter(c1_coords[0], c1_coords[1], s=10, c='tab:green')
c = plt.scatter(c2_coords[0], c2_coords[1], s=10, c='tab:red')
d = plt.scatter(c3_coords[0], c3_coords[1], s=10, c='tab:orange')
plt.legend((a, b, c, d), ('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'))
#plt.scatter(X[:,0], X[:,1], s=10, c='blue', alpha=0.5)
print('Cluster centers: ', kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.show()

# Shilouette method
# sil = []
# kmax = 10
# for k in range(2, kmax+1):
#   kmeans = KMeans(n_clusters = k).fit(x)
#   labels = kmeans.labels_
#   sil.append(silhouette_score(x, labels, metric = 'euclidean'))
# plt.plot(np.linspace(2, 10, num=9), sil, linewidth=3)
# plt.title('Silhouette Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette score')
# plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
# plt.show()

# den = [0, 0, 0, 0, 0]
# for i in range(len(pred_y)):
# 	den[pred_y[i]] += 1

# c0 = [[] for i in range(num_features)]
# c1 = [[] for i in range(num_features)]
# c2 = [[] for i in range(num_features)]
# c3 = [[] for i in range(num_features)]
# for i in range(num_drivers):
# 	k = pred_y[i]
# 	if k == 0:
# 		for j in range(num_features):
# 			c0[j].append(f[j][i])
# 	elif k == 1:
# 		for j in range(num_features):
# 			c1[j].append(f[j][i])
# 	elif k == 2:
# 		for j in range(num_features):
# 			c2[j].append(f[j][i])
# 	elif k == 3:
# 		for j in range(num_features):
# 			c3[j].append(f[j][i])
# mean0 = []
# mean1 = []
# mean2 = []
# mean3 = []
# std0 = []
# std1 = []
# std2 = []
# std3 = []
# for j in range(num_features):
# 	mean0.append(np.mean(c0[j]))
# 	mean1.append(np.mean(c1[j]))
# 	mean2.append(np.mean(c2[j]))
# 	mean3.append(np.mean(c3[j]))
# 	std0.append(np.std(c0[j]))
# 	std1.append(np.std(c1[j]))
# 	std2.append(np.std(c2[j]))
# 	std3.append(np.std(c3[j]))
# ctes = [mean2, mean1, mean3, mean0]
# error = [std2, std1, std3, std0]
# bx = np.arange(4)

# cluster_titles = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
# fig, ax = plt.subplots()
# rects1 = ax.bar(np.array(0), mean2[4] , yerr=error[0][4], align='center', label='Cluster 1')
# rects2 = ax.bar(np.array(1), mean1[4], yerr=error[1][4], align='center', label='Cluster 2')
# rects3 = ax.bar(np.array(2), mean3[4], yerr=error[2][4], align='center', label='Cluster 3')
# rects4 = ax.bar(np.array(3), mean0[4], yerr=error[3][4], align='center', label='Cluster 4')
# ax.xaxis.grid(True)
# ax.yaxis.grid(True)

# ax.set_ylabel('Maximum longitudinal velocity (m/s)')
# ax.set_title('Maximum longitudinal velocity by cluster')
# ax.set_xticks(bx)
# ax.set_xticklabels(cluster_titles)
# ax.legend()
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots()
# rects1 = ax.bar(np.array(0), mean2[1] , yerr=error[0][1], align='center', label='Cluster 1')
# rects2 = ax.bar(np.array(1), mean1[1], yerr=error[1][1], align='center', label='Cluster 2')
# rects3 = ax.bar(np.array(2), mean3[1], yerr=error[2][1], align='center', label='Cluster 3')
# rects4 = ax.bar(np.array(3), mean0[1], yerr=error[3][1], align='center', label='Cluster 4')
# ax.xaxis.grid(True)
# ax.yaxis.grid(True)

# ax.set_ylabel('Maximum lateral velocity (m/s)')
# ax.set_title('Maximum lateral velocity by cluster')
# ax.set_xticks(bx)
# ax.set_xticklabels(cluster_titles)
# ax.legend()
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots()
# rects1 = ax.bar(np.array(0), mean2[8] , yerr=error[0][8], align='center', label='Cluster 1')
# rects2 = ax.bar(np.array(1), mean1[8], yerr=error[1][8], align='center', label='Cluster 2')
# rects3 = ax.bar(np.array(2), mean3[8], yerr=error[2][8], align='center', label='Cluster 3')
# rects4 = ax.bar(np.array(3), mean0[8], yerr=error[3][8], align='center', label='Cluster 4')
# ax.xaxis.grid(True)
# ax.yaxis.grid(True)

# ax.set_ylabel('Relative velocity to traffic flow (m/s)')
# ax.set_title('Relative velocity to traffic flow by cluster')
# ax.set_xticks(bx)
# ax.set_xticklabels(cluster_titles)
# ax.legend()
# plt.tight_layout()
# plt.show()

# plot trajectories
# plt.scatter(vehicle[0].x,vehicle[0].y, s=10, c='green', alpha=0.5)
# plt.scatter(vehicle[270].x,vehicle[270].y, s=10, c='red', alpha=0.5)
# plt.xlabel('x coordinates')
# plt.ylabel('y coordinates')
# plt.show()

# Logistic Regression
print('Training Logistic Regression...')
x_train, x_test, y_train, y_test = train_test_split(x, pred_y, test_size=0.2, random_state=0)
logReg = LogisticRegression(solver='liblinear',max_iter=500, verbose=1)
# lr_time = []
# lr_accuracy = []
# start = time.time()
logReg.fit(x_train, y_train)
pred = logReg.predict(x_test)
# end = time.time()
# print('Time for logistic regression: ', end - start, ' seconds.')
score = logReg.score(x_test, y_test)
# print('Accuracy: ', score)
# lr_time.append(end - start)
# lr_accuracy.append(logReg.score(x_test, y_test))
lr_accuracy = logReg.score(x_test, y_test)


# K-Nearest Neighbors with k-fold cross validation
print('Training K-Nearest Neighbors...')
knn = KNeighborsClassifier(n_neighbors=3)
#param_grid = {'n_neighbors': np.arange(1, 40)}
#knn_gscv = GridSearchCV(knn, param_grid, cv=5)
# x_train = np.matmul(pcomponents, np.transpose(x_train)) 
# x_train = np.transpose(x_train)
# knn_time = []
# knn_accuracy = []

# start = time.time()
knn.fit(x_train, y_train)
knn.predict(x_test)
# end = time.time()
# print('Time for KNN: ', end - start, ' seconds.')

#print('Best number of neighbors: ', knn_gscv.best_params_)
#print('Mean score from KNN with CV: ', knn_gscv.best_score_)
print('Accuracy: ', knn.score(x_test, y_test))
#knn_time.append(end - start)

class MLP(torch.nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(MLP, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
		self.relu = torch.nn.ReLU()
		self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
		self.fc3 = torch.nn.Linear(self.hidden_size, self.output_size)

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		return x

input_size = num_features
hidden_size = 128
output_size = num_clusters

x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train,dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.long)

lr = 0.001
num_epochs = 200
mlp = MLP(input_size, hidden_size, output_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr = lr)

# mlp_time = []
# mlp_accuracy = []

#start = time.time()
for epoch in range(num_epochs):

	optimizer.zero_grad()

	y_pred = mlp(x_train)
	loss = criterion(y_pred, y_train)
	print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
	loss.backward()
	optimizer.step()

predictions = []
with torch.no_grad():
	for example in x_test:
		y_pred = mlp(example)
		predictions.append(np.argmax(y_pred))
#end = time.time()
#mlp_time.append(end - start)
#print('Time for MLP: ', end - start, 'time.')
num_correct = 0
for i in range(len(y_test)):
	if y_test[i] == predictions[i]:
		num_correct += 1
print('Accuracy: ', num_correct / len(y_test))

# with open('behavior.pkl', 'wb') as f:
# 	pickle.dump(behavior,f)

print('Done!\n')
#mlp_accuracy.append(num_correct / len(y_test))


# method_titles = ['KNN', 'Logistic Regression', 'Multilayer Perceptron']
# fig, ax = plt.subplots()
# rects1 = ax.bar(np.array(0), np.mean(knn_time), yerr=np.std(knn_time), align='center', label='KNN')
# rects2 = ax.bar(np.array(1), np.mean(lr_time), yerr=np.std(lr_time), align='center', label='Logistic Regression')
# rects3 = ax.bar(np.array(2), np.mean(mlp_time), yerr=np.std(mlp_time), align='center', label='Multilayer Perceptron')
# ax.xaxis.grid(True)
# ax.yaxis.grid(True)

# ax.set_ylabel('Runtime (s)')
# ax.set_title('Runtime by algorithm')
# ax.set_xticks(np.arange(3))
# ax.set_xticklabels(method_titles)
# ax.legend()
# #plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots()
# rects1 = ax.bar(np.array(0), np.mean(knn_accuracy), yerr=np.std(knn_accuracy), align='center', label='KNN')
# rects2 = ax.bar(np.array(1), np.mean(lr_accuracy), yerr=np.std(lr_accuracy), align='center', label='Logistic Regression')
# rects3 = ax.bar(np.array(2), np.mean(mlp_accuracy), yerr=np.std(mlp_accuracy), align='center', label='Multilayer Perceptron')
# ax.xaxis.grid(True)
# ax.yaxis.grid(True)

# ax.set_ylabel('Accuracy (%)')
# ax.set_title('Accuracy by algorithm')
# ax.set_xticks(np.arange(3))
# ax.set_xticklabels(method_titles)
# ax.legend()
# plt.tight_layout()
# plt.show()