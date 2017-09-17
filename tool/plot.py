import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot3d(func,min, max):
	fig = plt.figure()
	ax = Axes3D(fig)
	X = np.arange(min[0], max[0], 0.1)
	Y = np.arange(min[1], max[1], 0.1)
	X, Y = np.meshgrid(X, Y)
	Z = func(np.array([X,Y]))
	
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
	plt.show()

def plot_curve(X, Y):
	fig = plt.figure()
	plt.plot(X, Y, 'b*')
	plt.plot(X, Y, 'r')
	plt.show()

def plot_surface_curve(func,min, max, x, y,z):
	fig = plt.figure()
	ax = Axes3D(fig)
	X = np.arange(min[0], max[0], 0.1)
	Y = np.arange(min[1], max[1], 0.1)
	X, Y = np.meshgrid(X, Y)
	Z = func(np.array([X,Y]))
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')


	ax.plot(x, y, z, 'b*')
	ax.plot(x, y, z, 'r')

	plt.show()
