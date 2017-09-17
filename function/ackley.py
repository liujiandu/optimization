import numpy as np

def func(x):
	y1 = -20.0*np.exp(-0.2*np.sqrt(0.5*(x[0]**2+x[1]**2)))
	y2 = -np.exp(0.5*(np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1])))+  np.exp(1) + 20.0
	return y1+y2

def gfunc(x):
	return (func(x+0.001)-func(x))/0.001
if __name__ == "__main__":
	print func(np.array([0,0]))
