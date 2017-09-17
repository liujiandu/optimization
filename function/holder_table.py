import numpy as np

def func(x):
	return -abs(np.sin(x[0])*np.cos(x[1])*np.exp(abs(1-np.sqrt(x[0]**2+x[1]**2)/np.pi)))

def gfunc(x):
	return (func(x+0.001)-func(x))/0.001

if __name__ == "__main__":
	print func(np.array([8.05502, 9.66459]))
