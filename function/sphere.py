import numpy as np

def func(x):
	sum = 0.0
	n = x.shape[0]
	for i in range(n):	
		sum+=x[i]**2
	return sum

def gfunc(x):
	return 2*x

def hessian(x):
	return np.mat(np.eye(2))
