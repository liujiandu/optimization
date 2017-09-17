import numpy as np

def func(x):
	sum = 0.0
	n = x.shape[0]
	for i in range(n-1):
		sum+=(100*(x[i+1]-x[i]**2)**2+(x[i]+1)**2)
	
	return sum

def gfunc(x):
	return (func(x+0.001)-func(x))/0.001
