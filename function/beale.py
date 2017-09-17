import numpy as np

def func(x):
	return (1.5-x[0]+x[0]*x[1])**2 \
				+ (2.25-x[0]+x[0]*x[1]**2)**2 \
				+ (2.625-x[0]+x[0]*x[1]**3)**2 
def gfunc(x):
	return (func(x+0.001)-func(x))/0.001
