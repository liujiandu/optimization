import numpy as np

def func(x):
	return -(x[1]+47.0)*np.sin(np.sqrt(abs(0.5*x[0]+x[1]+47.0)))\
				-x[0]*np.sin(np.sqrt(abs(x[0]-(x[1]+47.0))))

def gfunc(x):
	return (func(x+0.001)-func(x))/0.001


if __name__ == "__main__":
	print func(np.array([512.0,404.2319]))
