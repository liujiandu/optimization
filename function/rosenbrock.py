import numpy as np

def func(x):
	sum = 0.0
	n = x.shape[0]
	for i in range(n-1):
		sum+=(100*(x[i+1]-x[i]**2)**2+(1-x[i])**2)
	
	return sum

def gfunc(x):
	return (func(x+0.001)-func(x))/0.001

if __name__=="__main__":
        import sys
        sys.path.append('../')
        from tool.plot import plot3d
        plot3d(func, np.array([-2, -2]), np.array([2,3]))
