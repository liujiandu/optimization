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

if __name__=="__main__":
        import sys
        sys.path.append('../')
        from tool.plot import plot3d
        plot3d(func, np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
