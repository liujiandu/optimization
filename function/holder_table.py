import numpy as np

def func(x):
	return -abs(np.sin(x[0])*np.cos(x[1])*np.exp(abs(1-np.sqrt(x[0]**2+x[1]**2)/np.pi)))

def gfunc(x):
	return (func(x+0.001)-func(x))/0.001

if __name__ == "__main__":
        import sys
        sys.path.append('../')
        from tool.plot import plot3d
        plot3d(func, np.array([-5.0, 5.0]), np.array([5.12, 5.12]))
	#print func(np.array([8.05502, 9.66459]))
