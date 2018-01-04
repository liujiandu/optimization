import numpy as np

def alg(init_x , func, gfunc, lr = 0.01, max_iter=100):
	lr = lr
	epsilon = 1e-8
	max_iter  = max_iter

	x = init_x.copy()
	fk = []
	X = []
	count = 0
	while count < max_iter:
		grad = gfunc(x)
		if abs(grad).max()<epsilon:
			break
		x = x-lr*grad
		count+=1
		fk.append(func(x))
		X.append(x)
	return (np.array(fk), np.array(X))
