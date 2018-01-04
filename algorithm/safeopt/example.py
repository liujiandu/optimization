import GPy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import safeopt
noise_var=0.05**2
bounds = [(-10, 10)]
parameter_set = safeopt.linearly_spaced_combinations(bounds, 1000)
kernel = GPy.kern.RBF(input_dim=len(bounds), variance=2, lengthscale=1.0, ARD=True)

def sample_safe_fun():
    while True:
        fun = safeopt.sample_gp_function(kernel, bounds, noise_var, 100)
        if fun(0, noise=False)>0.5:
            break
    return fun

x0 = np.zeros((1, len(bounds)))
fun = sample_safe_fun()

gp = GPy.models.GPRegression(x0, fun(x0), kernel, noise_var=noise_var)
opt = safeopt.SafeOptSwarm(gp, 0., bounds=bounds, threshold=0.2)

def plot_gp():
    opt.plot(1000)
    plt.plot(parameter_set, fun(parameter_set),  alpha=0.3)

for _ in range(100):
    x_next = opt.optimize()
    y_means = fun(x_next)
    opt.add_new_data_point(x_next, y_means)
plot_gp()
plt.show()

