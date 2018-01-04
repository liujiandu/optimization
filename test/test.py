import sys
sys.path.append('../')
import numpy as np

from function import rastrigin
from function import ackley
from function import sphere
from function import rosenbrock
from function import beale
from function import eggholder
from function import holder_table

from tool.plot import plot3d, plot_curve, plot_surface_curve

from algorithm import steepest_descent
from algorithm import pso

#plot3d(rastrigin.func, np.array([9, 13]), np.array([17,20]))
#plot3d(ackley.func, np.array([-5.0, -5.0]), np.array([5.12,5.12]))
#plot3d(sphere.func, np.array([-5.0, -5.0]), np.array([5.12,5.12]))
#plot3d(rosenbrock.func, np.array([-50, -50]), np.array([50,50]))
#plot3d(beale.func, np.array([-4.5, -4.5]), np.array([4.5,4.5]))
#plot3d(eggholder.func, np.array([-50, -50]), np.array([50,50]))
#plot3d(holder_table.func, np.array([-5.0, -5.0]), np.array([5.12,5.12]))


dim = 2
fk,X = pso.alg(5, (np.random.random((5, dim))-0.5)*20.0, (np.random.random((5, dim))-0.5)*20.0,holder_table.func, 100)

#fk,X = steepest_descent.alg(X[-1], holder_table.func, holder_table.gfunc, lr=0.001, max_iter=100)
plot_surface_curve(rastrigin.func, np.array([-5, -5]), np.array([5,5]), X[:,0], X[:,1], fk)
plot_curve(np.arange(len(fk)), fk)


