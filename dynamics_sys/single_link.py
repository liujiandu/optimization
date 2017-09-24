#coding=utf-8

import sys
sys.path.append('..')
from controller.pid import PID
from algorithm.grad_free import ops
import numpy as np

class SingleLink(object):
	def __init__(self, inertia=1.0):
		self.inertia = inertia
		self.dt = 0.001
	
	def reset(self, objective_theta, theta, omiga):
		self.theta = theta
		self.omiga = omiga
		self.objective_theta = objective_theta

	def step(self, torque):
		self.theta = self.omiga*self.dt+self.theta
		self.omiga = torque/self.inertia*self.dt+self.omiga
	
	def pid_episode(self, init_pid):	
		pid = PID()
		pid.set_pid(init_pid[0], init_pid[1], init_pid[2])

		for i in range(4000):
			if i==0:
				pid.pid0(self.objective_theta-self.theta)
			elif i==1:
				pid.pid1(self.objective_theta-self.theta)
			else:
				pid.pid(self.objective_theta-self.theta)
			self.step(pid.output)
		
		reward = (self.objective_theta-self.theta)**2+self.omiga**2
		return reward

if __name__=="__main__":
	from tool.plot import plot_curve

	sl = SingleLink()
	sl.reset(1.0, 0.0,0.0)
	init_pid = np.random.random((5,3))*10
	init_velocity = np.random.random((5,3))*10

	ops.alg(5, init_pid, init_velocity, sl.pid_episode, 20, omiga=0.9, lr1=0.5, lr2=0.7)
	#plot_curve(np.array(T)*sl.dt, X)

