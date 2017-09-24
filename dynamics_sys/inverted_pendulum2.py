#!coding=utf-8

import numpy as np

class InvertedPendulum(object):
	def __init__(self,M=1.096, m=0.109, b=0.1, I=0.0034, l=0.25):
		self.M = M
		self.b = b
		self.m = m
		self.l = l
		self.I = I
		self.g = 0.98
		self.dt = 0.001

		self.x = 0.0
		self.v = 0.0
		self.theta = 0.0
		self.omiga = 0.0
	
	def reset(self, x, v, theta, omiga):
		self.x = x
		self.v = v
		if theta>np.pi*0.5:
			self.theta = np.pi*0.5
		elif theta<-np.pi*0.5:
			self.theta = -np.pi*0.5
		else:
			self.theta = theta
		self.omiga = omiga

	def step(self, force):
		old_x = self.x
		old_v = self.v
		old_theta = self.theta
		old_omiga = self.omiga
		
		if self.theta<np.pi*0.5 and self.theta>-np.pi*0.5:
			self.theta = old_omiga*self.dt+old_theta
			
			if self.theta>=np.pi*0.5:
				t = (np.pi*0.5-old_theta)/old_omiga
				self.theta = 0.5*np.pi
			elif self.theta<=-0.5*np.pi:
				t = (-np.pi*0.5-old_theta)/old_omiga
				self.theta = -0.5*np.pi
			else:
				t = self.dt

			self.x = old_v*t + old_x

			v_dev = (force*(self.I+self.m*self.l**2)+self.m*self.l*(self.I+self.m*self.l**2)*np.sin(old_theta)*old_omiga**2-self.m**2*self.l**2*self.g*np.sin(old_theta)*np.cos(old_theta))/((self.I+self.m*self.l**2)*(self.M+self.m)-self.m**2*self.l**2*np.cos(old_theta)**2)
			self.v = v_dev*t+old_v

			omiga_dev = (self.m*self.l*np.cos(old_theta)*force+self.m**2*self.l**2*np.sin(old_theta)*np.sin(old_theta)*old_omiga**2-(self.M+self.m)*self.m*self.l*self.g*np.sin(old_theta))/(self.m**2*self.l**2*np.cos(old_theta)**2-(self.M+self.m)*(self.I+self.m*self.l**2))
			self.omiga = omiga_dev*t+old_omiga

		else:
			self.x = self.v*self.dt + old_x
			self.v = force/(self.M+self.b)*self.dt + old_v


if __name__ == "__main__":
	import sys
	sys.path.append('..')
	from tool.plot import plot_curve
	from controller.pid import PID
	from controller.pi import PI

	ip = InvertedPendulum()
	ip.reset(0.0, 0.0, 0.1, 0.0)
	pid = PID()
	pid.set_pid(36, 30, 10)

	####
	T = range(4000)
	X = []
	for i in T:	
		if i==0:
			pid.pid0(0.1-0.0)
		elif i==1:
			pid.pid1(ip.theta-0.00)
		else:
			pid.pid(ip.theta-0.00)
	
		ip.step(pid.output)
		X.append(ip.x)
	####

	plot_curve(np.array(T)*ip.dt, X)


		
		

