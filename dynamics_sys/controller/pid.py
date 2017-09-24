#!coding=utf-8
import numpy as np

class PID(object):
	def __init__(self):
		self.kp = 0.0
		self.ki = 0.0
		self.kd = 0.0
		self.output = 0.0
		self.error = 0.0
		self.error_1 = 0.0
		self.error_2 = 0.0

	def set_pid(self,kp, ki, kd, T=0.001):
		self.kp = kp
		self.ki = ki
		self.kd = kd
		self.T = T

	def pid0(self, error):
		self.error = error
		self.output = (self.kp+self.ki*self.T)*self.error
	
	def pid1(self,error):
		self.error_1 = self.error
		self.error = error
		self.output = self.kp*self.error+self.ki*self.T*(self.error_1+self.error)+self.kd/self.T*(self.error-self.error_1)

	def pid(self,error):
		self.error_2 = self.error_1
		self.error_1 = self.error
		self.error = error
		self.output = self.output + self.kp*(self.error-self.error_1)+self.ki*self.T*self.error_1+self.kd/self.T*(self.error-2*self.error_1+self.error_2)
	


	
