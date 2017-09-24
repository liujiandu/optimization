class PI(object):
	def __init__(self):
		self.kp = 0.0
		self.ki = 0.0
		self.error = 0.0
		self.error_1 = 0.0
		self.output = 0.0
	
	def set_pi(self, kp, ki):
		self.kp = kp
		self.ki = ki

	def pi0(self, error):
		self.output = (self.kp+self.ki)*error
	
	def pi(self, error):
		self.error_1 = self.error
		self.error = error
		self.output = self.output + (self.kp+self.ki)*self.error - self.kp*self.error_1
