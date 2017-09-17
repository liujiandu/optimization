import numpy as np

def alg(num,init_position, init_velocity, func, max_iter, omiga=0.9, lr1=0.5, lr2=0.7):
	omiga = omiga
	lr1 = lr1
	lr2 = lr2

	position = init_position.copy()
	velocity = init_velocity.copy()
	
	dim = position.shape[1]

	indivdual_best_fitness = [0]*num
	for i in range(num):
		indivdual_best_fitness[i] = func(position[i])
	indivdual_best_position = position.copy()
	group_best_fitness = min(indivdual_best_fitness)
	group_best_position =  position[indivdual_best_fitness.index(group_best_fitness)].copy()
	
	fk = []
	X = []
	count = 0
	while count< max_iter:
		velocity = omiga*velocity + lr1*np.multiply(np.random.random((num,dim)), indivdual_best_position-position) + lr2*np.multiply(np.random.random((num,dim)), -(position-group_best_position))
		position = position + velocity
		for i in range(num):
			fitness = func(position[i])
			if fitness<indivdual_best_fitness[i]:
				indivdual_best_fitness[i] = fitness
				indivdual_best_position[i] = position[i]
				if fitness<group_best_fitness:
					group_best_fitness = fitness
					group_best_position = position[i]
		fk.append(group_best_fitness)	
		X.append(group_best_position)
		count+=1

	return (np.array(fk), np.array(X))
		
		


	
	

