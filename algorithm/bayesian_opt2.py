import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from bayes_opt import BayesianOptimization

def target(x1, x2):
    y = np.exp(-(x1-2)**2)+np.exp(-(x2-6)**2/10)+1/(x1**2+x2**2+1)
    return y    

def posterior(bo, x, xmin=-2, xmax=10):
    bo.gp.fit(bo.X, bo.Y) 
    mu, sigma = bo.gp.predict(x, return_std=True)
    return mu, sigma

def plt_gp(bo, x, y):
    fif = plt.figure(figsize=(16,10))
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    mu, sigma = posterior(bo, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(bo.X.flatten(), bo.Y, 'D', markersize=8, color='r', label='Observation')
    axis.plot(x, mu, '--', color='k', label='Prediction')
    axis.fill(np.concatenate([x, x[::-1]]), np.concatenate([mu-1.96*sigma, (mu+1.96*sigma)[::-1]]), alpha=0.6, fc='c', ec='None')

    utility = bo.util.utility(x, bo.gp, 0)
    acq.plot(x, utility, label='Utility Function')
    plt.show()


def main():
    bo = BayesianOptimization(target, {'x1':(-2,10), 'x2':(-2, 10)})
    bo.maximize(init_points=2, n_iter=30, acq='ucb', kappa=5)
    
    #x = np.linspace(-2, 10, 100).reshape(-1,1)
    #y = target(x)
    #plt_gp(bo, x, y)
if __name__=="__main__":
    main()
