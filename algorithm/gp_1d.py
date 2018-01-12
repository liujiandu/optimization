import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

def target(x):
    y = np.exp(-(x-2)**2)+np.exp(-(x-6)**2/5)+1/(x**2+1)+0.1*np.sin(5*x)-0.5
    return y    

<<<<<<< HEAD:algorithm/gp.py

def plt_gp(gp, X, Y, x, y):
    fif = plt.figure(figsize=(16,10))
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
=======
def gp_fit(X, Y, x,y):
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5),n_restarts_optimizer=25)
    gp.fit(X, Y)
    mu, sigma = gp.predict(x, return_std=True)
    
    fig = plt.figure(figsize=(16,10))
    gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
>>>>>>> 50b378e545a252b49c8be6e07fbca8af83f99493:algorithm/gp_1d.py
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x, mu, '--', color='k', label='Prediction')
    axis.plot(x, np.zeros(x.shape[0]),  linewidth=3, color='r', label='Prediction')
<<<<<<< HEAD:algorithm/gp.py
    axis.plot(X.flatten(), Y, 'D', markersize=8, color='r', label='Observation')
    axis.fill(np.concatenate([x, x[::-1]]), np.concatenate([mu-1.96*sigma, (mu+1.96*sigma)[::-1]]), alpha=0.6, fc='c', ec='None')
=======
    #axis.fill(np.concatenate([x, x[::-1]]), np.concatenate([mu-1.96*sigma, (mu+1.96*sigma)[::-1]]), alpha=0.6, fc='c', ec='None')

>>>>>>> 50b378e545a252b49c8be6e07fbca8af83f99493:algorithm/gp_1d.py
    plt.show()


def main(): 
<<<<<<< HEAD:algorithm/gp.py
    gp = GaussianProcessRegressor(kernel=Matern(nu=25), n_restarts_optimizer=25)
    X = ((np.random.random(10)-0.5)*15).reshape(-1, 1)
    Y = target(X)

    x = np.linspace(-7.5, 7.5, 200).reshape(-1,1)
    y = target(x)
    plt_gp(gp, X, Y, x, y)
=======
    X = ((np.random.random(10)-0.5)*10).reshape(-1,1)
    Y = target(X)
    x = np.linspace(-5, 10, 200).reshape(-1,1)
    y = target(x)
    gp_fit(X, Y, x, y)
>>>>>>> 50b378e545a252b49c8be6e07fbca8af83f99493:algorithm/gp_1d.py

if __name__=="__main__":
    main()
