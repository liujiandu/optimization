import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from mpl_toolkits.mplot3d import Axes3D

def target(x, y):
    z = np.exp(-(x-2)**2)+np.exp(-(x-6)**2/5)+1/(x**2+1)+0.1*np.sin(5*x)-0.5
    z += np.exp(-(y-2)**2)+np.exp(-(y-6)**2/5)+1/(y**2+1)+0.1*np.sin(5*y)-0.5
    return z

def gp_fit(X, Y, x,y):
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5),n_restarts_optimizer=25)
    gp.fit(X, Y)
    mu, sigma = gp.predict(x, return_std=True)
    print mu.shape    
    '''
    fig = plt.figure(figsize=(16,10))
    #gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
    #axis = plt.subplot(gs[0])
    #acq = plt.subplot(gs[1])
    
    plt.plot(x, y, linewidth=3, label='Target')
    plt.plot(X.flatten(), Y, 'D', markersize=8, color='r', label='Observation')
    plt.plot(x, mu, '--', color='k', label='Prediction')
    plt.plot(x, np.zeros(x.shape[0]),  linewidth=3, color='r', label='Prediction')
    #axis.fill(np.concatenate([x, x[::-1]]), np.concatenate([mu-1.96*sigma, (mu+1.96*sigma)[::-1]]), alpha=0.6, fc='c', ec='None')

    plt.show()
    '''

def main(): 
    X = ((np.random.random((100, 2))-0.5)*15)+2.5
    Y = target(X[:,0], X[:,1])
    print X.shape
    print Y.shape
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5),n_restarts_optimizer=25)
    
    gp.fit(X, Y)
    
    a = np.arange(-5, 10, 0.2)
    b = np.arange(-5, 10, 0.2)
    x = []
    for i in a:
        for j in b:
            x.append([i,j])
    x= np.array(x)
    print x.shape
    mu, sigma = gp.predict(x, return_std=True)
    mu = mu.reshape((a.shape[0], b.shape[0]))

    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:,0], X[:,1], Y,s=40, c='r')
    
    x = np.arange(-5, 10,0.2)
    y = np.arange(-5, 10,0.2)
    
    x,y = np.meshgrid(x,y)
    z = target(x,y)
    print z.shape
    ax.plot_surface(x,y,mu,rstride=1, cstride=1, cmap='rainbow', alpha=0.5)
    plt.show()
    

if __name__=="__main__":
    main()
