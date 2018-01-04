import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from scipy.stats import norm
from scipy.optimize import minimize
from target_space import TargetSpace
from helpers import ensure_rng

def acq_max(ac, gp, y_max, bounds, random_state, n_warmup=100000, n_iter=250):
    #ward up eith random points
    x_tries = random_state.uniform(bounds[:,0], bounds[:,1], size=(n_warmup,bounds.shape[0]))
    ys = ac(x_tries, gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()
    
    #explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:,0], bounds[:,1], size=(n_iter, bounds.shape[0]))
    for x_try in x_seeds:
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B')
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]
    return np.clip(x_max, bounds[:,0], bounds[:,1])


class AcquisitionFunction(object):
    def __init__(self, kind, kappa, xi):
        self.kappa = kappa
        self.xi = xi
        if kind not in ['ucb', 'ei', 'poi']:
            err = 'the utility function {} has not been implement'.format(kind)
            raise NotImplementedError()
        self.kind = kind

    def acqf(self, x, gp, y_max):
        if self.kind=="ucb":
            return self._ucb(x, gp, self.kappa)
        if self.kind=="ei":
            return self._ei(x, gp, y_max, self.xi)
        if self.kind=="poi":
            return self._poi(x, gp, y_max, self.xi)
    
    @staticmethod
    def _ucb(x, gp, kappa):
        mean, std = gp.predict(x, return_std=True)
        return mean +kappa*std
    
    @staticmethod
    def _ei(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi)/std
        return (mean - y_max - xi)*norm.cdf(z) +std*norm.pdf(z)
    
    @staticmethod
    def _poi(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (mean -y_max -xi)/std
        return norm.cdf(z)


class BayesianOptimization(object):
    def __init__(self, f, pbounds, random_state=None):
        self.pbounds = pbounds
        self.random_state = ensure_rng(random_state)
        self.gp = GaussianProcessRegressor(kernel=Matern(nu=2.5),
                                           n_restarts_optimizer=25,
                                           random_state=self.random_state)
        self.init_points = []
        self.space = TargetSpace(f, pbounds, random_state)
        
        self.x_init = []
        self.y_init = []
        self.initialized=False
        self._acqkw = {'n_warmup':100000, 'n_iter':250}

    def init(self, init_points):
        #init points
        rand_points = self.space.random_points(init_points)
        self.init_points.extend(rand_points)

        #evaluate target function at all init points
        for x in self.init_points:
            y = self._observe_point(x)

        #add the points to the obsevations
        if self.x_init:
            x_init = np.vstack(self.x_init)
            y_init = np.hstack(self.y_init)
            for x, y in zip(x_init, y_init):
                self.space.add_observation(x,y)
        self.initialized = True

    def _observe_point(self, x):
        y = self.space.observe_point(x)
        return y


    def initialize(self, points_dict):
        self.y_init.extend(points_dict['target'])
        for i in range(len(points_dict['target'])):
            all_points = []
            for key in self.space.keys:
                all_points.append(points_dict[key][i])
            self.x_init_append(all_points)


    def maximize(self, 
                 init_points=5, 
                 n_iter=25, 
                 acq='ucb', 
                 kappa=2.576, 
                 xi=0.0, 
                 **gp_params):

        #get acquisition function
        self.util = AcquisitionFunction(kind=acq, kappa=kappa, xi=xi)
        
        #initialize
        if not self.initialized:
            self.init(init_points)
        y_max = self.space.Y.max()

        #set gp parameters
        self.gp.set_params(**gp_params)

        #gaussian process fit
        self.gp.fit(self.space.X, self.space.Y)

        #find argmax of the acquisition function
        x_max = acq_max(ac=self.util.acqf,
                        gp = self.gp,
                        y_max=y_max,
                        bounds = self.space.bounds,
                        random_state = self.random_state,
                        **self._acqkw)
        
        #Iterative process
        for i in range(n_iter):
            #Append most recently generated values to X and Y arrays
            y = self.space.observe_point(x_max)
            #updatging the Gp
            self.gp.fit(self.space.X, self.space.Y)

            #update maximum value to search for next probe point
            if self.space.Y[-1] > y_max:
                y_max = self.space.Y[-1]
            #Maximum acquasition function to find next probing point
            x_max = acq_max(ac=self.util.acqf,
                            gp = self.gp,
                            y_max=y_max,
                            bounds = self.space.bounds,
                            random_state = self.random_state,
                            **self._acqkw)
    @property
    def X(self):
        return self.space.X
    @property
    def Y(self):
        return self.space.Y

if __name__=="__main__":
    def target(x):
        y = np.exp(-(x-2)**2)+np.exp(-(x-6)**2/5)+1/(x**2+1)+0.1*np.sin(5*x)-0.5
        return y
    bo = BayesianOptimization(target, {'x':(-5, 10)})
    bo.maximize(init_points=2, n_iter=10, acq='ucb', kappa=5)






