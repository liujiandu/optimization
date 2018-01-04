import numpy as np
import matplotlib.pyplot as plt

def func1(x):
    y = np.exp(-(x-2)**2)+np.exp(-(x-6)**2/10)+1/(x**2+1)
    return y    


if __name__=="__main__":
    x = np.arange(-10, 10, 0.01)
    plt.plot(x, func1(x))
    plt.show()
