#-------------------------------------------
#particle swarm algorithm by pyswarm
#Author: Liujiandu
#Date: 2018/1/3
#-------------------------------------------
import numpy as np
from pyswarm import pso

import sys
sys.path.append('../')
from function import sphere

def main():
    xpot, fopt = pso(sphere.func, [-2, -5], [10, 10], swarmsize=10, maxiter=10)    
    print xpot
    print fopt
if __name__=="__main__":
    main()
