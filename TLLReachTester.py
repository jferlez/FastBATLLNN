

import numpy as np
import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
import scipy.optimize
import cdd
import TLLnet
import posetFastCharm
import TLLHypercubeReach
# import NodeCheckerLowerBdVerify
import time
import pickle



class Main(Chare):
    # @coro
    def __init__(self,args):

        with open('sizeVsTime_n2_input.p','rb') as fp:
            expers = pickle.load(fp)

        # Simple Example #1:
        # 
        # localLinearFns = [ \
        #         [np.array([[-1, 1]]),  np.array([0,-1])] \
        #     ]
        # selectorMats = [ \
        #         [np.eye(2)] \
        #     ]

        # Simple example #2: (see also Jupyter notebook)
        kern = np.ones(7)
        bias = np.ones(7)
        # l_1
        m = (1.2-3.0)/(0.17+1)
        kern[0] = m
        bias[0] = m*(-0.17)+1.2
        # l_2
        m = (4.1-1.2)/(0.78-0.17)
        kern[1] = m
        bias[1] = m*(-0.17)+1.2
        # l_3
        m = (0.3-4.1)/(2-0.78)
        kern[2] = m
        bias[2] = m*(-0.78)+4.1
        # l_4
        m = (1.7-0.3)/(2.07-2)
        kern[3] = m
        bias[3] = m*(-2)+0.3
        # l_5
        m = (1.62-1.7)/(2.13-2.07)
        kern[4] = m
        bias[4] = m*(-2.07)+1.7
        # l_6
        m = (8.0-1.62)/(5-2.13)
        kern[5] = m
        bias[5] = m*(-2.13)+1.62
        # l_7
        m = (0-8.0)/(7.2-5)
        kern[6] = m
        bias[6] = m*(-5)+8.0
        kern = np.array([kern])

        localLinearFns = [[kern,bias]]

        e = np.eye(7)
        selectorMats = [[] for k in range(8)]
        selectorMats[0] = np.vstack([ e[:,0], e[:,2], e[:,4], e[:,6], e[:,6], e[:,6], e[:,6] ]).T
        selectorMats[1] = np.vstack([ e[:,1], e[:,2], e[:,4], e[:,6], e[:,6], e[:,6], e[:,6] ]).T
        selectorMats[2] = np.vstack([ e[:,1], e[:,2], e[:,6], e[:,6], e[:,6], e[:,6], e[:,6] ]).T
        selectorMats[3] = np.vstack([ e[:,1], e[:,2], e[:,4], e[:,5], e[:,6], e[:,6], e[:,6] ]).T
        selectorMats[4] = np.vstack([ e[:,1], e[:,3], e[:,4], e[:,5], e[:,6], e[:,6], e[:,6] ]).T
        selectorMats[5] = np.vstack([ e[:,1], e[:,3], e[:,4], e[:,6], e[:,6], e[:,6], e[:,6] ]).T
        selectorMats[6] = np.vstack([ e[:,1], e[:,3], e[:,5], e[:,6], e[:,6], e[:,6], e[:,6] ]).T
        selectorMats[7] = np.vstack([ e[:,1], e[:,3], e[:,6], e[:,6], e[:,6], e[:,6], e[:,6] ]).T
        selectorMats = [selectorMats]


        # Specify the contraints for a 1-d input:
        constraints = [ \
                np.array([ [1 , -1]  ]).T,
                np.array([0, -1])
            ]
        
        sizeIdx = 7
        experIdx = 18
        localLinearFns = expers[sizeIdx][experIdx]['TLLparameters']['localLinearFunctions']
        selectorMats = expers[sizeIdx][experIdx]['TLLparameters']['selectorMatrices']
        constraints = [ expers[sizeIdx][experIdx]['inputPoly']['A'], expers[sizeIdx][experIdx]['inputPoly']['b'] ]

        pes = {'poset':[(0,3,1)],'hash':[(3,4,1)]}
        useQuery = True
        useBounding = True
        tllReach = Chare(TLLHypercubeReach.TLLHypercubeReach, args=[localLinearFns, selectorMats, constraints, 100, pes, useQuery, useBounding])
        # charm.awaitCreation(tllReach)

        t = time.time()
        lbFut = tllReach.searchBound(-0.135,lb=True,verbose=True,awaitable=True,ret=True)
        # lbFut = tllReach.verifyUB(100,ret=True) # verify NN >= 100
        # lbFut = tllReach.verifyLB(-100,ret=True) # verify NN <= 100
        lb = lbFut.get()
        t = time.time()-t

        print(' ')
        print('--------------  FOUND LOWER BOUND:  --------------')
        print(lb)
        print('Total time elapsed: ' + str(t) + ' (sec)')
        print(' ')
        try:
            print('Minimum of samples: ' + str(np.min(expers[sizeIdx][experIdx]['samples']['output'])))
        except NameError:
            pass
        print('--------------------------------------------------')
        print(' ')

        print(' ')
        t = time.time()
        lbFut = tllReach.searchBound(-135,lb=False,verbose=True,awaitable=True,ret=True)
        # lbFut = tllReach.verifyUB(4.1,ret=True) # verify NN <= 100
        ub = lbFut.get()
        t = time.time()-t

        print(' ')
        print('--------------  FOUND UPPER BOUND:  --------------')
        print(ub)
        print('Total time elapsed: ' + str(t) + ' (sec)')
        print(' ')
        try:
            print('Maximum of samples: ' + str(np.max(expers[sizeIdx][experIdx]['samples']['output'])))
        except NameError:
            pass
        print('--------------------------------------------------')
        print(' ')

        charm.exit()
        
charm.start(Main,modules=['posetFastCharm','TLLHypercubeReach','DistributedHash'])