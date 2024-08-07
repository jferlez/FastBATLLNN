import numpy as np
import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
import TLLnet
import posetFastCharm
import TLLHypercubeReach

import time
import pickle
import os


charm.options.local_msg_buf_size = 10000
if os.path.exists('./enable_profiling'):
    charm.options.profiling = True


class Main(Chare):

    def __init__(self,args):

        # Specify kernel/bias of 7 local linear functions:
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

        # These kernels/biases form local linear functions:
        localLinearFns = [[np.array([kern]).reshape(-1,1),np.array(bias)]]

        # Specify 8 selector sets for the TLL:
        selectorSets = [set([]) for k in range(8)]
        selectorSets[0] = set([0,2,4,6])
        selectorSets[1] = set([1,2,4,6])
        selectorSets[2] = set([1,2,6])
        selectorSets[3] = set([1,2,4,5,6])
        selectorSets[4] = set([1,3,4,5,6])
        selectorSets[5] = set([1,3,4,6])
        selectorSets[6] = set([1,3,5,6])
        selectorSets[7] = set([1,3,6])
        selectorSets = [selectorSets]

        # Create a TLL object with one intput/output and initialize it with the above localLinearFns/selectorSets:
        tll = TLLnet.TLLnet(linear_fns=7,uo_regions=8)
        tll.setLocalLinearFns(localLinearFns)
        tll.setSelectorSets(selectorSets)


        # Specify input constraints polytope: [-ext, ext]
        # (NOTE: we have a scalar input/scalar output TLL)
        ext = 2
        A_in = np.zeros((2*tll.n,tll.n))
        np.fill_diagonal(A_in[:tll.n], 1)
        np.fill_diagonal(A_in[tll.n:], -1)
        b_in = -ext * np.ones((2*tll.n,))
        constraints = [A_in , b_in]

        # Generate some input/output samples to get a sense of the TLL's max/min outputs:
        numSamples = 100
        samples, _, _ = tll.evalAt(2*ext*np.random.rand(numSamples,tll.n).T - ext)
        samples = samples.flatten()

        print('\n\nMax of Output Samples: ' + str(np.max(samples)))
        print('Min of Output Samples: ' + str(np.min(samples)))


        # Instantiate FastBATLLNN
        # Assume 4 cores (PEs) -- i.e. called with "charmrun +p4"
        pes = {'poset':[(0,charm.numPes(),1)],'hash':[(0,charm.numPes(),1)]}
        useQuery = False
        useBounding = False
        tllReach = Chare(TLLHypercubeReach.TLLHypercubeReach, args=[pes])
        charm.awaitCreation(tllReach)
        tllReach.initialize(tll , constraints, 100, useQuery, awaitable=True).get()

        optsDict = {'hashStoreUseBits':True,'prefilter':True,'solver':'glpk','verbose':True}

        print('\n\n----------------- VERIFYING LOWER BOUND:  -----------------')
        t = time.time()
        a = 0.299
        lbFut = tllReach.verifyLB(a,opts=optsDict,ret=True) # verify NN >= a: True/1 == SAT; False/0 == UNSAT
        lb = lbFut.get()
        t = time.time()-t
        print('TLL always >= ' + str(a) + ' on constraints? ' + str(bool(lb)))
        print('-----------------------------------------------------------')


        print('\n\n----------------- VERIFYING UPPER BOUND:  -----------------')
        t = time.time()
        b = 4.51
        ubFut = tllReach.verifyUB(b,ret=True) # verify NN <= b: True/1 == UNSAT; False/0 == SAT
        ub = ubFut.get()
        t = time.time()-t
        print('TLL always <= ' + str(b) + ' on constraints? ' + str(not bool(ub)))
        print('-----------------------------------------------------------')


        print('\n\n--------------- FINDING TIGHT LOWER BOUND:  ---------------')

        t = time.time()
        lbFut = tllReach.searchBound(-0.135,lb=True,verbose=True,awaitable=True,opts=optsDict,ret=True)
        lb = lbFut.get()
        t = time.time()-t

        print('\n\n------------------  FOUND LOWER BOUND:  -------------------')
        print(lb)
        print('Total time elapsed: ' + str(t) + ' (sec)')
        try:
            print('Minimum of samples: ' + str(np.min(samples)))
        except NameError:
            pass
        print('-----------------------------------------------------------')

        print('\n\n--------------- FINDING TIGHT UPPER BOUND:  ---------------')
        t = time.time()
        ubFut = tllReach.searchBound(-135,lb=False,awaitable=True,opts=optsDict,ret=True)
        ub = ubFut.get()
        t = time.time()-t

        print('\n\n------------------  FOUND UPPER BOUND:  -------------------')
        print(ub)
        print('Total time elapsed: ' + str(t) + ' (sec)')
        try:
            print('Maximum of samples: ' + str(np.max(samples)))
        except NameError:
            pass
        print('-----------------------------------------------------------')
        print(' ')

        if os.path.exists('./enable_profiling'):
            for pe in range(charm.numPes()):
                charm.thisProxy[pe].printStats(awaitable=True).get()
                charm.sleep(1)

        charm.exit()

charm.start(Main,modules=['posetFastCharm','TLLHypercubeReach','DistributedHash'])

