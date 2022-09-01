import numpy as np
import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Channel

import TLLnet

import TLLHypercubeReach

import encapsulateLP

import time
import pickle
import os
import re
from copy import copy, deepcopy


class LTITLLReach(Chare):

    def __init__(self, pes):

        self.tllReach = Chare(TLLHypercubeReach.TLLHypercubeReach, args=[pes])
        charm.awaitCreation(self.tllReach)

        self.defaultOpts = { \
                    'method':'fastLP', \
                    'solver':'glpk', \
                    'useQuery':False, \
                    'hashStore':'bits' \
                }
        self.lp = encapsulateLP.encapsulateLP()

    @coro
    def computeLTIReach(self,tllController=None,A=None,B=None,polytope=None,T=10,epsilon=0.01,opts={}):

        assert type(tllController) == TLLnet.TLLnet, 'tllController parameter must be a TLLnet object'

        self.tllController = tllController

        self.n = tllController.n
        self.m = tllController.m
        self.controllerLipschitz = 0

        assert type(A) == np.ndarray and type(B) == np.ndarray, 'A and B matrices must be numpy arrays'

        assert A.shape == (n,n), f'A must be an ({n} x {n}) matrix, where {n} is the number of TLL inputs'
        assert B.shape == (n,m), f'B must be an ({n} x {m}) dimensional matrix, where {n} is the number of TLL inputs and {m} is the number of TLL outputs'

        self.A = A.copy()
        self.B = B.copy()

        assert type(polytope) == dict and 'A' in polytope and 'b' in polytope and 'polyDefinition' in polytope, \
            'polytope must be a python dictionary with properties \'A\', \'b\' and \'polyDefinition\''
        assert polytope['polyDefinition'] == 'Ax >= b', 'Only \'Ax >= b\' polytopes are currently supported'

        assert type(T) == int and T > 0, 'Time horizon T must be an integer > 0'


        self.usedOpts = copy(self.defaultOpts)
        for ky in opts.keys():
            self.usedOpts[ky] = opts[ky]

        self.maxIts = 100

        constraints = [polytope['A'], polytope['b']]
        bBoxes = []

        for t in range(0,T):

            bBoxes.append( self.tllReach.computeReach(lbSeed=-1,ubSeed=1, tol=epsilon, ret=True).get() )

            # Now create a new set of linear constraints that one obtains from propagating the above
            # bounding box through the supplied LTI system

    @coro
    def computeLTIBbox(self, constraints, boxLike=False):
        # Function takes a polynomial constraint set of states as input
        # returns epsilon-tolerance bounding box for next state set subject to that constrained state set

        # Compute the bounding box for the current supplied state constraints
        if not boxLike:
            bboxIn = [[] for ii in range(d)]
            ed = np.zeros((self.n,1))
            for ii in range(self.n):
                for direc in [1,-1]:
                    ed[ii,0] = direc
                    status, x = self.lp.runLP( \
                        ed.flatten(), \
                        -constraints[0], -constraints[1], \
                        lpopts = {'solver':solver, 'fallback':'glpk'} if solver != 'glpk' else {'solver':'glpk'}, \
                        msgID = str(charm.myPe()) \
                    )
                    ed[ii,0] = 0

                    if status == 'optimal':
                        bboxIn[ii].append(np.array(x[ii,0]))
                    elif status == 'dual infeasible':
                        bboxIn[ii].append(-1*direc*np.inf)
                    else:
                        print('********************  PE' + str(charm.myPe()) + ' WARNING!!  ********************')
                        print('PE' + str(charm.myPe()) + ': Infeasible or numerical ill-conditioning detected while computing bounding box!')
                        return [set([]), 0]
        else:
            bboxIn = [[] for ii in range(d)]
            for  ii in range(self.n):
                for direc in [1,-1]:
                    idx = np.nonzero(constraints[0] == direc)[0]
                    bboxIn[ii].append(direc * constraints[1][idx])

        # Split the state bounding box into 2^d quadrants
        midpoints = np.array([ 0.5 * sum(dimBounds) for dimBounds in bboxIn ],dtype=np.float64)
        quadrants = np.eye(self.n, dtype=np.float64)

        # for each quadrant:
            # compute the controller reachable set subject to that quadrant as a state constraint
            # if the size of the controller reachable set multiplied by B is less than epsilon/2:
                # then state constraint quadrant times A + controller reachable set is an epsilon bounding box for the next state started from this quadrant
            # else:
                # recurse on this quadrant

        # Final return value is max/min coordinates of each quadrant

        self.tllReach.initialize(self.tllController, constraints, self.maxIts, self.usedOpts['useQuery'], awaitable=True ).get()

def int_to_np(myint,n):
    assert myint <= 2**n - 1, f'Integer {myint} can\'t be represented with only {n} bits!'
    numBytes = n//8  + (1 if n%8 !=0 else 0)
    bts = myint.to_bytes(numBytes,byteorder='little')
    retVal = np.zeros(n,dtype=np.bool8)

    for ii in range(numBytes):
        for b in range(8):
            if bts[ii] & (1<<b):
                retVal[8*ii + b] = 1
    return retVal


