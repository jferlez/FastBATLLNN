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

        assert A.shape == (self.n,self.n), f'A must be an ({self.n} x {self.n}) matrix, where {self.n} is the number of TLL inputs'
        assert B.shape == (self.n,self.m), f'B must be an ({self.n} x {self.m}) dimensional matrix, where {self.n} is the number of TLL inputs and {self.m} is the number of TLL outputs'

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
        # These will be updated with each time step
        self.lbSeed = -1
        self.ubSeed = 1

        # This needs to be set to account for the exponential growth and number of time steps T
        self.correctedEpsilon = epsilon

        constraints = [polytope['A'], polytope['b']]
        bBoxes = []

        for t in range(0,T):

            bboxStep = self.thisProxy.computeLTIBbox(constraints, boxLike=(False if t == 0 else True),ret=True).get()
            print(f'Bounding box at T={t} is {bboxStep}')
            constraints = [ \
                        np.vstack([ np.eye(self.n), -np.eye(self.n) ]), \
                        np.hstack([ bboxStep[:,0], -bboxStep[:,1] ]) \
                    ]


            # Now create a new set of linear constraints that one obtains from propagating the above
            # bounding box through the supplied LTI system

    @coro
    def computeLTIBbox(self, constraints, boxLike=False):
        # Function takes a polynomial constraint set of states as input
        # returns epsilon-tolerance bounding box for next state set subject to that constrained state set

        # Compute the bounding box for the current supplied state constraints
        if not boxLike:
            bboxIn = self.constraintBoundingBox(constraints)
        else:
            bboxIn = np.inf * np.ones((self.n,2),dtype=np.float64)# [[] for ii in range(d)]
            bboxIn[:,0] = -bboxIn[:,0]
            for  ii in range(self.n):
                for direc in [1,-1]:
                    idx = np.nonzero(constraints[0] == direc)[0]
                    bboxIn[ii,(0 if direc == 1 else 1)] = direc * constraints[1][idx]

        # Split the state bounding box into 2^d quadrants
        midpoints = np.array([ 0.5 * sum(dimBounds) for dimBounds in bboxIn ],dtype=np.float64)
        emat = np.eye(self.n, dtype=np.float64)

        # for each quadrant:
            # compute the controller reachable set subject to that quadrant as a state constraint
            # if the size of the controller reachable set multiplied by B is less than epsilon/2:
                # then state constraint quadrant times A + controller reachable set is an epsilon bounding box for the next state started from this quadrant
            # else:
                # recurse on this quadrant
        # This will track the coordinate-wise VERIFIED min and max values seen across ALL quadrants (possibly updated only after recursion)
        allQuadrantBox = np.inf * np.ones((self.n,2), dtype=np.float64)
        allQuadrantBox[:,1] = -allQuadrantBox[:,1]

        for quadrant in range(2**self.n):
            quadrantSel = int_to_np(quadrant, self.n)

            # print(constraints)
            # print(quadrantSel)
            # print(midpoints)

            quadrantConstraints = [ \
                            np.vstack([constraints[0], emat]), \
                            np.hstack([constraints[1], (-1)**quadrantSel * midpoints ]) \
                        ]

            bboxQuadrant = self.constraintBoundingBox(quadrantConstraints)

            self.tllReach.initialize( \
                        self.tllController, \
                        quadrantConstraints , \
                        self.maxIts, \
                        self.usedOpts['useQuery'], \
                        awaitable=True \
                    ).get()

            quadrantTLLReach = self.tllReach.computeReach(lbSeed=self.lbSeed,ubSeed=self.ubSeed, tol=self.correctedEpsilon, ret=True).get()

            controllerReachMidpoints = 0.5 * np.sum(quadrantTLLReach, axis=1)
            controllerReachBall = quadrantTLLReach[:,1] - controllerReachMidpoints # should be non-negative

            # This is the **l_1** error "added" to Ax + controllerReachMidpoints as a result of our bounding of B NN(x)
            nnError = (np.abs(self.B) @ controllerReachBall.reshape(-1,1)).flatten()

            if np.max(nnError) < self.correctedEpsilon/2:
                # the error is acceptably small, so update allQuadrantBox
                allQuadrantBox[:,0] = np.minimum(bboxQuadrant[:,0] + (self.B @ controllerReachMidpoints).flatten() + nnError, allQuadrantBox[:,0])
                allQuadrantBox[:,1] = np.maximum(bboxQuadrant[:,1] + (self.B @ controllerReachMidpoints).flatten() + nnError, allQuadrantBox[:,1])
            else:
                # recurse by calling computeLTIBbox on the current qudrant
                recurseBox = self.thisProxy.computeLTIBbox(quadrantConstraints,boxLike=boxLike,ret=True).get()
                allQuadrantBox[:,0] = np.minimum(recurseBox[:,0], allQuadrantBox[:,0])
                allQuadrantBox[:,1] = np.maximum(recurseBox[:,1], allQuadrantBox[:,1])

        # Final return value is max/min coordinates of each quadrant guaranteed up to self.correctedEpsilon
        return allQuadrantBox

    def constraintBoundingBox(self,constraints):
        solver = self.usedOpts['solver'] if 'solver' in self.usedOpts else 'glpk'
        #if constraints[0].shape[1] != 2:
        #print(f'constraints = {constraints}')
        #print(type(constraints[0][0]))
        bboxIn = np.inf * np.ones((self.n,2),dtype=np.float64)
        bboxIn[:,0] = -bboxIn[:,0]
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
                    bboxIn[ii,(0 if direc == 1 else 1)] = x[ii,0]
                elif status == 'dual infeasible':
                    bboxIn[ii,(0 if direc == 1 else 1)] = -1*direc*np.inf
                else:
                    print('********************  PE' + str(charm.myPe()) + ' WARNING!!  ********************')
                    print('PE' + str(charm.myPe()) + ': Infeasible or numerical ill-conditioning detected while computing bounding box!')
                    return bboxIn
        return bboxIn

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


