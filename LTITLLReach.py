import numpy as np
import scipy
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

        self.tllReach = []
        self.fastbatllnnLock = []
        self.suspendChannels = []
        for ii in range(5):
            self.tllReach.append(Chare(TLLHypercubeReach.TLLHypercubeReach, args=[pes]))
            charm.awaitCreation(self.tllReach[-1])
            self.fastbatllnnLock.append(False)
            self.suspendChannels.append(Channel(self,remote=self.thisProxy))
            self.suspendChannels[-1].send(-1)
            self.suspendChannels[-1].recv()

        self.defaultOpts = { \
                    'method':'fastLP', \
                    'solver':'glpk', \
                    'useQuery':False, \
                    'hashStore':'bits', \
                    'verbose': False, \
                    'restrictedVerbose':False \
                }
        self.lp = encapsulateLP.encapsulateLP()

        self.progressLevel = 3

        self.initialized = False

        self.usedOpts = copy(self.defaultOpts)

    def initialize(self,tllController=None,A=None,B=None,polytope=None):

        self.initialized = False

        assert issubclass(type(tllController),TLLnet.TLLnet) , 'tllController parameter must be a TLLnet object'

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

        self.constraints = [polytope['A'], polytope['b']]

        r, q = scipy.linalg.rq(self.A)
        self.r = r
        self.q = q

        # Create additional FastBATLLNN instances if necessary:
        if 2**self.n + 1 > len(self.tllReach):
            for ii in range(len(self.tllReach),2**self.n + 1):
                self.tllReach.append(Chare(TLLHypercubeReach.TLLHypercubeReach, args=[pes]))
                charm.awaitCreation(self.tllReach[-1])
                self.fastbatllnnLock.append(False)
                self.suspendChannels.append(Channel(self,remote=self.thisProxy))
                self.suspendChannels[-1].send(-1)
                self.suspendChannels[-1].recv()

        self.allQuadrantBox = np.full((self.n,2),np.inf,dtype=np.float64)
        self.allQuadrantBox[:,1] = -self.allQuadrantBox[:,1]

        self.progress = 0

        self.initialized = True

    @coro
    def computeLTIReach(self,T=10,epsilon=0.1,opts={}):

        assert self.initialized == True, 'Not initialized. Please call initialize method first...'

        assert type(T) == int and T > 0, 'Time horizon T must be an integer > 0'

        self.usedOpts = copy(self.defaultOpts)
        for ky in opts.keys():
            self.usedOpts[ky] = opts[ky]

        self.verbose = self.usedOpts['verbose']
        self.restrictedVerbose = self.usedOpts['restrictedVerbose']

        self.maxIts = 100
        # These will be updated with each time step
        self.lbSeed = -1
        self.ubSeed = 1

        # This needs to be set to account for the exponential growth and number of time steps T
        self.correctedEpsilon = epsilon/2

        bBoxes = []
        self.level = [0 for ii in range(len(self.tllReach))]

        constraints = [self.constraints[0].copy(), self.constraints[1].copy()]

        retDict = {}

        for t in range(0,T):

            tStart = time.time()
            self.skipBoxCount = 0
            self.numRefine = 0
            self.cornerRefined = [False for ii in range(len(self.tllReach))]
            self.cornerRefined[-1] = True
            self.allDone = [False for ii in range(len(self.tllReach)-1)]

            self.allQuadrantBox = np.full((self.n,2),np.inf,dtype=np.float64)
            self.allQuadrantBox[:,1] = -self.allQuadrantBox[:,1]
            self.progress = 0.0
            self.startTime = time.time()
            self.thisProxy.computeLTIBbox(constraints,boxLike=(False if t == 0 else True),ret=True).get()
            bboxStep = self.allQuadrantBox.copy()
            print(f'\nBounding box at T={t} is {bboxStep}')
            print(f'Number of boxes skipped: {self.skipBoxCount}; Total number of refinements: {self.numRefine}\n')
            constraints = [ \
                        np.vstack([ np.eye(self.n), -np.eye(self.n) ]), \
                        np.hstack([ bboxStep[:,0], -bboxStep[:,1] ]) \
                    ]
            self.level = [0 for ii in range(len(self.tllReach))]

            retDict[t] = {}
            retDict[t]['box'] = bboxStep
            retDict[t]['time'] = time.time() - tStart
            # Now create a new set of linear constraints that one obtains from propagating the above
            # bounding box through the supplied LTI system
        return retDict

    @coro
    def computeLTIBbox(self, constraints, order=None, batllnnInst=None, boxLike=False):
        if order is None and batllnnInst is None:
            firstRun = True
            order = 0
            batllnnInst = len(self.tllReach) - 1
        else:
            firstRun = False

        sig = np.random.rand()

        self.level[batllnnInst] += 1
        levelIndent = self.level[batllnnInst] * '    '

        # suspend here until our respective FastBATLLNN instance is free
        # while self.fastbatllnnLock[batllnnInst]:
        #     tempFut = Future()
        #     tempFut.send(1)
        #     tempFut.get()
        # self.fastbatllnnLock[batllnnInst] = True

        if self.verbose or self.restrictedVerbose:
            print(levelIndent + '***** DESCEND ONE LEVEL *****')
        # Function takes a polynomial constraint set of states as input
        # returns epsilon-tolerance bounding box for next state set subject to that constrained state set

        # Compute the axis-aligned bounding box for Ax where x is within the current supplied state constraints
        if not boxLike:
            bboxIn = self.constraintBoundingBox(constraints)
        else:
            bboxIn = np.inf * np.ones((self.n,2),dtype=np.float64)# [[] for ii in range(d)]
            bboxIn[:,0] = -bboxIn[:,0]
            for ii in range(self.n):
                for direc in [1,-1]:
                    idx = np.nonzero(constraints[0][:,ii] == direc)[0]
                    if direc == 1:
                        idx = idx[np.argmax(constraints[1][idx])]
                    else:
                        idx = idx[np.argmax(constraints[1][idx])]
                    #print(levelIndent + f'/\/\/\/\/\/ ***** idx is {idx}; constraints[1].shape = {constraints[1]}')
                    bboxIn[ii,(0 if direc == 1 else 1)] = direc * constraints[1][idx]

        bboxInConstraints = [ \
                        np.vstack([np.eye(self.n), -np.eye(self.n)]), \
                        np.hstack([bboxIn[:,0], -bboxIn[:,1]])
                    ]

        # Split the state bounding box into 2^d quadrants
        midpoints = np.array([ 0.5 * sum(dimBounds) for dimBounds in bboxIn ],dtype=np.float64)
        emat = np.eye(self.n, dtype=np.float64)

        if self.verbose or self.restrictedVerbose:
            print('\n' + levelIndent + '*****************************************************************************')
            print(levelIndent + f'bboxIn = {bboxIn.tolist()}; midpoints = {midpoints.tolist()}')
        # for each quadrant:
            # compute the controller reachable set subject to that quadrant as a state constraint
            # if the size of the controller reachable set multiplied by B is less than epsilon/2:
                # then state constraint quadrant times A + controller reachable set is an epsilon bounding box for the next state started from this quadrant
            # else:
                # recurse on this quadrant
        # This will track the coordinate-wise VERIFIED min and max values seen across ALL quadrants (possibly updated only after recursion)
        # allQuadrantBox = np.inf * np.ones((self.n,2), dtype=np.float64)
        # allQuadrantBox[:,1] = -allQuadrantBox[:,1]

        quadrantFutures = []
        waitForAllDone = []
        numDescents = 0
        for quadrant in range(2**self.n):
            quadrantSel = int_to_np( quadrant if firstRun else (quadrant + order) % 2**self.n , self.n)
            # print(constraints)
            # print(quadrantSel)
            # print(midpoints)
            # A box-like split quadrant of the previous state's bounding box
            if boxLike:
                quadrantConstraints = [ \
                            np.vstack([bboxInConstraints[0], (-1)**quadrantSel * emat ]), \
                            np.hstack([bboxInConstraints[1], (-1)**quadrantSel * midpoints])  \
                        ]
            else:
                quadrantConstraints = [ \
                            np.vstack([constraints[0], (-1)**quadrantSel * emat ]), \
                            np.hstack([constraints[1], (-1)**quadrantSel * midpoints])  \
                        ]
                status, _ = self.lp.runLP( \
                    np.ones(self.n), \
                    -quadrantConstraints[0], -quadrantConstraints[1], \
                    lpopts = {'solver':self.usedOpts['solver'], 'fallback':'glpk'} if self.usedOpts['solver'] != 'glpk' else {'solver':'glpk'}, \
                    msgID = str(charm.myPe()) \
                )
                if status == 'primal infeasible':
                    # The current quadrant box doesn't intersect the input constraint set, so we can skip this quadrant
                    print('Non-overlapping qudrant.... skipping...')
                    continue


            bboxQuadrant = bboxIn.copy()
            bboxQuadrant[quadrantSel,1] = midpoints[quadrantSel]
            bboxQuadrant[np.logical_not(quadrantSel),0] = midpoints[np.logical_not(quadrantSel)]

            # Compute a bounding box for the quadrant after passing it through the A matrix
            quadrantStateBBox = self.constraintBoundingBox([quadrantConstraints[0] @ self.q.T, quadrantConstraints[1]],basis=self.r)
            quadrantStateConstraints = [ \
                    np.vstack([np.eye(self.n), -np.eye(self.n)]), \
                    np.hstack([quadrantStateBBox[:,0], -quadrantStateBBox[:,1]])
                ]

            # We use the bounding box on the **PRIOR** state to bound the controller output
            try:
                self.tllReach[batllnnInst].initialize( \
                            self.tllController, \
                            quadrantConstraints , \
                            self.maxIts, \
                            self.usedOpts['useQuery'], \
                            awaitable=True \
                        ).get()
            except ValueError:
                if self.verbose or self.restrictedVerbose:
                    print(levelIndent + 'Unable to initialize tllReach; probably constraints with empty interior -- skipping this quadrant...')
                continue

            quadrantTLLReach = self.tllReach[batllnnInst].computeReach(lbSeed=self.lbSeed,ubSeed=self.ubSeed, tol=self.correctedEpsilon/(self.m*np.max(np.abs(self.B))), opts=self.usedOpts, ret=True).get()

            if self.verbose or self.restrictedVerbose:
                print(f'\n' + levelIndent + f'LEVEL={self.level[batllnnInst]}; QUAD={quadrant} --- quadrantTLLReach = {quadrantTLLReach}')

            controllerReachMidpoints = 0.5 * np.sum(quadrantTLLReach, axis=1)

            if self.verbose or self.restrictedVerbose:
                print(levelIndent + f'LEVEL={self.level[batllnnInst]}; QUAD={quadrant} --- controllerReachMidpoints = {controllerReachMidpoints}')

            controllerReachBall = quadrantTLLReach[:,1] - controllerReachMidpoints # should be non-negative

            # This is the **l_1** error "added" to Ax + controllerReachMidpoints as a result of our bounding of B NN(x)
            nnError = (np.abs(self.B) @ controllerReachBall.reshape(-1,1)).flatten()

            if self.verbose or self.restrictedVerbose:
                print(levelIndent + f'LEVEL={self.level[batllnnInst]}; QUAD={quadrant} --- nnError = {nnError}\n')

            if np.any(quadrantTLLReach[:,0] > quadrantTLLReach[:,1]):
                print(f'Error: nonsensical bounding box for quadrantTLLReach = {quadrantTLLReach} ... Exiting ...')
                charm.exit()

            nextStateBBox = np.zeros((self.n,2),dtype=np.float64)
            nextStateBBox[:,0] = quadrantStateBBox[:,0] + (self.B @ controllerReachMidpoints).flatten() - nnError
            nextStateBBox[:,1] = quadrantStateBBox[:,1] + (self.B @ controllerReachMidpoints).flatten() + nnError
            redundantBox = np.all(self.allQuadrantBox[:,0] <= nextStateBBox[:,0]) and np.all(nextStateBBox[:,1] <= self.allQuadrantBox[:,1])
            if np.max(nnError) < self.correctedEpsilon or redundantBox:
                if redundantBox and np.max(nnError) >= self.correctedEpsilon:
                    self.skipBoxCount += 1
                # the error is acceptably small, so update allQuadrantBox
                self.allQuadrantBox[:,0] = np.minimum(nextStateBBox[:,0], self.allQuadrantBox[:,0])
                self.allQuadrantBox[:,1] = np.maximum(nextStateBBox[:,1], self.allQuadrantBox[:,1])
                if not self.cornerRefined[batllnnInst]:
                    self.cornerRefined[batllnnInst] = True
                while not all(self.cornerRefined) and not firstRun:
                    # print(f'Waiting for corner visits on {batllnnInst}. QuadrantBox:\n{self.allQuadrantBox}')
                    tempFut = Future()
                    tempFut.send(1)
                    tempFut.get()
            else:
                # Recurse by refining the size box on the previous state
                self.numRefine += 1
                numDescents += 1
                if firstRun:
                    quadrantFutures.append( \
                                self.thisProxy.computeLTIBbox( \
                                    quadrantConstraints, \
                                    order=(quadrant if firstRun else order), \
                                    batllnnInst=(quadrant if firstRun else batllnnInst), \
                                    boxLike=boxLike, \
                                    ret=True \
                                ) \
                            )
                    waitForAllDone.append((quadrant if firstRun else batllnnInst))
                else:
                    self.thisProxy.computeLTIBbox( \
                                    quadrantConstraints, \
                                    order=order, \
                                    batllnnInst=batllnnInst, \
                                    boxLike=boxLike, \
                                    ret=True \
                                ).get()
                # allQuadrantBox[:,0] = np.minimum(recurseBox[:,0], allQuadrantBox[:,0])
                # allQuadrantBox[:,1] = np.maximum(recurseBox[:,1], allQuadrantBox[:,1])

        # We have finished using our instance of FastBATLLNN so unlock it for the next level down
        # self.fastbatllnnLock[batllnnInst] = False
        if len(waitForAllDone) > 0:
            for ii in range(len(self.level)-1):
                if ii not in waitForAllDone:
                    self.cornerRefined[ii] = True
                    self.allDone[ii] = True
        # Now wait for all quadrants to finish
        if len(quadrantFutures) > 0:
            while not all(self.allDone):
                tempFut2 = Future()
                tempFut2.send(-1)
                tempFut2.get()

            charm.wait(quadrantFutures)
        # for fut in quadrantFutures:
        #     print(f'Waiting for work to be done on box {bboxIn} {sig}')
        #     fut.get()
        if batllnnInst < len(self.tllReach) - 1:
            if self.level[batllnnInst] == self.progressLevel:
                # print(f'Level {self.level[batllnnInst]}')
                fineProgress = 1.0/(2.0**(self.n*(self.progressLevel)))
                self.progress += fineProgress
                print(f'{int(100*self.progress)}% complete... ({(time.time() - self.startTime):.3f} seconds elapsed)')
            elif self.level[batllnnInst] < self.progressLevel:
                # print(f'Level {self.level[batllnnInst]}')
                self.progress += (2**self.n - numDescents)/(2.0**(self.n*(self.level[batllnnInst]+1)))
                print(f'{int(100*self.progress)}% complete...')
        elif batllnnInst == len(self.tllReach)-1 and numDescents < 2**self.n:
            self.progress += (2**self.n - numDescents)/(2.0**(self.n))
            print(f'{int(100*self.progress)}% complete...')

        self.level[batllnnInst] -= 1
        if batllnnInst < len(self.tllReach) - 1 and self.level[batllnnInst] == 0:
            self.allDone[batllnnInst] = True
        return True

        # Final return value is max/min coordinates of each quadrant guaranteed up to self.correctedEpsilon
        # return allQuadrantBox

    def constraintBoundingBox(self,constraints,basis=None):
        solver = self.usedOpts['solver'] if 'solver' in self.usedOpts else 'glpk'
        if basis is None:
            bs = np.eye(constraints[0].shape[1])
        else:
            bs = basis.copy()
        #if constraints[0].shape[1] != 2:
        #print(f'constraints = {constraints}')
        #print(type(constraints[0][0]))
        bboxIn = np.inf * np.ones((self.n,2),dtype=np.float64)
        bboxIn[:,0] = -bboxIn[:,0]
        ed = np.zeros((self.n,1))
        for ii in range(self.n):
            for direc in [1,-1]:
                status, x = self.lp.runLP( \
                    direc * bs[ii,:], \
                    -constraints[0], -constraints[1], \
                    lpopts = {'solver':solver, 'fallback':'glpk'} if solver != 'glpk' else {'solver':'glpk'}, \
                    msgID = str(charm.myPe()) \
                )
                x = np.frombuffer(x)

                if status == 'optimal':
                    bboxIn[ii,(0 if direc == 1 else 1)] = np.dot(x,bs[ii,:])
                elif status == 'dual infeasible':
                    bboxIn[ii,(0 if direc == 1 else 1)] = -1*direc*np.inf
                else:
                    print('********************  PE' + str(charm.myPe()) + ' WARNING!!  ********************')
                    print('PE' + str(charm.myPe()) + ': Infeasible or numerical ill-conditioning detected while computing bounding box!')
                    return bboxIn
        return bboxIn

    def computeReachSamples(self,xIn,T=10,reachBoxes=None):
        sampleBoxes = {}
        approxBoxes = {}
        boxSamplesNum = 10000
        if reachBoxes is not None and 0 in reachBoxes:
            doBoxSamples = True
        else:
            doBoxSamples = False

        x = xIn.copy()
        for t in range(T):
            tllEval = np.zeros((x.shape[0],self.m),dtype=np.float64)
            for ii in range(x.shape[0]):
                tllEval[ii,:] = self.tllController.pointEval(x[ii,:])

            sampleBoxes[t] = {}

            bounds = np.array([np.min(tllEval,axis=0),np.max(tllEval,axis=0)]).T
            sampleBoxes[t]['controllerOutputBox'] = bounds

            x = (self.A @ x.T + self.B @ tllEval.T).T
            xBox = np.array([np.min(x,axis=0),np.max(x,axis=0)]).T
            sampleBoxes[t]['stateBoxInputSamples'] = xBox

            if reachBoxes is not None and t in reachBoxes and 'box' in reachBoxes[t] and reachBoxes[t]['box'].shape == (self.n, 2):
                boxTime = 0 if t == 0 else t-1
                xBoxSamples = np.random.uniform(low=reachBoxes[boxTime]['box'][:,0],high=reachBoxes[boxTime]['box'][:,1],size=(boxSamplesNum,self.n))
                if t == 0:
                    sampleBoxes[t]['stateBoxBoxSamples'] = np.array([np.min(xBoxSamples,axis=0),np.max(xBoxSamples,axis=0)]).T
                else:
                    tllEval = np.zeros((boxSamplesNum, self.m), dtype=np.float64)
                    for ii in range(boxSamplesNum):
                        tllEval[ii,:] = self.tllController.pointEval(xBoxSamples[ii,:])
                    xBoxSamples = (self.A @ xBoxSamples.T + self.B @ tllEval.T).T
                    sampleBoxes[t]['stateBoxBoxSamples'] = np.array([np.min(xBoxSamples,axis=0),np.max(xBoxSamples,axis=0)]).T

        return sampleBoxes

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


