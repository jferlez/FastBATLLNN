
# import TLLnet
import numpy as np
import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Channel
import cdd
import cvxopt
import posetFastCharm
from copy import copy,deepcopy
import time
import encapsulateLP
import DistributedHash
import numba as nb
import posetFastCharm_numba
import itertools
import random
import region_helpers

try:
    from simple2xSuccessorWorker import simple2xSuccessorWorker
    simple2xAvailable = True
except ImportError:
    simple2xAvailable = False


cvxopt.solvers.options['show_progress'] = False

# All hyperplanes assumed to be specified as A x >= b

# For checking on *Hash* Nodes
class PosetNodeTLLVer(DistributedHash.Node):
    def init(self):
        self.constraints, self.selectorSetsFull, self.nodeIntMask, self.out = self.localProxy[self.storePe].getConstraints(ret=True).get()
    def check(self):
        regSet = np.full(self.constraints.allN, True, dtype=bool)
        regSet[tuple(self.constraints.flipMapSet),] = np.full(len(self.constraints.flipMapSet),False,dtype=bool)
        if self.constraints.N == self.constraints.allN:
            regSet[self.nodeBytes,] = np.full(len(self.nodeBytes),False,dtype=bool)
            unflipped = posetFastCharm_numba.is_in_set(self.constraints.flipMapSetNP,list(self.nodeBytes))
        else:
            sel = self.constraints.nonRedundantHyperplanes[self.nodeBytes,]
            regSet[sel,] = np.full(len(sel),False,dtype=bool)
            unflipped = posetFastCharm_numba.is_in_set(self.constraints.flipMapSetNP,sel.tolist())
        regSet[unflipped,] = np.full(len(unflipped),True,dtype=bool)
        regSet = np.nonzero(regSet)[0]

        val = False
        for sSet in self.selectorSetsFull[self.out]:
            if not posetFastCharm_numba.is_non_empty_intersection(regSet,sSet):
                val = True
                break

        return val

# NUMBA jit-able versions of the functions used above; they are slower then the compiled versions
# @nb.cfunc(nb.int64[:](nb.int64[:],nb.int64[:]) )
# def is_in_set_idx(a, b):
#     a = a.ravel()
#     n = len(a)
#     result = np.full(n, 0)
#     set_b = set(b)
#     idx = 0
#     for i in range(n):
#         if a[i] in set_b:
#             result[idx] = i
#             idx += 1
#     return result[0:idx].flatten()
# @nb.cfunc(nb.types.boolean(nb.int64[:],nb.types.Set(nb.int64, reflected=True)) )
# def is_non_empty_intersection(a, set_b):
#     retVal = False
#     a = a.ravel()
#     n = len(a)
#     # set_b = set(b)
#     for i in range(n):
#         if a[i] in set_b:
#             retVal = True
#             return retVal
#     return retVal

class setupCheckerVars(Chare):
    def init(self,selectorSetsFull,hashPElist):
        self.schedCount = 0
        self.skip = False
        if not hashPElist is None:
            if charm.myPe() in hashPElist:
                self.selectorSetsFull = selectorSetsFull
            else:
                self.selectorSetsFull = [set()]
        else:
            self.selectorSetsFull = selectorSetsFull
        # self.selectorSetsFull = [[] for k in range(len(selectorMats))]
        # # Convert the matrices to sets of 'used' hyperplanes
        # for k in range(len(selectorMats)):
        #     self.selectorSetsFull[k] = list( \
        #             map( \
        #                 lambda x: frozenset(np.flatnonzero(np.count_nonzero(x, axis=0)>0)), \
        #                 selectorMats[k] \
        #             ) \
        #         )
    def setConstraint(self,constraints, out):
        self.out = out
        # self.selectorSets = self.selectorSetsFull[out]
        self.constraints = constraints
        self.nodeIntMask = [(2**(self.constraints.N+1))-1]
    def getConstraints(self):
        return (self.constraints, self.selectorSetsFull, self.nodeIntMask, self.out)
    @coro
    def getSchedCount(self):
        return self.schedCount



# For checking on *Poset* nodes
class PosetNodeTLLVerOriginCheck(DistributedHash.Node):
    # def init(self):
    #     self.posetSuccGroupProxy, self.posetPElist = self.localProxy[self.storePe].getPosetSuccGroupProxy(ret=True).get()
    def check(self):
        # print(self.posetSuccGroupProxy)
        # self.posetSuccGroupProxy[self.data[0]].checkNode(self.nodeBytes)
        self.localProxy[ self.localProxy[self.storePe].schedRandomPosetPe(ret=True).get() ].checkNode(self.nodeBytes)
        return True

class setupCheckerVarsOriginCheck(Chare):
    def init(self,succGroupProxy,posetPElist):
        self.posetSuccGroupProxy = succGroupProxy
        self.posetPElist = posetPElist
        self.schedCount = 0
        self.skip = False

    def initialize(self,selectorSetsFull):
        self.selectorSetsFull = selectorSetsFull

    def getPosetSuccGroupProxy(self):
        return (self.posetSuccGroupProxy, self.posetPElist)
    @coro
    def schedRandomPosetPe(self):
        self.schedCount += 1
        return random.choice(self.posetPElist)
    
    def setSkip(self,val):
        # print('Executing setSkip on PE ' + str(charm.myPe()))
        self.skip = val
        # return 37
    @coro
    def reset(self):
        self.skip = False
        self.schedCount = 0
    @coro
    def getSchedCount(self):
        return self.schedCount
    
    # Legacy methods
    def setConstraint(self,constraints, out):
        self.out = out
        # self.posetSuccGroupProxy.setProperty('out',out)
        # self.selectorSets = self.selectorSetsFull[out]
        self.flippedConstraints = constraints
        self.N = self.flippedConstraints.N
        self.allN = self.flippedConstraints.allN
        self.nodeIntMask = [(2**(self.N+1))-1]
        self.schedCount = 0
        self.skip = False
    def getConstraints(self):
        return (self.flippedConstraints, self.selectorSetsFull, self.nodeIntMask, self.out)

# class successorWorkerCheck(posetFastCharm.successorWorker,Chare):
    
    @coro
    def checkNode(self,nodeBytes):
        temp = self.skip
        if not temp:
            regSet = np.full(self.allN, True, dtype=bool)
            regSet[tuple(self.flippedConstraints.flipMapSet),] = np.full(len(self.flippedConstraints.flipMapSet),False,dtype=bool)
            if self.N == self.allN:
                regSet[nodeBytes,] = np.full(len(nodeBytes),False,dtype=bool)
                unflipped = posetFastCharm_numba.is_in_set(self.flippedConstraints.flipMapSetNP,list(nodeBytes))
            else:
                sel = self.flippedConstraints.nonRedundantHyperplanes[nodeBytes,]
                regSet[sel,] = np.full(len(sel),False,dtype=bool)
                unflipped = posetFastCharm_numba.is_in_set(self.flippedConstraints.flipMapSetNP,sel.tolist())
            regSet[unflipped,] = np.full(len(unflipped),True,dtype=bool)
            regSet = np.nonzero(regSet)[0]

            val = False
            for sSet in self.selectorSetsFull[self.out]:
                if not posetFastCharm_numba.is_non_empty_intersection(regSet,sSet):
                    val = True
                    break
            # print('Done check; val = ' + str(val))
            if not val:
                # This **MUST** be an ordinary method: if it's a @coro, the entire system will fail, even with suitable .get() calls
                # This behavior is totally inexplicable: for some reason, it will fail with the infamous: "No pending future with fid= ... 
                # A common reason is sending to a future that already received its value(s)" message.
                self.thisProxy.setSkip(True)
                self.posetSuccGroupProxy[self.thisIndex].sendAll(-4,ret=True).get()

        self.schedCount -= 1
    
class TLLHypercubeReach(Chare):
    # @coro
    def __init__(self,pes):
        self.usePosetChecking = True
        self.posetPElist = list(itertools.chain.from_iterable( \
               [list(range(r[0],r[1],r[2])) for r in pes['poset']] \
            ))
        self.hashPElist = list(itertools.chain.from_iterable( \
               [list(range(r[0],r[1],r[2])) for r in pes['hash']] \
            ))
        
        if self.usePosetChecking:
            # For poset checking
            self.checkerLocalVars = Group(setupCheckerVarsOriginCheck,args=[])
        else:
            # For node checking
            self.checkerLocalVars = Group(setupCheckerVars,args=[])
        
        charm.awaitCreation(self.checkerLocalVars)

        self.poset = Chare(posetFastCharm.Poset,args=[],onPE=charm.myPe())

        if self.usePosetChecking:
             # For poset checking:    
            self.poset.init(pes, PosetNodeTLLVerOriginCheck, self.checkerLocalVars, (simple2xSuccessorWorker if simple2xAvailable else None),awaitable=True).get()
        else:
            # For node checking;
            self.poset.init(pes, PosetNodeTLLVer, self.checkerLocalVars, (simple2xSuccessorWorker if simple2xAvailable else None),awaitable=True).get()
       
        charm.awaitCreation(self.poset)

        succGroupProxy = self.poset.getSuccGroupProxy(ret=True).get()
        self.checkerLocalVars.init(succGroupProxy,self.posetPElist)

        self.ubCheckerGroup = Group(minGroupFeasibleUB)
        charm.awaitCreation(self.ubCheckerGroup)

    @coro
    def initialize(self, localLinearFns, selectorMats, inputConstraints, maxIts, useQuery, useBounding):
        self.maxIts = maxIts
        self.useQuery = useQuery
        self.useBounding = useBounding

        # Transpose local linear function kernels and selector matrices to correct for
        # Keras' multiply-on-the-right convention
        self.localLinearFns = list(map( lambda x: [np.array(x[0]).T, np.array(x[1]).reshape( (len(x[1]),1) )] ,  localLinearFns))
        self.selectorMats = [ list(map( lambda x: np.array(x).T, selectorMats[k] )) for k in range(len(selectorMats)) ]

        self.numOutputs = len(localLinearFns)
        self.n = len(localLinearFns[0][0])
        self.N = len(localLinearFns[0][0][0])
        self.M = len(selectorMats[0])
        self.m = len(localLinearFns)

        self.inputConstraintsA = np.array(inputConstraints[0])
        self.inputConstraintsb = np.array(inputConstraints[1]).reshape( (len(inputConstraints[1]),1) )

        # Find a point in the middle of the polyhedron
        self.pt = region_helpers.findInteriorPoint(np.hstack((-self.inputConstraintsb,self.inputConstraintsA)))
        if self.pt is None:
            raise ValueError('Input polytope has empty interior!')
        # self.pt = np.full(self.n,0,dtype=np.float64).reshape(-1,1)

        self.selectorSetsFull = [[] for k in range(len(selectorMats))]
        # Convert the matrices to sets of 'used' hyperplanes
        for k in range(len(self.selectorMats)):
            self.selectorSetsFull[k] = list( \
                    map( \
                        lambda x: set(np.flatnonzero(np.count_nonzero(x, axis=0)>0)), \
                        self.selectorMats[k] \
                    ) \
                )

        # Defunct
        # self.poset.setSuccessorCommonProperty('selectorSetsFull',self.selectorSetsFull,awaitable=True).get()
        if self.usePosetChecking:
            # For poset checking:
            self.checkerLocalVars.initialize(self.selectorSetsFull)
        else:
            # For node checking:
            self.checkerLocalVars.init(self.selectorSetsFull,self.hashPElist)
        
        
        stat = self.poset.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, awaitable=True)
        stat.get()
        
        stat = self.ubCheckerGroup.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, self.selectorMats, awaitable=True)
        stat.get()

        self.copyTime = 0
        self.posetTime = 0
        self.workerInitTime = 0
        

    @coro
    def computeReach(self, lbSeed=-1, ubSeed=1, tol=1e-3):
        self.hypercube = np.ones((self.m, 2))
        print('m = ' + str(self.m))
        for out in range(self.m):
            self.hypercube[out,0] = self.searchBound(lbSeed,out=out,lb=True,tol=tol)
            self.hypercube[out,1] = self.searchBound(ubSeed,out=out,lb=False,tol=tol)
        return self.hypercube

    @coro
    def searchBound(self,seedBd,out=0,lb=True,tol=1e-3,verbose=False):
        if out >= self.m:
            raise ValueError('Output ' + str(out) + ' is greater than m = ' + str(self.m))
        # lb2ub = 1
        # if not lb:
        #     lb2ub = -1
        straddle = False
        windLB = -np.inf
        windUB = seedBd
        searchDir = 0
        prevBD = seedBd
        itCnt = self.maxIts
        
        while itCnt > 0:
            bdToCheck = windUB if windLB==-np.inf else 0.5*(windLB + windUB)
            ver = self.verifyLB( bdToCheck, out=out) if lb else self.verifyUB( bdToCheck,out=out)
            
            if verbose:
                print( 'Iteration ' + str(itCnt) +  ': ' + str(bdToCheck) + ' is ' + ('a VALID' if ver else 'an INVALID') + ' lower bound!')
            if windLB == -np.inf:
                # If this is the first pass, decide which way to start looking
                # based on ver:
                if searchDir == 0:
                    searchDir = 1 if ver else -1
                if ver and searchDir > 0:
                    # We're searching right, which means prevBD was a valid lower bound
                    # windUB is a valid lower bound, too, so keep searching right
                    prevBD = windUB
                    searchDir = 1
                    windUB += np.exp(self.maxIts-itCnt)
                elif ver and searchDir < 0:
                    # we were searching left, which means prevBD was NOT a lower bound
                    # Hence, we're now straddling the actual lower bound
                    windLB = windUB
                    windUB = prevBD
                    straddle = True
                elif not ver and searchDir > 0:
                    # We were searching right, which means prevBD WAS a lower bound
                    # Hence, we're now straddling the actual lower bound
                    windLB = prevBD
                    straddle = True
                elif not ver and searchDir < 0:
                    # We're searching left, which means prevBD was not a lower bound
                    # windUB is still not a lower bound, so keep searching left
                    prevBD = windUB
                    searchDir = -1
                    windUB -= np.exp(self.maxIts-itCnt)
            else:
                # Now we know that windLB < actual bound < windUB, and we called the verify function
                # with the midpoint 0.5*(windLB + windUB)
                if ver:
                    windLB = bdToCheck
                else:
                    windUB = bdToCheck
                if np.abs(windUB-windLB) < tol:
                    break
            
            itCnt -= 1
        if not straddle:
            if lb:
                windLB = -np.inf if searchDir < 0 else windUB
            else:
                windUB = np.inf if searchDir > 0 else windUB
        if verbose:
            print('**********    ' + ('verifyLB on LB' if lb else 'verifyUB on UB') + ' processing times:   **********')
            if lb:
                print('Total time required to initialize the new lb problem: ' + str(self.copyTime))
                # collectTimeFut = Future()
                # self.checkerGroup.workerInitTime(collectTimeFut)
                # self.workerInitTime = collectTimeFut.get()
                print('Total time required for region check workers to initialize: ' + str(self.workerInitTime))
                print('Total time required for (partial) poset calculation: ' + str(self.posetTime))
            print('Iterations used: ' + str(self.maxIts - itCnt))
            if not lb:
                print('Total number of LPs used for Upper Bound verification: ' + str(sum(self.ubCheckerGroup.getLPcount(ret=True).get())))
            print('***********************************************************')
        return windLB if lb else windUB

    @coro
    def verifyLB(self,lb, out=0, timeout=None, method='fastLP'):
        if out >= self.m:
            raise ValueError('Output ' + str(out) + ' is greater than m = ' + str(self.m))
        
        t = time.time()
        
        stat = self.poset.setConstraint(lb, out=out, timeout=timeout, awaitable=True)
        stat.get()
        self.checkerLocalVars.setConstraint(self.poset.getConstraintsObject(ret=True).get(),out,awaitable=True).get()

        self.copyTime += time.time() - t # Total time across all PEs to set up a new problem

        t = time.time()
        retVal = self.poset.populatePoset(method=method,solver='glpk',findAll=False,useQuery=self.useQuery,useBounding=self.useBounding,ret=True).get() # specify retChannelEndPoint=self.thisProxy to send to a channel as follows
        self.posetTime += time.time() - t

        return retVal
    
    @coro
    def verifyUB(self,ub,out=0, timeout=None):
        if out >= self.m:
            raise ValueError('Output ' + str(out) + ' is greater than m = ' + str(self.m))
        self.ubCheckerGroup.reset(timeout,awaitable=True).get()
        timedOut = self.ubCheckerGroup.checkMinGroup(ub,out, ret=True)
        minCheckFut = Future()
        self.ubCheckerGroup.collectMinGroupStats(minCheckFut,ret=True)
        
        retVal = minCheckFut.get()
        timedOut = any(timedOut.get())
        print('Upper Bound verifiction used ' + str(sum(self.ubCheckerGroup.getLPcount(ret=True).get())) + ' total LPs.')
        if timedOut:
            retVal = None
            print('Upper bound verification timed out.')
        return retVal


class minGroupFeasibleUB(Chare):

    def initialize(self, AbPairs, pt, fixedA, fixedb, selectorMats):
        self.constraints = None
        self.AbPairs = AbPairs
        self.pt = pt
        self.fixedA = fixedA
        self.fixedb = fixedb
        self.N = len(self.AbPairs[0][0])
        self.n = len(self.AbPairs[0][0][0])
        self.selectorMatsFull = selectorMats
        
        self.selectorSetsFull = [[] for k in range(len(selectorMats))]
        # Convert the matrices to sets of 'used' hyperplanes
        for k in range(len(selectorMats)):
            self.selectorSetsFull[k] = list( \
                    map( \
                        lambda x: frozenset(np.flatnonzero(np.count_nonzero(x, axis=0)>0)), \
                        self.selectorMatsFull[k] \
                    ) \
                )
        
        self.lp = encapsulateLP.encapsulateLP()

        self.selectorIndex = -1
        self.loopback = Channel(self,remote=self.thisProxy[self.thisIndex])
        self.workDone = False
        pes = list(range(charm.numPes()))
        pes.pop(charm.myPe())
        self.otherProxies = [self.thisProxy[k] for k in pes]
        self.tol = 1e-10
    @coro
    def reset(self,timeout):
        self.workDone = False
        self.clockTimeout = time.time() + timeout if timeout is not None else None
    @coro
    def checkMinGroup(self, ub, out):
        self.status = Future()

        for mySelector in range(charm.myPe(),len(self.selectorSetsFull[out]),charm.numPes()):
            self.loopback.send(1)
            self.loopback.recv()
            if self.workDone:
                break
            if self.clockTimeout is not None and time.time() > self.clockTimeout:
                self.status.send(False)
                return True
            n = self.AbPairs[out][0].shape[1]
            # Actually do the feasibility check:
            ubShift = self.AbPairs[out][1][list(self.selectorSetsFull[out][mySelector]),:]
            ubShift = ubShift - ub*np.ones(ubShift.shape)
            bVec = np.vstack([ ubShift , -1*self.fixedb ]).T.flatten()
            selHypers = self.AbPairs[out][0][list(self.selectorSetsFull[out][mySelector]),:]
            status, sol = self.lp.runLP( \
                    np.ones(self.n,dtype=np.float64), \
                    -1*np.vstack([ selHypers, self.fixedA ]), \
                    bVec, \
                    lpopts = {'solver':'glpk'}
                )
            # TO DO: account for intersections that are on the boundary of the input polytope
            if status == 'optimal':
                full = np.vstack([ selHypers, self.fixedA ]) 
                actHypers = np.nonzero(np.abs( full @ sol + bVec) <= self.tol)[0]
                # print('actHypers = ' + str(actHypers))
                # print(sol)
                if len(actHypers) == 0 or np.all((selHypers @ sol + ubShift.flatten()) + self.tol >= 0):
                    for pxy in self.otherProxies:
                        pxy.setDone()
                    self.status.send(True)
                    return False
                distinctCount = 1
                solList = [np.array(sol)]
                # print(solList)
                for k in actHypers:
                    # Try to get away from the kth active hyperplane
                    newStatus, newSol = self.lp.runLP( \
                                -full[k,:], \
                                -1*full, \
                                bVec, \
                                lpopts = {'solver':'glpk'}
                            )
                    # print('newSol = ' + str(newSol))
                    # print('Solution difference: '  + str(np.abs(newSol - sol)))
                    if k < len(selHypers) and np.abs(selHypers[k,:] @ newSol + ubShift[k]) <= self.tol \
                        and np.abs(selHypers[k,:] @ solList[-1] + ubShift[k]) <= self.tol \
                        and all([np.linalg.norm(prevSol - newSol) > self.tol for prevSol in solList]):
                        # The feasible set contains a local linear function that is always equal to the upper bound
                        # we're testing, hence the min of this selector set is exactly equal to that upper bound
                        # Hence, this min term does not generate a violation, so we should move on to the next min term
                        print('Degeneracy condition: ' + str(np.abs(selHypers[k,:] @ newSol + ubShift[k])))
                        print('Solution difference: '  + str(np.linalg.norm(newSol - sol)))
                        print('newSol = ' + str(newSol))
                        print('Degenerate upper bound detected')
                        break
                    # print('solList internal: ' + str(solList))
                    if all([np.linalg.norm(prevSol - newSol) > self.tol for prevSol in solList]):
                        # This is a new solution
                        distinctCount += 1
                        solList.append(np.array(newSol))
                        interiorPoint = np.sum(np.hstack(solList),axis=1)/(n+1)
                        # print('Violation condition: ' + str((selHypers @ interiorPoint)+ubShift.flatten()  ) + ' Distinct count ' + str(distinctCount) + ' ' + str(n))
                        # print('LHS = ' + str(selHypers @ np.hstack(solList)) + ' RHS = '  + str(ubShift))
                        # print('Compare LHS = ' + str((np.transpose(selHypers @ np.hstack(solList)) + ubShift.flatten()) + self.tol >= 0) )
                        # print('solList ' + str(solList))
                        # print('selHypers @ interiorPoint = ' + str((selHypers @ interiorPoint) + ubShift.flatten()))
                        if (distinctCount == n + 1 and \
                            np.all(selHypers @ interiorPoint + ubShift.flatten() > self.tol)) or \
                            np.all((selHypers @ newSol + ubShift.flatten()) + self.tol >= 0):
                            # This feasible set has a nonempty interior, so we have a violation
                            # print('sending true')
                            for pxy in self.otherProxies:
                                pxy.setDone()
                            self.status.send(True)
                            return False
        self.status.send(False)
        return False

    @coro
    def collectMinGroupStats(self, stat_result):
        self.reduce(stat_result, self.status.get(), Reducer.logical_or)
    @coro
    def getLPcount(self):
        return self.lp.lpCount
    @coro
    def setDone(self):
        self.workDone = True



# Helper functions:
