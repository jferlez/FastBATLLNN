
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
import TLLnet

# try:
#     from simple2xSuccessorWorker import simple2xSuccessorWorker
#     from simple2xPosetFastCharm import PosetSimple2x
#     simple2xAvailable = True
# except ImportError:
simple2xAvailable = False


cvxopt.solvers.options['show_progress'] = False

# All hyperplanes assumed to be specified as A x >= b

# For checking on *Hash* Nodes
class PosetNodeTLLVer(DistributedHash.Node):
    def init(self):
        self.constraints, self.selectorSetsFull, self.nodeIntMask, self.out = self.localProxy[self.storePe].getConstraints(ret=True).get()
    def check(self):
        if type(self.nodeBytes) == bytearray:
            nodeBytes = tuple(posetFastCharm.bytesToList(self.nodeBytes,self.constraints.wholeBytes,self.constraints.tailBits))
        else:
            nodeBytes = self.nodeBytes

        regSet = self.constraints.translateRegion(nodeBytes,allN=True)

        val = False
        for sSet in self.selectorSetsFull[self.out]:
            if not posetFastCharm_numba.is_non_empty_intersection(regSet,sSet):
                val = True
                break

        return val

# NUMBA jit-able versions of the functions used above; they are slower than the compiled versions
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
    def initialize(self,selectorSetsFull,hashPElist):
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
    @coro
    def getCounterExample(self):
        return None



# For checking on *Poset* nodes
class PosetNodeTLLVerOriginCheck(DistributedHash.Node):
    # Optional init that copies (by reference) the list of poset PEs to make available to Node methods
    # def init(self):
    #     self.posetPElist = self.localProxy[self.storePe].getPosetPEList(ret=True).get()
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
        self.counterExample = None

    def initialize(self,selectorSetsFull):
        self.selectorSetsFull = selectorSetsFull
        self.counterExample = None

    def getPosetSuccGroupProxy(self):
        return (self.posetSuccGroupProxy, self.posetPElist)
    @coro
    def schedRandomPosetPe(self):
        # self.schedCount += 1
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
    def getCounterExample(self):
        return self.counterExample

    # Legacy methods
    @coro
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
        self.counterExample = None
    def getConstraints(self):
        return (self.flippedConstraints, self.selectorSetsFull, self.nodeIntMask, self.out)
    def getPosetPEList(self):
        return self.posetPElist

# class successorWorkerCheck(posetFastCharm.successorWorker,Chare):

    @coro
    def checkNode(self,nodeBytes):
        temp = self.skip
        if not temp:
            if type(nodeBytes) == bytearray:
                nodeBytes = tuple(posetFastCharm.bytesToList(nodeBytes,self.flippedConstraints.wholeBytes,self.flippedConstraints.tailBits))
            regSet = self.flippedConstraints.translateRegion(nodeBytes,allN=True)

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
                self.counterExample = copy(nodeBytes)
                self.posetSuccGroupProxy[self.thisIndex].sendAll(-4,ret=True).get()

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

        if simple2xAvailable and 'gpu' in pes:
            self.poset = Chare(PosetSimple2x,args=[],onPE=charm.myPe())
        else:
            self.poset = Chare(posetFastCharm.Poset,args=[],onPE=charm.myPe())

        charm.awaitCreation(self.poset)

        if self.usePosetChecking:
             # For poset checking:
            self.poset.init(pes, PosetNodeTLLVerOriginCheck, self.checkerLocalVars, (simple2xSuccessorWorker if simple2xAvailable else None),awaitable=True).get()
        else:
            # For node checking;
            self.poset.init(pes, PosetNodeTLLVer, self.checkerLocalVars, (simple2xSuccessorWorker if simple2xAvailable else None),awaitable=True).get()


        succGroupProxy = self.poset.getSuccGroupProxy(ret=True).get()
        if self.usePosetChecking:
            self.checkerLocalVars.init(succGroupProxy,self.posetPElist)

        self.ubCheckerGroup = Group(minGroupFeasibleUB)
        charm.awaitCreation(self.ubCheckerGroup)
        self.lpObj = encapsulateLP.encapsulateLP()
        # self.lpObj.initSolver(solver='glpk')

    @coro
    def initialize(self, tll, inputConstraints, maxIts, useQuery):
        self.maxIts = maxIts
        self.useQuery = useQuery

        # Transpose local linear function kernels and selector matrices to correct for
        # Keras' multiply-on-the-right convention
        # self.localLinearFns = list(map( lambda x: [np.array(x[0]).T, np.array(x[1]).reshape( (len(x[1]),1) )] ,  localLinearFns))
        # self.selectorMats = [ list(map( lambda x: np.array(x).T, selectorMats[k] )) for k in range(len(selectorMats)) ]

        # self.numOutputs = len(localLinearFns)
        # self.n = len(localLinearFns[0][0])
        # self.N = len(localLinearFns[0][0][0])
        # self.M = len(selectorMats[0])
        # self.m = len(localLinearFns)
        self.tll = tll

        self.localLinearFns = [ [kernBias[0].copy(), kernBias[1].copy().reshape( (-1,1) )] for kernBias in tll.localLinearFns ]
        self.selectorSetsFull = deepcopy(tll.selectorSets)

        self.numOutputs = tll.m
        self.n = tll.n
        self.N = tll.N
        self.M = tll.M
        self.m = tll.m

        self.inputConstraintsA = np.array(inputConstraints[0])
        self.inputConstraintsb = np.array(inputConstraints[1]).reshape( (len(inputConstraints[1]),1) )

        # Find a point in the middle of the polyhedron
        self.pt = region_helpers.findInteriorPoint(np.hstack((-self.inputConstraintsb,self.inputConstraintsA)))
        if self.pt is None:
            raise ValueError('Input polytope has empty interior!')
        # self.pt = np.full(self.n,0,dtype=np.float64).reshape(-1,1)

        if self.usePosetChecking:
            # For poset checking:
            self.checkerLocalVars.initialize(self.selectorSetsFull)
        else:
            # For node checking:
            self.checkerLocalVars.initialize(self.selectorSetsFull,self.hashPElist)


        stat = self.poset.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, awaitable=True)
        stat.get()

        stat = self.ubCheckerGroup.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, self.selectorSetsFull, awaitable=True)
        stat.get()

        self.copyTime = 0
        self.posetTime = 0
        self.workerInitTime = 0
        self.cePoint = None
        self.cePointVal = None


    @coro
    def computeReach(self, lbSeed=-1, ubSeed=1, tol=1e-3, opts={}):
        hypercube = np.ones((self.m, 2))
        for out in range(self.m):
            hypercube[out,0] = self.thisProxy.searchBound(lbSeed,out=out,lb=True,tol=tol,opts=opts,ret=True).get()
            hypercube[out,1] = self.thisProxy.searchBound(ubSeed,out=out,lb=False,tol=tol,opts=opts,ret=True).get()
        return hypercube

    @coro
    def searchBound(self,seedBd,out=0,lb=True,tol=1e-3,opts={}):
        if 'verbose' in opts:
            verbose = opts['verbose']
        else:
            verbose = False
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
            ver = self.verifyLB( bdToCheck, out=out, opts=opts) if lb else self.verifyUB( bdToCheck,out=out, verbose=verbose)

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
    def verifyLB(self,lb, out=0, timeout=None, opts={}):
        if out >= self.m:
            raise ValueError('Output ' + str(out) + ' is greater than m = ' + str(self.m))
        self.cePoint = None
        self.cePointVal = None
        self.prefilter = True
        if 'prefilter' in opts:
            self.prefilter = opts['prefilter']

        t = time.time()

        stat = self.poset.setConstraint(lb, out=out, timeout=timeout, prefilter=self.prefilter, awaitable=True)
        stat.get()
        self.checkerLocalVars.setConstraint(self.poset.getConstraintsObject(ret=True).get(),out,awaitable=True).get()

        self.copyTime += time.time() - t # Total time across all PEs to set up a new problem

        opts['useQuery'] = self.useQuery

        t = time.time()
        retVal = self.poset.populatePoset(opts, ret=True).get() # specify retChannelEndPoint=self.thisProxy to send to a channel as follows
        self.posetTime += time.time() - t

        if not retVal:
            t = time.time()
            ceList = self.checkerLocalVars.getCounterExample(ret=True).get()
            #print(f'Waited {time.time() - t} seconds to receive counterexample.')
            for ce in ceList:
                if ce is not None:
                    # ce is now a list of flipped hyperplanes corresponding to a counterexample region (w.r.t. the ORIGINAL constraints)
                    t = time.time()
                    self.cePoint = self.poset.getConstraintsObject(ret=True).get().regionInteriorPoint(ce)
                    #print(f'Used {time.time()-t} seconds to find an interior point.')
                    t = time.time()
                    self.cePointVal = self.tll.pointEval(self.cePoint)
                    #print(f'Used {time.time() - t} seconds to evaluate TLL at interior point.')
                    #print(f'Found counterexample TLL({self.cePoint}) = {self.cePointVal}')
                    break
        return retVal

    @coro
    def verifyUB(self,ub,out=0, timeout=None, verbose=False, **kwargs):
        if out >= self.m:
            raise ValueError('Output ' + str(out) + ' is greater than m = ' + str(self.m))
        self.cePoint = None
        self.cePointVal = None
        self.ubCheckerGroup.reset(timeout,awaitable=True).get()
        timedOut = self.ubCheckerGroup.checkMinGroup(ub,out, ret=True)
        minCheckFut = Future()
        self.ubCheckerGroup.collectMinGroupStats(minCheckFut,ret=True)

        retVal = minCheckFut.get()
        timedOut = any(timedOut.get())
        if verbose:
            print('Upper Bound verifiction used ' + str(sum(self.ubCheckerGroup.getLPcount(ret=True).get())) + ' total LPs.')
        if timedOut:
            retVal = None
            print('Upper bound verification timed out.')
        if retVal:
            t = time.time()
            ceList = self.ubCheckerGroup.getCounterExample(ret=True).get()
            if verbose:
                print(f'Used {time.time()-t} seconds to receive counterexample.')
            self.cePoint = None
            self.cePointVal = None
            for ce in ceList:
                if ce is not None:
                    self.cePoint = np.array(ce,dtype=np.float64).reshape(self.n,1)
                    t = time.time()
                    self.cePointVal = self.tll.pointEval(self.cePoint)
                    if verbose:
                        print(f'Used {time.time() - t} seconds to evaluate TLL at interior point.')
                        print(f'Found counterexample TLL({self.cePoint}) = {self.cePointVal}')
                    break
        return retVal

    @coro
    def getCounterExamplePoint(self):
        return [copy(self.cePoint), copy(self.cePointVal)]


class minGroupFeasibleUB(Chare):

    def initialize(self, AbPairs, pt, fixedA, fixedb, selectorSets):
        self.constraints = None
        self.AbPairs = deepcopy(AbPairs)
        self.pt = pt
        self.fixedA = fixedA.copy()
        self.fixedb = fixedb.copy()
        self.N = len(self.AbPairs[0][0])
        self.n = len(self.AbPairs[0][0][0])
        self.selectorSetsFull = selectorSets
        # self.selectorMatsFull = selectorMats

        # self.selectorSetsFull = [[] for k in range(len(selectorMats))]
        # # Convert the matrices to sets of 'used' hyperplanes
        # for k in range(len(selectorMats)):
        #     self.selectorSetsFull[k] = list( \
        #             map( \
        #                 lambda x: frozenset(np.flatnonzero(np.count_nonzero(x, axis=0)>0)), \
        #                 self.selectorMatsFull[k] \
        #             ) \
        #         )

        self.lp = encapsulateLP.encapsulateLP()

        self.selectorIndex = -1
        self.loopback = Channel(self,remote=self.thisProxy[self.thisIndex])
        self.workDone = False
        pes = list(range(charm.numPes()))
        pes.pop(charm.myPe())
        self.otherProxies = [self.thisProxy[k] for k in pes]
        self.tol = 1e-10
        self.cePoint = None
    @coro
    def reset(self,timeout):
        self.workDone = False
        self.clockTimeout = time.time() + timeout if timeout is not None else None
    @coro
    def checkMinGroup(self, ub, out):
        self.status = Future()
        self.cePoint = None
        for mySelector in range(charm.myPe(),len(self.selectorSetsFull[out]),charm.numPes()):
            self.loopback.send(1)
            self.loopback.recv()
            #tempFut = Future()
            #tempFut.send(-1)
            #tempFut.get()
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

            status, sol = region_helpers.findInteriorPointFull( \
                                np.hstack([ \
                                    bVec.reshape(-1,1), np.vstack([ selHypers, self.fixedA]) \
                                ]), \
                                lpObj=self.lp \
                            )
            if status == 'optimal':
                if sol[-1] > self.tol or np.all(((selHypers @ sol[:-1].reshape(-1,1)).flatten() + ubShift.flatten()) - self.tol >= 0):
                    self.cePoint = sol[:-1].reshape(-1,1)
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
    @coro
    def getCounterExample(self):
        return self.cePoint



# Helper functions:

