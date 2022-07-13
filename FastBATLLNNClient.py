import TLLnet
import numpy as np
import pickle
import sys
import os
import re
import copy
import csv
from vnnlib import read_vnnlib_simple
import requests

serverURL = 'http://localhost:8000/'

def setProblem(onnxFile,vnnlibFile):

    spec = read_vnnlib_simple(vnnlibFile,2,1)
    # print(spec)
    # [
    #     (
    #         [[-2.0, 2.0], [-2.0, 2.0]], 
    #         [
    #             (array([[1.]]), array([-1.74903255]))
    #         ]
    #     )
    # ]

    tll = TLLnet.TLLnet.fromONNX(onnxFile)
    tllDict = tll.save()

    n = tllDict['n']
    A_in = np.zeros((2*n,n))
    np.fill_diagonal(A_in[:n], 1)
    np.fill_diagonal(A_in[n:], -1)
    b_in = np.ones((2*n,))
    b_in[:n] = np.array([inputConst[0] for inputConst in spec[0][0]])
    b_in[n:] = np.array([-inputConst[1] for inputConst in spec[0][0]])
    
    tllDict['A_in'] = A_in.tolist()
    tllDict['b_in'] = b_in.tolist()

    A_out, b_out = spec[0][1][0]
    tllDict['A_out'] = float(A_out)
    tllDict['b_out'] = float(b_out)

    tllDict['localLinearFns'] = localLinToPy(tllDict['localLinearFns'])
    tllDict['selectorSets'] = selectorsToList(tllDict['selectorSets'])

    tllDict['id'] = '[' + onnxFile + '][' + vnnlibFile + ']'
    tllDict['COMMAND'] = 'NEW_PROBLEM'
    # print(tllDict)
    r = requests.post(serverURL + 'post', json=tllDict)
    return tllDict

def getResult(onnxFile,vnnlibFile):
    tllDict = {}
    tllDict['id'] = '[' + onnxFile + '][' + vnnlibFile + ']'
    tllDict['COMMAND'] = 'GO'
    r = requests.post(serverURL + 'post', json=tllDict)

    result = requests.get(serverURL)

    return result


def shutdown():
    tllDict = {}
    tllDict['COMMAND'] = 'SHUTDOWN'
    r = requests.post(serverURL + 'post', json=tllDict)


def localLinToPy(localLinearFns):
    return [
        [ x.tolist() for x in output ] for output in localLinearFns
    ]
def selectorsToList(selectorSets):
    return [
        [ [int(idx) for idx in x] for x in output ] for output in selectorSets
    ]