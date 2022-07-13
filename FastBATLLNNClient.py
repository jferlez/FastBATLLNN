import sys
import requests
import json
import time
serverURL = 'http://localhost:8000/'

if len(sys.argv) >= 2 and sys.argv[1] == 'setProblem' or sys.argv[0][-min(20,len(sys.argv[0])):] != 'FastBATLLNNClient.py':

    import TLLnet
    import numpy as np
    from vnnlib import read_vnnlib_simple

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
        t = time.time()
        tll = TLLnet.TLLnet.fromONNX(onnxFile)
        tllDict = tll.save()
        print(f'Used {time.time()-t} seconds to import ONNX file...')

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


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        sys.exit(1)
    
    command = sys.argv[1]

    if command == 'shutdown':
        result = shutdown()
        sys.exit()
    elif command == 'getResult':
        if len(sys.argv) >= 4:
            result = getResult(sys.argv[2], sys.argv[3])
            result = json.loads(result)
            if 'RESULT' in result:
                print(result['RESULT'])
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            sys.exit(1)
    elif command == 'setProblem':
        if len(sys.argv) >= 4:
            result = setProblem(sys.argv[2], sys.argv[3])
        else:
            sys.exit(1)
    else:
        sys.exit(1)