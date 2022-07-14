import numpy as np
import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
import TLLnet
import posetFastCharm
import TLLHypercubeReach

import time
import pickle

# Server code from https://gist.github.com/mdonkers/63e115cc0c79b4f6b8b3a6b797e485c7
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import json

class ShutdownServer(Exception):
    pass



class Server(Chare):
    def __init__(self,processorRemote):

        toProcessorsChannel = Channel(self,processorRemote)
        fromProcessorsChannel = Channel(self,processorRemote)

        workingBase = [False]
        problem_setBase = [False]

        self.toProcessorsChannel = toProcessorsChannel
        self.fromProcessorsChannel = fromProcessorsChannel

        self.working = workingBase
        self.problem_set = problem_setBase

        class S(BaseHTTPRequestHandler):
            toProcChan = toProcessorsChannel
            fromProcChan = fromProcessorsChannel

            working = workingBase
            problem_set = problem_setBase
            
            def _set_response(self,content=None):
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                if not content is None:
                    self.wfile.write(content)
                    # self.wfile.close()

            def do_GET(self):

                if not self.working[0] or not self.problem_set[0]:
                    logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
                    self._set_response()
                    self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))
                else:
                    result = self.fromProcChan.recv()
                    dataContent = bytes(json.dumps(result,separators=(',', ':')), encoding='utf8')
                    print(f'Sending {dataContent}')
                    self._set_response(content=dataContent)
                    self.working[0] = False
                    self.problem_set[0] = False

            def do_POST(self):
                content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
                post_data = self.rfile.read(content_length) # <--- Gets the data itself
                data = json.loads(post_data.decode('utf-8'))
                logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                        str(self.path), str(self.headers), data)
                
                self._set_response()
                self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))
                if 'COMMAND' in data:
                    if data['COMMAND'] == 'NEW_PROBLEM':
                        if not self.working[0]:
                            self.toProcChan.send(data)
                            self.problem_set[0] = True
                        else:
                            print('ERROR: recieved new problem before finishing with the last one')
                    elif data['COMMAND'] == 'GO':
                        print([self.problem_set[0],self.working[0]])
                        if self.problem_set[0] and not self.working[0]:
                            self.toProcChan.send(data)
                            self.working[0] = True
                        else:
                            print('ERROR: no problem has been set yet')

                    elif data['COMMAND'] == 'SHUTDOWN':
                        self.toProcChan.send(data)
                        raise KeyboardInterrupt()
        self.handler_class = S
    @coro
    def run(self,server_class=HTTPServer, port=8080):
        logging.basicConfig(level=logging.INFO)
        server_address = ('', port)
        httpd = server_class(server_address, self.handler_class)
        logging.info('Starting httpd...\n')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        httpd.server_close()
        logging.info('Stopping httpd...\n')

def localLinToNumpy(localLinearFns):
    return [
        [ np.array(x,dtype=np.float64) for x in output ] for output in localLinearFns
    ]

def selectorsToSet(selectorSets):
    return [
        [ set(x) for x in output ] for output in selectorSets
    ]

class FastBATLLNNServer(Chare):
    
    def __init__(self,args):

        # Instantiate HTTP server
        serverTask = Chare(Server, args=[self.thisProxy], onPE=charm.numPes()-1)
        charm.awaitCreation(serverTask)
        # Create channels to/from HTTP server PE
        fromServerChannel = Channel(self,serverTask)
        toServerChannel = Channel(self,serverTask)

        # Instantiate FastBATLLNN
        pes = {'poset':[(0,charm.numPes()-1,1)],'hash':[(0,charm.numPes()-1,1)]}
        useQuery = False
        useBounding = False
        tllReach = Chare(TLLHypercubeReach.TLLHypercubeReach, args=[pes])
        charm.awaitCreation(tllReach)

        # Start listening for HTTP connnections
        serverDone = serverTask.run(awaitable=True)    

        # Get the first command via HTTP
        msg = fromServerChannel.recv()

        timeout=300

        while msg and type(msg) is dict:

            if not 'COMMAND' in msg:
                # ignore this message, and get the next
                msg = fromServerChannel.recv()
                continue
            
            if msg['COMMAND'] == 'SHUTDOWN':
                toServerChannel.send({})
                break
            
            if msg['COMMAND'] == 'NEW_PROBLEM':

                validProc = True
                problemID = 'NULL'
                for k in ['A_in','b_in','A_out','b_out','n','N','M','m','localLinearFns','selectorSets','TLLFormatVersion','id']:
                    if not k in msg:
                        validProc = False
                
                # We got a valid new problem (more or less), so set things up to run FastBATLLNN
                if validProc:
                    msg['localLinearFns'] = localLinToNumpy(msg['localLinearFns'])
                    msg['selectorSets'] = selectorsToSet(msg['selectorSets'])
                    for k in ['A_in', 'b_in']:
                        msg[k] = np.array(msg[k],dtype=np.float64)

                    tll = TLLnet.TLLnet.fromTLLFormat(msg)
                    constraints = [msg['A_in'] , msg['b_in']]
                    problemID = msg['id']
                    A_out = msg['A_out']
                    b_out = msg['b_out']

                    tllReach.initialize(tll , constraints, 100, useQuery, useBounding,awaitable=True).get()
                
                # Now wait for either a "GO" or "SHUTDOWN" command
                msg = fromServerChannel.recv()
                while msg:
                    if type(msg) is dict and 'COMMAND' in msg:
                        if msg['COMMAND'] == 'SHUTDOWN' or msg['COMMAND'] == 'GO':
                            break
                    else:
                        msg = fromServerChannel.recv()
                
                if msg['COMMAND'] == 'SHUTDOWN':
                    toServerChannel.send({})
                    break

                # If we got a GO command for some other problem, then return an invalid result
                if not 'id' in msg or msg['id'] != problemID:
                    validProc = False
                if 'timeout' in msg:
                    timeout = msg['timeout']
                # We recieved a GO command
                if not validProc:
                    toServerChannel.send({'id':problemID,'RESULT':'INVALID'})
                else:
                    # Here is where we will actually run FastBATLLNN
                    
                    if A_out < 0:
                        result = bool(tllReach.verifyLB(b_out,timeout=(timeout if timeout > 0 else None),ret=True).get()) # verify NN >= a: True/1 == SAT; False/0 == UNSAT
                    else:
                        result = not bool(tllReach.verifyUB(b_out,timeout=(timeout if timeout > 0 else None),ret=True).get()) # verify NN <= b: True/1 == UNSAT; False/0 == SAT

                    toServerChannel.send({'id':problemID,'RESULT':'UNSAT' if result else 'SAT'})
            
            # Now wait for either a "GO" or "SHUTDOWN" command
            msg = fromServerChannel.recv()

        serverDone.get()

        charm.exit()
        
charm.start(FastBATLLNNServer,modules=['posetFastCharm','TLLHypercubeReach','DistributedHash'])