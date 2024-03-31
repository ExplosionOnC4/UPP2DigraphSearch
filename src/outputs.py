# Stores any code which was used to output important data in one file

# In case not run from sagemath
try:
    from genUPPDigraphs import *
except ModuleNotFoundError:
    pass

from switchingsUPP import *
from utilsUPP import *
from knuthianUPP import *
import os
import numpy as np
import pynauty as nauty

baseDir = os.path.split(os.path.dirname(__file__))[0]
dataDir = os.path.join(baseDir, 'data')

def writeMajorConnectComponent(k: int):
    stand = createStandardCentralDigraph(k)
    standComponent = switchConnectedComponentFromVertex(stand)
    print(len(standComponent))
    with open(os.path.join(dataDir, f'majorGk{k}.txt'), 'w') as f:
        f.write(str(standComponent))

def writeCompleteSearch(k: int):
    ls = genUPPByBlockDFS(k)
    print(len(ls))
    file = os.path.join(dataDir, f'completeSearch{k}')
    np.savez(file, *ls)
    npz = np.load(file+'.npz')
    for arr in npz.files[:-3]:
        print(npz[arr])

def writeCompleteKnuthian(k: int):
    hash = set()
    with open(os.path.join(dataDir, f'completeKnuthian{k}.txt'), 'a') as f:
        for tab in genProductTables(k):
            adj = getKnuthianAdjMatrix(tab)
            if nauty.certificate(createNautyGraphFromAdjMatrix(adj)) not in hash:
                f.write(convertAdjMatrixToBinaryString(adj) + '\n')
                hash.add(nauty.certificate(createNautyGraphFromAdjMatrix(adj)))

if __name__=='__main__':
    # writeMajorConnectComponent(3)
    # writeCompleteSearch(4)
    writeCompleteKnuthian(6)