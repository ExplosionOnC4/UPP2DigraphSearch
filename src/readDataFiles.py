import numpy as np
import os
import ast
from utilsUPP import *

baseDir = os.path.split(os.path.dirname(__file__))[0]
dataDir = os.path.join(baseDir, 'data')

def readOrderedComplete4() -> list[np.ndarray]:
    '''
    Reads in all the 4-CDGs from the file of their ordered binary representations (the canonical list)
    '''

    with open(os.path.join(dataDir, 'orderedComplete4'), 'r') as f:
        ls = f.read().split('\n')[:-1]
    return list(map(lambda binRep : getAdjacencyMatrixFromSet(convertBinaryToSetRepresentation(binRep), 16), ls))

def readMajorGk(k: int, keepCerts=False) -> list[np.ndarray]:
    '''
    Read in all CDGs from file containing all CDG in the major component of G_k by switchings
    '''

    with open(os.path.join(dataDir, f'majorGk{k}.txt'), 'r') as f:
        dict = ast.literal_eval(f.read())
    if keepCerts:
        return [cert for cert in dict]
    else:
        return [recoverAdjMatrixFromNautyGraph(recoverGraphFromNautyCert(cert, k ** 2)) for cert in dict]

if __name__ == '__main__':
    ls = readOrderedComplete4()