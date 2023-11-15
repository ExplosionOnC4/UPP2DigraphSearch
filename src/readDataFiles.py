import numpy as np
import os
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
