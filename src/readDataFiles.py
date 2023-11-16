import numpy as np
import os
import ast
import collections
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
    # Messing around with results/trying to find patterns

    # the 4-CDGs not switch equivalent to canonical 4-CDG are 
    # [0, 174, 258, 288, 290, 298, 301, 309, 310, 317, 501, 653, 684, 1591, 1966, 1967, 2027, 2028, 2029, 2087, 2300]

    # major = readMajorGk(4, keepCerts=True)
    # certList = list(map(lambda adj : nauty.certificate(createNautyGraphFromAdjMatrix(adj)), readOrderedComplete4()))
    # print([i for i in range(len(certList)) if certList[i] not in major])

    # 4-CDGs which have equivalent underlying undirected graph
    # Frequency information (key is number of times a specific undirected graph appears, value is how many undirected graphs appear key times):
    # Counter({'4': 193, '8': 106, '2': 76, '12': 37, '6': 23, '24': 18, '16': 10, '10': 7, '28': 5, '22': 3, '32': 3, '3': 3, '1': 3, '7': 2, '18': 1, '56': 1, '14': 1, '40': 1, '20': 1})

    undirectedList = [nx.DiGraph(adj).to_undirected() for adj in readOrderedComplete4()]
    certList = list(map(lambda adj : nauty.certificate(createNautyGraphFromAdjMatrix(adj)), undirectedList))
    dict = {}
    for i, cert in enumerate(certList):
        if cert in dict:
            dict[cert].append(i)
        else:
            dict[cert] = [i]
    # print('\n'.join([str(i) for i in dict.values()]))
    # print('\n'.join([str(len(i)) for i in dict.values()]))
    print(collections.Counter([str(len(i)) for i in dict.values()]))