# File to store research directions and random experiments/ideas before being categorised.

import numpy as np
import os
import ast
import collections
from itertools import combinations
from readDataFiles import *
from utilsUPP import *
from switchingsUPP import *

def getInducedUndirectedCanonAdj(adj: np.ndarray) -> np.ndarray:
    '''
    For a given CDG, return an undirected graph where the edge set is the arc set with di-gons being represented by a single edge.
    
    We conjecture if di-gons are represented by a multigraph then there is exactly two possible base CDGs, the original and its reverse.

    TODO fix, very broken
    '''

    undirectedAdj = nx.to_numpy_array(nx.DiGraph(adj).to_undirected())
    relabel = nauty.canon_label(createNautyGraphFromAdjMatrix(undirectedAdj))
    perm = dict(zip(range(np.shape(adj)[0]), relabel))
    # perm = dict(zip(relabel, np.roll(relabel, -1)))
    D = nx.relabel_nodes(nx.DiGraph(adj), perm)
    return nx.to_numpy_array(D, nodelist=[*range(np.shape(adj)[0])])

def calcAverageCommonNeighboursLoopVertices(adj: np.ndarray) -> tuple:
    inAverage, outAverage, i = 0, 0, 0
    loops = [i for i in range(np.shape(adj)[0]) if int(adj[i][i]) == 1]
    for l1, l2 in combinations(loops, 2):
        inAverage += np.dot(adj[l1, :], adj[l2, :])
        outAverage += np.dot(adj[:, l1], adj[:, l2])
        i += 1
    return (inAverage / i, outAverage / i)

if __name__ == '__main__':

    ls = readOrderedComplete4()

    # the 4-CDGs not switch equivalent to canonical 4-CDG are 
    # [0, 174, 258, 288, 290, 298, 301, 309, 310, 317, 501, 653, 684, 1591, 1966, 1967, 2027, 2028, 2029, 2087, 2300]

    # major = readMajorGk(4, keepCerts=True)
    # certList = list(map(lambda adj : nauty.certificate(createNautyGraphFromAdjMatrix(adj)), readOrderedComplete4()))
    # print([i for i in range(len(certList)) if certList[i] not in major])

    # The reverses of the 4-CDGs
    # The 34 self-reverse 4-CDGs are
    # ['185', '850', '907', '942', '1696', '2223', '2300', '2310', '2353', '2465', '2466', '2514', '2517', '2522', '2649', '2677', '2678',
    # '2728', '2731', '2737', '2739', '2816', '3150', '3257', '3263', '3265', '3287', '3378', '3379', '3380', '3469', '3478', '3488', '3491']

    # reverseList = [np.transpose(adj) for adj in readOrderedComplete4()]
    # reverseCertList = list(map(lambda adj : nauty.certificate(createNautyGraphFromAdjMatrix(adj)), reverseList))
    # certList = list(map(lambda adj : nauty.certificate(createNautyGraphFromAdjMatrix(adj)), readOrderedComplete4()))
    # reverseDict = {}
    # for i in range(len(certList)):
    #     reverseDict[i] = reverseCertList.index(certList[i])
    # selfReverseList = [i for i in reverseDict.keys() if reverseDict[i] == i]
    # print(selfReverseList)

    # 4-CDGs which have equivalent underlying undirected graph
    # By underlying directed digraph, we mean removing digons, as keeping digons in a multigraph appears to produce only the original CDG and its reverse.
    # Frequency information (key is number of times a specific undirected graph appears, value is how many undirected graphs appear key times):
    # Counter({'4': 193, '8': 106, '2': 76, '12': 37, '6': 23, '24': 18, '16': 10, '10': 7, '28': 5, '22': 3, '32': 3, '3': 3, '1': 3, '7': 2, '18': 1, '56': 1, '14': 1, '40': 1, '20': 1})

    # undirectedList = [nx.DiGraph(adj).to_undirected() for adj in readOrderedComplete4()]
    # undirectedCertList = list(map(lambda adj : nauty.certificate(createNautyGraphFromAdjMatrix(adj)), undirectedList))
    # undirectedDict = {}
    # for i, cert in enumerate(undirectedCertList):
    #     if cert in undirectedDict:
    #         undirectedDict[cert].append(i)
    #     else:
    #         undirectedDict[cert] = [i]
    # print('\n'.join([str(i) for i in undirectedDict.values()]))
    # print('\n'.join([str(len(i)) for i in undirectedDict.values()]))
    # print(collections.Counter([str(len(i)) for i in undirectedDict.values()]))
    
    # print(collections.Counter([str(undirectedDict[undirectedCertList[i]]) for i in selfReverseList]))

    # print(ls[0])
    # print()
    # print(ls[reverseDict[0]])
    # print()
    # print(getInducedUndirectedCanonAdj(ls[0]) + getInducedUndirectedCanonAdj(ls[reverseDict[0]]))

    # productList = [np.matmul(adj, np.transpose(adj)) for adj in ls]
    # print(productList[1])
    # print([i for i in range(len(productList)) if int(productList[i].sum()) != 256])
    # print(len([i for i in range(len(productList)) if int(np.max(productList[i] - 4*np.eye(16))) < 4]))

    # print([calcAverageCommonNeighboursLoopVertices(adj) for adj in ls][-10:])
    # print("max common in: " + str(max([calcAverageCommonNeighboursLoopVertices(adj)[0] for adj in ls])))
    # print("max common out: " + str(max([calcAverageCommonNeighboursLoopVertices(adj)[1] for adj in ls])))
    # print("max common neigh: " + str(max([calcAverageCommonNeighboursLoopVertices(adj)[0] + calcAverageCommonNeighboursLoopVertices(adj)[1] for adj in ls])))
    # print([calcAverageCommonNeighboursLoopVertices(adj)[0] for adj in ls].index(1.5))
    # print(ls[2])

    # there are 1299 max rank 4-CDGs
    print(len([i for i in range(len(ls)) if np.linalg.matrix_rank(ls[i]) == 8]))

    pass

