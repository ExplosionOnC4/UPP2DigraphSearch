import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pynauty as nauty
from queue import Queue
from utilsUPP import *

def validSwitch(adj: np.ndarray, p: int, q: int, r: int, s: int) -> bool:
    '''
    Checks if the provided parameters form a valid switch on the adjacency matrix A = adj.

    By Fletcher (as was said in https://doi.org/10.1016/j.jcta.2011.03.009), we require\n
    \t(1) 0 <= p < q <= k^2, 0 <= r < s <= k^2\n
    \t(2) {p,q}\cap{r,s}=\emptyset\n 
    \t(3) columns p,q are equal in A, rows r,s are equal in A\n
    \t(4) The 2x2 submatrix A[p,q;r,s] = I or J-I\n
    If all of these conditions hold then switching A[p,q;r,s] between I and J-I preserves the unique path property
    '''

    return (0 <= p < q <= np.shape(adj)[0] and 0 <= r < s <= np.shape(adj)[1] and
            not {p,q}.intersection({r,s}) and
            np.array_equal(adj[r], adj[s]) and np.array_equal(adj[:,p], adj[:,q]) and
            adj[p][r] == adj[q][s] and adj[p][s] == adj[q][r] and adj[p][r] != adj[p][s])
    
def performSwitch(adj: np.ndarray, p: int, q: int, r: int, s: int):
    '''
    Switches the 2x2 submatrix A[p,q;r,s] between I and J-I if Fletcher conditions hold

    NOTE this is a mutator method, does NOT create new adjacency matrix

    :raise Exception: if not valid parameters for a switch
    '''

    if not validSwitch(adj,p,q,r,s):
        raise Exception("Not a valid switch")
    else:
        for i in (p,q):
            for j in (r,s):
                adj[i][j] = (adj[i][j] + 1) % 2


def switchConnectedComponentFromVertex(adj: np.ndarray) -> dict[list]:
    '''
    Uses BFS to find the entire connected component containing the provided graph in G_k, i.e. finds all switching equivalent graphs to the seed graph

    :return: a dictionary of lists, where keys are canonical labelling of graph, and the list contain all canonical labellings of graphs that can be
    achieved with one switch from the key graph
    '''

    # immediately return singelton point if it allows no switches
    if not containsIdenticalCols(adj) or not containsIdenticalCols(adj):
        return {nauty.canon_label(createNautyGraphFromAdjMatrix(adj)): []}

    q = Queue()
    isomorphHash = {}

    q.put(adj)
    while not q.empty():
        tempAdj = q.get()
        # lists aren't hashable
        tempLabel = str(nauty.canon_label(createNautyGraphFromAdjMatrix(tempAdj)))

        # TODO make this quicker, eg only check on partitions of rows/columns
        for rows in list(itertools.combinations(range(np.shape(tempAdj)[0]), 2)):
            for cols in list(itertools.combinations(range(np.shape(tempAdj)[1]), 2)):
                if validSwitch(tempAdj, *rows, *cols):
                    newAdj = np.copy(tempAdj)
                    performSwitch(newAdj, *rows, *cols)
                    newLabel = str(nauty.canon_label(createNautyGraphFromAdjMatrix(newAdj)))

                    # For a central digraph, look at the digraphs attainable from one switch and add them to the neighbourhood
                    if tempLabel in isomorphHash and newLabel not in isomorphHash[tempLabel]:
                        isomorphHash[tempLabel].append(newLabel)
                    else:
                        isomorphHash[tempLabel] = [newLabel]
                    
                    # Put switched digraph in queue if has not been seen before and add to hash
                    if newLabel not in isomorphHash:
                        q.put(newAdj)
                        isomorphHash[newLabel] = [tempLabel]

    return isomorphHash


if __name__ == '__main__':
    stand = createStandardCentralDigraph(3)
    # print(nauty.canon_label(createNautyGraphFromAdjMatrix(stand)))
    print(switchConnectedComponentFromVertex(stand))
    # print(containsIdenticalCols(stand))
    # print([i for i in partitionEqualColIndices(stand).values()])