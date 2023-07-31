import networkx as nx
import itertools
import numpy as np
import matplotlib.pyplot as plt

def genPossibleUPPRows(k: int) -> list:
    '''
    Generate all possible rows of length k^2 with exactly k 1's
    '''

    rows = []
    for ls in list(itertools.combinations(range(k ** 2), k)):
        rows.append(createRowVectorFromIndexList(k, ls))
    return rows

def checkMostSinglePath(adj: np.ndarray) -> bool:
    '''
    Checks that the graph with the given adjacency matrix has at most one path of length 2 between any two points
    '''

    return np.max(np.linalg.matrix_power(adj, 2)) <= 1

def createBasisCentralDigraph(k: int) -> np.ndarray:
    '''
    Makes the partially filled basis for a central digraph, i.e. creates the adjacency matrix where a loop vertex is chosen as the intial point,
    and all of its neighbours at distance 2 are written out, labeled in the standard order.
    '''

    # TODO add the k-1 remaining 1's to the correct places in the first column
    D_adj = nx.to_numpy_array(nx.empty_graph(k ** 2))
    for i in range(k):
        D_adj[i] = createRowVectorFromIndexList(k, range(i * k, (i + 1) * k))
    return D_adj

def createStandardCentralDigraph(k: int) -> np.ndarray:
    '''
    Creates the standard example of a central digraph on k^2 vertices, which is consecutive 1s being shifted cyclically over the rows
    '''

    D_adj = nx.to_numpy_array(nx.empty_graph(k ** 2))
    for i in range(k ** 2):
        D_adj[i] = createRowVectorFromIndexList(k, range((i % k) * k, ((i % k) + 1) * k))
    return D_adj

def createRowVectorFromIndexList(k: int, ls: list) -> np.ndarray:
    '''
    Given a list of indices, creates a k^2 dim {0-1} vector with 1's in the specified indices
    '''

    row = np.zeros(k ** 2)
    for i in ls:
        row[i] = 1
    return row



def attemptGenUPPRowByRow(adj: np.ndarray, i: int, k: int, possibleRows: list, _temp=[]) -> list[np.ndarray]:
    '''
    Recursively generates all possible k-central digraphs in a row by row manner, i.e. we attempt to add a row to a interim matrix and see if it has at most 1 path
    length 2 at each step.

    Does not remove isomorphs
    '''

    if i == k ** 2:
        # For some reason yield didn't work, would be nicer with it
        _temp.append(np.copy(adj))
    else:
        for row in possibleRows:
            adj[i] = row
            if checkMostSinglePath(adj):
                attemptGenUPPRowByRow(adj, i + 1, k, possibleRows, _temp)
            adj[i] = np.zeros(k ** 2)
    
    return _temp

def genUPPMatrices(k: int):
    '''
    Generates list of all k-central digraphs
    '''

    possibleRows = genPossibleUPPRows(k)
    base = createBasisCentralDigraph(k)
    return attemptGenUPPRowByRow(base, k, k, possibleRows)


def __main__():
    # nx.draw(nx.from_numpy_array(createStandardCentralDigraph(4), create_using=nx.DiGraph))
    # plt.show()
    ls = genUPPMatrices(3)
    # print(ls[-6:])
    # print(len(ls))

if __name__ == '__main__':
    __main__()