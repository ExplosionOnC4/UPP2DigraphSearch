import networkx as nx
import itertools
import numpy as np

# Possible rows of length k^2 with exactly k 1's
def genPossibleCentralRows(k):
    rows = []
    for ls in list(itertools.combinations(range(k ** 2), k)):
        rows.append(createRowVectorFromIndexList(k, ls))
    return rows

def checkMostSinglePath(adj):
    return np.max(np.linalg.matrix_power(adj, 2)) <= 1

def createBaseUPPDigraph(k):
    D_adj = nx.to_numpy_array(nx.empty_graph(k ** 2))
    for i in range(k):
        D_adj[i] = createRowVectorFromIndexList(k, range(i * k, (i + 1) * k))
    return D_adj

# Slow, use np array instead
def createRowVectorFromIndexList(k, ls):
    row = []
    for i in range(k ** 2):
        row.append(int(i in ls))
    return row

Possible4Rows = genPossibleCentralRows(4)

def attemptGenUPPRowByRow4(adj, i):
    if i == 16:
        print(adj)
        return adj
    else:
        for row in Possible4Rows:
            adj[i] = row
            if checkMostSinglePath(adj):
                attemptGenUPPRowByRow4(adj, i + 1)
            adj[i] = np.zeros(16)

def genUPPMatrices4():
    base = createBaseUPPDigraph(4)
    UPPs = []
    UPPs.append(attemptGenUPPRowByRow4(base, 4))
    return UPPs

ls = genUPPMatrices4()