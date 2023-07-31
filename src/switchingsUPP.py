import networkx as nx
import itertools
import numpy as np
import matplotlib.pyplot as plt

def validSwitch(adj: np.ndarray, p: int, q: int, r: int, s: int) -> bool:
    '''
    Checks if the provided parameters form a valid switch on the adjacency matrix A = adj.

    By Fletcher [<insert ref here>], we require\n
    \t(1) 0 <= p < q <= k^2, 0 <= r < s <= k^2\n
    \t(2) {p,q}\cap{r,s}=\emptyset\n 
    \t(3) columns p,q are equal in A, rows r,s are equal in A\n
    \t(4) The 2x2 submatrix A[p,q;r,s] = I or J-I\n
    If all of these conditions hold then switching A[p,q;r,s] between I and J-I preserves the unique path property
    '''

    return (0 <= p < q <= np.shape(adj)[0] and 0 <= r < s <= np.shape(adj)[0] and
            not {p,q}.intersection({r,s}) and
            np.array_equal(adj[r], adj[s]) and np.array_equal(adj[:,p], adj[:,q]) and
            adj[p][r] == adj[q][s] and adj[p][s] == adj[q][r] and adj[p][r] != adj[p][s])
    
def performSwitch(adj: np.ndarray, p: int, q: int, r: int, s: int):
    '''
    Switches the 2x2 submatrix A[p,q;r,s] between I and J-I if Fletcher conditions hold
    '''

    if validSwitch(adj,p,q,r,s):
        for i in (p,q):
            for j in (r,s):
                adj[i][j] = (adj[i][j] + 1) % 2

def paritionEqualRowIndices(adj: np.ndarray) -> map:
    '''
    Partitions the set of row indices into equivalence classes based on equality, i.e. two rows will lie in the same class if they are equal
    '''

    rowsHashMap={}
    for i in range(np.shape(adj)[0]):
        key = adj[i].tobytes()
        if key in rowsHashMap:
            rowsHashMap[key].append(i)
        else:
            rowsHashMap[key] = [i]
    return rowsHashMap


def partitionEqualColIndices(adj: np.ndarray) -> map:
    '''
    Partitions the set of column indices into equivalence classes based on equality, i.e. two rows will lie in the same class if they are equal
    '''

    return paritionEqualRowIndices(np.transpose(adj))

def containsIdenticalRows(adj: np.ndarray) -> bool:
    '''
    Checks that there exits at least two rows in the adjacency matrix that are equal
    '''

    parts = paritionEqualRowIndices(adj)
    for _, v in parts.items():
        if len(v) > 1:
            return True
    return False

def containsIdenticalCols(adj: np.ndarray) -> bool:
    '''
    Checks that there exits at least two columns in the adjacency matrix that are equal

    This will be useful for verifying the conjecture that there are no central digraphs with all columns different
    '''

    return containsIdenticalRows(np.transpose(adj))

if __name__ == '__main__':
    import genUPPDigraphs as gen
    stand = gen.createStandardCentralDigraph(3)
    # print(containsIdenticalCols(stand))
    # print([i for i in partitionEqualColIndices(stand).values()])