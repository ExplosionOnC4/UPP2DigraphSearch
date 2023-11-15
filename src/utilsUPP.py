import numpy as np
import networkx as nx
import pynauty as nauty
import subprocess

def checkMostSinglePath(adj: np.ndarray) -> bool:
    '''
    Checks that the graph with the given adjacency matrix has at most one path of length 2 between any two points
    '''

    return np.max(np.linalg.matrix_power(adj, 2)) <= 1

def createRowVectorFromIndexList(k: int, ls: list) -> np.ndarray:
    '''
    Given a list of indices, creates a k^2 dim {0-1} vector with 1's in the specified indices
    '''

    row = np.zeros(k ** 2)
    for i in ls:
        row[i] = 1
    return row

def createStandardCentralDigraph(k: int) -> np.ndarray:
    '''
    Creates the standard example of a central digraph on k^2 vertices, which is consecutive 1s being shifted cyclically over the rows
    '''

    D_adj = nx.to_numpy_array(nx.empty_graph(k ** 2))
    for i in range(k ** 2):
        D_adj[i] = createRowVectorFromIndexList(k, range((i % k) * k, ((i % k) + 1) * k))
    return D_adj

def createBasisBlockMatrix(i: int, k: int) -> np.ndarray:
    '''
    For a given i, returns a the i-th kxk block submatrix from first row of standard central digraph.
    '''

    adj = np.zeros((k, k))
    adj[i] = np.ones(k)
    return adj

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


def convertAdjMatrixToNeighbourList(adj: np.ndarray) -> dict[list]:
    return nx.to_dict_of_lists(nx.DiGraph(adj))

def createNautyGraphFromAdjMatrix(adj: np.ndarray) -> nauty.Graph:
    G = nx.DiGraph(adj)
    return nauty.Graph(number_of_vertices=G.order(), directed=True, adjacency_dict=nx.to_dict_of_lists(G))

def recoverGraphFromNautyCert(cert, numVerts: int) -> nauty.Graph:
    '''
    Recovers the Nauty graph from its binary string form given by nauty.certificate

    From https://github.com/pdobsan/pynauty/issues/30#issuecomment-1564066767 although original post is broken
    '''

    set_length = len(cert)
    lenVertString = int(set_length / numVerts)

    sets = [cert[lenVertString*k:lenVertString*(k+1)] for k in range(numVerts)]
    neighbors = [[i for i in range(lenVertString * 8) if st[-1 - i//8] & (1 << (7 - i%8))] for st in sets]
    return nauty.Graph(number_of_vertices=numVerts, directed=True, adjacency_dict={i: neighbors[i] for i in range(numVerts)})

def getSetRepresentationAdjacencyMatrix(adj: np.ndarray) -> list[int]:
    '''
    Get the set representation of graph by labelling the entries of the adjacency matrix from 1 to n^2. Returns the set containing the non-zero entries.

    For use with induced permutation of S_{n} on {1, ..., n} x {1, ..., n}
    '''

    set = []
    for i in range(np.shape(adj)[0]):
        for j in range(np.shape(adj)[1]):
            if adj[i][j] != 0:
                set.append(np.shape(adj)[1] * i + j + 1)
    return set

def getAdjacencyMatrixFromSet(set: list[int], n: int) -> np.ndarray:
    '''
    The reverse of getSetRepresentationAdjacencyMatrix(), recovers the adjacency matrix
    '''

    adj = np.zeros((n, n))
    for i in set:
        adj[int((i-1) / n)][(i-1) % n] = 1

    return adj

def isLexMinAdjacencyMatrix(adj: np.ndarray) -> bool:
    '''
    Finds the adjacency matrix of the lexicographically minimal graph labelling of a given graph
    '''

    set = getSetRepresentationAdjacencyMatrix(adj)
    n = np.shape(adj)[0]
    proc = subprocess.run(args=['sh', 'lexMin.sh', f'{set}', f'{n}'], capture_output=True, text=True)
    return proc.stdout.strip() == 'true'

def filterIsomorphs(ls: list[np.ndarray]) -> list[np.ndarray]:
    '''
    For an input list of adjacency matrices, returns the list with isomorphs removed.
    '''

    out = []
    certs = set()
    for adj in ls:
        cert = nauty.certificate(createNautyGraphFromAdjMatrix(adj))
        if cert not in certs:
            out.append(adj)
            certs.add(cert)

    return out

def convertBinaryToSetRepresentation(binRep: str) -> list[int]:
    '''
    Helper function to read in orderedComplete4, convert binary representation of digraph to its set representation
    '''

    return [i+1 for i in range(len(binRep)) if binRep[i] == '1']