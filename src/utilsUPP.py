import numpy as np
import networkx as nx
import pynauty as nauty
import math
import subprocess
import itertools
from collections.abc import Iterable

def checkMostSinglePath(adj: np.ndarray) -> bool:
    '''
    Checks that the graph with the given adjacency matrix has at most one path of length 2 between any two points
    '''

    return np.max(np.linalg.matrix_power(adj, 2)) <= 1

def isUPP(adj: np.ndarray) -> bool:
    '''
    Checks directly if `A^2=J`.
    '''

    return np.array_equal(np.linalg.matrix_power(adj, 2), np.ones(np.shape(adj)))

def getLoopVerts(adj: np.ndarray) -> list[int]:
    '''
    Returns the list of indices of all looped vertices in the graph.
    '''

    return [i for i in range(np.shape(adj)[0]) if int(adj[i][i]) == 1]

def findInneighbours(v: int, adj: np.ndarray) -> list[int]:
    '''
    For a given vertex `v`, find all of its inneighbours in the CDG given by 
    '''

    return [i for i in range(np.shape(adj)[0]) if adj[:,v][i] == 1]

def findOutneighbours(v: int, adj: np.ndarray) -> list[int]:
    '''
    For a given vertex `v`, find all of its outneighbours in the CDG given by 
    '''

    return [i for i in range(np.shape(adj)[0]) if adj[v,:][i] == 1]

def findConnectionVertex(u: int, v: int, adj: np.ndarray) -> int:
    '''
    Returns the unique intermidiate vertex between `u` and `v` in the CDG given by `adj`.
    '''

    if not isUPP(adj):
        raise(ValueError('Provided adjacency is not UPP_2'))
    return set(findOutneighbours(u, adj)).intersection(set(findInneighbours(v, adj))).pop()

def findDigonTwin(v: int, adj: np.ndarray) -> int:
    '''
    Returns the adjacent vertex of the digon system containing `v`.
    '''

    if v in getLoopVerts(adj):
        raise(ValueError('Provided vertex is a loop'))
    else:
        return findConnectionVertex(v, v, adj)
    
def getUPPClosureOfSubset(includedVerts: Iterable[int], adj: np.ndarray, excludedVerts=set()) -> np.ndarray | None:
    '''
    Finds the UPP_2 closure of a subset of vertices of CDG given by `adj`.
    The optional argument `excludedVerts` acts as an early abort condition if closure contains an excluded vertex, in which case returns None
    '''

    # Two approaches, either start with k-1 loop vertices and see if UPP2 closure contains the k-th one,
    # OR delete every choice of 2k-1 row/columns from original k-CDG (=> O(k*[k^2-1 C 2k-2]))
    # Approach 1 seems much faster, since two loop vertices cannot be adjacent, it should find all vertices in closure relatively quickly
    lastVertexSet = set()
    vertexSet = set(includedVerts)
    while lastVertexSet != vertexSet and not vertexSet.intersection(set(excludedVerts)):
        lastVertexSet = vertexSet.copy()
        intermidVerts = set()
        # for any perm of 2 elements from vertex set, add intermid vertex (cartesian product to be completely safe but should be unnecessary)
        # IDEA have list of VxV, if both elements are in vertex set then find intermid vertex and delete from VxV to avoid recompute.
        for v1, v2 in itertools.permutations(vertexSet, 2):
            intermidVerts.add(findConnectionVertex(v1, v2, adj))
        vertexSet.update(intermidVerts)
    if not vertexSet.intersection(set(excludedVerts)):
        # transform back into adjacency
        scdg = np.zeros((len(vertexSet), len(vertexSet)))
        for i, vi in enumerate(vertexSet):
            for j, vj in enumerate(vertexSet):
                scdg[i][j] = adj[vi][vj]
        return scdg
    return
    

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

def partitionEqualRowIndices(adj: np.ndarray) -> map:
    '''
    Partitions the set of row indices into equivalence classes based on equality, i.e. two rows will lie in the same class if they are equal

    :return: dictionary where keys are byte representation of rows, and values are the row indices which are equal to the given row.
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

    return partitionEqualRowIndices(np.transpose(adj))

def containsIdenticalRows(adj: np.ndarray) -> bool:
    '''
    Checks that there exits at least two rows in the adjacency matrix that are equal
    '''

    parts = partitionEqualRowIndices(adj)
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

def satisfiesNeighbourhoodCondition(adj: np.ndarray) -> bool:
    '''
    Checks if a k-CDG satisfies the neighbourhood condition, as described in Propostion 5 of https://doi.org/10.1016/j.jcta.2011.03.009.

    A k-CDG satisfies the neighbourhood condition if it contains a set of k-1 non-loop vertices which either all share the same inneighbourhood or outneighbourhood.

    A necessary condition for a k-CDG to be reducible.
    '''

    k = int(math.sqrt(np.shape(adj)[0]))
    neighbourhoodCond = False
    loops = {i for i in range(np.shape(adj)[0]) if int(adj[i][i]) == 1}
    rowPartitions = partitionEqualColIndices(adj)
    colPartitions = partitionEqualRowIndices(adj)
    for part in rowPartitions.values():
        if len(set(part).difference(loops)) >= k - 1:
            neighbourhoodCond = True
            break
    for part in colPartitions.values():
        if len(set(part).difference(loops)) >= k - 1:
            neighbourhoodCond = True
            break

    return neighbourhoodCond

def convertAdjMatrixToNeighbourList(adj: np.ndarray) -> dict[list]:
    return nx.to_dict_of_lists(nx.DiGraph(adj))

def createNautyGraphFromAdjMatrix(adj: np.ndarray) -> nauty.Graph:
    G = nx.DiGraph(adj)
    return nauty.Graph(number_of_vertices=G.order(), directed=True, adjacency_dict=nx.to_dict_of_lists(G))

def recoverAdjMatrixFromNautyGraph(G: nauty.Graph) -> np.ndarray:
    return nx.to_numpy_array(nx.from_dict_of_lists(G.adjacency_dict, create_using=nx.DiGraph))

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

def convertAdjMatrixToBinaryString(adj: np.ndarray) -> str:
    '''
    Convert a nxn adjacency matrix into a binary string of length n^2, by concatenating its rows
    '''

    return ''.join(str(int(i)) for l in adj for i in l)

def convertBinaryToSetRepresentation(binRep: str) -> list[int]:
    '''
    Helper function to read in orderedComplete4, convert binary representation of digraph to its set representation
    '''

    return [i+1 for i in range(len(binRep)) if binRep[i] == '1']