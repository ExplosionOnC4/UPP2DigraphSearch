import itertools
import numpy as np
import networkx as nx
from utilsUPP import *

def genProductTables(k: int) -> list[np.ndarray]:
    '''
    Generates all possible product tables for the set S={0..k-1} for which x*0=0 and unique z such that x*y=z. These are the underlying structure of Knuthian systems.

    # TODO make lazy
    '''

    firstRow = np.array([i for i in range(k)])
    out = []

    # Generate k-1 permutations of {1..k-1}
    allPerms = list(itertools.permutations(range(1,k)))
    permSubsets = list(itertools.product(range(len(allPerms)), repeat=k-1))
    for permSubset in permSubsets:
        perms = np.array([[0] + list(allPerms[i]) for i in permSubset])
        
        # Create multiplication table
        productTable = np.block([[firstRow],[perms]])
        out.append(productTable)
    return out

def makeKnuthianDigraph(table: np.ndarray) -> nx.DiGraph:
    '''
    Return the (central) digraph of the Knuthian system defined by the given product table
    '''

    k = table.shape[0]
    elems = list(itertools.product(range(k), range(k)))

    D = nx.DiGraph()
    D.add_nodes_from(elems)
    D.add_edges_from([(e1, e2) for e1 in elems for e2 in elems if (e1[1] == e2[0] and e2[1] != 0) or (e2[1] == 0 and table[e1[0],e1[1]] == e2[0])])

    return D

def getKnuthianAdjMatrix(table: np.ndarray) -> np.ndarray:
    '''
    Return the adjacency matrix of the Knuthian system defined by the given product table
    '''

    return nx.to_numpy_array(makeKnuthianDigraph(table))

def getMultiplicationSubtable(table: np.ndarray) -> np.ndarray:
    '''
    Returns the multiplication table with the trivial first row and columns removed
    '''

    return np.delete(np.delete(table, 0, 1), 0, 0)

def applySymmetricActionOnSubtable(subtable: np.ndarray, perm: dict) -> np.ndarray:
    '''
    Apply the action of the group S_{k-1} on the k-1 x k-1 multiplication subtable given in https://cklixx.people.wm.edu/reu02.pdf.

    Let \hat{T} be a multiplication subtable. Then an element \sigma \in S_{k-1} acts on the set such possible subtables as follows:

    `$$\sigma(\hat{T}) = P_\sigma \overline{\sigma}(\hat{T}) P_\sigma^{-1}$$` where `$\overline{\sigma}(\hat{T})_{ij} = \sigma(\hat{T}_{ij})$`, 
    i.e. it permutes the rows, permutes the columns and maps the elements of the table.

    The paper proves that the k x k multiplication tables T_1 and T_2 define isomorphic operations if there exists \sigma \in S_{k-1} such that
    \sigma(\hat{T}_1) = \hat{T}_2.
    '''

    newTable = np.copy(subtable)
    for i in range(len(subtable)):
        for j in range(len(subtable[0])):
            newTable[perm[i+1]-1][perm[j+1]-1] = perm[subtable[i][j]]  # As S_n does not include 0
    return newTable

def genSymmetricGroupMaps(k: int) -> list[dict]:
    '''
    Return a list containing every element of S_k, represented as a map.
    '''

    return [dict(zip([i for i in range(1, k)], perm)) for perm in itertools.permutations([i for i in range(1, k)])]

def areKnuthianIsomorphic(table1: np.ndarray, table2: np.ndarray) -> bool:
    '''
    Checks whether two product tables are isomorphic.
    
    By result in https://doi.org/10.1016/S0021-9800(70)80032-1, two Knuthian systems are isomorphic iff the underlying product tables are isomorphic (modulo other conditions).

    Uses @applySymmetricActionOnSubtable() and the result from https://cklixx.people.wm.edu/reu02.pdf to check for isomorphism.
    '''

    # Do not see anything better than checking all k! possible permutations of the elements of k right now
    if np.shape(table1) != np.shape(table2):
        return False
    symmetricGroup = genSymmetricGroupMaps(len(table1))
    for sigma in symmetricGroup:
        if np.array_equal(applySymmetricActionOnSubtable(getMultiplicationSubtable(table1), sigma), getMultiplicationSubtable(table2)):
            return True
    return False

def findFixedSubtablesOfPermutation(perm: dict, subtables=None) -> list[np.ndarray]:
    '''
    Returns all of the product subtables which are fixed by the given element of S_{k-1}

    Slow as purely bruteforce
    '''

    if not subtables:
        subtables = list(map(getMultiplicationSubtable, genProductTables(len(perm) + 1)))
    fixed = []
    for subtable in subtables:
        if np.array_equal(applySymmetricActionOnSubtable(subtable, perm), subtable):
            fixed.append(subtable)
    return fixed

def calcNumNonIsomorphKnuthianDigraphs(k: int) -> int:
    '''
    Counts the number of non-isomorphic of size k using Burnside lemma
    '''

    # For some reason if left as map object return exactly (k-1)!^{k-2} which is exactly the assumption that average |X^g|=1
    # TODO make lazy to not use so much memory
    subtables = list(map(getMultiplicationSubtable, genProductTables(k)))
    symmetricGroup = genSymmetricGroupMaps(k)
    sum = 0
    for sigma in symmetricGroup:
        sum += len(findFixedSubtablesOfPermutation(sigma, subtables))
    return int(sum / len(symmetricGroup))

def getSymmetricGroupOrbitOfSubtable(subtable: np.ndarray) -> list[np.ndarray]:
    '''
    Returns the orbit of a given product subtable for the action of S_{k-1} described in https://cklixx.people.wm.edu/reu02.pdf.
    '''

    orbit = []
    symmetricGroup = genSymmetricGroupMaps(len(subtable) + 1)
    for sigma in symmetricGroup:
        image = applySymmetricActionOnSubtable(subtable, sigma)
        if not any((image == x).all() for x in orbit):
            orbit.append(image)
    return orbit

def removeNumpyArrFromList(ls: list[np.ndarray], arr: np.ndarray) -> None:
    '''
    Helper function, removes first instance of (numpy array) `arr` in `ls`.

    :throw StopIteration if `arr` not in `ls`
    '''

    index = next((i for i, x in enumerate(ls) if np.array_equal(arr, x)))
    ls.pop(index)
    return 

def findAllOrbitsSymmetricGroupSubtables(k: int) -> list[list]:
    '''
    Returns the orbits of the set of product subtables under the symmetric group action described in https://cklixx.people.wm.edu/reu02.pdf.

    :return list for which each sublist is an orbit.

    # TODO will break if subtables changed to lazy eval
    '''

    orbitsSet = []
    subtables = list(map(getMultiplicationSubtable, genProductTables(k)))
    while subtables:
        sub = subtables[0]
        orbit = getSymmetricGroupOrbitOfSubtable(sub)
        for elem in orbit:
            removeNumpyArrFromList(subtables, elem)
        orbitsSet.append(orbit)
    return orbitsSet

if __name__ == '__main__':
    print(len(findAllOrbitsSymmetricGroupSubtables(4)))
    pass
