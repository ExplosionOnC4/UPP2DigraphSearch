import itertools
import numpy as np
import networkx as nx
from collections.abc import Generator
from utilsUPP import *
# import random

def genProductTables(k: int) -> Generator[np.ndarray, None, None]:
    '''
    Lazily generates all possible product tables for the set S={0..k-1} for which x*0=0 and unique z such that x*y=z. These are the underlying structure of Knuthian systems.
    '''

    firstRow = np.array([i for i in range(k)])

    # Generate k-1 permutations of {1..k-1}
    allPerms = list(itertools.permutations(range(1,k)))
    permSubsets = itertools.product(range(len(allPerms)), repeat=k-1)
    for permSubset in permSubsets:
        perms = np.array([[0] + list(allPerms[i]) for i in permSubset])
        
        # Create multiplication table
        productTable = np.block([[firstRow],[perms]])
        yield productTable

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

def recoverKnuthianMultiplicationTable(adj: np.ndarray) -> np.ndarray:
    '''
    From a given adjacency matrix of a Knuthian CDG, restores a multiplication table for a binary operation that yielded the CDG.
    '''

    loops = getLoopVerts(adj)
    # This was added for testing, as hypothetically we can assign the indices randomly and should still get isomorphic tables.
    # random.shuffle(loops)

    # Find the loop vertex that must be labelled (0,0), this is explained in the case analysis later (symbols code snippet below or Issue #3 on github)
    # TODO these heuristics are not guaranteed and can return non-isomorphic tables if they don't apply, find a way to completely tell what the (0,0) vertex should be.
    zeroLoop = -1
    # Condition 1:
    # LEMMA: If a CDG is Knuthian then it has at most 1 loop vertex such that the multiset of intermediate vertices between it and the other loop vertices can contain duplicates
    # i.e. if (a,a) -> (x,y) -> (b,b) and (c,c) -> (x,y) -> (b,b) [where (x,y) is the repeated intermid vertex] then necessarily (b,b) = (0,0) and a*a = c*c.
    for i in loops:
        intermidVerts = [findConnectionVertex(j, i, adj) for j in loops]
        if len(intermidVerts) != len(set(intermidVerts)):
            zeroLoop = i
            break
    
    if zeroLoop == -1:
        possibleZeroLoops = list(loops)
        # Condition 2:
        # If a loop vertex has an inneighbour such that its digon twin is not an outneighbour then that loop cannot be assigned (0,0)
        removeLoops = []
        for loop in possibleZeroLoops:
            for out in findOutneighbours(loop, adj):
                if out != loop and findDigonTwin(out, adj) not in findInneighbours(loop, adj):
                    removeLoops.append(loop)
                    break
        for l in removeLoops:
            possibleZeroLoops.remove(l)

        # Condition 3: 
        # By Curtis et. al. in https://cklixx.people.wm.edu/reu02.pdf, a Knuthian CDG is reducible and contains the standard (k-1)-CDG as a subCDG
        # The proof removes (0,0) and all of its neighbours to yield the required subCDG.
        # Thus if a loop vertex is assignable (0,0) then the UPP closure without it must be the standard subCDG.
        removeLoops = []
        for loop in possibleZeroLoops:
            scdg = getUPPClosureOfSubset(set(loops).difference({loop}), adj, {loop})
            if scdg is None or not nx.is_isomorphic(nx.DiGraph(scdg), nx.DiGraph(createStandardCentralDigraph(len(loops) - 1))):
                removeLoops.append(loop)
        for l in removeLoops:
            possibleZeroLoops.remove(l)      

        if not possibleZeroLoops:
            raise(ValueError('Provided adjacency matrix is not a Knuthian CDG'))
    else:
        possibleZeroLoops = [zeroLoop]

    for zeroLoop in possibleZeroLoops:
        # Assign each vertex in the graph a tuple which respects the product rule
        # All loops have form (a,a)
        loops.insert(0, loops.pop(loops.index(zeroLoop)))
        cartesianProductRepresentation = {}
        for i, loop in enumerate(loops):
            cartesianProductRepresentation[loop] = (i,i)
        
        # intermid vertex between loops (a,a) and (b,b) has form (a,b) if b is not 0
        # Thus label of intermid vertex is entirely determined by choice of looped vertices
        for i, j in itertools.permutations(loops, 2):
            if j != zeroLoop:
                cartesianProductRepresentation[findConnectionVertex(i,j, adj)] = (cartesianProductRepresentation[i][1], cartesianProductRepresentation[j][0])

        # If (a,a) -> (x,y) -> (0,0) then (x,y) = (a * a, 0) which covers all inneighbours of (0,0)
        # Now if (x,y) -> (z*z,0) => x * y = z * z so every inneigbour of (z*z,0) shares its symbol in the product table
        # Further, all their inneighbours must be disjoint to obey UPP_2, so they cover all of the vertices not of the form (x,0)
        # However it is possible that a*a = b*b and so two loops share an intermidiate vertex.
        symbols = []
        for u in findInneighbours(zeroLoop, adj):
            if u != zeroLoop:
                shareSymbol = []
                for v in findInneighbours(u, adj):
                    shareSymbol.append(cartesianProductRepresentation[v])
                symbols.append(shareSymbol)

        # Insert symbols into product table such that first row is of standard form.
        numElems = int(math.sqrt(len(adj)))
        productTable = np.zeros((numElems, numElems))
        for i in range(1, numElems):
            sameSymbolList = [ls for ls in symbols if (0,i) in ls][0]
            for x,y in sameSymbolList:
                productTable[x][y] = i
        if nx.is_isomorphic(makeKnuthianDigraph(productTable), nx.DiGraph(adj)):
            return productTable
    
    raise(ValueError('Provided adjacency matrix is not a Knuthian CDG'))

def isKnuthianCDG(adj: np.ndarray) -> bool:
    if not isUPP(adj):
        raise(ValueError('Provided adjacency matrix is not a central digraph'))
    try:
        recoverKnuthianMultiplicationTable(adj)
        return True
    except ValueError:
        return False

def getMultiplicationSubtable(table: np.ndarray) -> np.ndarray:
    '''
    Returns the multiplication table with the trivial first row and columns removed
    '''

    return np.delete(np.delete(table, 0, 1), 0, 0)

def extendMultiplicationSubtable(subtable: np.ndarray) -> np.ndarray:
    '''
    Undoes getMultiplicationSubtable() by adding back the superfluous first row and column.
    '''

    return np.block([[np.arange(len(subtable) + 1)], [np.zeros((len(subtable), 1)), subtable]])

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
            newTable[perm[i+1]-1][perm[j+1]-1] = perm[subtable[i][j]]  # As S_n does not include 0 as an index
    return newTable

def genSymmetricGroupMaps(k: int) -> list[dict]:
    '''
    Return a list containing every element of S_k, represented as a map.
    '''

    return [dict(zip([i for i in range(1, k + 1)], perm)) for perm in itertools.permutations([i for i in range(1, k + 1)])]

def areKnuthianIsomorphic(table1: np.ndarray, table2: np.ndarray) -> bool:
    '''
    Checks whether two product tables are isomorphic.
    
    By result in https://doi.org/10.1016/S0021-9800(70)80032-1, two Knuthian systems are isomorphic iff the underlying product tables are isomorphic (modulo other conditions).

    Uses @applySymmetricActionOnSubtable() and the result from https://cklixx.people.wm.edu/reu02.pdf to check for isomorphism.
    '''

    # Do not see anything better than checking all k! possible permutations of the elements of k right now
    if np.shape(table1) != np.shape(table2):
        return False
    symmetricGroup = genSymmetricGroupMaps(len(table1) - 1)
    for sigma in symmetricGroup:
        if np.array_equal(applySymmetricActionOnSubtable(getMultiplicationSubtable(table1), sigma), getMultiplicationSubtable(table2)):
            return True
    return False

def findFixedSubtablesOfPermutation(perm: dict, subtables=None) -> list[np.ndarray]:
    '''
    Returns all of the product subtables which are fixed by the given element of S_{k-1}

    Slow as purely bruteforce

    TODO make an actual algorithm
    '''

    if not subtables:
        subtables = map(getMultiplicationSubtable, genProductTables(len(perm) + 1))
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
    # TODO make findFixedSubtables not bruteforce to avoid precomputing subtables
    subtables = list(map(getMultiplicationSubtable, genProductTables(k)))
    symmetricGroup = genSymmetricGroupMaps(k - 1)
    sum = 0
    for sigma in symmetricGroup:
        sum += len(findFixedSubtablesOfPermutation(sigma, subtables))
    return int(sum / len(symmetricGroup))

def getSymmetricGroupOrbitOfSubtable(subtable: np.ndarray) -> list[np.ndarray]:
    '''
    Returns the orbit of a given product subtable for the action of S_{k-1} described in https://cklixx.people.wm.edu/reu02.pdf.
    '''

    orbit = []
    symmetricGroup = genSymmetricGroupMaps(len(subtable))
    for sigma in symmetricGroup:
        image = applySymmetricActionOnSubtable(subtable, sigma)
        if not any((image == x).all() for x in orbit):
            orbit.append(image)
    return orbit

def findIndexNumpyArrInList(ls: list[np.ndarray], arr: np.ndarray, default=None) -> int:
    '''
    Finds first instance of (numpy array) `arr` in `ls` and returns its index.

    Can pass a default to return in case `arr` not in `ls`. Otherwise throws StopIteration
    '''

    matches = [i for i, x in enumerate(ls) if np.array_equal(arr, x)]
    if len(matches) == 0 and default is None:
        raise(StopIteration)
    return next(iter(matches), default)

def doesNumpyArrayContainArr(ls: list[np.ndarray], arr: np.ndarray) -> bool:
    return findIndexNumpyArrInList(ls, arr, default=-1) != -1

def removeNumpyArrFromList(ls: list[np.ndarray], arr: np.ndarray) -> None:
    '''
    Helper function, removes first instance of (numpy array) `arr` in `ls`.

    :throw StopIteration if `arr` not in `ls`
    '''

    index = findIndexNumpyArrInList(ls, arr)
    ls.pop(index)
    return 

def findAllOrbitsSymmetricGroupSubtables(k: int) -> list[list]:
    '''
    Returns the orbits of the set of product subtables under the symmetric group action described in https://cklixx.people.wm.edu/reu02.pdf.

    :return list for which each sublist is an orbit.
    '''

    orbitsSet = []
    foundSubs = []
    subtables = map(getMultiplicationSubtable, genProductTables(k))
    for sub in subtables:
        if not doesNumpyArrayContainArr(foundSubs, sub):
            orbit = getSymmetricGroupOrbitOfSubtable(sub)
            for elem in orbit:
                if not np.array_equal(elem, sub):
                    foundSubs.append(elem)
            orbitsSet.append(orbit)

        # This case is added as a heuristic to save on memory
        # worst case is still O([k-1]!^[k-1]) memory where we find exactly 1 representative from each orbit at beginning
        # this is done at cost of additional compute of removing from list. Can comment out to make faster.
        else:
            removeNumpyArrFromList(foundSubs, sub)

    return orbitsSet

if __name__ == '__main__':
    # print(len(findAllOrbitsSymmetricGroupSubtables(4)))
    # print(calcNumNonIsomorphKnuthianDigraphs(5))

    # tab = np.array([[0,1,2,3],[0,1,3,2],[0,3,2,1],[0,2,1,3]])
    # adj = getKnuthianAdjMatrix(tab)
    # print(recoverKnuthianMultiplicationTable(adj))

    tab = np.array([[0,3,1,2],[0,1,3,2],[0,2,1,3],[0,1,2,3]])
    adj = getKnuthianAdjMatrix(tab)
    print(adj)
    # for i in range(len(adj)):
    #     for j in range(len(adj)):
    #         if adj[len(adj)-i-1][j] == 1:
    #             print(f"\draw [draw=black, fill=black] ({j},{i}) rectangle ({j+1},{i+1});")

    # print(findAllOrbitsSymmetricGroupSubtables(4))
    pass
