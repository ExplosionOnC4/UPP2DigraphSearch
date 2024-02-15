import networkx as nx
import itertools
import numpy as np
import matplotlib.pyplot as plt
from utilsUPP import *
from sageGAP import PreCompInducedPermGroup

def genPossibleUPPRows(k: int) -> list:
    '''
    Generate all possible rows of length k^2 with exactly k 1's
    '''

    rows = []
    for ls in list(itertools.combinations(range(k ** 2), k)):
        rows.append(createRowVectorFromIndexList(k, ls))
    return rows

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


def genPossibleBlocksGivenEdgeConditions(rowSums: list[int], colSums: list[int], rowExact: bool, colExact: bool, _block, _i=0, _j=0, _blocks=[]) -> list[np.ndarray]:
    '''
    Given a maximum sum on each row and column of a {0-1} matrix, finds all possible such matrices that has row sums and column sums less than or equal to the
    provided bound. If rowExact or colExact are true then the corresponding bounds have to be met.
    '''

    if np.min(rowSums) < 0 or np.min(colSums) < 0:
        return []

    # Terminate immediately if both row and column exact but sums are different as each 1 adds to both rowSum and colSum
    if rowExact and colExact and np.sum(rowSums) != np.sum(colSums):
        return []
    # Further initial impossible condition.
    if (np.sum(rowSums) < np.sum(colSums) and colExact) or (np.sum(colSums) < np.sum(rowSums) and rowExact):
        return []

    # Ideally calculate all exact solutions first and then get non-exact solutions by deleting 1's but not sure that hits all of them.

    # This algorithm is O({k^2 \choose maxOnes}), where maxOnes <= k^2 but will be used on maxOnes = k as we will require exactly 1 one in each column
    # TODO Since this algorithm is equivalent to just choosing the maxOnes 1's in the matrix, rewrite using itertools.combinations if exact maybe

    # NOTE this is definitely some puzzle game already if assuming exactness. It is a strictly harder version of Picross/Nonograms (no intermidiate information)
    # which is already NP-complete.
    maxOnes = min(np.sum(rowSums), np.sum(colSums))
    if maxOnes > 0:
        for i in range(_i, len(rowSums)):
            for j in range(_j * int(i == _i), len(colSums)):
                if rowSums[i] * colSums[j] > 0 and _block[i][j] != 1:
                    
                    _block[i][j] = 1
                    rowSums[i] -= 1
                    colSums[j] -= 1
                    if (not rowExact and not colExact) or (rowExact and np.sum(rowSums) == 0) or (colExact and np.sum(colSums) == 0):
                        _blocks.append(np.copy(_block))
                        # BUG does not add zero block for not exact case

                    genPossibleBlocksGivenEdgeConditions(rowSums, colSums, rowExact, colExact, _block, i, j, _blocks)

                    rowSums[i] += 1
                    colSums[j] += 1
                    _block[i][j] = 0

    return _blocks

def findRowSums(matrix: np.ndarray) -> list[int]:
    rowSums = []
    for i in range(np.shape(matrix)[0]):
        rowSums.append(sum(matrix[i]))
    return rowSums

def findColSums(matrix: np.ndarray) -> list[int]:
    return findRowSums(np.transpose(matrix))

def appendBlocksIntermidMatrix(matrix: np.ndarray, blockList: list[np.ndarray], k: int, i: int) -> np.ndarray:
    '''
    Helper function for genUPPByBlockDFS(). Goes from the ixi block matrix to the (i+1)x(i+1) block matrix by appending the block matrices in the list
    to the (i+1)-th row/column. If blockList has length less than 2i+1 then fill remaining blocks with the zero matrix.
    '''

    # Consider rewriting to use np.block() or np.concatenate()

    zeroBlock = np.zeros((k, k))
    # fill in empty blocks with zero
    zerodBlockList = list(blockList)
    for j in range(2 * i + 1 - len(blockList)):
        zerodBlockList.append(zeroBlock)

    fullMatrix = np.zeros((k*(i+1), k*(i+1)))
    fullMatrix[:k*i,:k*i] = matrix
    for j in range(i):
        fullMatrix[k*j:k*(j+1),k*i:] = zerodBlockList[j]
    for j in range(i + 1):
        fullMatrix[k*i:,k*j:k*(j+1)] = zerodBlockList[j + i]
    return fullMatrix

def genUPPByBlockDFS(k: int) -> list[np.ndarray]:
    '''
    Based of algorithm in appendix A.2 in https://doi.org/10.1016/j.jcta.2011.03.009
    '''

    intermidMatrices = [createBasisBlockMatrix(0, k)] # add M_1
    zeroBlock = np.zeros((k, k))
    colEdgeCondition = [1 for i in range(k)]

    for i in range(1, k):
        # print(len(intermidMatrices))
        # if i == k-1:
        #     np.savez('./intermid34', *intermidMatrices)
        group = PreCompInducedPermGroup((i + 1) * k)

        # the (i+1)x(i+1) submatrices
        temp = []
        for subMatrix in intermidMatrices:
            # store the array of added blocks in stack as they generate the new matrix
            stackDFS = [[]]
            # attempt to add new block to intermidiate matrix if there are open slots remaining
            # for each step from ixi -> (i+1)x(i+1) there are 2i+1 available slots for blocks

            # We store list of added blocks, and determine which block to add next by its current length. We can construct the next intermid matrix
            # By calling np.block() on the previous submatrix with the added blocks at the correct array subindices. 
            while stackDFS:
                filledBlocks = stackDFS.pop()

                if len(filledBlocks) == 0:
                    stackDFS.append([createBasisBlockMatrix(i, k)])

                elif len(filledBlocks) < i: # filling in column i+1
                    rowSum = findRowSums(subMatrix[k * len(filledBlocks):k * (len(filledBlocks) + 1),:])
                    canBlocks = genPossibleBlocksGivenEdgeConditions(rowSums=[k - i for i in rowSum], colSums=colEdgeCondition, rowExact=(i == k - 1), colExact=True, _block=zeroBlock, _i=0, _j=0, _blocks=[])
                    for block in canBlocks:
                        tempBlockList = list(filledBlocks) + [block]
                        mat = appendBlocksIntermidMatrix(subMatrix, tempBlockList, k, i)
                        if checkMostSinglePath(mat) and group.isLexMinAdjacencyMatrixSage(mat):
                            stackDFS.append(tempBlockList)

                else: # filling in row i+1
                    horizBlocks = filledBlocks[i:] # Get all blocks already inserted into row i+1
                    if horizBlocks:
                        rows = [findRowSums(arr) for arr in horizBlocks]
                        rowSum = [np.sum(x) for x in zip(*rows)]
                    else:
                        rowSum = [0 for i in range(k)]

                    # the bottom right corner block -> has to be row exact too if last i
                    canBlocks = genPossibleBlocksGivenEdgeConditions(rowSums=[k - i for i in rowSum], colSums=colEdgeCondition, rowExact=(i == k - 1 and len(filledBlocks) >= 2 * i), colExact=True, _block=zeroBlock, _i=0, _j=0, _blocks=[])
                    for block in canBlocks:
                        tempBlockList = list(filledBlocks) + [block]
                        mat = appendBlocksIntermidMatrix(subMatrix, tempBlockList, k, i)
                        if checkMostSinglePath(mat):
                            if len(tempBlockList) < 2 * i + 1:
                                stackDFS.append(tempBlockList)

                            # Can only check if lexMin if the entire last row has been filled, for example:
                            #
                            # [[1,1,1,0,0,0],[0,0,0,1,1,1],[0,0,0,0,0,0],[1,1,0,0,1,0],[0,0,0,0,0,0],[0,0,1,1,0,1]] is lexMin while
                            # [[1,1,1,0,0,0],[0,0,0,1,1,1],[0,0,0,0,0,0],[1,1,0,0,0,0],[0,0,0,0,0,0],[0,0,1,0,0,0]] is not as the permutation (56)
                            # makes the adjacency matrix lexicographically smaller
                            #
                            # Checking lexMin on the (i+1)th column *should* be safe (as if not lexMin on insert then will not be lexMin after the entire submatrix
                            # has been filled in) but should be formally proved.
                            # The assumption will drastically speed up the algorithm as we early abort big subtrees.
                            elif group.isLexMinAdjacencyMatrixSage(mat):
                                temp.append(mat)            

        intermidMatrices = temp

    return intermidMatrices

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

    $$\sigma(\hat{T}) = P_\sigma \overline{\sigma}(\hat{T}) P_\sigma^{-1}$$ where $\overline{\sigma}(\hat{T})_{ij} = \sigma(\hat{T}_{ij})$, 
    i.e. it permutes the rows, permutes the columns and maps the elements of the table.

    The paper proves that the k x k multiplication tables T_1 and T_2 define isomorphic operations if there exists \sigma \in S_{k-1} such that
    \sigma(\hat{T}_1) = \hat{T}_2.
    '''

    newTable = np.copy(subtable)
    for i in range(len(subtable)):
        for j in range(len(subtable[0])):
            newTable[perm[i+1]-1][perm[j+1]-1] = perm[subtable[i][j]]  # As S_n does not include 0
    return newTable

def areKnuthianIsomorphic(table1: np.ndarray, table2: np.ndarray) -> bool:
    '''
    Checks whether two product tables are isomorphic.
    
    By result in https://doi.org/10.1016/S0021-9800(70)80032-1, two Knuthian systems are isomorphic iff the underlying product tables are isomorphic (modulo other conditions).

    Uses @applySymmetricActionOnSubtable() and the result from https://cklixx.people.wm.edu/reu02.pdf to check for isomorphism.
    '''

    # Do not see anything better than checking all k! possible permutations of the elements of k right now
    if np.shape(table1) != np.shape(table2):
        return False
    symmetricGroup = [dict(zip([i for i in range(1, len(table1))], perm)) for perm in itertools.permutations([i for i in range(1, len(table1))])]
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
    symmetricGroup = [dict(zip([i for i in range(1, k)], perm)) for perm in itertools.permutations([i for i in range(1, k)])]
    sum = 0
    for sigma in symmetricGroup:
        sum += len(findFixedSubtablesOfPermutation(sigma, subtables))
    return int(sum / len(symmetricGroup))

def getSymmetricGroupOrbitOfSubtable(subtable: np.ndarray) -> list[np.ndarray]:
    '''
    Returns the orbit of a given product subtable for the action of S_{k-1} described in https://cklixx.people.wm.edu/reu02.pdf.
    '''

    orbit = []
    symmetricGroup = [dict(zip([i for i in range(1, len(subtable) + 1)], perm)) for perm in itertools.permutations([i for i in range(1, len(subtable) + 1)])]
    for sigma in symmetricGroup:
        image = applySymmetricActionOnSubtable(subtable, sigma)
        if not any((image == x).all() for x in orbit):
            orbit.append(image)
    return orbit

def removeNumpyArrFromList(ls: list[np.ndarray], arr: np.ndarray) -> None:
    '''
    Helper function, removes first instance of (numpy array) arr in ls.

    :throw StopIteration if arr not in ls
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
    

def __main__():
    # nx.draw(nx.from_numpy_array(createStandardCentralDigraph(4), create_using=nx.DiGraph))
    # plt.show()
    # ls = genUPPMatrices(3)
    ls2 = genUPPByBlockDFS(3)
    print(len(ls2))
    print(ls2)
    # print(ls[-6:])
    # print(len(ls))

if __name__ == '__main__':
    __main__()