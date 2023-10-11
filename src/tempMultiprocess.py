import multiprocessing as mp

import numpy as np
from utilsUPP import *
from genUPPDigraphs import *
from sageGAP import PreCompInducedPermGroup


# From Thomas Lux on https://stackoverflow.com/questions/489861/locking-a-file-in-python
import fcntl, os
def lock_file(f):
    if f.writable(): fcntl.lockf(f, fcntl.LOCK_EX)
def unlock_file(f):
    if f.writable(): fcntl.lockf(f, fcntl.LOCK_UN)

class AtomicOpen:
    # Open the file with arguments provided by user. Then acquire
    # a lock on that file object (WARNING: Advisory locking).
    def __init__(self, path, *args, **kwargs):
        # Open the file and acquire a lock on the file before operating
        self.file = open(path,*args, **kwargs)
        # Lock the opened file
        lock_file(self.file)

    # Return the opened file object (knowing a lock has been obtained).
    def __enter__(self, *args, **kwargs): return self.file

    # Unlock the file and close the file object.
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):        
        # Flush to make sure all buffered contents are written to file.
        self.file.flush()
        os.fsync(self.file.fileno())
        # Release the lock on the file.
        unlock_file(self.file)
        self.file.close()
        # Handle exceptions that may have come up during execution, by
        # default any exceptions are raised to the user.
        if (exc_type != None): return False
        else:                  return True  


def loadIntermidMatrices(file) -> list[np.ndarray]:
    f = np.load(file)
    return [f[i] for i in f]

def multi_genUPPByBlockDFS4(subMatrix):
    '''
    Same alg as genUPPDigraphs.py:genUPPByBlockDFS() except now implements multiprocessing and only on k=4, starting at the 12x12 submatrices

    #TODO rewrite original function to use multiprocessing from beginning
    '''

    k = 4

    matrixOut = 'completeSearch4'
    processedIndices = 'procIndex'


    zeroBlock = np.zeros((k, k))
    colEdgeCondition = [1 for i in range(k)]

    for i in range(k-1, k):
        # print(len(intermidMatrices))
        # if i == k-1:
        #     np.savez('./intermid34', *intermidMatrices)
        group = PreCompInducedPermGroup(16)

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

                        elif group.isLexMinAdjacencyMatrixSage(mat):
                            with AtomicOpen(matrixOut, 'ab') as mout:
                                np.save(mout, mat)
    
    # write index to file of processed indices
    with AtomicOpen(processedIndices, 'ab') as iout:
        np.save(iout, subMatrix)

if __name__ == "__main__":
    intermid = loadIntermidMatrices('intermid34.npz')
    p = mp.Pool(mp.cpu_count() - 1)
    p.map(multi_genUPPByBlockDFS4, intermid)
