import numpy as np
from sage.all import *
from utilsUPP import getSetRepresentationAdjacencyMatrix

libgap.LoadPackage("images")

class PreCompInducedPermGroup:
    def __init__(self, n: int):
        self.computeInducedGroup(n)

    def isLexMin(self, set: list[int], n: int) -> bool:
        if self.n != n:
            self.computeInducedGroup(n)
        
        return libgap.IsMinimalImage(self.inducedGroup, set, libgap.OnSets).sage()

    def computeInducedGroup(self, n: int):
        '''
        Pre-compute the induced permutation group of S_n on the nxn matrix postitions
        '''

        elems = libgap.Cartesian([i for i in range(1,n+1)], [i for i in range(1,n+1)])
        gens = []
        for gen in libgap.GeneratorsOfGroup(libgap.SymmetricGroup(n)):
            ls = [libgap.OnPairs(elem, gen) for elem in elems]
            gens.append(libgap.PermListList(elems, ls))

        self.inducedGroup = libgap.GroupByGenerators(gens)
        self.n = n

    def isLexMinAdjacencyMatrixSage(self, adj: np.ndarray):
        set = getSetRepresentationAdjacencyMatrix(adj)
        n = np.shape(adj)[0]
        return self.isLexMin(set, n)

if __name__ == "__main__":
    p = PreCompInducedPermGroup(8)
    print(p.isLexMin([1,2,3,4,13], 8))