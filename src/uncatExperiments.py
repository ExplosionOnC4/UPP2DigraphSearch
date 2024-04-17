# File to store research directions and random experiments/ideas before being categorised. Expect large commented out sections for past calculations

import numpy as np
import collections
import pynauty as nauty
from itertools import combinations, permutations
from readDataFiles import *
from utilsUPP import *
from switchingsUPP import *
import matplotlib.pyplot as plt
from knuthianUPP import *

def getInducedUndirectedCanonAdj(adj: np.ndarray) -> np.ndarray:
    '''
    For a given CDG, return an undirected graph where the edge set is the arc set with di-gons being represented by a single edge.
    
    We conjecture if di-gons are represented by a multigraph then there is exactly two possible base CDGs, the original and its reverse.
    '''

    undirectedAdj = nx.to_numpy_array(nx.DiGraph(adj).to_undirected())
    relabel = nauty.canon_label(createNautyGraphFromAdjMatrix(undirectedAdj))
    perm = dict(zip(relabel, range(np.shape(adj)[0])))
    # perm = dict(zip(relabel, np.roll(relabel, -1)))
    D = nx.relabel_nodes(nx.DiGraph(adj), perm)
    return nx.to_numpy_array(D, nodelist=[*range(np.shape(adj)[0])])

def findAllSubcentralDigraphs(adj: np.ndarray, excludeNeighbours=False) -> list[np.ndarray]:
    '''
    For a CDG, return a list of all of its subcentral digraphs.
    '''

    # a necessary condition, if a CDG is reducible then it satisfies the neighbourhood condition.
    if not satisfiesNeighbourhoodCondition(adj):
        return []

    SCDGs = []
    loops = getLoopVerts(adj)
    for i in loops:
        excludeSet = {i} if not excludeNeighbours else set(findInneighbours(i, adj)).union(set(findOutneighbours(i, adj)))
        scdg = getUPPClosureOfSubset(set(loops).difference(excludeSet), adj, {i})
        if scdg is not None:
            SCDGs.append(scdg)
    return SCDGs

def calcAverageCommonNeighboursLoopVertices(adj: np.ndarray) -> tuple:
    '''
    Idea for probabilistic proof of conjecture described in getInducedUndirectedCanonAdj(). Didn't work.
    '''

    inAverage, outAverage, i = 0, 0, 0
    loops = [i for i in range(np.shape(adj)[0]) if int(adj[i][i]) == 1]
    for l1, l2 in combinations(loops, 2):
        inAverage += np.dot(adj[l1, :], adj[l2, :])
        outAverage += np.dot(adj[:, l1], adj[:, l2])
        i += 1
    return (inAverage / i, outAverage / i)

def checkKnuthianSwitchComponent(adj, knuthCertList):
    q = Queue()
    isomorphHash = {}

    q.put(adj)
    while not q.empty():
        tempAdj = q.get()
        # lists aren't hashable
        tempLabel = nauty.certificate(createNautyGraphFromAdjMatrix(tempAdj))

        # TODO make this quicker, eg only check on partitions of rows/columns
        for rows in list(itertools.combinations(range(np.shape(tempAdj)[0]), 2)):
            for cols in list(itertools.combinations(range(np.shape(tempAdj)[1]), 2)):
                if validSwitch(tempAdj, *rows, *cols):
                    newAdj = np.copy(tempAdj)
                    performSwitch(newAdj, *rows, *cols)
                    newLabel = nauty.certificate(createNautyGraphFromAdjMatrix(newAdj))
                    if newLabel in knuthCertList:

                        # For a central digraph, look at the digraphs attainable from one switch and add them to the neighbourhood
                        if tempLabel in isomorphHash and newLabel not in isomorphHash[tempLabel]:
                            isomorphHash[tempLabel].append(newLabel)
                        else:
                            isomorphHash[tempLabel] = [newLabel]
                        
                        # Put switched digraph in queue if has not been seen before and add to hash
                        if newLabel not in isomorphHash:
                            q.put(newAdj)
                        isomorphHash[newLabel] = [tempLabel]

    return isomorphHash

if __name__ == '__main__':

    ls = readOrderedComplete4()

    # the 4-CDGs not switch equivalent to canonical 4-CDG are 
    # [0, 174, 258, 288, 290, 298, 301, 309, 310, 317, 501, 653, 684, 1591, 1966, 1967, 2027, 2028, 2029, 2087, 2300]

    # major = readMajorGk(4, keepCerts=True)
    # certList = list(map(lambda adj : nauty.certificate(createNautyGraphFromAdjMatrix(adj)), readOrderedComplete4()))
    # nonMajorCDGs = [i for i in range(len(certList)) if certList[i] not in major]
    # print(nonMajorCDGs)

    # The reverses of the 4-CDGs
    # The 34 self-reverse 4-CDGs are
    # ['185', '850', '907', '942', '1696', '2223', '2300', '2310', '2353', '2465', '2466', '2514', '2517', '2522', '2649', '2677', '2678',
    # '2728', '2731', '2737', '2739', '2816', '3150', '3257', '3263', '3265', '3287', '3378', '3379', '3380', '3469', '3478', '3488', '3491']

    reverseList = [np.transpose(adj) for adj in readOrderedComplete4()]
    reverseCertList = list(map(lambda adj : nauty.certificate(createNautyGraphFromAdjMatrix(adj)), reverseList))
    certList = list(map(lambda adj : nauty.certificate(createNautyGraphFromAdjMatrix(adj)), readOrderedComplete4()))
    reverseDict = {}
    for i in range(len(certList)):
        reverseDict[i] = reverseCertList.index(certList[i])
    selfReverseList = [i for i in reverseDict.keys() if reverseDict[i] == i]
    # print(selfReverseList)

    # 4-CDGs which have equivalent underlying undirected graph
    # By underlying directed digraph, we mean removing digons, as keeping digons in a multigraph appears to produce only the original CDG and its reverse.
    # !! Most self-reverse 4-CDGs lie in the same equivalence classes as other self-reverse 4-CDGs !!
    # Frequency information (key is number of times a specific undirected graph appears, value is how many undirected graphs appear key times):
    # Counter({'4': 193, '8': 106, '2': 76, '12': 37, '6': 23, '24': 18, '16': 10, '10': 7, '28': 5, '22': 3, '32': 3, '3': 3, '1': 3, '7': 2, '18': 1, '56': 1, '14': 1, '40': 1, '20': 1})

    undirectedList = [nx.DiGraph(adj).to_undirected() for adj in readOrderedComplete4()]
    undirectedCertList = list(map(lambda adj : nauty.certificate(createNautyGraphFromAdjMatrix(adj)), undirectedList))
    undirectedDict = {}
    for i, cert in enumerate(undirectedCertList):
        if cert in undirectedDict:
            # filter out reverses as they always lie in the same equivalence class
            if reverseDict[i] not in undirectedDict[cert]:
                undirectedDict[cert].append(i)
        else:
            undirectedDict[cert] = [i]
    # print('\n'.join([str(i) for i in undirectedDict.values()])) 
    # print('\n'.join([str(i) for i in undirectedDict.values() if set(i).intersection(set(selfReverseList))]))
    # print('\n'.join([str(len(i)) for i in undirectedDict.values()]))
    # print(collections.Counter([str(len(i)) for i in undirectedDict.values()]))
    
    # print(collections.Counter([str(undirectedDict[undirectedCertList[i]]) for i in selfReverseList]))

    # Finding what arcs where flipped between 4-CDGs lying in the same undirected equivalence class
    flippedSubgraphs = {}
    showShadowArcs = False
    mult = 1 if not showShadowArcs else -1

    for cert, eqClass in undirectedDict.items():
        if len(eqClass) >= 2:
            for i1, i2 in combinations(eqClass, 2):
                # on comparison, select the CDG or reverse that leaves a smaller number of flipped arcs.
                # TODO consider drawing the -1 edges in a different colour instead of zeroing out
                flips1 = np.clip(mult * (getInducedUndirectedCanonAdj(ls[i1]) - getInducedUndirectedCanonAdj(ls[i2])), a_min=0, a_max=None)
                flips2 = np.clip(mult * (getInducedUndirectedCanonAdj(ls[i1]) - getInducedUndirectedCanonAdj(ls[reverseDict[i2]])), a_min=0, a_max=None)
                # need this in case second CDG is self-reverse
                flips3 = np.clip(mult * (getInducedUndirectedCanonAdj(ls[reverseDict[i1]]) - getInducedUndirectedCanonAdj(ls[i2])), a_min=0, a_max=None)
                # flippedComponent = flips1 if np.sum(flips1) <= np.sum(flips2) else flips2
                # flippedComponent = [flip for flip in [flips1, flips2, flips3] if np.sum(flip) == min(np.sum(flips1), np.sum(flips2), np.sum(flips3))][0]
                # cdgIndices = (i1, i2) if np.sum(flips1) <= np.sum(flips2) else (i1, reverseDict[i2])
                if np.sum(flips1) == min(np.sum(flips1), np.sum(flips2), np.sum(flips3)):
                    flippedComponent = flips1
                    cdgIndices = (i1, i2)
                elif np.sum(flips2) == min(np.sum(flips1), np.sum(flips2), np.sum(flips3)):
                    flippedComponent = flips2
                    cdgIndices = (i1, reverseDict[i2])
                else:
                    flippedComponent = flips3
                    cdgIndices = (reverseDict[i1], i2)
                if cert in flippedSubgraphs:
                    flippedSubgraphs[cert].append((nx.from_numpy_array(flippedComponent, create_using=nx.DiGraph), cdgIndices))
                else:
                    flippedSubgraphs[cert] = [(nx.from_numpy_array(flippedComponent, create_using=nx.DiGraph), cdgIndices)]
    
    # !! The flipped arcs subgraph is connected only if both the CDGs in the same undirected equivalence class are self-reverse !!
    connected = []
    for val in flippedSubgraphs.values():
        for s, iis in val:
            if nx.number_strongly_connected_components(s) == 1:
                connected.append(iis)
    # print(connected)

    # Drawing flipped components
    # Nearly every single pair that does not yield a strongly connected flipped subCDG is flipped only on digons
    # (?=>?) there exists a switch operation for digons where u1 <-> v1 and u2 <-> v2 becomes u1 <-> u2 and v1 <-> v2
    # There are some exception which have additional flipped arcs between such a digon switch.
    # TODO find conditions for when a digon switch is possible
    for _, v in flippedSubgraphs.items():
        for subgraph, indices in v:
            if indices not in connected:
                plt.figure()
                plt.title(f'Flipped arc subgraph of 4-CDG {indices[0]} and {indices[1]}:')
                showDigons = True

                if showDigons:
                    # Very hacky section, probably doesn't work for general k-CDGs
                    g1, g2 = (nx.from_numpy_array(getInducedUndirectedCanonAdj(ls[i]), create_using=nx.DiGraph) for i in indices)
                    for i, (g, c, r) in enumerate([(g1, 'red', 0.1), (g2, 'blue', 0.15)]):
                        digonSubgraph = nx.DiGraph()
                        digonSubgraph.add_nodes_from(subgraph)
                        digons = [(u, v) for u, v in g.edges if (v, u) in g.edges and u != v]
                        digonSubgraph.add_edges_from(digons, color=c)
                        colors = [digonSubgraph[u][v]['color'] for u,v in digonSubgraph.edges]
                        nx.draw(digonSubgraph, pos=nx.circular_layout(subgraph), edge_color=colors, with_labels=True, connectionstyle=f"arc3,rad={r}")
                        plt.text(-1.4, 1-i/5, f"{c}: {indices[i]}")

                # TODO make these edges thicker if showDigons is set to true
                nx.draw(subgraph, pos=nx.circular_layout(subgraph), with_labels=True)
                plt.show()


    # productList = [np.matmul(adj, np.transpose(adj)) for adj in ls]
    # print(productList[1])
    # print([i for i in range(len(productList)) if int(productList[i].sum()) != 256])
    # print(len([i for i in range(len(productList)) if int(np.max(productList[i] - 4*np.eye(16))) < 4]))

    # print([calcAverageCommonNeighboursLoopVertices(adj) for adj in ls][-10:])
    # print("max common in: " + str(max([calcAverageCommonNeighboursLoopVertices(adj)[0] for adj in ls])))
    # print("max common out: " + str(max([calcAverageCommonNeighboursLoopVertices(adj)[1] for adj in ls])))
    # print("max common neigh: " + str(max([calcAverageCommonNeighboursLoopVertices(adj)[0] + calcAverageCommonNeighboursLoopVertices(adj)[1] for adj in ls])))
    # print([calcAverageCommonNeighboursLoopVertices(adj)[0] for adj in ls].index(1.5))
    # print(ls[2])

    # there are 1299 max rank 4-CDGs
    # maxRanks = [i for i in range(len(ls)) if np.linalg.matrix_rank(ls[i]) == 8]
    # print(len(maxRanks))

    # there are 328 irreducible 4-CDGs, and removing neighbourhood condition does not yield more
    # Further, there is exactly one 4-CDG with 4 subCDGs, being the standard one (1-index: 3492)
    # Curtis et. al. managed to prove in https://cklixx.people.wm.edu/reu02.pdf that the standard k-CDG always has k subCDGS
    # CONJECTURE: above is an iff.
    # irreducible4CDGs = [i for i in range(len(ls)) if len(findAllSubcentralDigraphs(ls[i])) == 0]
    # print(len(irreducible4CDGs))
    # print(len(nonMajorCDGs))
    # print(set(nonMajorCDGs).difference(set(irreducible4CDGs)))
    # print(len(set(irreducible4CDGs).difference(set(maxRanks))))

    # !! Both of below are equal => all subcentral digraphs are achieved by removing loop + all its neighbours !!
    # print(collections.Counter([len(findAllSubcentralDigraphs(i)) for i in ls]))
    # print(collections.Counter([len(findAllSubcentralDigraphs(i, excludeNeighbours=True)) for i in ls]))

    # 328 CDGs do not satisfy the neighbourhood condition, which are all exactly the irreducible ones.
    # CONJECTURE: NC is necessary and sufficient to be reducible
    # nonSFC = [i for i in range(len(ls)) if not satisfiesNeighbourhoodCondition(ls[i])]
    # print(len(nonSFC))
                
    # There are 698 (>> 44) 4-CDGs which contain the canonical 3-CDG as a subCDG.
    # Thus contains canonical (k - 1)-CDG as a subCDG =/=> Knuthian
    # containStandardSubCDG = [i for i, adj in enumerate(readOrderedComplete4()) if any([nx.is_isomorphic(nx.DiGraph(createStandardCentralDigraph(3)), nx.DiGraph(sub)) for sub in findAllSubcentralDigraphs(adj)])]
    # print(len(containStandardSubCDG))
                
    knuthianList4 = [getKnuthianAdjMatrix(extendMultiplicationSubtable(orb[0])) for orb in findAllOrbitsSymmetricGroupSubtables(4)]
    knuthianCertList = list(map(lambda adj : nauty.certificate(createNautyGraphFromAdjMatrix(adj)), knuthianList4))
    knuthianIndices = [certList.index(cert) for cert in knuthianCertList]
    # print(knuthianIndices)

    # !!! The induced subgraph on Knuthian 4-CDGs of the switching graph is connected !!!
    # print(len(checkKnuthianSwitchComponent(createStandardCentralDigraph(4), knuthianCertList)))

    knuthianReverseList = [np.transpose(adj) for adj in knuthianList4]
    knuthianReverseCertList = list(map(lambda adj : nauty.certificate(createNautyGraphFromAdjMatrix(adj)), knuthianReverseList))
    knuthianReverseDict = {}
    for i in range(len(knuthianCertList)):
        if knuthianCertList[i] in knuthianReverseCertList:
            knuthianReverseDict[i] = knuthianReverseCertList.index(knuthianCertList[i])
        else:
            knuthianReverseDict[i] = -1
    knuthianSelfReverseList = [i for i in knuthianReverseDict.keys() if knuthianReverseDict[i] == i]
    # !! Reverse of Knuthian CDG is Knuthian only if it is self-reverse !!
    # print([reverseDict[i] in knuthianIndices for i in knuthianIndices])
    # print(knuthianReverseDict)
    # print(knuthianSelfReverseList)
    selfReverseTables = [recoverKnuthianMultiplicationTable(knuthianList4[i]) for i in knuthianSelfReverseList]
    print('\n\n'.join([np.array_str(i) for i in selfReverseTables]))

    pass

