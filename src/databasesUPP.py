import os
import sqlite3

from utilsUPP import *
from knuthianUPP import *
import numpy as np
import pynauty as nauty

baseDir = os.path.split(os.path.dirname(__file__))[0]
dataDir = os.path.join(baseDir, 'data')
file = os.path.join(dataDir, 'knuthianUPPs.db')

con = sqlite3.connect(file)
cur = con.cursor()

def initKnuthian(k):
    cur.execute("""
                CREATE TABLE knuthian{} (
                    certificate TEXT PRIMARY KEY,
                    binRep TEXT,
                    selfReverse INTEGER,
                    conjecturedSelfRev INTEGER,
                    rank INTEGER
                );
                """.format(k))

def writeKnuthianToDB(k: int):
    numComputed = 0
    insert = 'INSERT OR IGNORE INTO knuthian{} VALUES (?, ?, ?, ?, ?);'.format(k)
    for tab in genProductTables(k):
        adj = getKnuthianAdjMatrix(tab)

        args = (nauty.certificate(createNautyGraphFromAdjMatrix(adj)),
                convertAdjMatrixToBinaryString(adj),
                int(nx.is_isomorphic(nx.DiGraph(adj), nx.DiGraph(np.transpose(adj)))),
                int((len(partitionEqualRowIndices(tab)) == 2 and min([len(i) for i in partitionEqualRowIndices(tab).values()]) <= 1) or len(partitionEqualRowIndices(tab)) == 1),
                np.linalg.matrix_rank(adj))
        
        # for row in cur.execute('SELECT * FROM sqlite_master WHERE name = "knuthian5";'):
        #     print(row)

        cur.execute(insert, args)
        numComputed += 1
        if numComputed % 5000 == 0:
            con.commit()
            with open(os.path.join(dataDir, 'dbProgress'), 'w') as f:
                f.write(str(numComputed))

if __name__ == '__main__':
    size = 5
    initKnuthian(size)
    # for row in cur.execute('SELECT COUNT(certificate) FROM knuthian5;'):
    #     print(row)
    writeKnuthianToDB(size)
    pass