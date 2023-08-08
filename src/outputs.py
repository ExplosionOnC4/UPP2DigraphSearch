# Stores any code which was used to output important data in one file

from genUPPDigraphs import *
from switchingsUPP import *
from utilsUPP import *
import os

baseDir = os.path.split(os.path.dirname(__file__))[0]
dataDir = os.path.join(baseDir, 'data')

def writeMajorConnectComponent(k: int):
    stand = createStandardCentralDigraph(k)
    standComponent = switchConnectedComponentFromVertex(stand)
    print(len(standComponent))
    with open(os.path.join(dataDir, f'majorGk{k}.txt'), 'w') as f:
        f.write(str(standComponent))

if __name__=='__main__':
    writeMajorConnectComponent(3)