import sys
from typing import List

datasetFile: str = "";
if len(sys.argv) == 0:
    datasetFile= "sonar.csv"
else:
    datasetFile= sys.argv[0]

f = open(f"./dataset/${datasetFile}", 'r')

y: List[str] =[]
X: List[List[float]]=[[]]

for line in f.readlines():
    fields = line.split(',')
    y.append(fields[len(fields)])
    try:
        X.append([float(param) for param in fields[:-1]])
    except:
        #Logging


    

f.close()

