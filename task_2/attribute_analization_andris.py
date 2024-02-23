import sys
sys.path.insert(0, '')
from utils.read_dataset import read

attributeNames, X = read()
print(X)