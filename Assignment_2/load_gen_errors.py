import numpy as np

root = "output/ANN-class/"

def display(name):
    arr = np.load(root + name)
    print(arr)



display("E_gen_o.npy")
display("nums_of_hidden.npy")