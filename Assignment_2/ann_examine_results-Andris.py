import numpy as np

root = "Assignment_2/output/ANN-class/"

def display(name):
    arr = np.load(root + name)
    print(arr)


#arr1 = np.load(root + "fold_9_y_est.npy")
#arr2 = np.load(root + "fold_9_y_true.npy")
#print(arr2 - arr1)

display("E_gen_o.npy")
display("nums_of_hidden.npy")