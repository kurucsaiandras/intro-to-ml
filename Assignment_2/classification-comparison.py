import numpy as np
from scipy import stats
import math
from dtuimldmtools import mcnemar

root = "output/"

def load_and_concatenate(root, model, type):
    arr = []
    for i in range(1, 11):
        arr.append(np.load(f'{root}/{model}/fold_{i}_y_{type}.npy'))
    return np.concatenate(arr)

def compare_models(model_1, model_2, alpha):
    y_est_1 = load_and_concatenate(root, model_1, 'est')
    y_tru_1 = load_and_concatenate(root, model_1, 'true')
    y_est_2 = load_and_concatenate(root, model_2, 'est')
    y_tru_2 = load_and_concatenate(root, model_2, 'true')


    print(y_est_1, y_est_2)

    if not (y_est_1.shape == y_est_2.shape == y_tru_1.shape == y_tru_2.shape):
        print('Invalid data: the array lengths differ.')
        return
    
    n = y_est_1.shape[0]

    if (y_tru_1 != y_tru_2).sum():
        print('Invalid data: the two true arrays are not the same.')
        return
    
    print("Comparing A: ", model_1, " and B: ", model_2)

    e1_true = (y_tru_1 == y_est_1)
    e2_true = (y_tru_2 == y_est_2)

    n11 = ( e1_true &  e2_true).sum()
    n12 = ( e1_true &  ~e2_true).sum()
    n21 = (~e1_true &  e2_true).sum()
    n22 = (~e1_true & ~e2_true).sum()

    print(n11, n12, n21, n22)

    acc_diff = (n12 - n21) / n


    # confidence interval and p-value using mcNemar's test
    Q = (n**2 * (n+1)*(acc_diff+1)*(1-acc_diff) 
        / (n*(n12+n21)-(n12-n21)**2)
    )
    f = ((acc_diff +1)*0.5)*(Q-1)
    g = ((1-acc_diff)*0.5)*(Q-1)

    interval = stats.beta.interval(1-alpha, a=f,b=g) 
    z_L = interval[0] * 2 - 1
    z_U = interval[1] * 2 - 1
    p = 2*stats.binom.cdf(min(n12,n21),n= n12 + n21, p=0.5)

    

    return z_L, acc_diff, z_U, p

LB, z, UB, p = compare_models('Logistic-class', 'baseline-class', 0.05)

print(f'Confidence interval: ({LB:.5f}, {UB:.5f})')
print(f'Estimated difference: {z:.5f}')
print(f"p-value: {p}")