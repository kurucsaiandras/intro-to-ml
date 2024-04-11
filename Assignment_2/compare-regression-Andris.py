import numpy as np
from scipy import stats
import math

root = "Assignment_2/output"

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

    if not (y_est_1.shape == y_est_2.shape == y_tru_1.shape == y_tru_2.shape):
        print('Invalid data: the array lengths differ.')
        return
    
    n = y_est_1.shape[0]

    if (y_tru_1 != y_tru_2).sum():
        print('Invalid data: the two true arrays are not the same.')
        return
    
    y_tru = y_tru_1

    z_1 = np.square(y_tru - y_est_1)
    z_2 = np.square(y_tru - y_est_2)
    z = z_1 - z_2
    z_hat = z.mean()
    std_hat = math.sqrt(np.square(z - z_hat).sum() / (n * (n - 1)))

    z_L = stats.t.ppf(alpha / 2, n - 1, z_hat, std_hat)
    z_U = stats.t.ppf(1 - alpha / 2, n - 1, z_hat, std_hat)

    p = 2 * stats.t.cdf(-abs(z_hat), n - 1, 0, std_hat)

    return z_L, z_hat, z_U, p

print('Baseline errors:')
print(np.load(root + '/baseline-regr/E_gen_o.npy'))
print()

print('ANN errors:')
print(np.load(root + '/ANN-regr/E_gen_o.npy'))
print('ANN hiddens:')
print(np.load(root + '/ANN-regr/nums_of_hidden.npy'))
print()

print('Linear errors:')
print(np.load(root + '/linear-regr/E_gen_o.npy'))
print('Linear lambdas:')
print(np.load(root + '/linear-regr/optimal_lambdas.npy'))
print()

print(compare_models('ANN-regr', 'baseline-regr', 0.05))
print(compare_models('linear-regr', 'baseline-regr', 0.05))
print(compare_models('linear-regr', 'ANN-regr', 0.05))