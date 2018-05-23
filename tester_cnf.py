import numpy as np
from experiments import get_cnf_mat

Y_test = np.load('Y_test.npy')
class_pred = np.load('class_pred.npy')

cnf_matrix = get_cnf_mat(Y_test,class_pred)