# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import os
import pickle
import scipy.io

def main():
    # input_mat_path = "./N64_M256_SIDCO.mat"
    # output_pickle_path = "./N64_M256_SIDCO.pickle"
    #
    input_mat_path = "./N160_M256_SIDCO.mat"
    output_pickle_path = "./N160_M256_SIDCO.pickle"

    # Load
    Dictionary_mat = scipy.io.loadmat(input_mat_path)
    X = Dictionary_mat["A_QCS"]

    # Save
    with open(output_pickle_path, "wb") as f:
        pickle.dump(X, f)

main()