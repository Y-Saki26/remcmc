import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pystan
import arviz

if __name__=="__main__":
    with open("data_linear-3.bin", "rb") as f:
        data = pickle.load(f)
    
    N,M,X,y_obs,y_true,beta_true = data.values()

    stan_data = {
        "N": N,
        "D": M,
        "X": X,
        "y": y_obs
    }

    save_model_name = "sampling_model_horseshoe.bin"
    if os.path.exists(save_model_name):
        with open(save_model_name, "rb") as f:
            sm = pickle.load(f)
    else:
        sm = pystan.StanModel(file="sparse_linear_model_horseshoe.stan")
        with open(save_model_name, "wb") as f:
            pickle.dump(sm, f)
    print(sm)

    save_sample_name = "samples_data-3_horseshoe.bin"
    if os.path.exists(save_sample_name):
        with open(save_sample_name, "rb") as f:
            fit = pickle.load(f)
    else:
        fit = sm.sampling(data=stan_data, iter=2000, chains=5)
        with open(save_sample_name, "wb") as f:
            pickle.dump(fit, f)
    print(fit)

    arviz.plot_trace(fit)
    plt.savefig("hist_trace-3_horseshoe.png")
    plt.cla()
    plt.clf()
    plt.close()
