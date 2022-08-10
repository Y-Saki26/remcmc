import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pystan
import arviz

if __name__=="__main__":
    for b in [0.1, 0.2, 0.25, 0.5, 0.75, 7/8, 1]:
        with open("data_linear-3.bin", "rb") as f:
            data = pickle.load(f)
        
        N,M,X,y_obs,y_true,beta_true = data.values()

        stan_data = {
            "N": N,
            "D": M,
            "X": X,
            "y": y_obs,
            "b": b
        }

        save_model_name = "sampling_model_middle.bin"
        if os.path.exists(save_model_name):
            with open(save_model_name, "rb") as f:
                sm = pickle.load(f)
        else:
            sm = pystan.StanModel(file="sparse_linear_model_middle.stan")
            with open(save_model_name, "wb") as f:
                pickle.dump(sm, f)
        print(sm)

        save_sample_name = f"samples_data-3_middle-{b}.bin"
        if os.path.exists(save_sample_name):
            with open(save_sample_name, "rb") as f:
                fit = pickle.load(f)
        else:
            fit = sm.sampling(data=stan_data, iter=2000, chains=5)
            with open(save_sample_name, "wb") as f:
                pickle.dump(fit, f)
        print(fit)

        arviz.plot_trace(fit)
        plt.savefig(f"hist_trace-3_middle-{b}.png")
        plt.cla()
        plt.clf()
        plt.close()
