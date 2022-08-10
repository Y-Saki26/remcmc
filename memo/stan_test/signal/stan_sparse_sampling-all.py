import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pystan
import arviz

if __name__=="__main__":
    suffixs = [*[f"lasso-{r}" for r in [0.1, 0.5, 1, 2, 5]], "horseshoe", "jeffreys"]
    for suffix in suffixs:
        print(suffix)
        if "lasso" in suffix:
            save_model_name = f"sampling_model-lasso.bin"
        else:
            save_model_name = f"sampling_model-{suffix}.bin"
        save_sample_name = f"samples_data_big-{suffix}.bin"
        save_fig_name = f"hist_cul_trace_big-{suffix}.png"
        with open("data.bin", "rb") as f:
            y_true, y_obs = pickle.load(f)

        stan_data = {
            "N": len(y_obs),
            "y": y_obs
        }
        if "lasso" in suffix:
            stan_data["a"] = float(suffix.split("-")[-1])
        if os.path.exists(save_model_name):
            with open(save_model_name, "rb") as f:
                sm = pickle.load(f)
        else:
            assert False
        print(sm)

        if os.path.exists(save_sample_name):
            with open(save_sample_name, "rb") as f:
                fit = pickle.load(f)
        else:
            fit = sm.sampling(data=stan_data, iter=10000, chains=7)
            with open(save_sample_name, "wb") as f:
                pickle.dump(fit, f)
        print(fit)

        arviz.plot_dist(fit, cumulative=True)
        plt.savefig(save_fig_name)
        plt.cla()
        plt.clf()
        plt.close()
