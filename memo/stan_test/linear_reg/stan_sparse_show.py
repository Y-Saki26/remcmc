import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pystan
import arviz

if __name__=="__main__":
    suffix = "horseshoe"
    #for suffix in ["jeffreys", "horseshoe", "lasso_ver", *[f"middle-{b}" for b in [0.1, 0.2, 0.25, 0.5, 0.75, 7/8, 1]],
    #        *[f"lasso_part-{b}" for b in [0.2, 0.5, 2, 5, 200, 500, 1000,0.1, 1, 10, 20, 50, 100]]]:
    for suffix in ["jeffreys", "horseshoe", ]:#"lasso_ver", *[f"middle-{b}" for b in [0.1, 0.2, 0.25, 0.5, 0.75, 7/8, 1]],
            #*[f"lasso_part-{b}" for b in [0.2, 0.5, 2, 5, 200, 500, 1000,0.1, 1, 10, 20, 50, 100]]]:
        if "lasso_part" in suffix: 
            save_model_name = f"sampling_model_lasso_part.bin"
        elif "lasso" in suffix: 
            save_model_name = f"sampling_model_lasso_ver.bin"
        elif "horseshoe" in suffix: 
            save_model_name = f"sampling_model_horseshoe.bin"
        elif "jeffreys" in suffix: 
            save_model_name = f"sampling_model_jeffreys.bin"
        elif "middle" in suffix: 
            save_model_name = f"sampling_model_middle.bin"
        
        save_sample_name = f"samples_data-3_{suffix}.bin"
        save_fig_name = f"results-3_{suffix}.png"

        with open("data_linear-3.bin", "rb") as f:
            data = pickle.load(f)
        N,D,X,y_obs,y_true,beta_true = data.values()
        with open(save_model_name, "rb") as f:
            sm = pickle.load(f)
        with open(save_sample_name, "rb") as f:
            fit = pickle.load(f)

        if False:
            arviz.plot_trace(fit)
            plt.savefig("hist_trace.png")

        betas = np.array([
            fit[f"beta[{i+1}]"] for i in range(D)
        ])
        
        plt.plot(
            beta_true, "o", markersize=3, color="k", label="true"
        )
        if False:
            plt.plot(
                y_obs, "o", markersize=3, color="r", label="observed"
            )
        lower,middle,upper = np.quantile(betas, [0.25,0.5,0.75], axis=-1)
        plt.errorbar(
            range(D), middle, yerr=(middle - lower, upper - middle), capsize=5, color="C0", fmt="x", label="predicted 25 - 75 %"
        )
        #lower,middle,upper = np.quantile(betas, [0.01,0.5,0.99], axis=-1)
        #plt.errorbar(
        #    range(D), middle, yerr=(middle - lower, upper - middle), capsize=5, color="C0", fmt=".", alpha=0.5, label="1 - 99 %"
        #)
        plt.hlines([0], *plt.xlim(), color="k")
        noise = np.mean(fit["sigma"])
        plt.hlines([-noise, noise], *plt.xlim(), linestyle="--", color="k", linewidth=1, label="noise level")
        plt.legend()
        plt.savefig(save_fig_name)
        plt.cla()
        plt.clf()
        plt.close()

        #fit = sm.sampling(data=stan_data, iter=1000, chains=4)
        #print(fit)
        #with open("data.bin", "wb") as f:
        #    pickle.dump(fit, f)

