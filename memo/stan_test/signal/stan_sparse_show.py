import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pystan
import arviz

if __name__=="__main__":
    suffix = "_big-jeffreys"
    if "lasso" in suffix: 
        save_model_name = f"sampling_model-lasso.bin"
    elif "horseshoe" in suffix: 
        save_model_name = f"sampling_model-horseshoe.bin"
    elif "jeffreys" in suffix: 
        save_model_name = f"sampling_model-jeffreys.bin"
    save_sample_name = f"samples_data{suffix}.bin"
    save_fig_name = f"results{suffix}.png"

    with open("data.bin", "rb") as f:
        y_true, y_obs = pickle.load(f)
    with open(save_model_name, "rb") as f:
        sm = pickle.load(f)
    with open(save_sample_name, "rb") as f:
        fit = pickle.load(f)

    #sm = pystan.StanModel(model_code=stan_code)
    #print(sm)
    if False:
        arviz.plot_trace(fit)
        plt.savefig("hist_trace.png")

    betas = np.array([
        fit[f"beta[{i+1}]"] for i in range(20)
    ])
    
    plt.plot(
        y_true, "o", markersize=3, color="k", label="true"
    )
    plt.plot(
        y_obs, "o", markersize=3, color="r", label="observed"
    )
    lower,middle,upper = np.quantile(betas, [0.25,0.5,0.75], axis=-1)
    plt.errorbar(
        range(20), middle, yerr=(middle - lower, upper - middle), capsize=5, fmt="x", label="predicted"
    )
    plt.hlines([0], *plt.xlim(), color="k")
    noise = np.mean(fit["sigma"])
    plt.hlines([-noise, noise], *plt.xlim(), linestyle="--", color="k", linewidth=1, label="noise level")
    plt.legend()
    plt.savefig(save_fig_name)

    #fit = sm.sampling(data=stan_data, iter=1000, chains=4)
    #print(fit)
    #with open("data.bin", "wb") as f:
    #    pickle.dump(fit, f)

