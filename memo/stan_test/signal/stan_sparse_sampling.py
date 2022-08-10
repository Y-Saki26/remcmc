import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pystan
import arviz

stan_code = """
data {
    int<lower=0> N;
    vector[N] y;
    real a;
}
parameters {
    real<lower=0> sigma;
    vector[N] beta;
}
model {
    target += 1/sigma;
    for(i in 1:N){
        beta[i] ~ double_exponential(0, a);
        y[i] ~ normal(beta[i], sigma);
    }
}
"""

if __name__=="__main__":
    with open("data.bin", "rb") as f:
        y_true, y_obs = pickle.load(f)

    stan_data = {
        "N": len(y_obs),
        "y": y_obs,
        "a": 1
    }

    save_model_name = "sampling_model.bin"
    if os.path.exists(save_model_name):
        with open(save_model_name, "rb") as f:
            sm = pickle.load(f)
    else:
        sm = pystan.StanModel(model_code=stan_code)
        with open(save_model_name, "wb") as f:
            pickle.dump(sm, f)
    print(sm)

    save_sample_name = "samples_data.bin"
    if os.path.exists(save_sample_name):
        with open(save_sample_name, "rb") as f:
            fit = pickle.load(f)
    else:
        fit = sm.sampling(data=stan_data, iter=1000, chains=4)
        with open(save_sample_name, "wb") as f:
            pickle.dump(fit, f)
    print(fit)

    arviz.plot_trace(fit)
    plt.savefig("hist_trace.png")
    plt.cla()
    plt.clf()
    plt.close()
