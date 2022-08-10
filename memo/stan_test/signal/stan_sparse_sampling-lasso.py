import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pystan
import arviz

stan_code = """
functions {
    real kappa_prior_lpdf(real x, real a){
        real alpha = a^2 / 2;
        return log(alpha / x^2)- alpha * (1-x) / x ;
    }
}
data {
    int<lower=0> N;
    vector[N] y;
    real a;
}
parameters {
    real<lower=0> sigma;
    vector<lower=0>[N] lambda;
    vector[N] beta;
}
transformed parameters {
    vector<lower=0, upper=1>[N] kappa;
    for(i in 1:N)
        kappa[i] = 1 / (1 + lambda[i] ^ 2);
}
model {
    target += 1/sigma;
    for(i in 1:N){
        target += kappa_prior_lpdf(kappa[i] | a);
        beta[i] ~ normal(0, sigma*lambda[i]);
        y[i] ~ normal(beta[i], sigma);
    }
}
"""

if __name__=="__main__":
    suffix = "-lasso"
    save_model_name = f"sampling_model{suffix}.bin"
    save_sample_name = f"samples_data{suffix}-2.bin"
    save_fig_name = f"hist_trace{suffix}-2.png"
    with open("data.bin", "rb") as f:
        y_true, y_obs = pickle.load(f)

    stan_data = {
        "N": len(y_obs),
        "y": y_obs,
        "a": 2
    }

    if os.path.exists(save_model_name):
        with open(save_model_name, "rb") as f:
            sm = pickle.load(f)
    else:
        sm = pystan.StanModel(model_code=stan_code)
        with open(save_model_name, "wb") as f:
            pickle.dump(sm, f)
    print(sm)

    if os.path.exists(save_sample_name):
        with open(save_sample_name, "rb") as f:
            fit = pickle.load(f)
    else:
        fit = sm.sampling(data=stan_data, iter=1000, chains=4)
        with open(save_sample_name, "wb") as f:
            pickle.dump(fit, f)
    print(fit)

    arviz.plot_trace(fit)
    plt.savefig(save_fig_name)
    plt.cla()
    plt.clf()
    plt.close()
