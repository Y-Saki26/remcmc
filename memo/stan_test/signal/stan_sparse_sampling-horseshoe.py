import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pystan
import arviz

if False:
    # 全域用
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
            target += beta_lpdf(kappa[1] | 0.5, 0.5);
            beta[i] ~ normal(0, sigma*lambda[i]);
            y[i] ~ normal(beta[i], sigma);
        }
    }
    """

if False:
    # 範囲を絞った
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
    }
    parameters {
        real<lower=0, upper=10> sigma;
        vector<lower=0, upper=5>[N] lambda;
        vector<lower=-10, upper=10>[N] beta;
    }
    transformed parameters {
        vector<lower=0, upper=1>[N] kappa;
        for(i in 1:N)
            kappa[i] = 1 / (1 + lambda[i] ^ 2);
    }
    model {
        target += 1/sigma;
        for(i in 1:N){
            target += beta_lpdf(kappa[i] | 0.5, 0.5);
            beta[i] ~ normal(0, sigma*lambda[i]);
            y[i] ~ normal(beta[i], sigma);
        }
    }
    """

# 範囲を絞った
stan_code = """
data {
    int<lower=0> N;
    vector[N] y;
}
parameters {
    real<lower=0, upper=10> sigma;
    vector<lower=0, upper=1>[N] kappa;
    vector<lower=-10, upper=10>[N] beta;
}
transformed parameters {
    vector<lower=0, upper=5>[N] lambda;
    for(i in 1:N)
        lambda[i] = sqrt(1 / kappa[i] - 1);
}
model {
    // prior
    target += 1/sigma;
    for(i in 1:N){
        kappa[i] ~ beta(0.5, 0.5);
    }

    // likelihood
    for(i in 1:N){
        beta[i] ~ normal(0, sigma*lambda[i]);
        y[i] ~ normal(beta[i], sigma);
    }
}
"""

if __name__=="__main__":
    suffix = "-horseshoe"
    save_model_name = f"sampling_model{suffix}.bin"
    save_sample_name = f"samples_data{suffix}.bin"
    save_fig_name = f"hist_trace{suffix}.png"
    with open("data.bin", "rb") as f:
        y_true, y_obs = pickle.load(f)

    stan_data = {
        "N": len(y_obs),
        "y": y_obs
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
        fit = sm.sampling(data=stan_data, iter=2000, chains=4)
        with open(save_sample_name, "wb") as f:
            pickle.dump(fit, f)
    print(fit)

    arviz.plot_trace(fit)
    plt.savefig(save_fig_name)
    plt.cla()
    plt.clf()
    plt.close()
