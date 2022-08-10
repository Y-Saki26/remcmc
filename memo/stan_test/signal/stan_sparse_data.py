import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
#import pystan

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
    np.random.seed(10)
    N = 20
    hit = 10
    sigma = 1.2
    y_true = np.random.randn(N)*5
    y_true[np.random.choice(N, N-hit, replace=False)]=0
    y_obs = y_true + np.random.randn(N) * sigma

    plt.plot(y_true, "o", label="True")
    plt.plot(y_obs, "o", label="Observed")
    plt.legend()
    plt.show()

    with open("data.bin", "wb") as f:
        pickle.dump((y_true.tolist(), y_obs.tolist()), f)

    stan_data = {
        "N": y_obs.size,
        "y": y_obs,
        "a": 1
    }

    #sm = pystan.StanModel(model_code=stan_code)
    #print(sm)

    #fit = sm.sampling(data=stan_data, iter=1000, chains=4)
    #print(fit)
    #with open("data.bin", "wb") as f:
    #    pickle.dump(fit, f)

