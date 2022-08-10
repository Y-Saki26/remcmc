import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
#import pystan

if __name__=="__main__":
    np.random.seed(10)
    N = 30 # sample size
    M = 10 # parameter dim
    hit = 5
    sigma = 1.0
    X = np.random.rand(N,M)*8-4 # -4 ~ 4 の一様
    beta_true = np.array((np.random.rand(hit) * 10 - 5).tolist() + np.zeros(M-hit).tolist())
    N = 15 # sample size
    X = np.random.rand(N,M)*8-4 # -4 ~ 4 の一様
    y_true = X @ beta_true
    y_obs = y_true + np.random.randn(N) * sigma

    plt.plot(beta_true, "o", label="True")
    plt.hlines([0], *plt.xlim())
    plt.legend()
    plt.savefig("data-3.png")
    #plt.show()

    with open("data_linear-3.bin", "wb") as f:
        pickle.dump({
            "N": N, 
            "M": M,
            "X": X,
            "y_obs": y_obs,
            "y_true": y_true,
            "beta_true": beta_true
        }, f)

    #sm = pystan.StanModel(model_code=stan_code)
    #print(sm)

    #fit = sm.sampling(data=stan_data, iter=1000, chains=4)
    #print(fit)
    #with open("data.bin", "wb") as f:
    #    pickle.dump(fit, f)
