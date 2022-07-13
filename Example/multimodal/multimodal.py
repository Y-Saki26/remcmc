import sys
import numpy as np

sys.path.append('../../lib/')
from kernel import VectorSampling
from utils import target_function

np.random.seed(42)
dim = 3
print("simple MCMC")
n_samples = 10**4
for i in range(8):
    beta_k = [1]
    init = [(-1)**(i//4),(-1)**((i//4)//2),(-1)**(i%2)]
    print(init)
    sp = VectorSampling(
        dimention=dim,
        log_likelifood_function=target_function,
        beta_k=beta_k,
        eps_j_k=[[1 for _ in range(dim)] for _k in beta_k],
        exchange_step=10**4,
        prior_center=[0 for _ in range(dim)],
        prior_width=[10 for _ in range(dim)],
        init=[
            {f"x_{i}": x_i for i,x_i in enumerate(init)}
            for _k in beta_k]
    )
    sp.sampling(n_samples)
    sp.save("test_mhmcmc.bin", timestamp=True)

print("Replica exchange MCMC")
beta_k = np.logspace(-5, 5, 21)
init = [2.5 for _ in range(dim)]
sp = VectorSampling(
    dimention=dim,
    log_likelifood_function=target_function,
    beta_k=beta_k,
    eps_j_k=[
        [eps for _ in range(dim)]
        for eps in np.logspace(np.log10(10), np.log10(0.1), beta_k.size)],
    exchange_step=5,
    prior_center=[0 for _ in range(dim)],
    prior_width=[10 for _ in range(dim)],
    init=[
        {f"x_{i}": x_i for i,x_i in enumerate(init)}
        for _k in beta_k]
)
sp.sampling(n_samples)
sp.save("test_remcmc.bin", timestamp=True)
