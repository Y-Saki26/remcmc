import numpy as np
from numba import njit, f8, i8
import remcmc_base

@njit("f8(f8[:])")
def target_function(X: list[f8]) -> f8:
    r = 0
    for x in X:
        r += x ** 4 - 16 * x ** 2 + 5 * x
    return -r / 2

class BenchmarkSampling(remcmc_base.ReplicaExchangeBase):
    def __init__(
        self,
        dimention,
        beta_k,
        eps_j_k,
        exchange_step,
        pri_mid,
        pri_width,
        init,
        *params, **kwargs
    ):
        param_name_j = [f"x_{i}" for i in range(dimention)]
        static_params = {
            "pri_mid": f8(pri_mid),
            "pri_width": f8(pri_width)
        }
        param_init_j_k = [
            [init[k][pn] for _j,pn in enumerate(param_name_j)]
            for k,_ in enumerate(beta_k)
        ]

        super().__init__(
            param_name_j=param_name_j,
            beta_k=beta_k,
            eps_j_k=eps_j_k,
            exchange_step=exchange_step,
            param_init_j_k=param_init_j_k,
            data=dict(),
            static_params=static_params,
            *params, **kwargs
        )

    def log_priorprob(self, param_j) -> float:
        """prior probability"""
        param_j = f8(param_j)
        return self._log_priorprob(
            param_j,
            self.static_params["pri_mid"],
            self.static_params["pri_width"]
        )

    @staticmethod
    @njit("f8(f8[:], f8[:], f8[:])")
    def _log_priorprob(param_j, prior_mid, prior_width) -> float:
        """
        calc log prior probability
        """
        return sum(-(param_j - prior_mid)**2/2/prior_width**2)

    def log_condprob(self, param_j) -> float:
        """prior probability"""
        return self._log_condprob(f8(param_j))
        
    @staticmethod
    @njit("f8(f8[:])")
    def _log_condprob(param_j) -> float:
        """
        calc log likelihood by numba
        """
        return target_function(param_j)

if __name__=="__main__":
    np.random.seed(42)
    dim = 3
    print("simple MCMC")
    n_samples = 10**4
    for _ in range(6):
        beta_k = [1]
        init = [np.random.randn()*10 for _ in range(dim)]
        print(init)
        sp = BenchmarkSampling(
            dimention=dim,
            beta_k=beta_k,
            eps_j_k=[[1 for _ in range(dim)] for _k in beta_k],
            exchange_step=10**4,
            pri_mid=[0 for _ in range(dim)],
            pri_width=[10 for _ in range(dim)],
            init=[
                {f"x_{i}": x_i for i,x_i in enumerate(init)}
                for _k in beta_k]
        )
        sp.sampling(n_samples)
        sp.save("test_mhmcmc.bin", timestamp=True)
    
    print("Replica exchange MCMC")
    beta_k = np.logspace(-5, 5, 21)
    init = [2.5 for _ in range(dim)]
    sp = BenchmarkSampling(
        dimention=dim,
        beta_k=beta_k,
        eps_j_k=[
            [eps for _ in range(dim)]
            for eps in np.logspace(np.log10(10), np.log10(0.1), beta_k.size)],
        exchange_step=5,
        pri_mid=[0 for _ in range(dim)],
        pri_width=[10 for _ in range(dim)],
        init=[
            {f"x_{i}": x_i for i,x_i in enumerate(init)}
            for _k in beta_k]
    )
    sp.sampling(n_samples)
    sp.save("test_remcmc.bin", timestamp=True)
