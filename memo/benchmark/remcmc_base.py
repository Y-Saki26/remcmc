import pickle
import os
import datetime
from abc import abstractmethod
import numpy as np
from numba import njit, f8, i8

def shape(obj):
    r = ()
    if obj == []:
        return (0,)
    while hasattr(obj, "__iter__"):
        r = (*r, len(obj))
        obj = obj[0]
    return r

class ReplicaExchangeBase(object):
    """
    sampler by replica exchange
    """
    def __init__(
        self,
        param_name_j,
        beta_k,
        eps_j_k,
        exchange_step: int,
        param_init_j_k=None,
        data=None,
        static_params=None,
        random_state=None
    ):
        """init"""
        # initialize member variables
        self.param_name_j = param_name_j
        self.beta_k = beta_k
        self.eps_j_k = eps_j_k
        self.beta2eps = {beta: eps_j for beta, eps_j in zip(beta_k, eps_j_k)}
        self.exchange_step = exchange_step
        self.data = data
        self.static_params = static_params
        self.random_init = random_state
        if random_state is not None:
            np.random.seed(random_state)
        self.n_dim = len(self.param_name_j)
        self.n_temp = len(self.beta_k)
        if shape(param_init_j_k)!=(self.n_temp, self.n_dim):
            raise ValueError(
                f"param_init_j_k must be in shape (K, J) but this is {shape(param_init_j_k)}"
            )

        # get first sample
        if param_init_j_k is None:
            param_init_j_k = self.suggestion(
                [np.random.rand() for pn in self.param_name_j],
                self.eps_j_k[-1], 0)

        # initialize model
        # calc log likelihood at initial value
        log_cond_k = [
            self.log_condprob(param_init_j_k[k])
            for k,_beta in enumerate(self.beta_k)
        ]
        log_pri_k = [
            self.log_priorprob(param_init_j_k[k])
            for k,_beta in enumerate(self.beta_k)
        ]
        assert shape(log_cond_k)==(self.n_temp,), "log_cnd_k"
        assert shape(log_pri_k)==(self.n_temp,), "log_pri_k"
        if any(np.isinf(log_cond_k[k]) for k in range(self.n_temp)):
            raise ValueError(
                "conditional probability infinite value"
                f"{log_cond_k}"
                ", at init: "f"{param_init_j_k}"
            )
        for k,lp in enumerate(log_pri_k):
            if np.isinf(lp):
                raise ValueError(
                    "prior probability infinite value"
                    f"{k}-th: {lp}"
                    ", at init: "f"{param_init_j_k[k]}"
                )
        if any(np.isinf(log_pri_k[k]) for k in range(self.n_temp)):
            raise ValueError(
                "prior probability infinite value"
                f"{log_pri_k}"
                ", at init: "f"{param_init_j_k}"
            )
        # store initial values
        self.sample_j_n_k = [
            [param_init_j_k[k]]
            for k in range(self.n_temp)
        ]
        assert shape(self.sample_j_n_k)==(self.n_temp, 1, self.n_dim), "sample_j_n_k"
        self.lcp_sample_n_k = [
            [log_cond_k[k]]
            for k in range(self.n_temp)
        ]
        self.lpp_sample_n_k = [
            [log_pri_k[k]]
            for k in range(self.n_temp)
        ]
        self.ll_sample_n_k = [
            [log_cond_k[k] * beta + log_pri_k[k]]
            for k,beta in enumerate(self.beta_k)
        ]
        assert shape(self.lcp_sample_n_k)==(self.n_temp, 1), "cond_prob_sample_n_k"
        assert shape(self.lpp_sample_n_k)==(self.n_temp, 1), "pri_prob_sample_n_k"
        assert shape(self.ll_sample_n_k)==(self.n_temp, 1), "ll_sample_n_k"
        self.accept_j_n_k = [
            [[-1]*self.n_dim]
            for k,_ in enumerate(self.beta_k)
        ]
        assert shape(self.accept_j_n_k)==(self.n_temp, 1, self.n_dim)
        self.exchange_accept_n_k = [
            [-1] for k in range(self.n_temp - 1)
        ]
        assert shape(self.exchange_accept_n_k) in {(self.n_temp - 1, 1), (0,)}

        self.loop_count = 1
        self.exchange_count = 0

    # base algorithm
    def sampling(
        self, loopcount=1000,
        verbose=True, verbose_count=10
    ):
        """
        execute simulation

        Parameters
        ----------
        loopcount: int
            number of sample to get
        verbose: bool, default=True
            verbose output
        verbose_count: int, default=10
            number of verbose output
        """
        last_verbose = self.loop_count
        verbose_interval = loopcount // verbose_count
        while self.loop_count < loopcount:
            self.update_parallel(self.exchange_step - 1)
            self.update_exhange()
            if verbose and self.loop_count >= last_verbose + verbose_interval:
                self.verbose()
                last_verbose = self.loop_count
        if verbose and self.loop_count == last_verbose + verbose_interval:
            self.verbose()

    def update_parallel(self, batch):
        """
        update new sample with M-H step for each betas
        Calculate the column of `beta_k` X times in a row, and then calculate the next temperature.

        Parameters
        ----------
        batch: int
            number of continuus calculation
        """
        #print("update parallel")
        for k,beta in enumerate(self.beta_k):
            param_j_s, lcp_s, lpp_s, ll_s, accept_j_s = self.get_batch(
                self.sample_j_n_k[k][-1],
                self.lcp_sample_n_k[k][-1],
                self.lpp_sample_n_k[k][-1],
                self.ll_sample_n_k[k][-1],
                beta, batch
            )
            assert shape(param_j_s)==(batch, self.n_dim),\
                f"param_j_s {np.array(param_j_s).shape}"
            assert shape(lcp_s)==(batch,),\
                f"lcp_s {np.array(lcp_s).shape}"
            assert shape(lpp_s)==(batch,),\
                f"lpp_s {np.array(lpp_s).shape}"
            assert shape(ll_s)==(batch,),\
                f"ll_s {np.array(ll_s).shape}"
            self.sample_j_n_k[k] += param_j_s
            self.lcp_sample_n_k[k] += lcp_s
            self.lpp_sample_n_k[k] += lpp_s
            self.ll_sample_n_k[k] += ll_s
            self.accept_j_n_k[k] += accept_j_s
            if k<self.n_temp-1:
                self.exchange_accept_n_k[k] += [-1] * batch
        self.loop_count += batch
        assert shape(self.sample_j_n_k)==(self.n_temp, self.loop_count, self.n_dim),\
            f"sample_j_n_k {shape(self.sample_j_n_k)}, {np.array(self.sample_j_n_k)}, {self.loop_count}"
        assert shape(self.ll_sample_n_k)==(self.n_temp, self.loop_count),\
            f"ll_sample_n_k {shape(self.ll_sample_n_k)}, {np.array(self.ll_sample_n_k)}, {self.loop_count}"

    def get_batch(self, param_j_pre, lcp_pre, lpp_pre, ll_pre, beta, batch):
        """
        get new sample at beta_k[k] with M-H step for each betas

        Parameters
        ----------
        param_j_pre: list[(float|int)] (N_dim,)
            previous param sample
        ll_pre: float
            previous log likelihood
        beta: float (in self.beta_k)
            Target temperature for calculation
        batch: int
            number of continuus calculation

        Returns
        -------
        param_j_s: list[list[(float|int)]] (N_dim, batch)
        ll_s: list[float] (batch,)
        accept_j_s: list[list[bool]] (N_dim, batch)
        """
        param_j_s = []
        lcp_s = []
        lpp_s = []
        ll_s = []
        accept_j_s = []
        #loop_count = self.loop_count
        for _ in range(batch):
            param_j_pre, lcp_pre, lpp_pre, ll_pre, accept_j = self.get_next(
                param_j_pre, lcp_pre, lpp_pre, ll_pre, beta
            )
            param_j_s += [param_j_pre]
            lcp_s += [lcp_pre]
            lpp_s += [lpp_pre]
            ll_s += [ll_pre]
            accept_j_s += [accept_j]
        assert shape(param_j_s)==(batch, self.n_dim), param_j_s
        assert shape(lcp_s)==shape(lpp_s)==shape(ll_s)==(batch,), (lcp_s, lpp_s, ll_s)
        assert shape(accept_j_s)==(batch, self.n_dim), accept_j_s
        return param_j_s, lcp_s, lpp_s, ll_s, accept_j_s

    def get_next(self, param_j_pre, lcp_pre, lpp_pre, ll_pre, beta):
        """
        suggest and Metropolis test

        Parameters
        ----------
        param_j_pre: list[(float|int)] (N_dim,)
            previous param sample
        ll_pre: float
            previous log likelihood
        beta: float (in self.beta_k)
            Target temperature for calculation

        Returns
        -------
        param_j_new: list[(float|int)] (N_dim,)
            next param sample
        ll: float
            next ll
        accept_j: list[bool] (N_dim,)
            Whether the step has been accepted for each parameter.
        """
        assert beta in self.beta_k
        accept_j = [0] * len(self.param_name_j)
        for j,_ in enumerate(self.param_name_j):
            param_j_new = self.suggestion(param_j_pre, self.beta2eps[beta], j)
            lcp_new = self.log_condprob(param_j_new)
            lpp_new = self.log_priorprob(param_j_new)
            ll_new = lcp_new * beta + lpp_new
            if self.metropolis_test(ll_pre, ll_new):
                param_j_pre = param_j_new
                lcp_pre = lcp_new
                lpp_pre = lpp_new
                ll_pre = ll_new
                accept_j[j] = 1
        assert shape(param_j_pre)==(self.n_dim,), param_j_pre
        assert shape(accept_j)==(self.n_dim,), accept_j
        return param_j_pre, lcp_pre, lpp_pre, ll_pre, accept_j

    @staticmethod
    def metropolis_test(ll_pre, ll_new):
        """whether metropolis test accepted or rejected"""
        return ll_new >= ll_pre or np.random.rand() <= np.exp(ll_new - ll_pre)

    def update_exhange(self):
        """
        update new sample with exchange step
        """
        if self.n_temp==1:
            return
        #print("update exchange")
        param_new_j_k = [
            self.sample_j_n_k[k][-1]
            for k,_ in enumerate(self.beta_k)
        ]
        lcp_new_k = [
            self.lcp_sample_n_k[k][-1]
            for k,_ in enumerate(self.beta_k)
        ]
        lpp_new_k = [
            self.lpp_sample_n_k[k][-1]
            for k,_ in enumerate(self.beta_k)
        ]
        ll_new_k = [
            self.ll_sample_n_k[k][-1]
            for k,_ in enumerate(self.beta_k)
        ]
        ex_accept_k = [0] * (self.n_temp - 1)

        # 偶数/奇数番目を交互に交換
        for k2, beta2 in enumerate(self.beta_k):
            if not(0 < k2 < self.n_temp):
                continue
            k1 = k2 - 1
            if k1 % 2 != self.exchange_count % 2:
                continue
            beta1 = self.beta_k[k1]
            lcp_pre1 = self.lcp_sample_n_k[k1][-1]
            lcp_pre2 = self.lcp_sample_n_k[k2][-1]
            lcp_new1, lcp_new2 = lcp_pre2, lcp_pre1
            if self.metropolis_test(
                lcp_pre1 * beta1 + lcp_pre2 * beta2,
                lcp_new1 * beta1 + lcp_new2 * beta2
            ):
                # swap param, likelihoods
                param_new_j_k[k1], param_new_j_k[k2] = param_new_j_k[k2], param_new_j_k[k1]
                lcp_new_k[k1], lcp_new_k[k2] = lcp_new1, lcp_new2
                lpp_new_k[k1], lpp_new_k[k2] = lpp_new_k[k2], lpp_new_k[k1]
                ll_new_k[k1] = lcp_new1 * beta1 + lpp_new_k[k1]
                ll_new_k[k2] = lcp_new2 * beta2 + lpp_new_k[k2]
                assert 0<=k1==k2-1<self.n_temp-1, (k1,k2)
                assert shape(ex_accept_k)==(self.n_temp-1,), (ex_accept_k, shape(ex_accept_k), self.n_dim)
                assert ex_accept_k[k1] == 0
                ex_accept_k[k1] = 1
        assert shape(param_new_j_k)==(self.n_temp, self.n_dim), param_new_j_k
        assert shape(lcp_new_k)==shape(lpp_new_k)==shape(ll_new_k)==(self.n_temp,),lcp_new_k
        # store data
        for k,_ in enumerate(self.beta_k):
            self.sample_j_n_k[k] += [param_new_j_k[k]]
            self.lcp_sample_n_k[k] += [lcp_new_k[k]]
            self.lpp_sample_n_k[k] += [lpp_new_k[k]]
            self.ll_sample_n_k[k] += [ll_new_k[k]]
            self.accept_j_n_k[k] += [[-1] * self.n_dim]
            if k < self.n_temp - 1:
                self.exchange_accept_n_k[k] += [ex_accept_k[k]]
        self.loop_count += 1
        self.exchange_count += 1

    def suggestion(self, param_j_pre, eps_j, j_index):
        """
        suggest a new sample

        Parameters
        ----------
        param_j_pre: list[(float|int)] (N_dim,)
            previous param sample
        eps_j: list[(float|int)] (N_dim,)
            step width
        j_index: int
            Index of the parameter to be changed

        Returns
        -------
        param_j_new: list[(float|int)] (N_dim,)
            suggested param sample
        """
        param_j_new = self._suggestion(param_j_pre, eps_j, j_index)
        assert shape(param_j_new)==(self.n_dim,), (param_j_new, len(param_j_new[0]), len(self.sample_j_n_k[0]))
        return param_j_new

    @staticmethod #@njit
    @abstractmethod
    def _suggestion(param_j_pre, eps_j, j_index):
        """
        suggest a new sample

        Parameters
        ----------
        param_j_pre: list[(float|int)] (N_dim,)
            previous param sample
        eps_j: list[(float|int)] (N_dim,)
            step width
        j_index: int
            Index of the parameter to be changed

        Returns
        -------
        param_j_new: list[(float|int)] (N_dim,)
            suggested param sample
        """
        param_j_new = [param for param in param_j_pre]
        param_j_new[j_index] += np.random.normal(0, eps_j[j_index])
        return param_j_new

    def log_condprob(self, param_j) -> float:
        """
        calc log likelihood, passing params to staticmethod
        """
        return self._log_condprob(*[f8(p) for p in param_j])

    @staticmethod #@njit
    @abstractmethod
    def _log_condprob(*args) -> f8:
        """
        calc log likelihood by numba

        Parameters
        ----------
        args: any numba objects
            appropriate arguments

        Returns
        -------
        numba.f8
            log conditional probability
        """
        pass

    def log_priorprob(self, param_j) -> float:
        """
        calc log prior probability, passing params to staticmethod
        """
        return self._log_priorprob(*[f8(p) for p in param_j])

    @staticmethod #@njit
    @abstractmethod
    def _log_priorprob(*args) -> f8:
        """
        calc log prior probability by numba

        Parameters
        ----------
        args: any numba objects
            appropriate arguments

        Returns
        -------
        float
            log prior probability
        """
        pass
    
    def verbose(self):
        """verbose status"""
        print(
            self.loop_count,
            self.exchange_count,
            len(self.sample_j_n_k[0]))
        #print(
        #    "  ",
        #    ", ".join([
        #        f"{self.accept_j_n_k[k] models[beta]['accept_count']/(self.loop_count - self.exchange_count):.3f}"
        #        for beta in self.beta_k
        #    ]))
        #print("  ", ", ".join([f"{self.models[beta]['ex_accept_count']/self.exchange_count:.3f}"
        #                    for beta in self.beta_k]))
        for j,pn in enumerate(self.param_name_j):
            print(
                f"   {pn.ljust(8)}"\
                f"{', '.join(['%.3f' % (self.sample_j_n_k[k][-1][j]) for k,_ in enumerate(self.beta_k)])}")
        print(
            f"   {'ll'.ljust(8)}"\
            f"{', '.join(['%.3f' % (self.ll_sample_n_k[k][-1]) for k,_ in enumerate(self.beta_k)])}")
    
    def save(self, name:str, timestamp=False):
        """save data"""
        name = os.path.abspath(name)
        try:
            os.makedirs(os.path.dirname(name))
        except FileExistsError:
            pass
        while os.path.exists(name) or timestamp:
            timestamp = False
            root, ext = os.path.splitext(name)
            name = root\
                + "_" + datetime.datetime.now().strftime("%y%m%d-%H%M%S-%f")\
                + ext
        with open(name, mode='wb') as f:
            pickle.dump({
                "param_name_j": self.param_name_j,
                "beta_k": self.beta_k,
                "eps_j_k": self.eps_j_k,
                "exchange_step": self.exchange_step,
                "accept_j_n_k": self.accept_j_n_k,
                "exchange_accept_n_k": self.exchange_accept_n_k,
                "sample_j_n_k": self.sample_j_n_k,
                "lcp_sample_n_k": self.lcp_sample_n_k,
                "lpp_sample_n_k": self.lpp_sample_n_k,
                "ll_sample_n_k": self.ll_sample_n_k
            }, f)
