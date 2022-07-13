import numpy as np
#from numba import njit

def prior_prob_unif(params):
    return 0

def gauss_noise(func, params, X, Y, sigma):
    return -((Y-func(X, *params))**2).sum() / 2 / sigma**2 - np.log(2 * np.pi * sigma**2) * X.size

class MH(object):
    def __init__(self, func, X, Y, init, eps, prior_prob=prior_prob_unif):
        self.func = func
        self.X = X
        self.Y = Y
        self.eps = np.array(eps)
        self.prior_prob = prior_prob
        
        self.pre_params = init
        self.samples = [self.pre_params]
        self.pre_ll = self.log_prob(init)
        self.ll_samples = [self.pre_ll]
        
    def log_prob(self, params):
        p = self.prior_prob(params)
        if p==0: return np.NINF
        return self.log_likelihood(params) + np.log(p)
        
    def log_likelihood(self, params):
        return gauss_noise(self.func, params[:-1], self.X, self.Y, params[-1])
        #return -((Y - self.func(self.X, *params[:-1]))**2).sum() / 2 / params[-1]**2 - np.log(2*np.pi*params[-1]**2) * X.size 
    
    def sampling(self, sample_size):
        for _ in range(sample_size):
            new_params = [p+e for p,e in zip(self.pre_params, self.eps*np.random.randn(self.eps.size))]
            new_ll = self.log_prob(new_params)
            if new_ll>self.pre_ll or np.random.rand() < np.exp(new_ll - self.pre_ll):
                self.pre_params = new_params
                self.pre_ll = new_ll
            self.samples += [self.pre_params]
            self.ll_samples += [self.pre_ll]

class ReplicaExchange(object):
    def __init__(self, func, X, Y, init, eps, betas, exchange_step=10, prior_prob=prior_prob_unif):
        self.func = func
        self.X = X
        self.Y = Y
        self.eps = np.array(eps)
        self.betas = np.array(betas)
        self.exchange_step = exchange_step
        self.prior_prob = prior_prob
        
        self.pre_params = {beta: init for beta in self.betas}
        self.samples = {beta: [self.pre_params[beta]] for beta in self.betas}
        self.pre_ll = {beta: self.log_prob(self.pre_params[beta], beta) for beta in self.betas}
        self.ll_samples = {beta: [self.pre_ll[beta]] for beta in self.betas}
    
    def log_prob(self, params, beta):
        p = self.prior_prob(params)
        if p==0: return np.NINF
        return self.log_likelihood(params) * beta + np.log(p)
        
    def log_likelihood(self, params):
        return gauss_noise(self.func, params[:-1], self.X, self.Y, params[-1])
    
    def sampling(self, sample_size):
        for i in range(1,sample_size):
            if i%self.exchange_step:
                self.mh_sampling()
            elif i%(self.exchange_step*2):
                self.exchange_sampling_odd()
            else:
                self.exchange_sampling_even()
    
    def mh_sampling(self):
        for beta in self.betas:
            new_params_beta = [p+e for p,e in zip(self.pre_params[beta], self.eps*np.random.randn(self.eps.size))]
            new_ll_beta = self.log_prob(new_params_beta, beta)
            if new_ll_beta>self.pre_ll[beta] or np.random.rand() < np.exp(new_ll_beta - self.pre_ll[beta]):
                self.pre_params[beta] = new_params_beta
                self.pre_ll[beta] = new_ll_beta
            self.samples[beta] += [self.pre_params[beta]]
            self.ll_samples[beta] += [self.pre_ll[beta]]
    
    def exchange_sampling_even(self):
        for k, beta1 in enumerate(self.betas):
            if not(k%2==0 and k+1<self.betas.size): continue
            beta2 = self.betas[k+1]
            new_params_beta1 = [p for p in self.pre_params[beta2]]
            new_params_beta2 = [p for p in self.pre_params[beta1]]
            new_ll_beta1 = self.log_prob(new_params_beta1, beta1)
            new_ll_beta2 = self.log_prob(new_params_beta2, beta2)
            if (new_ll_beta1 + new_ll_beta2 > self.pre_ll[beta1] + self.pre_ll[beta2] 
                    or np.random.rand() < np.exp(new_ll_beta1 + new_ll_beta2 - self.pre_ll[beta1] - self.pre_ll[beta2])):
                self.pre_params[beta1] = new_params_beta1
                self.pre_params[beta2] = new_params_beta2
                self.pre_ll[beta1] = new_ll_beta1
                self.pre_ll[beta2] = new_ll_beta2
            self.samples[beta1] += [self.pre_params[beta1]]
            self.samples[beta2] += [self.pre_params[beta2]]
            self.ll_samples[beta1] += [self.pre_ll[beta1]]
            self.ll_samples[beta2] += [self.pre_ll[beta2]]
        if self.betas.size%2==1:
            beta = self.betas[-1]
            self.samples[beta] += [self.pre_params[beta]]
            self.ll_samples[beta] += [self.pre_ll[beta]]
    
    def exchange_sampling_odd(self):
        for k, beta1 in enumerate(self.betas):
            if not(k%2==1 and k+1<self.betas.size): continue
            beta2 = self.betas[k+1]
            new_params_beta1 = [p for p in self.pre_params[beta2]]
            new_params_beta2 = [p for p in self.pre_params[beta1]]
            new_ll_beta1 = self.log_prob(new_params_beta1, beta1)
            new_ll_beta2 = self.log_prob(new_params_beta2, beta2)
            if (new_ll_beta1 + new_ll_beta2 > self.pre_ll[beta1] + self.pre_ll[beta2] 
                    or np.random.rand() < np.exp(new_ll_beta1 + new_ll_beta2 - self.pre_ll[beta1] - self.pre_ll[beta2])):
                self.pre_params[beta1] = new_params_beta1
                self.pre_params[beta2] = new_params_beta2
                self.pre_ll[beta1] = new_ll_beta1
                self.pre_ll[beta2] = new_ll_beta2
            self.samples[beta1] += [self.pre_params[beta1]]
            self.samples[beta2] += [self.pre_params[beta2]]
            self.ll_samples[beta1] += [self.pre_ll[beta1]]
            self.ll_samples[beta2] += [self.pre_ll[beta2]]
            
        beta = self.betas[0]
        self.samples[beta] += [self.pre_params[beta]]
        self.ll_samples[beta] += [self.pre_ll[beta]]
        if self.betas.size%2==0:
            beta = self.betas[-1]
            self.samples[beta] += [self.pre_params[beta]]
            self.ll_samples[beta] += [self.pre_ll[beta]]
        


