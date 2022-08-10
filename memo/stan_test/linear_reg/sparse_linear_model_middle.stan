functions {
    real jeffreys_prior_lpdf(real x){
        return log(1 / x / (1-x));
    }
    real lasso_prior_lpdf(real x, real a){
        real alpha = a^2 / 2;
        return log(alpha / x^2)- alpha * (1-x) / x ;
    }
    vector zero_vector(int n){
        return rep_vector(0, n);
    }
    vector one_vector(int n){
        return rep_vector(1, n);
    }
    matrix id_matrix(int n){
        return diag_matrix(one_vector(n));
    }
}
data {
    int<lower=0> N;
    int<lower=0> D;
    matrix[N, D] X;
    vector[N] y;
    // b=0.5: horseshoe, b->0: jeffreys
    real<lower=0> b;
}
parameters {
    real<lower=0, upper=10> sigma;
    vector<lower=-10, upper=10>[D] beta;
    vector<lower=0, upper=1>[D] kappa;
}
transformed parameters {
    vector<lower=0>[D] lambda;
    for(i in 1:D)
        lambda[i] = sqrt(1 / kappa[i] - 1);
}
model {
    // prior
    target += 1/sigma;
    for(i in 1:D){
        kappa[i] ~ beta(b, b);
    }

    // likelihood
    beta ~ multi_normal(zero_vector(D), sigma*diag_matrix(lambda));
    y ~ multi_normal(X * beta, sigma*id_matrix(N));
}
