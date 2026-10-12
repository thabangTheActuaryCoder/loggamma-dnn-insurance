# loggamma.R -- the loggamma GLM, fitted by custom maximum likelihood.
#
# Definition (Mamba & Chikobvu 2025, Atmosphere 16:1273). A positive random
# variable Y follows a loggamma law iff W = log(Y) follows a gamma law:
#
#     W = log Y ~ Gamma(shape = alpha, rate = beta),   so   Y = exp(W),  y > 1.
#
# The density of Y carries the Jacobian 1/y of the transformation w = log y:
#
#     f_Y(y) = f_W(log y) / y
#            = beta^alpha / Gamma(alpha) * (log y)^{alpha-1} * y^{-(beta+1)},  y > 1.
#
# This is why the loggamma GLM is NOT a base-R glm() on log(y): to obtain an
# AIC/BIC on the Y scale that is directly comparable with the gamma and
# lognormal GLMs, the likelihood must be evaluated as the density of Y, with
# the Jacobian included. We therefore fit it by direct maximum likelihood.
#
# Regression structure (log link on the mean of W, matching the gamma GLM):
#     E[W_i] = alpha / beta_i = exp(eta_i),   eta_i = x_i' b,
# so beta_i = alpha / exp(eta_i), with a single shape alpha shared across i.

# Log-density of Y at y given shape alpha and rate beta (vectorised).
dloggamma_log <- function(y, alpha, beta) {
  w <- log(y)
  # gamma log-density of w, minus log(y) Jacobian
  alpha * log(beta) - lgamma(alpha) + (alpha - 1) * log(w) - beta * w - log(y)
}

# CDF of Y (probability integral transform input): P(Y <= y) = P(W <= log y).
ploggamma <- function(y, alpha, beta) {
  pgamma(log(y), shape = alpha, rate = beta)
}

# Mean of Y = E[exp(W)] = gamma MGF at t = 1 = (beta / (beta - 1))^alpha,
# finite only when beta > 1. Returns NA where beta <= 1.
mean_loggamma <- function(alpha, beta) {
  out <- rep(NA_real_, length(beta))
  ok <- beta > 1
  out[ok] <- (beta[ok] / (beta[ok] - 1))^alpha
  out
}

# Fit the loggamma GLM by maximum likelihood.
#   X      : model matrix (with intercept column)
#   y      : response (claim severity, > 1)
# Returns coefficients b, shape alpha, maximised log-likelihood, npar, AIC/BIC.
fit_loggamma_glm <- function(X, y, start_b = NULL) {
  stopifnot(all(y > 1))
  p <- ncol(X)
  logy <- log(y)

  # Sensible starts: gamma GLM of log(y) on X gives b; shape from CV of log(y).
  if (is.null(start_b)) {
    g0 <- suppressWarnings(
      glm.fit(X, logy, family = Gamma(link = "log")))
    start_b <- g0$coefficients
  }
  start_logalpha <- log(max(1, 1 / (sd(logy) / mean(logy))^2))

  negll <- function(par) {
    b <- par[1:p]
    alpha <- exp(par[p + 1])
    eta <- as.vector(X %*% b)
    muW <- exp(eta)                 # E[W] = alpha / beta
    beta <- alpha / muW             # rate_i
    ll <- dloggamma_log(y, alpha, beta)
    if (any(!is.finite(ll))) return(1e10)
    -sum(ll)
  }

  opt <- optim(c(start_b, start_logalpha), negll,
               method = "BFGS",
               control = list(maxit = 2000, reltol = 1e-10))

  b <- opt$par[1:p]
  alpha <- exp(opt$par[p + 1])
  eta <- as.vector(X %*% b)
  beta <- alpha / exp(eta)
  loglik <- -opt$value
  npar <- p + 1                     # betas + shape
  n <- length(y)

  list(coef = setNames(b, colnames(X)), alpha = alpha,
       loglik = loglik, npar = npar, n = n,
       aic = -2 * loglik + 2 * npar,
       bic = -2 * loglik + log(n) * npar,
       convergence = opt$convergence,
       eta = eta, beta = beta)
}

# Predict mean severity E[Y|x] for new design matrix Xnew.
predict_loggamma_mean <- function(fit, Xnew) {
  eta <- as.vector(Xnew %*% fit$coef)
  beta <- fit$alpha / exp(eta)
  mean_loggamma(fit$alpha, beta)
}

# PIT values u_i = F(y_i | x_i) on a (possibly new) sample.
pit_loggamma <- function(fit, Xnew, ynew) {
  eta <- as.vector(Xnew %*% fit$coef)
  beta <- fit$alpha / exp(eta)
  ploggamma(ynew, fit$alpha, beta)
}
