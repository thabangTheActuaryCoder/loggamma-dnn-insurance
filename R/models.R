# models.R -- gamma and lognormal GLMs, with log-likelihood, mean prediction
# and PIT all evaluated on the Y (severity) scale so that AIC/BIC and
# diagnostics are directly comparable with the loggamma GLM.

# ---- Gamma GLM (log link) -------------------------------------------------
# Y_i ~ Gamma(mean = mu_i, shape = nu), mu_i = exp(eta_i). The MLE of the
# shape nu differs from the moment/deviance dispersion glm() reports, so we
# profile it out by maximum likelihood for a like-for-like comparison.
fit_gamma_glm <- function(form, train) {
  g <- glm(form, data = train, family = Gamma(link = "log"))
  mu <- fitted(g)
  y  <- model.response(model.frame(form, train))
  # MLE of shape nu given mu (Gamma: var = mu^2 / nu).
  nll_nu <- function(lognu) {
    nu <- exp(lognu)
    -sum(dgamma(y, shape = nu, rate = nu / mu, log = TRUE))
  }
  nu <- exp(optimize(nll_nu, c(-5, 10))$minimum)
  loglik <- sum(dgamma(y, shape = nu, rate = nu / mu, log = TRUE))
  npar <- length(coef(g)) + 1
  n <- length(y)
  list(glm = g, nu = nu, loglik = loglik, npar = npar, n = n,
       aic = -2 * loglik + 2 * npar,
       bic = -2 * loglik + log(n) * npar)
}

predict_gamma_mean <- function(fit, newdata) {
  predict(fit$glm, newdata = newdata, type = "response")
}
pit_gamma <- function(fit, newdata, ynew) {
  mu <- predict(fit$glm, newdata = newdata, type = "response")
  pgamma(ynew, shape = fit$nu, rate = fit$nu / mu)
}

# ---- Lognormal GLM --------------------------------------------------------
# log Y_i ~ Normal(eta_i, sigma^2); density of Y carries the 1/y Jacobian, so
# loglik_Y = sum[ dnorm(log y; eta, sigma, log) - log y ].
fit_lognormal_glm <- function(form, train) {
  logform <- update(form, log(.) ~ .)
  m <- lm(logform, data = train)
  y <- model.response(model.frame(form, train))
  eta <- fitted(m)
  sigma <- sqrt(mean(residuals(m)^2))          # MLE of sigma
  loglik <- sum(dnorm(log(y), eta, sigma, log = TRUE) - log(y))
  npar <- length(coef(m)) + 1
  n <- length(y)
  list(lm = m, sigma = sigma, loglik = loglik, npar = npar, n = n,
       aic = -2 * loglik + 2 * npar,
       bic = -2 * loglik + log(n) * npar)
}

predict_lognormal_mean <- function(fit, newdata) {
  eta <- predict(fit$lm, newdata = newdata)
  exp(eta + fit$sigma^2 / 2)                    # E[Y] = exp(eta + sigma^2/2)
}
pit_lognormal <- function(fit, newdata, ynew) {
  eta <- predict(fit$lm, newdata = newdata)
  pnorm((log(ynew) - eta) / fit$sigma)
}

# ---- Shared accuracy metrics ---------------------------------------------
# RMSE / MAE of predicted mean severity, overall and restricted to the upper
# tail of ACTUAL severity (top (1 - q) fraction).
accuracy <- function(y, yhat) {
  ok <- is.finite(yhat)
  c(rmse = sqrt(mean((y[ok] - yhat[ok])^2)),
    mae  = mean(abs(y[ok] - yhat[ok])))
}
tail_accuracy <- function(y, yhat, q) {
  thr <- quantile(y, q)
  idx <- y >= thr
  accuracy(y[idx], yhat[idx])
}
