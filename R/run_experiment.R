# run_experiment.R: the main experiment. For each dataset (thesis light-tailed,
# severity heavy-tailed) we fit the four response laws both as linear GLMs and
# as deep distributional heads, and compare them by the held-out predictive
# negative log-likelihood (NLL, a proper score), the robust median MAE, and the
# Kolmogorov and Smirnov statistic on the held-out PIT. The head with the
# smallest held-out NLL is the one the data select.

here <- function(...) file.path("R", ...)
source(here("data.R")); source(here("models.R")); source(here("loggamma.R"))
source(here("deepdist.R"))
dir.create("outputs/tables", showWarnings = FALSE, recursive = TRUE)

ksD <- function(u) suppressWarnings(ks.test(u, "punif")$statistic)

# Held-out mean NLL of a fitted GLM on the cost scale.
glm_heldout <- function(fam, fit, test, X_te, y) {
  switch(fam,
    normal = { mu <- predict_normal_mean(fit, test)
      list(nll = mean(-dnorm(y, mu, fit$sigma, log = TRUE)),
           mae = mean(abs(y - mu)), pit = pit_normal(fit, test, y)) },
    gamma = { mu <- predict_gamma_mean(fit, test)
      list(nll = mean(-dgamma(y, shape = fit$nu, rate = fit$nu/mu, log = TRUE)),
           mae = mean(abs(y - qgamma(0.5, fit$nu, rate = fit$nu/mu))),
           pit = pit_gamma(fit, test, y)) },
    lognormal = { eta <- predict(fit$lm, newdata = test)
      list(nll = mean(-(dnorm(log(y), eta, fit$sigma, log = TRUE) - log(y))),
           mae = mean(abs(y - exp(eta))), pit = pit_lognormal(fit, test, y)) },
    loggamma = { eta <- as.vector(X_te %*% fit$coef); b <- fit$alpha/exp(eta)
      list(nll = mean(-dloggamma_log(y, fit$alpha, b)),
           mae = mean(abs(y - exp(qgamma(0.5, fit$alpha, rate = b)))),
           pit = pit_loggamma(fit, X_te, y)) })
}

run_dataset <- function(which, deep_epochs) {
  dat <- prepare_data(which, seed = 1); train <- dat$train; test <- dat$test
  form <- dat$form; y_tr <- dat$y_train; y_te <- dat$y_test
  mf <- model.frame(form, train); X_tr <- model.matrix(form, mf)
  X_te <- model.matrix(form, model.frame(form, test, xlev = lapply(mf, levels)))
  Xtr <- X_tr[, -1, drop = FALSE]; Xte <- X_te[, -1, drop = FALSE]
  ct <- colMeans(Xtr); sdv <- apply(Xtr, 2, sd); sdv[sdv == 0] <- 1
  Ztr <- scale(Xtr, ct, sdv); Zte <- scale(Xte, ct, sdv)

  # Linear GLM heads.
  glms <- list(normal = fit_normal_glm(form, train), gamma = fit_gamma_glm(form, train),
               lognormal = fit_lognormal_glm(form, train), loggamma = fit_loggamma_glm(X_tr, y_tr))
  grows <- lapply(names(glms), function(f) {
    h <- glm_heldout(f, glms[[f]], test, X_te, y_te)
    data.frame(approach = "GLM", family = f, nll = h$nll, mae = h$mae, ks = ksD(h$pit)) })

  # Deep distributional heads.
  drows <- lapply(c("normal","gamma","lognormal","loggamma"), function(f) {
    d <- fit_deep_dist(Ztr, y_tr, Zte, y_te, f, epochs = deep_epochs, seed = 1)
    data.frame(approach = "Deep", family = f, nll = d$nll, mae = d$mae_med, ks = ksD(d$pit)) })

  res <- do.call(rbind, c(grows, drows)); res$dataset <- which
  res[, c("nll","mae","ks")] <- round(res[, c("nll","mae","ks")], 3)
  sel_glm <- res$family[res$approach == "GLM"][which.min(res$nll[res$approach=="GLM"])]
  sel_deep <- res$family[res$approach == "Deep"][which.min(res$nll[res$approach=="Deep"])]
  cat(sprintf("\n===== %s =====\n", which)); print(res[, c("approach","family","nll","mae","ks")], row.names = FALSE)
  cat(sprintf("Selected by held-out NLL:  GLM -> %s,  Deep -> %s\n", sel_glm, sel_deep))
  res
}

set.seed(1)
r_sev <- run_dataset("severity", deep_epochs = 250)
r_the <- run_dataset("thesis",   deep_epochs = 70)
all <- rbind(r_the, r_sev)
write.csv(all, "outputs/tables/experiment.csv", row.names = FALSE)
saveRDS(all, "outputs/experiment.rds")
cat("\nWrote outputs/tables/experiment.csv\n")
