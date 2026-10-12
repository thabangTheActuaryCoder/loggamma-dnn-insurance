# run_glm_core.R -- fit the three distributional GLMs, select among them by
# AIC/BIC + KS(PIT), and score held-out accuracy in the body and the tail.

here <- function(...) file.path("R", ...)
source(here("data.R")); source(here("loggamma.R")); source(here("models.R"))

dat <- prepare_data(seed = 1)
train <- dat$train; test <- dat$test; form <- dat$form
y_tr <- dat$y_train; y_te <- dat$y_test

# Shared design matrices (intercept included), consistent factor coding.
mf_tr <- model.frame(form, train)
X_tr  <- model.matrix(form, mf_tr)
X_te  <- model.matrix(form, model.frame(form, test, xlev = lapply(mf_tr, levels)))

cat(sprintf("Train n = %d, Test n = %d, predictors (design cols) = %d\n",
            nrow(X_tr), nrow(X_te), ncol(X_tr)))

# ---- Fit ------------------------------------------------------------------
fg  <- fit_gamma_glm(form, train)
fln <- fit_lognormal_glm(form, train)
flg <- fit_loggamma_glm(X_tr, y_tr)
cat(sprintf("loggamma MLE convergence code: %d (0 = OK); shape alpha = %.3f\n",
            flg$convergence, flg$alpha))

# ---- Model selection table (in-sample fit) --------------------------------
sel <- data.frame(
  model  = c("Gamma GLM", "Lognormal GLM", "Loggamma GLM"),
  npar   = c(fg$npar, fln$npar, flg$npar),
  loglik = c(fg$loglik, fln$loglik, flg$loglik),
  AIC    = c(fg$aic, fln$aic, flg$aic),
  BIC    = c(fg$bic, fln$bic, flg$bic)
)
sel$dAIC <- sel$AIC - min(sel$AIC)
sel$dBIC <- sel$BIC - min(sel$BIC)

# ---- KS test on PIT (held-out) -------------------------------------------
u_g  <- pit_gamma(fg, test, y_te)
u_ln <- pit_lognormal(fln, test, y_te)
u_lg <- pit_loggamma(flg, X_te, y_te)
ks <- function(u) { k <- suppressWarnings(ks.test(u, "punif")); c(D = unname(k$statistic), p = k$p.value) }
sel$KS_D <- c(ks(u_g)["D"], ks(u_ln)["D"], ks(u_lg)["D"])
sel$KS_p <- c(ks(u_g)["p"], ks(u_ln)["p"], ks(u_lg)["p"])

sel[,-1] <- round(sel[,-1], 3)
cat("\n== Model selection (fit on train, KS on test PIT) ==\n"); print(sel, row.names = FALSE)

# ---- Held-out accuracy: overall and in the tail ---------------------------
yh_g  <- predict_gamma_mean(fg, test)
yh_ln <- predict_lognormal_mean(fln, test)
yh_lg <- predict_loggamma_mean(flg, X_te)

acc <- function(yh) c(accuracy(y_te, yh),
                      tail90 = tail_accuracy(y_te, yh, 0.90)["rmse"],
                      tail95 = tail_accuracy(y_te, yh, 0.95)["rmse"],
                      tail99 = tail_accuracy(y_te, yh, 0.99)["rmse"])
acctab <- rbind(`Gamma GLM` = acc(yh_g),
                `Lognormal GLM` = acc(yh_ln),
                `Loggamma GLM` = acc(yh_lg))
colnames(acctab) <- c("RMSE", "MAE", "RMSE_tail90", "RMSE_tail95", "RMSE_tail99")
cat("\n== Held-out accuracy (predicted mean severity) ==\n")
print(round(acctab, 1))

# ---- Persist --------------------------------------------------------------
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
write.csv(sel, "outputs/tables/model_selection.csv", row.names = FALSE)
write.csv(data.frame(model = rownames(acctab), round(acctab, 2)),
          "outputs/tables/glm_accuracy.csv", row.names = FALSE)
saveRDS(list(fg = fg, fln = fln, flg = flg,
             pit = list(gamma = u_g, lognormal = u_ln, loggamma = u_lg)),
        "outputs/glm_fits.rds")
cat("\nWrote outputs/tables/model_selection.csv, glm_accuracy.csv, glm_fits.rds\n")
