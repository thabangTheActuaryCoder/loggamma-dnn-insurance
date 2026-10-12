# run_shapley.R -- tail-conditional Shapley attribution.
#
# For both the flexible model (XGBoost, exact tree SHAP) and the best
# distributional GLM (loggamma, exact additive Shapley), we rank feature
# importance globally and conditional on the model's own predicted severity
# exceeding its 90th/95th/99th percentile, then ask whether the drivers in
# the tail differ from the drivers on average (cf. Article 1).

suppressMessages({ library(xgboost) })
here <- function(...) file.path("R", ...)
source(here("data.R")); source(here("models.R")); source(here("loggamma.R"))

dat <- prepare_data(seed = 1); train <- dat$train; test <- dat$test; form <- dat$form
mf_tr <- model.frame(form, train)
Xtr_full <- model.matrix(form, mf_tr)
Xte_full <- model.matrix(form, model.frame(form, test, xlev = lapply(mf_tr, levels)))
assign <- attr(Xtr_full, "assign")                 # 0 = intercept
feat_names <- attr(terms(form), "term.labels")
group_of <- ifelse(assign == 0, "(Intercept)", feat_names[assign])

# Collapse a per-design-column SHAP matrix to per-original-feature.
group_shap <- function(S) {
  cols <- colnames(S); cols[is.na(cols)] <- "BIAS"
  g <- group_of[match(cols, colnames(Xte_full))]
  g[is.na(g)] <- "BIAS"
  t(rowsum(t(S), group = g))
}

# Mean |SHAP| per feature over a row subset -> normalised importance.
importance <- function(S, idx) {
  m <- colMeans(abs(S[idx, , drop = FALSE]))
  m <- m[names(m) != "BIAS" & names(m) != "(Intercept)"]
  sort(m / sum(m), decreasing = TRUE)
}
rank_table <- function(S, pred, label) {
  thr <- quantile(pred, c(0.90, 0.95, 0.99))
  sets <- list(Global = rep(TRUE, length(pred)),
               Tail90 = pred >= thr[1], Tail95 = pred >= thr[2],
               Tail99 = pred >= thr[3])
  imp <- lapply(sets, function(ix) importance(S, ix))
  feats <- names(imp$Global)
  out <- sapply(imp, function(v) v[feats]); rownames(out) <- feats
  cat(sprintf("\n== %s: normalised mean|SHAP| by subset ==\n", label))
  print(round(out, 3))
  cat(sprintf("Top feature  Global: %s | Tail99: %s\n",
              names(which.max(imp$Global)), names(which.max(imp$Tail99))))
  as.data.frame(cbind(feature = feats, round(out, 4)))
}

# ---- XGBoost tree SHAP ----------------------------------------------------
bst <- xgb.load("outputs/xgb.model")
Xte_ml <- Xte_full[, -1, drop = FALSE]
pred_xgb <- predict(bst, xgb.DMatrix(Xte_ml))
S_xgb <- predict(bst, xgb.DMatrix(Xte_ml), predcontrib = TRUE)
colnames(S_xgb) <- c(colnames(Xte_ml), "BIAS")
S_xgb_g <- group_shap(S_xgb)
tab_xgb <- rank_table(S_xgb_g, pred_xgb, "XGBoost (tree SHAP)")

# ---- Loggamma GLM additive Shapley ---------------------------------------
# For an additive linear predictor eta = Xb, the exact Shapley value of a
# feature group on eta is the centred contribution of its columns.
g <- readRDS("outputs/glm_fits.rds"); flg <- g$flg
b <- flg$coef
Xc <- sweep(Xte_full, 2, colMeans(Xtr_full), "-")   # centre at train means
contrib <- sweep(Xc, 2, b, "*")                      # per-column Shapley on eta
colnames(contrib) <- colnames(Xte_full)
S_glm_g <- group_shap(contrib)
pred_glm <- predict_loggamma_mean(flg, Xte_full)
tab_glm <- rank_table(S_glm_g, pred_glm, "Loggamma GLM (additive Shapley)")

# ---- Agreement between the two models on the global ranking ---------------
r_x <- rank(-importance(S_xgb_g, rep(TRUE, nrow(S_xgb_g))))
common <- intersect(names(r_x), names(rank(-importance(S_glm_g, rep(TRUE, nrow(S_glm_g))))))
r_g <- rank(-importance(S_glm_g, rep(TRUE, nrow(S_glm_g))))[common]
rho <- suppressWarnings(cor(r_x[common], r_g, method = "spearman"))
cat(sprintf("\nSpearman rank correlation of global feature importance (XGB vs loggamma GLM): %.2f\n", rho))

write.csv(tab_xgb, "outputs/tables/shap_xgboost.csv", row.names = FALSE)
write.csv(tab_glm, "outputs/tables/shap_loggamma.csv", row.names = FALSE)
cat("\nWrote outputs/tables/shap_xgboost.csv and shap_loggamma.csv\n")
