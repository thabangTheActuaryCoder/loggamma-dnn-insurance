# run_ml.R -- flexible-model benchmarks: a deep neural network (torch) and
# gradient boosting (xgboost), scored against the GLMs on held-out accuracy
# in the body and the tail.

suppressMessages({ library(xgboost); library(torch) })
here <- function(...) file.path("R", ...)
source(here("data.R")); source(here("models.R"))

dat <- prepare_data(seed = 1)
train <- dat$train; test <- dat$test; form <- dat$form
y_tr <- dat$y_train; y_te <- dat$y_test

# Numeric design matrices (drop intercept for ML; standardise columns).
mf_tr <- model.frame(form, train)
Xtr <- model.matrix(form, mf_tr)[, -1, drop = FALSE]
Xte <- model.matrix(form, model.frame(form, test, xlev = lapply(mf_tr, levels)))[, -1, drop = FALSE]
ctr <- colMeans(Xtr); csd <- apply(Xtr, 2, sd); csd[csd == 0] <- 1
Ztr <- scale(Xtr, center = ctr, scale = csd)
Zte <- scale(Xte, center = ctr, scale = csd)

# ---- Deep neural network (torch) -----------------------------------------
# Target: standardised log-severity; mean back-transformed with Duan's
# smearing estimator so predictions are on the severity scale.
set.seed(1); torch_manual_seed(1)
lt <- log(y_tr); lm_ <- mean(lt); ls_ <- sd(lt)
Xt <- torch_tensor(Ztr, dtype = torch_float())
Yt <- torch_tensor(matrix((lt - lm_) / ls_, ncol = 1), dtype = torch_float())

net <- nn_module(
  initialize = function(p) {
    self$fc1 <- nn_linear(p, 64); self$fc2 <- nn_linear(64, 32)
    self$fc3 <- nn_linear(32, 16); self$out <- nn_linear(16, 1)
    self$drop <- nn_dropout(0.15)
  },
  forward = function(x) {
    x <- self$drop(nnf_relu(self$fc1(x)))
    x <- self$drop(nnf_relu(self$fc2(x)))
    x <- nnf_relu(self$fc3(x)); self$out(x)
  })
model <- net(ncol(Ztr))
opt <- optim_adam(model$parameters, lr = 1e-3, weight_decay = 1e-4)
for (epoch in 1:400) {
  model$train(); opt$zero_grad()
  loss <- nnf_mse_loss(model(Xt), Yt); loss$backward(); opt$step()
}
model$eval()
pred_std <- as.numeric(as.matrix(model(torch_tensor(Zte, dtype = torch_float()))))
pred_logy <- pred_std * ls_ + lm_
# Duan smearing on training residuals for an unbiased mean back-transform.
res_tr <- lt - (as.numeric(as.matrix(model(Xt))) * ls_ + lm_)
smear <- mean(exp(res_tr))
yh_dnn <- exp(pred_logy) * smear

# ---- Gradient boosting (xgboost, gamma objective) -------------------------
dtr <- xgb.DMatrix(Xtr, label = y_tr); dte <- xgb.DMatrix(Xte, label = y_te)
set.seed(1)
bst <- xgb.train(params = list(objective = "reg:gamma", eta = 0.05,
                               max_depth = 3, subsample = 0.8,
                               colsample_bytree = 0.8, min_child_weight = 5),
                 data = dtr, nrounds = 300, verbose = 0)
yh_xgb <- predict(bst, dte)

# ---- Accuracy table (reuse GLM predictions from the core run) ------------
g <- readRDS("outputs/glm_fits.rds")
yh_g  <- predict_gamma_mean(g$fg, test)
yh_ln <- predict_lognormal_mean(g$fln, test)
source(here("loggamma.R"))
yh_lg <- predict_loggamma_mean(g$flg, model.matrix(form, model.frame(form, test, xlev = lapply(mf_tr, levels))))

acc <- function(yh) c(accuracy(y_te, yh),
                      t90 = unname(tail_accuracy(y_te, yh, 0.90)["rmse"]),
                      t95 = unname(tail_accuracy(y_te, yh, 0.95)["rmse"]),
                      t99 = unname(tail_accuracy(y_te, yh, 0.99)["rmse"]))
tabM <- rbind(`Gamma GLM` = acc(yh_g), `Lognormal GLM` = acc(yh_ln),
              `Loggamma GLM` = acc(yh_lg), `XGBoost` = acc(yh_xgb),
              `DNN` = acc(yh_dnn))
colnames(tabM) <- c("RMSE","MAE","RMSE_t90","RMSE_t95","RMSE_t99")
cat("\n== Held-out accuracy: GLMs vs DNN vs XGBoost ==\n"); print(round(tabM, 1))

write.csv(data.frame(model = rownames(tabM), round(tabM, 2)),
          "outputs/tables/all_accuracy.csv", row.names = FALSE)
xgb.save(bst, "outputs/xgb.model")
cat("\nWrote outputs/tables/all_accuracy.csv and outputs/xgb.model\n")
