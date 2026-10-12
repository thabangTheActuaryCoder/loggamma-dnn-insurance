# run_figures.R: figures for the deep distributional experiment, plus the
# tail-conditional Shapley attribution of the selected GLM head per dataset.

here <- function(...) file.path("R", ...)
source(here("data.R")); source(here("models.R")); source(here("loggamma.R")); source(here("deepdist.R"))
dir.create("outputs/figures", showWarnings = FALSE, recursive = TRUE)
cols <- c(normal = "#6A3D9A", gamma = "#B2182B", lognormal = "#2166AC", loggamma = "#1B7837")

prep <- function(which) {
  dat <- prepare_data(which, seed = 1); form <- dat$form
  mf <- model.frame(form, dat$train); X_tr <- model.matrix(form, mf)
  X_te <- model.matrix(form, model.frame(form, dat$test, xlev = lapply(mf, levels)))
  Xtr <- X_tr[,-1,drop=FALSE]; Xte <- X_te[,-1,drop=FALSE]
  ct <- colMeans(Xtr); sdv <- apply(Xtr,2,sd); sdv[sdv==0]<-1
  list(dat=dat, form=form, mf=mf, X_tr=X_tr, X_te=X_te,
       Ztr=scale(Xtr,ct,sdv), Zte=scale(Xte,ct,sdv))
}

# ---- Fig 1: deep-head PIT calibration per dataset -------------------------
png("outputs/figures/fig_deep_pit.png", width = 2100, height = 1050, res = 230)
par(mfrow = c(1,2), mar = c(4.2,4.2,2.4,1))
for (which in c("severity","thesis")) {
  P <- prep(which); ep <- if (which=="severity") 250 else 70
  plot(c(0,1),c(0,1),type="n",xlab="Theoretical uniform quantile",
       ylab="Empirical PIT quantile",
       main=sprintf("Deep heads: %s", if(which=="severity") "heavy-tailed severity" else "light-tailed benchmark"))
  abline(0,1,lty=2,col="grey50")
  for (f in c("normal","gamma","lognormal","loggamma")) {
    d <- fit_deep_dist(P$Ztr, P$dat$y_train, P$Zte, P$dat$y_test, f, epochs=ep, seed=1)
    u <- sort(d$pit); lines(ppoints(length(u)), u, col=cols[f], lwd=2)
  }
  legend("topleft", legend=c("Normal","Gamma","Lognormal","Loggamma"), col=cols, lwd=2, bty="n")
}
dev.off(); cat("wrote fig_deep_pit.png\n")

# ---- Fig 2: held-out NLL, GLM vs Deep, per dataset ------------------------
ex <- read.csv("outputs/tables/experiment.csv")
png("outputs/figures/fig_nll.png", width = 2100, height = 1000, res = 230)
par(mfrow = c(1,2), mar = c(5,4.2,2.4,1))
for (which in c("severity","thesis")) {
  s <- ex[ex$dataset==which,]
  M <- rbind(GLM = s$nll[s$approach=="GLM"], Deep = s$nll[s$approach=="Deep"])
  colnames(M) <- s$family[s$approach=="GLM"]
  barplot(M, beside=TRUE, las=2, col=c("#9ECAE1","#08519C"), ylab="held-out mean NLL",
          main=sprintf("%s", if(which=="severity") "heavy-tailed severity" else "light-tailed benchmark"),
          ylim=c(min(s$nll)-0.3, max(s$nll)+0.3), xpd=FALSE)
  legend("topright", c("GLM","Deep"), fill=c("#9ECAE1","#08519C"), bty="n")
}
dev.off(); cat("wrote fig_nll.png\n")

# ---- Tail-conditional Shapley of the selected GLM head per dataset --------
add_shap <- function(which, head) {
  P <- prep(which); form <- P$form
  assign <- attr(P$X_tr,"assign"); feat <- attr(terms(form),"term.labels")
  group_of <- rep("(Intercept)", length(assign)); group_of[assign>0] <- feat[assign[assign>0]]
  gshap <- function(S){ cols<-colnames(S); g<-group_of[match(cols,colnames(P$X_te))]; t(rowsum(t(S),g)) }
  if (head=="normal") { fit <- fit_normal_glm(form, P$dat$train); b <- coef(fit$lm)[colnames(P$X_te)]
    pred <- predict_normal_mean(fit, P$dat$test)
  } else { fit <- fit_loggamma_glm(P$X_tr, P$dat$y_train); b <- fit$coef[colnames(P$X_te)]
    pred <- as.vector(P$X_te %*% fit$coef) }       # rank by linear predictor
  Xc <- sweep(P$X_te, 2, colMeans(P$X_tr), "-"); contrib <- sweep(Xc, 2, b, "*")
  colnames(contrib) <- colnames(P$X_te); S <- gshap(contrib)
  imp <- function(ix){ m<-colMeans(abs(S[ix,,drop=FALSE])); m<-m[!names(m) %in% c("BIAS","(Intercept)")]; sort(m/sum(m),decreasing=TRUE)}
  thr <- quantile(pred, 0.99)
  g <- imp(rep(TRUE,nrow(S))); t99 <- imp(pred>=thr)[names(g)]
  data.frame(feature=names(g), global=round(as.numeric(g),3), tail99=round(as.numeric(t99),3))
}
sev_sh <- add_shap("severity","loggamma"); the_sh <- add_shap("thesis","normal")
write.csv(sev_sh, "outputs/tables/shap_severity_loggamma.csv", row.names=FALSE)
write.csv(the_sh, "outputs/tables/shap_thesis_normal.csv", row.names=FALSE)
cat("\n== Severity (loggamma GLM) Shapley ==\n"); print(sev_sh, row.names=FALSE)
cat("\n== Thesis (normal GLM) Shapley ==\n"); print(the_sh, row.names=FALSE)

png("outputs/figures/fig_shap_tail.png", width=2100, height=1050, res=230)
par(mfrow=c(1,2), mar=c(8,4,2.4,1))
pp <- function(df,title){ o<-order(-df$global); df<-df[o,]; M<-rbind(Global=df$global,Tail99=df$tail99)
  colnames(M)<-df$feature; barplot(M,beside=TRUE,las=2,col=c("#9ECAE1","#08519C"),
  ylab="normalised mean|SHAP|",main=title,cex.names=0.75)
  legend("topright",c("Global","Tail (>P99)"),fill=c("#9ECAE1","#08519C"),bty="n")}
pp(sev_sh,"Severity: loggamma GLM"); pp(the_sh,"Benchmark: Normal GLM")
dev.off(); cat("wrote fig_shap_tail.png\n")
