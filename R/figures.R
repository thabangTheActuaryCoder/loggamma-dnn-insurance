# figures.R -- two manuscript figures from saved outputs.
here <- function(...) file.path("R", ...)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

# ---- Fig 1: PIT uniformity QQ for the three GLMs --------------------------
g <- readRDS("outputs/glm_fits.rds")
pit <- g$pit
png("outputs/figures/fig_pit_qq.png", width = 1500, height = 1500, res = 240)
par(mar = c(4.2, 4.2, 2, 1))
cols <- c(gamma = "#B2182B", lognormal = "#2166AC", loggamma = "#1B7837")
plot(c(0,1), c(0,1), type = "n", xlab = "Theoretical uniform quantile",
     ylab = "Empirical PIT quantile", main = "PIT calibration (held-out)")
abline(0, 1, lty = 2, col = "grey50")
for (nm in names(pit)) {
  u <- sort(pit[[nm]]); p <- ppoints(length(u))
  lines(p, u, col = cols[nm], lwd = 2)
}
legend("topleft", legend = c("Gamma GLM","Lognormal GLM","Loggamma GLM"),
       col = cols, lwd = 2, bty = "n")
dev.off()

# ---- Fig 2: Global vs Tail99 importance, XGBoost and loggamma GLM ---------
sx <- read.csv("outputs/tables/shap_xgboost.csv")
sg <- read.csv("outputs/tables/shap_loggamma.csv")
png("outputs/figures/fig_shap_tail.png", width = 2100, height = 1050, res = 230)
par(mfrow = c(1, 2), mar = c(7, 4, 2.4, 1))
plot_pair <- function(df, title) {
  o <- order(-df$Global); df <- df[o, ]
  m <- rbind(Global = df$Global, Tail99 = df$Tail99)
  colnames(m) <- df$feature
  barplot(m, beside = TRUE, las = 2, col = c("#9ECAE1", "#08519C"),
          ylab = "normalised mean|SHAP|", main = title, cex.names = 0.8)
  legend("topright", c("Global","Tail (>P99)"), fill = c("#9ECAE1","#08519C"), bty = "n")
}
plot_pair(sx, "XGBoost (tree SHAP)")
plot_pair(sg, "Loggamma GLM (additive Shapley)")
dev.off()
cat("Wrote outputs/figures/fig_pit_qq.png and fig_shap_tail.png\n")
