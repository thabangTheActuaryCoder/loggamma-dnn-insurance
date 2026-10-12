# loggamma-dnn-insurance

Reproducible analysis for *The Loggamma Generalised Linear Model for Heavy-Tailed
Insurance Cost Prediction: A Benchmark Against Deep Neural Networks with
Tail-Conditional Shapley Attribution*.

The study fits gamma, lognormal and **loggamma** generalised linear models (GLMs)
to heavy-tailed motor claim severity, selects among them by AIC/BIC and held-out
calibration, benchmarks the best GLM against a deep neural network and gradient
boosting, and attributes predictions with Shapley values conditional on the tail.

## Data

`insuranceData::dataCar` (De Jong & Heller, 2008): 67,856 Australian motor policies,
4,624 with a positive claim. The response is claim severity (`claimcst0 > 0`), which
is strongly right-skewed (skewness 5.0, excess kurtosis 40.2).

## Layout

```
R/
  data.R           load & prepare the severity sample; train/test split
  loggamma.R       loggamma density, CDF, mean, and custom-MLE GLM fit
  models.R         gamma & lognormal GLMs; cost-scale log-likelihood; accuracy
  run_glm_core.R   fit the three GLMs; AIC/BIC + KS(PIT) selection; accuracy
  run_ml.R         torch DNN + XGBoost benchmarks
  run_shapley.R    tail-conditional tree SHAP (XGB) & additive Shapley (GLM)
  figures.R        PIT calibration and Shapley figures
run_all.R          runs the whole pipeline
outputs/
  tables/          model_selection.csv, all_accuracy.csv, shap_*.csv
  figures/         fig_pit_qq.png, fig_shap_tail.png
```

## Reproduce

```r
# from the repository root
install.packages(c("insuranceData", "xgboost", "torch"))
torch::install_torch()        # one-time libtorch download
source("run_all.R")           # or: Rscript run_all.R
```

Tested with R 4.6.0.

## Key results

- The **loggamma GLM** is selected over the lognormal and gamma GLMs by AIC and BIC
  (ΔAIC 193.9 and 1174.9) and has the smallest held-out KS statistic.
- The GLMs **match or beat** the DNN and XGBoost on held-out RMSE; the DNN is least
  accurate. Distributional choice matters more for calibration than model class does
  for point accuracy.
- The flexible model and the GLM **disagree sharply** on feature importance (global
  Spearman rank correlation 0.14), and the leading driver **intensifies in the tail**
  for both.

## Method note: the loggamma GLM

`log(Y) ~ Gamma`, so `Y` is loggamma. The loggamma GLM is *not* a base-R `glm()` of a
gamma model on `log(y)`: to compare AIC/BIC with the gamma and lognormal GLMs, the
likelihood is evaluated as the density of `Y` (with the `1/y` Jacobian) and maximised
directly (`R/loggamma.R`).
