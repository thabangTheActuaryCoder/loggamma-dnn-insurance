# loggamma-dnn-insurance

Reproducible analysis for *Deep Distributional Regression for Insurance Cost: Learning the
Response Law from Normal to Loggamma, with Tail-Conditional Shapley Attribution*.

The study makes the **response law a learnable, selectable head** of a neural network. A
shared trunk feeds four maximum-likelihood heads, **Normal, Gamma, Lognormal and Loggamma**,
each trained end to end on the cost scale; the data select the head by the held-out
predictive negative log-likelihood (a proper score), checked by the Kolmogorov-Smirnov
statistic on the probability integral transform. This generalises Chikobvu's loggamma GLM
distributional selection (Mamba & Chikobvu 2025) to a deep, non-linear mean.

## Key result

On two datasets of opposite character, the framework recovers the correct law unaided:

| Dataset | Character | Selected law (GLM / Deep) | Loggamma KS |
|---|---|---|---|
| Kaggle benchmark | light-tailed (skew 0.13) | **Normal** / **Normal** | 0.38 / 0.43 (worst) |
| dataCar severity | heavy-tailed (skew 5.0) | **Loggamma** / **Loggamma** | 0.09 / **0.06** (best) |

The response law, not the linear-vs-deep distinction, governs predictive likelihood and
calibration. A squared-error network is the Normal head, hence the worst choice for
heavy-tailed cost.

## Datasets

- **Thesis benchmark**: Kaggle "Insurance Data for Machine Learning" (1e6 records,
  light-tailed). The 97 MB CSV is not committed; place `insurance_dataset.csv` in the repo
  root or set `INSURANCE_CSV`.
- **Severity**: `insuranceData::dataCar` motor claims (heavy-tailed), installed from CRAN.

## Layout

```
R/
  data.R            load either dataset through one interface
  loggamma.R        loggamma density/CDF/mean and custom-MLE GLM
  models.R          Normal, Gamma, Lognormal GLM heads (cost-scale likelihood)
  deepdist.R        deep distributional heads in torch (NLL, mean, median, PIT)
  run_experiment.R  main experiment: GLM vs deep heads on both datasets
  run_figures.R     PIT, NLL figures and tail-conditional Shapley
  figures.R         two manuscript figures from saved outputs
  run_glm_core.R    distributional GLMs selected by AIC/BIC and KS(PIT)
  run_ml.R          DNN (torch) and XGBoost benchmarks against the GLMs
  run_shapley.R     tail-conditional Shapley for XGBoost and the loggamma GLM
run_all.R           runs the experiment and figures
outputs/tables/     experiment.csv, shap_*.csv
outputs/figures/    fig_deep_pit.png, fig_nll.png, fig_shap_tail.png
```

The scripts `run_glm_core.R`, `run_ml.R` and `run_shapley.R` provide
the detailed single-dataset analyses used for robustness and coefficient tables.

## Reproduce

```r
install.packages(c("insuranceData", "xgboost", "torch"))
torch::install_torch()
Sys.setenv(INSURANCE_CSV = "/path/to/insurance_dataset.csv")
source("run_all.R")
```

Tested with R 4.6.0.

## Method note: deep loggamma head

`log(Y) ~ Gamma(alpha, beta(x))`, so `Y` is loggamma. The head is trained on the cost-scale
density `f_Y(y) = dgamma(log y; alpha, beta)/y` (with the `1/y` Jacobian), so its likelihood
is comparable with the Normal and Gamma heads (see `R/deepdist.R`).
