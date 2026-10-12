# run_all.R -- reproduce every table and figure for the article.
# Usage:  Rscript run_all.R   (from the repository root)

stopifnot(dir.exists("R"))
message("1/4  Distributional GLMs (gamma, lognormal, loggamma) + selection")
source("R/run_glm_core.R")
message("2/4  Flexible benchmarks: deep neural network + gradient boosting")
source("R/run_ml.R")
message("3/4  Tail-conditional Shapley attribution")
source("R/run_shapley.R")
message("4/4  Figures")
source("R/figures.R")
message("Done. See outputs/tables and outputs/figures.")
