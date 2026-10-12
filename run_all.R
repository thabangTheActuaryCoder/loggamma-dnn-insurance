# run_all.R: reproduce the deep distributional regression experiment.
# Requires the thesis CSV (set INSURANCE_CSV) and the insuranceData package.
stopifnot(dir.exists("R"))
message("1/2  Main experiment: GLM vs deep heads (Normal/Gamma/Lognormal/Loggamma) on both datasets")
source("R/run_experiment.R")
message("2/2  Figures (PIT, NLL) and tail-conditional Shapley of the selected head")
source("R/run_figures.R")
message("Done. See outputs/tables and outputs/figures.")
