# data.R: load either of the two cost datasets through a common interface.
#
#   "thesis"   : the Kaggle "Insurance Data for Machine Learning" benchmark used
#                in the companion thesis (1e6 records, light-tailed, near-linear).
#   "severity" : insuranceData::dataCar motor claim severity (positive claims
#                only, heavy-tailed, skewness ~5).
#
# Each returns a list with train, test, a shared model formula, and the
# train/test responses. The large thesis CSV is not committed; place it in the
# repo root, in data/, or set the INSURANCE_CSV environment variable.

CSV_PATH_CANDS <- function()
  c(Sys.getenv("INSURANCE_CSV", ""), "insurance_dataset.csv",
    "data/insurance_dataset.csv", "../insurance_dataset.csv")

resolve_csv <- function() {
  cands <- CSV_PATH_CANDS(); hit <- cands[cands != "" & file.exists(cands)]
  if (length(hit) == 0) stop("insurance_dataset.csv not found; set INSURANCE_CSV.")
  hit[1]
}

load_thesis_sample <- function(N = 120000, cache = "outputs/sample.rds") {
  if (file.exists(cache)) return(readRDS(cache))
  d <- read.csv(resolve_csv())
  set.seed(42); if (nrow(d) > N) d <- d[sample.int(nrow(d), N), ]
  for (f in c("gender","smoker","region","medical_history","family_medical_history",
              "exercise_frequency","occupation","coverage_level")) d[[f]] <- factor(d[[f]])
  rownames(d) <- NULL
  dir.create(dirname(cache), showWarnings = FALSE, recursive = TRUE)
  saveRDS(d, cache); d
}

load_severity <- function(rare_thresh = 0.02) {
  suppressMessages(requireNamespace("insuranceData", quietly = TRUE))
  e <- new.env(); utils::data("dataCar", package = "insuranceData", envir = e)
  d <- e$dataCar
  d <- d[d$claimcst0 > 0, c("claimcst0","veh_value","exposure","veh_body",
                            "veh_age","gender","area","agecat")]
  d$veh_value <- pmax(d$veh_value, 0.1)
  tab <- prop.table(table(d$veh_body)); rare <- names(tab)[tab < rare_thresh]
  d$veh_body <- as.character(d$veh_body); d$veh_body[d$veh_body %in% rare] <- "OTHER"
  for (f in c("veh_body","veh_age","gender","area","agecat")) d[[f]] <- factor(d[[f]])
  names(d)[1] <- "cost"; rownames(d) <- NULL; d
}

prepare_data <- function(which = "thesis", seed = 1, test_frac = 0.30) {
  if (which == "thesis") {
    d <- load_thesis_sample()
    names(d)[names(d) == "charges"] <- "cost"
    form <- cost ~ age + gender + bmi + children + smoker + region +
      medical_history + family_medical_history + exercise_frequency +
      occupation + coverage_level
  } else if (which == "severity") {
    d <- load_severity()
    form <- cost ~ log(veh_value) + exposure + veh_body + veh_age +
      gender + area + agecat
  } else stop("unknown dataset: ", which)

  set.seed(seed); n <- nrow(d)
  test_id <- sample.int(n, size = floor(test_frac * n))
  list(train = d[-test_id, ], test = d[test_id, ], form = form,
       y_train = d[-test_id, "cost"], y_test = d[test_id, "cost"])
}
