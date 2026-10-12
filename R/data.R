# data.R -- load and prepare the heavy-tailed motor claim severity data.
#
# Source: insuranceData::dataCar (De Jong & Heller 2008), Australian motor
# third-party claims. We model claim SEVERITY, i.e. claimcst0 for policies
# with at least one claim (claimcst0 > 0).

suppressMessages(library(insuranceData))

prepare_data <- function(seed = 1, test_frac = 0.30, rare_thresh = 0.02) {
  data(dataCar, package = "insuranceData", envir = environment())
  d <- dataCar

  # Severity sample: positive claim cost only.
  d <- d[d$claimcst0 > 0, ]

  # Predictors used throughout.
  d <- d[, c("claimcst0", "veh_value", "exposure", "veh_body",
             "veh_age", "gender", "area", "agecat")]

  # veh_value is in $10,000 units; a few zeros exist -- floor at a small value.
  d$veh_value <- pmax(d$veh_value, 0.1)

  # Lump rare vehicle-body levels (< rare_thresh of rows) into "OTHER" for
  # GLM stability.
  tab <- prop.table(table(d$veh_body))
  rare <- names(tab)[tab < rare_thresh]
  d$veh_body <- as.character(d$veh_body)
  d$veh_body[d$veh_body %in% rare] <- "OTHER"
  d$veh_body <- factor(d$veh_body)

  # veh_age and agecat are ordinal codes; treat as factors.
  d$veh_age <- factor(d$veh_age)
  d$agecat  <- factor(d$agecat)
  d$gender  <- factor(d$gender)
  d$area    <- factor(d$area)

  # Train/test split.
  set.seed(seed)
  n <- nrow(d)
  test_id <- sample.int(n, size = floor(test_frac * n))
  train <- d[-test_id, ]
  test  <- d[ test_id, ]

  # Common model formula (shared by every model).
  form <- claimcst0 ~ log(veh_value) + exposure + veh_body +
    veh_age + gender + area + agecat

  list(train = train, test = test, form = form,
       y_train = train$claimcst0, y_test = test$claimcst0)
}
