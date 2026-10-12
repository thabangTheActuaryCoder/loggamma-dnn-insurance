# deepdist.R: deep distributional regression with selectable likelihood heads.
#
# A shared feed-forward trunk maps the standardised covariates to a single
# location signal r_i; a family-specific link turns r_i into the location
# parameter of the chosen response law, and a global learnable parameter gives
# the law's second parameter. The network is trained end to end by minimising
# the family's negative log-likelihood (NLL) on the cost scale, so the heads are
# directly comparable with the corresponding GLMs and with each other by the
# held-out predictive NLL (a proper score).
#
# Families: "normal", "gamma", "lognormal", "loggamma".
# Loggamma uses the cost-scale density f_Y(y) = dgamma(log y; alpha, beta) / y
# (Mamba & Chikobvu 2025), with log y > 0 required (satisfied: both datasets
# have cost > e).

suppressMessages(library(torch))

# Softplus with a floor, for positive second parameters.
.spos <- function(x) nnf_softplus(x) + 1e-4
# Inverse softplus, for initialising a raw parameter at a target positive value.
.inv_sp <- function(s) if (s > 20) s else log(expm1(s))

# Family specification: location link (r -> location parameter), the second
# parameter transform sig(raw), a sensible raw initial value, and the nll, mean
# and pit on the cost scale. st = list(ym, ys, lym, lys) are training moments.
dist_family <- function(family, st) {
  lcv <- st$lys / st$lym                     # coefficient of variation of log y
  switch(family,
    normal = list(
      loc = function(r) st$ym + st$ys * r,
      sig = function(raw) st$ys * .spos(raw),            # cost-scale sd
      raw_init = .inv_sp(1),                             # start sd ~ ys
      nll = function(y, mu, s) 0.5*torch_log(2*pi*s*s) + (y - mu)^2/(2*s*s),
      mean = function(mu, s) as.numeric(as.matrix(mu)),
      med = function(mu, s) as.numeric(as.matrix(mu)),
      pit = function(y, mu, s) pnorm((y - as.numeric(as.matrix(mu)))/as.numeric(s))),
    gamma = list(
      loc = function(r) torch_exp(log(st$ym) + st$lys * r),
      sig = function(raw) .spos(raw),                    # shape nu
      raw_init = .inv_sp((st$ym/st$ys)^2),
      nll = function(y, mu, nu) { lam <- nu/mu
        -(nu*torch_log(lam) - torch_lgamma(nu) + (nu-1)*torch_log(y) - lam*y) },
      mean = function(mu, nu) as.numeric(as.matrix(mu)),
      med = function(mu, nu) { nu <- as.numeric(nu); mu <- as.numeric(as.matrix(mu))
        qgamma(0.5, shape = nu, rate = nu/mu) },
      pit = function(y, mu, nu) { nu <- as.numeric(nu); mu <- as.numeric(as.matrix(mu))
        pgamma(y, shape = nu, rate = nu/mu) }),
    lognormal = list(
      loc = function(r) st$lym + st$lys * r,
      sig = function(raw) .spos(raw),                    # log-scale sd
      raw_init = .inv_sp(st$lys),
      nll = function(y, mlog, s) 0.5*torch_log(2*pi*s*s) + (torch_log(y)-mlog)^2/(2*s*s) + torch_log(y),
      mean = function(mlog, s) { s <- as.numeric(s); exp(as.numeric(as.matrix(mlog)) + s*s/2) },
      med = function(mlog, s) exp(as.numeric(as.matrix(mlog))),
      pit = function(y, mlog, s) pnorm((log(y)-as.numeric(as.matrix(mlog)))/as.numeric(s))),
    loggamma = list(
      loc = function(r) torch_exp(log(st$lym) + lcv * r),  # E[log Y] via log link
      sig = function(raw) .spos(raw),                       # shape alpha
      raw_init = .inv_sp((st$lym/st$lys)^2),
      nll = function(y, muW, a) { b <- a/muW; w <- torch_log(y)
        -(a*torch_log(b) - torch_lgamma(a) + (a-1)*torch_log(w) - b*w) + torch_log(y) },
      mean = function(muW, a) { a <- as.numeric(a); muW <- as.numeric(as.matrix(muW)); b <- a/muW
        out <- rep(NA_real_, length(b)); ok <- b > 1; out[ok] <- (b[ok]/(b[ok]-1))^a; out },
      med = function(muW, a) { a <- as.numeric(a); muW <- as.numeric(as.matrix(muW))
        exp(qgamma(0.5, shape = a, rate = a/muW)) },
      pit = function(y, muW, a) { a <- as.numeric(a); muW <- as.numeric(as.matrix(muW))
        pgamma(log(y), shape = a, rate = a/muW) }),
    stop("unknown family"))
}

make_trunk <- function(p) nn_module(
  initialize = function(p) {
    self$fc1 <- nn_linear(p, 64); self$fc2 <- nn_linear(64, 32)
    self$out <- nn_linear(32, 1) },
  forward = function(x) self$out(nnf_relu(self$fc2(nnf_relu(self$fc1(x)))))
)(p)

# Fit one deep distributional head. Returns held-out mean NLL, RMSE, R2, PIT,
# predicted means, and the fitted global second parameter.
fit_deep_dist <- function(Ztr, ytr, Zte, yte, family,
                          epochs = 120, bs = 1024, lr = 2e-3, seed = 1) {
  st <- list(ym = mean(ytr), ys = sd(ytr), lym = mean(log(ytr)), lys = sd(log(ytr)))
  fam <- dist_family(family, st)
  torch_manual_seed(seed); set.seed(seed)
  net <- make_trunk(ncol(Ztr))
  raw <- nn_parameter(torch_tensor(fam$raw_init))  # global 2nd parameter (pre-transform)
  Xt <- torch_tensor(Ztr, dtype = torch_float())
  Yt <- torch_tensor(matrix(ytr, ncol = 1), dtype = torch_float())
  opt <- optim_adam(c(net$parameters, list(raw)), lr = lr)
  n <- nrow(Ztr)
  for (ep in 1:epochs) {
    net$train(); perm <- sample.int(n)
    for (s in seq(1, n, by = bs)) {
      b <- perm[s:min(s+bs-1, n)]; opt$zero_grad()
      loc <- fam$loc(net(Xt[b, ]))
      loss <- torch_mean(fam$nll(Yt[b, , drop = FALSE], loc, fam$sig(raw)))
      loss$backward(); opt$step()
    }
  }
  net$eval()
  loc_te <- fam$loc(net(torch_tensor(Zte, dtype = torch_float())))
  p2t <- fam$sig(raw)
  nll_te <- as.numeric(torch_mean(fam$nll(
    torch_tensor(matrix(yte, ncol = 1), dtype = torch_float()), loc_te, p2t)))
  p2 <- as.numeric(p2t)
  yhat <- fam$mean(loc_te, p2t)
  ymed <- fam$med(loc_te, p2t)
  u <- fam$pit(yte, loc_te, p2t)
  ok <- is.finite(yhat)
  rmse <- sqrt(mean((yte[ok] - yhat[ok])^2))
  r2 <- 1 - sum((yte[ok]-yhat[ok])^2) / sum((yte[ok]-mean(yte[ok]))^2)
  mae_med <- mean(abs(yte - ymed))                 # robust point accuracy (median)
  list(family = family, nll = nll_te, rmse = rmse, r2 = r2, mae_med = mae_med,
       pit = u, yhat = yhat, ymed = ymed, param2 = p2)
}
