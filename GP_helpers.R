
rbf_kernel <- function(X, Y, ls, var){
  X <- X / ls
  Y <- Y / ls
  N1 <- nrow(X)
  N2 <- nrow(Y)
  s1 <- rowSums(X**2)
  s2 <- rowSums(Y**2)
  square <- matrix(s1, N1, N2) + matrix(s2, N1, N2, byrow=T) - 2 * X %*% t(Y)
  var * exp(-0.5 * square)
}

# helper function for fitting GP using gpflow

fit_GP <- function(x, y, x_star, n_draws = 10L){
  k <- gpflow$kernels$RBF(1L, lengthscales = 1.0)
  k$variance$prior <- gpflow$priors$Gamma(1.0, 1.0)
  k$lengthscales$prior <- gpflow$priors$Gamma(1.0, 1.0)
  m <- gpflow$gpr$GPR(cbind(x), cbind(y), k)
  m$likelihood$variance <- 0.05
  m$optimize()
  # print(k$lengthscales)
  pred_mat <- m$predict_f_samples(cbind(x_star), n_draws)
  pred_mat %>%
    reshape2::melt() %>%
    rename(index = Var2, draw = Var1, y = value) %>%
    mutate(x = x_star[index])
}

fit_and_plot_GP <- function(x0, y0, x_star){
  df_pred <- fit_GP(x0, y0, x_star, n_draws = 50L)
  df_obs <- data.frame(x = x0, y = y0)
  df_pred %>%
    ggplot(aes(x, y)) +
    geom_line(aes(group = draw), alpha = 0.25) +
    geom_point(data = df_obs, col = "#b2182b", size = 3) +
    theme_classic()
}
