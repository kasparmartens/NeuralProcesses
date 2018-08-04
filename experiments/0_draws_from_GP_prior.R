library(tidyverse)
source("GP_helpers.R")

GP_prior_draws <- function(ls, n_draws, N = 200){
  x <- seq(-3, 3, length=N)
  Y <- matrix(0, n_draws, N)
  for(i in 1:n_draws){
    K <- rbf_kernel(cbind(x), cbind(x), ls = ls, var = 1.0) + 1e-6*diag(N)
    Y[i, ] <- as.numeric(mvtnorm::rmvnorm(1, sigma = K))
  }
  Y %>%
    reshape2::melt() %>%
    rename(draw = Var1, index = Var2, f = value) %>%
    mutate(x = x[index], ls = ls)
}

df1 <- GP_prior_draws(ls = 1, n_draws = 20)
df2 <- GP_prior_draws(ls = 2, n_draws = 20)
df3 <- GP_prior_draws(ls = 3, n_draws = 20)

bind_rows(df1, df2, df3) %>%
  mutate(label = sprintf("lengthscale = %d", ls)) %>%
  ggplot(aes(x, f, group=draw)) +
  geom_line(alpha = 0.2) +
  # scale_color_viridis_c() +
  theme_classic() +
  facet_wrap(~ label) +
  theme(legend.position = "none") +
  labs(title = "Draws from the GP prior", y = "Function value")

ggsave("fig/GP_draws.png", width = 7.5, height = 2.5)
