library(tidyverse)
library(tensorflow)
library(patchwork)

source("NP_helpers.R")
source("NP_core.R")
source("GP_helpers.R")
source("helpers_for_plotting.R")

dim_r_values <- c(1, 2, 4, 8)
plot_list <- list()
for(i in seq_along(dim_r_values)){
  dim_r <- dim_r_values[i]
  
  sess <- tf$Session()
  # create all NN weights
  weights <- init_weights(dim_r = dim_r, dim_h_hidden = 8L, dim_g_hidden = 8L)
  # initialise
  init <- tf$global_variables_initializer()
  sess$run(init)
  
  x_star <- seq(-4, 4, length=100)
  prior_predict_op <- prior_predict(weights, cbind(x_star), n_draws = 50L)
  y_star_mat <- sess$run(prior_predict_op$mu)
  df_pred <- reshape_predictions(y_star_mat, x_star) 
  
  plot_list[[i]] <- df_pred %>% 
    ggplot(aes(x, y, group=rep_index)) +
    geom_line(alpha = 0.2) +
    theme_classic() +
    labs(title = sprintf("dim(z) = %d", dim_r), y = "Function value")
  
}

p <- wrap_plots(plot_list, nrow = 1) + plot_annotation(title = "Function draws from the NP prior")
ggsave("fig/draws_from_prior.png", width = 10, height = 3)
