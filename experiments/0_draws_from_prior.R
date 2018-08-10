library(tidyverse)
library(tensorflow)
library(patchwork)

source("NP_core.R")
source("GP_helpers.R")
source("helpers_for_plotting.R")
source("NP_architecture1.R")


dim_r_values <- c(1, 2, 4, 8)
plot_list <- list()
for(i in seq_along(dim_r_values)){
  dim_r <- dim_r_values[i]
  dim_z <- dim_r
  dim_h_hidden <- 8L
  dim_g_hidden <- 8L
  
  sess <- tf$Session()
  
  # placeholders for training inputs
  x_context <- tf$placeholder(tf$float32, shape(NULL, 1))
  y_context <- tf$placeholder(tf$float32, shape(NULL, 1))
  x_target <- tf$placeholder(tf$float32, shape(NULL, 1))
  y_target <- tf$placeholder(tf$float32, shape(NULL, 1))
  
  # set up NN
  train_op_and_loss <- init_NP(x_context, y_context, x_target, y_target, learning_rate = 0.001)
  
  # initialise
  init <- tf$global_variables_initializer()
  sess$run(init)
  
  x_star <- seq(-4, 4, length=100)
  prior_predict_op <- prior_predict(cbind(x_star), n_draws = 50L)
  y_star_mat <- sess$run(prior_predict_op$mu)
  df_pred <- reshape_predictions(y_star_mat, x_star) 
  
  plot_list[[i]] <- df_pred %>% 
    ggplot(aes(x, y, group=rep_index)) +
    geom_line(alpha = 0.2) +
    theme_classic() +
    labs(title = sprintf("dim(z) = %d", dim_r), y = "Function value")
  
  tf$reset_default_graph()
}

p <- wrap_plots(plot_list, nrow = 1) + plot_annotation(title = "Function draws from the NP prior")
p
ggsave("fig/draws_from_prior.png", p, width = 10, height = 3)
