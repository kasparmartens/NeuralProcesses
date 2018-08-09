library(tidyverse)
library(tensorflow)
library(patchwork)

source("NP_core.R")
source("GP_helpers.R")
source("helpers_for_plotting.R")
source("NP_architecture1.R")

# global variables for training the model
dim_r <- 2L
dim_z <- 2L
dim_h_hidden <- 32L
dim_g_hidden <- 32L

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

n_iter <- 300000

for(iter in 1:n_iter){
  N <- 20
  x_obs <- runif(N, -3, 3)
  ls <- sample(c(1, 2, 3), 1)
  K <- rbf_kernel(cbind(x_obs), cbind(x_obs), ls, 1.0) + 1e-5 * diag(N)
  y_obs <- as.numeric(mvtnorm::rmvnorm(1, sigma = K))
  
  # sample N_context for training
  N_context <- sample(1:15, 1)
  feed_dict <- helper_context_and_target(x_obs, y_obs, N_context, x_context, y_context, x_target, y_target)
  a <- sess$run(train_op_and_loss, feed_dict = feed_dict)
  if(iter %% 1e3 == 0){
    cat(sprintf("loss = %1.3f\n", a[[2]]))
  }
}


# create a gif

library(gganimate)

x_star <- seq(-4, 4, length=100)
z1 <- seq(-15, 15, length=31)
z2 <- seq(-15, 15, length=31)
eps_value <- as.matrix(expand.grid(z1, z2))
eps <- tf$constant(eps_value, dtype = tf$float32)
prior_predict_op <- prior_predict(weights, cbind(x_star), epsilon = eps)
y_star_mat <- sess$run(prior_predict_op$mu)

df_pred <- y_star_mat %>%
  reshape_predictions(x_star) %>% 
  mutate(z1 = eps_value[, 1][rep_index], 
         z2 = eps_value[, 2][rep_index])

p1 <- df_pred %>%
  ggplot(aes(x, y, group=z2, col = z2)) +
  geom_line() +
  transition_time(as.integer(z1)) +
  labs(title = "NP trained on GP draws", subtitle = 'z1 = {frame_time}') +
  scale_color_viridis_c() +
  theme_classic()
animate(p1, nframes = 31, fps = 7, width=300, height=175)
# anim_save("fig/experiment3_1.gif")

p2 <- df_pred %>%
  ggplot(aes(x, y, group=z1, col = z1)) +
  geom_line() +
  transition_time(as.integer(z2)) +
  labs(title = " ", subtitle = 'z2 = {frame_time}') +
  scale_color_viridis_c() +
  theme_classic()
animate(p2, nframes = 31, fps = 7, width=300, height=175)
# anim_save("fig/experiment3_2.gif")


# Predictions for context points
x_star <- seq(-4, 4, length=100)
true_f <- function(x) -1.0 * sin(0.5*x)


x0 <- seq(-3, 3, length=3)
y0 <- true_f(x0)
p1 <- plot_posterior_draws(x0, y0, x_star, n_draws = 50L) +
  labs(title = "NP predictions")

x0 <- seq(-3, 3, length=5)
y0 <- true_f(x0)
p2 <- plot_posterior_draws(x0, y0, x_star, n_draws = 50L)

x0 <- seq(-3, 3, length=11)
y0 <- true_f(x0)
p3 <- plot_posterior_draws(x0, y0, x_star, n_draws = 50L)

p1 + p2 + p3

# GP predictions for the same set of points
library(gpflowr)

x0 <- seq(-3, 3, length=3)
y0 <- true_f(x0)
gp1 <- fit_and_plot_GP(x0, y0, x_star) +
  labs(title = "GP predictions")

x0 <- seq(-3, 3, length=5)
y0 <- true_f(x0)
gp2 <- fit_and_plot_GP(x0, y0, x_star)

x0 <- seq(-3, 3, length=11)
y0 <- true_f(x0)
gp3 <- fit_and_plot_GP(x0, y0, x_star)

p <- ((p1 | p2 | p3) * coord_cartesian(ylim = c(-2, 2)) / 
    ((gp1 | gp2 | gp3) * coord_cartesian(ylim = c(-2, 2))))

ggsave("fig/experiment3_pred1.png", p, width = 8, height = 4.5)
# ggsave("fig/experiment3_pred2.png", p, width = 8, height = 4.5)

