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

n_iter <- 50000

for(iter in 1:n_iter){
  N <- 20
  x_obs <- runif(N, -3, 3)
  a <- runif(1, -2, 2)
  y_obs <- a * sin(x_obs)
  
  # sample N_context for training
  N_context <- sample(1:10, 1)
  feed_dict <- helper_context_and_target(x_obs, y_obs, N_context, x_context, y_context, x_target, y_target)
  a <- sess$run(train_op_and_loss, feed_dict = feed_dict)
  if(iter %% 1e3 == 0){
    cat(sprintf("loss = %1.3f\n", a[[2]]))
  }
}


# create a grid (z1, z2) and plot predictions
x_star <- seq(-4, 4, length=100)
z1 <- seq(-4, 4, length=9)
z2 <- seq(-4, 4, length=9)
eps_value <- as.matrix(expand.grid(z1, z2))
eps <- tf$constant(eps_value, dtype = tf$float32)
prior_predict_op <- prior_predict(cbind(x_star), epsilon = eps)
y_star_mat <- sess$run(prior_predict_op$mu)
reshape_predictions(y_star_mat, x_star) %>% 
  mutate(z1 = eps_value[, 1][rep_index], 
         z2 = eps_value[, 2][rep_index]) %>%
  ggplot(aes(x, y, group=rep_index, col = z1+z2)) +
  geom_line() +
  facet_grid(z1 ~ z2) +
  theme_classic() +
  theme(legend.position = "none") +
  scale_y_continuous(breaks = c(-2, 0, 2))
ggsave("fig/experiment2_latent_space.png", width=8, height=5)


# create a gif

library(gganimate)

x_star <- seq(-4, 4, length=100)
z1 <- seq(-4, 4, length=41)
z2 <- seq(-4, 4, length=41)
eps_value <- as.matrix(expand.grid(z1, z2))
eps <- tf$constant(eps_value, dtype = tf$float32)
prior_predict_op <- prior_predict(cbind(x_star), epsilon = eps)
y_star_mat <- sess$run(prior_predict_op$mu)

df_pred <- y_star_mat %>%
  reshape_predictions(x_star) %>% 
  mutate(z1 = eps_value[, 1][rep_index], 
         z2 = eps_value[, 2][rep_index])

p1 <- df_pred %>%
  ggplot(aes(x, y, group=z2, col = z2)) +
  geom_line() +
  transition_time(z1) +
  labs(title = "NP for a*sin(x)", subtitle = 'z1 = {frame_time}') +
  scale_color_viridis_c() +
  theme_classic()
animate(p1, nframes = 50, width=300, height=250)
# anim_save("fig/sin_z1.gif")

p2 <- df_pred %>%
  ggplot(aes(x, y, group=z1, col = z1)) +
  geom_line() +
  transition_time(z2) +
  labs(title = " ", subtitle = 'z2 = {frame_time}') +
  scale_color_viridis_c() +
  theme_classic()
animate(p2, nframes = 50, width=300, height=250)
# anim_save("fig/sin_z2.gif")


# Static plots for predictions
x_star <- seq(-4, 4, length=100)

# start with point (0, 0) and then expand context set
x0 <- c(0)
y0 <- 1*sin(x0)
p1 <- plot_posterior_draws(x0, y0, x_star, n_draws = 50L)

x0 <- c(0, 1)
y0 <- 1*sin(x0)
p2 <- plot_posterior_draws(x0, y0, x_star, n_draws = 50L)

x0 <- c(0, 1, -1, 2, -2)
y0 <- 1*sin(x0)
p3 <- plot_posterior_draws(x0, y0, x_star, n_draws = 50L)

p <- (p1 + p2 + p3) * coord_cartesian(ylim = c(-1, 1)) + plot_layout(nrow = 1)
p
ggsave("fig/experiment2_pred.png", p, width = 9, height = 3)


# generalisation / model misspecification
x0 <- seq(-2, 2, length=10)
y0 <- 2.5*sin(x0)
p1 <- plot_posterior_draws(x0, y0, x_star, n_draws = 50L) +
  labs(subtitle = "2.5 sin(x)")

y0 <- abs(sin(x0))
p2 <- plot_posterior_draws(x0, y0, x_star, n_draws = 50L) +
  coord_cartesian(ylim = c(-1, 1)) +
  labs(subtitle = "abs(sin(x))")


p <- (p1 + p2) + plot_layout(nrow = 1)
ggsave("fig/experiment2_misspecification.png", p, width = 7, height = 2.5)
