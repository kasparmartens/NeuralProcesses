library(tidyverse)
library(tensorflow)
library(patchwork)

source("NP_core.R")
source("GP_helpers.R")
source("helpers_for_plotting.R")
source("NP_architecture1.R")

# generate data
N <- 5
x <- c(-2, -1, 0, 1, 2)
y <- sin(x)
df_obs <- data.frame(x, y)

# plot data
p0 <- df_obs %>%
  ggplot(aes(x, y)) +
  geom_point(col = "#377EB8", size=3) +
  theme_classic() +
  coord_cartesian(xlim = c(-3, 3)) +
  labs(title = "Observed data")
p <- plot_spacer() + p0 + plot_spacer()
# ggsave("fig/observed_data.png", p, width=10, height=2.5)


### Now fitting the NP

# global variables for training the model
dim_r <- 2L
dim_z <- 2L
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

n_iter <- 5000
plot_freq <- 200

# Plotting functionality (to record training)
# (using fixed source of randomness over all iters)
n_draws <- 50L
x_star <- seq(-4, 4, length=100)
eps_value <- matrix(rnorm(n_draws*dim_r), n_draws, dim_r)
epsilon <- tf$constant(eps_value, dtype = tf$float32)
predict_op <- posterior_predict(cbind(x), cbind(y), cbind(x_star), epsilon)



df_pred_list <- list()
for(iter in 1:n_iter){
  N_context <- sample(1:4, 1)
  # create feed_dict containing context and target sets
  feed_dict <- helper_context_and_target(x, y, N_context, x_context, y_context, x_target, y_target)
  # optimisation step
  a <- sess$run(train_op_and_loss, feed_dict = feed_dict)
  
  # plotting
  if(iter %% plot_freq == 0){
    y_star_mat <- sess$run(predict_op$mu)
    df_pred <- y_star_mat %>% 
      reshape_predictions(x_star) %>%
      mutate(iter = iter)
    df_pred_list[[iter]] <- df_pred
  }
}
df_pred <- bind_rows(df_pred_list)


# gif

library(gganimate)

df_obs_rep <- crossing(df_obs, 
                       iter = df_pred$iter, 
                       rep_index = df_pred$rep_index)

p <- df_pred %>%
  ggplot(aes(x, y)) +
  geom_line(aes(group=rep_index), alpha = 0.2) +
  geom_point(data = df_obs_rep, col = "#377EB8") +
  transition_time(iter) +
  labs(title = "Training a Neural Process", subtitle = "Iteration: {frame_time}", y = "Function value") +
  coord_cartesian(ylim = c(-2, 2)) +
  theme_classic()

animate(p, nframes = 50, width=400, height=250)

anim_save("fig/experiment1.gif")


# prediction for a different set of context points
y2 <- 1 + sin(x)
df_obs2 <- data.frame(x, y = y2)
predict_op2 <- posterior_predict(weights, cbind(x), cbind(y2), cbind(x_star), epsilon)

y_star_mat <- sess$run(predict_op2$mu)
df_pred2 <- y_star_mat %>% 
  reshape_predictions(x_star)

df_pred2 %>%
  ggplot(aes(x, y)) +
  geom_line(aes(group=rep_index), alpha = 0.2) +
  geom_point(aes(col = "Context points at training time"), data = df_obs, size=3) +
  geom_point(aes(col = "Context points at prediction time"), data = df_obs2, size=3) +
  scale_color_brewer("", palette = "Set1") +
  theme_classic()

ggsave("fig/experiment1.png", width = 8, height = 3)
