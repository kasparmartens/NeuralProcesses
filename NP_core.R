
# helper function to map (x, y) -> z directly without intermediate steps
map_xy_to_z_params <- function(x, y){
  list(x, y) %>%
    tf$concat(axis = 1L) %>%
    h() %>%
    aggregate_r() %>%
    get_z_params()
}

# set up the NN architecture with train_op and loss
init_NP <- function(x_context, y_context, x_target, y_target, learning_rate = 0.001){
  
  # concatenate context and target
  x_all <- tf$concat(list(x_context, x_target), axis = 0L)
  y_all <- tf$concat(list(y_context, y_target), axis = 0L)
  
  # map input to z
  z_context <- map_xy_to_z_params(x_context, y_context)
  z_all <- map_xy_to_z_params(x_all, y_all)
  
  # sample z using reparametrisation, z = mu + sigma*eps
  epsilon <- tf$random_normal(shape(7L, dim_z))
  z_sample <- epsilon %>%
    tf$multiply(z_all$sigma) %>%
    tf$add(z_all$mu)
  
  # map (z, x*) to y*
  y_pred_params <- g(z_sample, x_target)
  
  # ELBO
  loglik <- loglikelihood(y_target, y_pred_params)
  KL_loss <- KLqp_gaussian(z_all$mu, z_all$sigma, z_context$mu, z_context$sigma)
  loss <- tf$negative(loglik) + KL_loss
  
  # optimisation
  optimizer <- tf$train$AdamOptimizer(learning_rate)
  train_op <- optimizer$minimize(loss)
  
  # return train_op and loss
  list(train_op, loss)
}

prior_predict <- function(x_star_value, epsilon = NULL, n_draws = 1L){
  N_star <- nrow(x_star_value)
  x_star <- tf$constant(x_star_value, dtype = tf$float32)
  
  # the source of randomness can be optionally passed as an argument
  if(is.null(epsilon)){
    epsilon <- tf$random_normal(shape(n_draws, dim_z))
  }
  # draw z ~ N(0, 1)
  z_sample <- epsilon
  
  # y ~ g(z, x*)
  y_star <- g(z_sample, x_star)
  
  y_star
}


posterior_predict <- function(x, y, x_star_value, epsilon = NULL, n_draws = 1L){
  # inputs for prediction time
  x_obs <- tf$constant(x, dtype = tf$float32)
  y_obs <- tf$constant(y, dtype = tf$float32)
  x_star <- tf$constant(x_star_value, dtype = tf$float32)
  
  # for out-of-sample new points
  z_params <- map_xy_to_z_params(x_obs, y_obs)
  
  # the source of randomness can be optionally passed as an argument
  if(is.null(epsilon)){
    epsilon <- tf$random_normal(shape(n_draws, dim_z))
  }
  # sample z using reparametrisation
  z_sample <- epsilon %>%
    tf$multiply(z_params$sigma) %>%
    tf$add(z_params$mu)
  
  # predictions
  y_star <- g(z_sample, x_star)
  
  y_star
}

# KLqp helper
KLqp_gaussian <- function(mu_q, sigma_q, mu_p, sigma_p){
  sigma2_q <- tf$square(sigma_q) + 1e-16
  sigma2_p <- tf$square(sigma_p) + 1e-16
  temp <- sigma2_q / sigma2_p + tf$square(mu_q - mu_p) / sigma2_p - 1.0 + tf$log(sigma2_p / sigma2_q + 1e-16)
  0.5 * tf$reduce_sum(temp)
}

# for ELBO
loglikelihood <- function(y_star, y_pred_params){
  
  p_normal <- tf$distributions$Normal(loc = y_pred_params$mu, scale = y_pred_params$sigma)
  
  loglik <- y_star %>%
    p_normal$log_prob() %>%
    # sum over data points
    tf$reduce_sum(axis=0L) %>%
    # average over n_draws
    tf$reduce_mean()
  
  loglik
}


# for training
helper_context_and_target <- function(x, y, N_context, x_context, y_context, x_target, y_target){
  N <- length(y)
  context_set <- sample(1:N, N_context)
  dict(
    x_context = cbind(x[context_set]), 
    y_context = cbind(y[context_set]),
    x_target = cbind(x[-context_set]), 
    y_target = cbind(y[-context_set])
  )
}

