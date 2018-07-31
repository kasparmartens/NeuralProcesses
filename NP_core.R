# create and return all weight tensors used in the NP model
init_weights <- function(dim_r = 8L, dim_z = dim_r, dim_h_hidden = 32L, dim_g_hidden = 32L){
  # weights for the encoder h, mapping (x, y) -> hidden -> r
  W_h1 <- tf$Variable(tf$random_normal(shape(2L, dim_h_hidden)))
  W_h2 <- tf$Variable(tf$random_normal(shape(dim_h_hidden, dim_r)))
  
  # weights for mapping r to (mu_z, sigma_z)
  W_z_mu <- tf$Variable(tf$random_normal(shape(dim_r, dim_z)))
  W_z_sigma <- tf$Variable(tf$random_normal(shape(dim_r, dim_z)))
  
  # weights for the decoder g, mapping (z, x) -> hidden -> y
  W_g1 <- tf$Variable(tf$random_normal(shape(dim_z + 1L, dim_g_hidden)))
  W_g2 <- tf$Variable(tf$random_normal(shape(dim_g_hidden, 1L)))
  
  # return all weights
  list(W_h1 = W_h1,
       W_h2 = W_h2, 
       W_z_mu = W_z_mu, 
       W_z_sigma = W_z_sigma, 
       W_g1 = W_g1, 
       W_g2 = W_g2, 
       dim_z = dim_z)
}

# helper function to map (x, y) -> z directly without intermediate steps
map_xy_to_z_params <- function(x, y, weights){
  list(x, y) %>%
    tf$concat(axis = 1L) %>%
    h(weights$W_h1, weights$W_h2) %>%
    aggregate_r() %>%
    get_z_params(weights$W_z_mu, weights$W_z_sigma)
}

# set up the NN architecture with train_op and loss
init_NP <- function(weights, x_context, y_context, x_target, y_target){
  
  # concatenate context and target
  x_all <- tf$concat(list(x_context, x_target), axis = 0L)
  y_all <- tf$concat(list(y_context, y_target), axis = 0L)
  
  # map input to z
  z_context <- map_xy_to_z_params(x_context, y_context, weights)
  z_all <- map_xy_to_z_params(x_all, y_all, weights)
  
  # sample z using reparametrisation, z = mu + sigma*eps
  epsilon <- tf$random_normal(shape(7L, weights$dim_z))
  z_sample <- epsilon %>%
    tf$multiply(z_all$sigma) %>%
    tf$add(z_all$mu)
  
  # map (z, x*) to y*
  y_pred_params <- g(z_sample, x_target, weights$W_g1, weights$W_g2)
  
  # ELBO
  loglik <- loglikelihood(y_target, y_pred_params)
  KL_loss <- KLqp_gaussian(z_all$mu, z_all$sigma, z_context$mu, z_context$sigma)
  loss <- tf$negative(loglik) + KL_loss
  
  # optimisation
  optimizer <- tf$train$AdamOptimizer()
  train_op <- optimizer$minimize(loss)
  
  # return train_op and loss
  list(train_op, loss)
}

prior_predict <- function(weights, x_star_value, n_draws = 1L){
  N_star <- nrow(x_star_value)
  x_star <- tf$constant(x_star_value, dtype = tf$float32)
  
  # draw z ~ N(0, 1)
  z_sample <- tf$random_normal(shape(n_draws, weights$dim_z))
  # y ~ g(z, x*)
  y_star <- g(z_sample, x_star, weights$W_g1, weights$W_g2)
  
  y_star
}


posterior_predict <- function(weights, x, y, x_star_value, epsilon = NULL, n_draws = 1L){
  # inputs for prediction time
  x_obs <- tf$constant(x, dtype = tf$float32)
  y_obs <- tf$constant(y, dtype = tf$float32)
  x_star <- tf$constant(x_star_value, dtype = tf$float32)
  
  # for out-of-sample new points
  z_params <- map_xy_to_z_params(x_obs, y_obs, weights)
  
  # the source of randomness can be optionally passed as an argument
  if(is.null(epsilon)){
    epsilon <- tf$random_normal(shape(n_draws, weights$dim_z))
  }
  # sample z using reparametrisation
  z_sample <- epsilon %>%
    tf$multiply(z_params$sigma) %>%
    tf$add(z_params$mu)
  
  # predictions
  y_star <- g(z_sample, x_star, weights$W_g1, weights$W_g2)
  
  y_star
}

