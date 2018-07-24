
init_weights <- function(dim_r = 32L, dim_z = dim_r, dim_hidden = 32L){
  # weights for the encoder h
  W_h <- tf$Variable(tf$random_normal(shape(2L, dim_r), mean = 0, stddev = 1))
  
  # weights for mapping r to (mu_z, sigma_z)
  W_z_mu <- tf$Variable(tf$random_normal(shape(dim_r, dim_z), mean = 0, stddev = 1))
  W_z_sigma <- tf$Variable(tf$random_normal(shape(dim_r, dim_z), mean = 0, stddev = 1))
  
  # weights for the decoder g
  W_g_mu <- tf$Variable(tf$random_normal(shape(dim_z + 1L, dim_hidden), mean = 0, stddev = 1))
  W_g_sigma <- tf$Variable(tf$random_normal(shape(dim_hidden, 1L), mean = 0, stddev = 1))
  
  # return all weights
  list(W_h = W_h, 
       W_z_mu = W_z_mu, 
       W_z_sigma = W_z_sigma, 
       W_g_mu = W_g_mu, 
       W_g_sigma = W_g_sigma, 
       dim_z = dim_z)
}

init_NP <- function(weights, N_context, N_target){
  
  # inputs for training time
  x_context <- tf$placeholder(tf$float32, shape(N_context, 1))
  y_context <- tf$placeholder(tf$float32, shape(N_context, 1))
  x_target <- tf$placeholder(tf$float32, shape(N_target, 1))
  y_target <- tf$placeholder(tf$float32, shape(N_target, 1))
  
  # concatenate context and target
  x_all <- tf$concat(list(x_context, x_target), axis = 0L)
  y_all <- tf$concat(list(y_context, y_target), axis = 0L)
  
  # map input to z
  z_context <- list(x_context, y_context) %>%
    h(weights$W_h) %>%
    aggregate_r() %>%
    get_z_params(weights$W_z_mu, weights$W_z_sigma)
  
  z_context_and_target <- list(x_all, y_all) %>%
    h(weights$W_h) %>%
    aggregate_r() %>%
    get_z_params(weights$W_z_mu, weights$W_z_sigma)
  
  # sample z using reparametrisation, z = mu + sigma*eps
  epsilon <- tf$random_normal(shape(1L, weights$dim_z))
  z_sample <- epsilon %>%
    tf$multiply(z_context_and_target$sigma) %>%
    tf$add(z_context_and_target$mu)
  
  # map (z, x*) to y*
  y_pred_params <- g(z_sample, x_target, weights$W_g_mu, weights$W_g_sigma)
  
  # ELBO
  loglik <- loglikelihood(y_target, y_pred_params)
  KL_loss <- KLqp_gaussian(z_context_and_target$mu, z_context_and_target$sigma, z_context$mu, z_context$sigma)
  loss <- tf$negative(loglik) + KL_loss
  
  # optimisation
  optimizer <- tf$train$AdamOptimizer()
  train_op <- optimizer$minimize(loss)
  
  # return train_op and loss
  list(train_op = train_op, 
       loss = loss)
}


condition_and_predict <- function(weights, data_test, n_draws = 10L){
  # inputs for prediction time
  N_obs <- nrow(data_test$x)
  N_star <- nrow(data_test$x_star)
  x_obs <- tf$placeholder(tf$float32, shape(N_obs, 1))
  y_obs <- tf$placeholder(tf$float32, shape(N_obs, 1))
  x_star <- tf$placeholder(tf$float32, shape(N_star, 1))
  
  # for out-of-sample new points
  z_params <- list(x_obs, y_obs) %>%
    h(weights$W_h) %>%
    aggregate_r() %>%
    get_z_params(weights$W_z_mu, weights$W_z_sigma)
  # reparametrisation
  epsilon <- tf$random_normal(shape(1L, weights$dim_z))
  z_sample <- epsilon %>%
    tf$cast(tf$float32) %>%
    tf$multiply(z_params$sigma) %>%
    tf$add(z_params$mu)
  
  y_star <- g(z_sample, x_star, weights$W_g_mu, weights$W_g_sigma)
  
  feed_dict_test <- dict(
    x_obs = data_test$x, 
    y_obs = data_test$y, 
    x_star = data_test$x_star
  )
  
  df_pred_list <- list()
  
  for(i in 1:n_draws){
    y_star_pred <- sess$run(y_star$mu, feed_dict = feed_dict_test)
    
    df_pred_list[[i]] <- data.frame(x = as.numeric(data_test$x_star), y = y_star_pred, rep = i)
  }
  
  df_pred <- bind_rows(df_pred_list)
  df_pred
}
