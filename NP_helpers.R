# Helper functions for Neural Processes

# encoder h -- map inputs (x_i, y_i) to r_i
h <- function(input, W1, W2){
  input %>%
    tf$matmul(W1) %>%
    tf$nn$sigmoid() %>%
    tf$matmul(W2)
}

# aggregate the output of h (i.e. values of r_i) to a single vector r
aggregate_r <- function(input){
   input %>%
    tf$reduce_mean(axis=0L) %>%
    tf$reshape(shape(1L, -1L))
}

# map aggregated r to (mu_z, sigma_z)
get_z_params <- function(input_r, W_mu, W_sigma){
  mu <- input_r %>%
    tf$matmul(W_mu)
  
  sigma <- input_r %>%
    tf$matmul(W_sigma) %>%
    tf$nn$softplus()
  
  list(mu = mu, sigma = sigma)
}


# decoder g -- map (z, x*) -> hidden -> y*
g <- function(z_sample, x_star, W1, W2){
  # inputs dimensions
  # z_sample has dim [n_draws, dim_z]
  # x_star has dim [N_star, dim_x]
  
  n_draws <- z_sample$get_shape()$as_list()[1]
  N_star <- tf$shape(x_star)[1]
  
  # z_sample_rep will have dim [n_draws, N_star, dim_z]
  z_sample_rep <- z_sample %>%
    tf$expand_dims(axis = 1L) %>%
    tf$tile(c(1L, N_star, 1L))
  
  # x_star_rep will have dim [n_draws, N_star, dim_x]
  x_star_rep <- x_star %>%
    tf$expand_dims(axis = 0L) %>%
    tf$tile(shape(n_draws, 1L, 1L))
  
  # concatenate x* and z
  input <- list(x_star_rep, z_sample_rep) %>%
    tf$concat(axis = 2L)
  
  # batch matmul
  W1_rep <- W1 %>%
    tf$expand_dims(axis=0L) %>%
    tf$tile(shape(n_draws, 1L, 1L))
  
  W2_rep <- W2 %>%
    tf$expand_dims(axis=0L) %>%
    tf$tile(shape(n_draws, 1L, 1L))
  
  # hidden layer
  hidden <- input %>%
    tf$matmul(W1_rep) %>%
    tf$nn$sigmoid()
  
  # mu will be of the shape [N_star, n_draws]
  mu_star <- hidden %>%
    tf$matmul(W2_rep) %>%
    tf$squeeze(axis = 2L) %>%
    tf$transpose()
  
  # for the toy example, assume y* ~ N(mu, sigma) with fixed sigma = 0.05
  sigma_star <- tf$constant(0.05, dtype = tf$float32)
  
  list(mu = mu_star, sigma = sigma_star)
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

