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
  N_star <- x_star$get_shape()$as_list()[1]
  
  z_sample_rep <- z_sample %>%
    tf$tile(shape(N_star, 1L))
  
  input <- list(x_star, z_sample_rep) %>%
    tf$concat(axis = 1L)
  
  hidden <- input %>%
    tf$matmul(W1) %>%
    tf$nn$sigmoid()
  
  mu_star <- hidden %>%
    tf$matmul(W2) %>%
    tf$reshape(shape(-1L))
  
  # for the toy example, assume y* ~ N(mu, sigma) with fixed sigma = 0.1
  sigma_star <- tf$constant(0.1, dtype = tf$float32)
  
  list(mu = mu_star, sigma = sigma_star)
}

# sample from Gaussian helper
sample_gaussian <- function(obj, shape){
  obj$mu + obj$sigma * tf$random_normal(shape)
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
    tf$reshape(shape(-1L)) %>%
    p_normal$log_prob() %>%
    tf$reduce_sum()
}
