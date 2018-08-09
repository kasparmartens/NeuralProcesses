# Architecture for the Neural Process

# encoder h -- map inputs (x_i, y_i) to r_i
h <- function(input){
  input %>%
    tf$layers$dense(dim_h_hidden, tf$nn$sigmoid, name = "encoder_layer1", reuse = tf$AUTO_REUSE) %>%
    tf$layers$dense(dim_r, name = "encoder_layer2", reuse = tf$AUTO_REUSE)
}

# aggregate the output of h (i.e. values of r_i) to a single vector r
aggregate_r <- function(input){
  input %>%
    tf$reduce_mean(axis=0L) %>%
    tf$reshape(shape(1L, -1L))
}

# map aggregated r to (mu_z, sigma_z)
get_z_params <- function(input_r){
  
  hidden <- input_r
  # hidden <- input_r %>%
  #   tf$layers$dense(dim_r, tf$nn$sigmoid, name = "z_params_layer1", reuse = tf$AUTO_REUSE)
  
  mu <- hidden %>%
    tf$layers$dense(dim_z, name = "z_params_mu", reuse = tf$AUTO_REUSE)
  
  sigma <- hidden %>%
    tf$layers$dense(dim_z, name = "z_params_sigma", reuse = tf$AUTO_REUSE) %>%
    tf$nn$softplus()
  
  list(mu = mu, sigma = sigma)
}


# decoder g -- map (z, x*) -> hidden -> y*
g <- function(z_sample, x_star, noise_sd = 0.05){
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
  
  # hidden layer
  hidden <- input %>%
    tf$layers$dense(dim_g_hidden, tf$nn$sigmoid, name = "decoder_layer1", reuse = tf$AUTO_REUSE)
  
  # mu will be of the shape [N_star, n_draws]
  mu_star <- hidden %>%
    tf$layers$dense(1L, name = "decoder_layer2", reuse = tf$AUTO_REUSE) %>%
    tf$squeeze(axis = 2L) %>%
    tf$transpose()
  
  # for the toy example, assume y* ~ N(mu, sigma) with fixed sigma
  sigma_star <- tf$constant(noise_sd, dtype = tf$float32)
  
  list(mu = mu_star, sigma = sigma_star)
}

