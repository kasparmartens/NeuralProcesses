reshape_predictions <- function(y_star_mat, x_star){
  y_star_mat %>%
    reshape2::melt() %>%
    rename(index = Var1, rep_index = Var2, y = value) %>%
    mutate(x = x_star[index]) %>%
    select(-index)
}

plot_posterior_draws <- function(x, y, x_star, n_draws = 50L){
  df_obs <- data.frame(x = x, y = y)
  predict_op <- posterior_predict(weights, cbind(x), cbind(y), cbind(x_star), n_draws = n_draws)
  y_star_mat <- sess$run(predict_op$mu)
  df_pred <- reshape_predictions(y_star_mat, x_star)
  
  df_pred %>%
    ggplot(aes(x, y)) +
    geom_line(aes(group=rep_index), alpha = 0.2) +
    geom_point(data = df_obs, col = "#b2182b", size = 3) +
    theme_classic()
}