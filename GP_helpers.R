
rbf_kernel <- function(X, Y, ls, var){
  X <- X / ls
  Y <- Y / ls
  N1 <- nrow(X)
  N2 <- nrow(Y)
  s1 <- rowSums(X**2)
  s2 <- rowSums(Y**2)
  square <- matrix(s1, N1, N2) + matrix(s2, N1, N2, byrow=T) - 2 * X %*% t(Y)
  var * exp(-0.5 * square)
}

