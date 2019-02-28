### SOFTMAX IMPLEMENTATION 

#Iris data set including 4 features and 3 different target
x<- as.matrix(iris[,1:4])

#One hot encoding
y<- model.matrix(~Species-1, iris)
y<- as.matrix(as.data.frame(y))


#Softmax for multiple classification
softmax <- function(x) {
  exp(x)/sum(exp(x))
}

#Leaky Rectified Linear Unit
lyrelu <- function(x) {
  ifelse(x>=0, x, 0.01*x)
}

#Leaky RELU derivative
dlyrelu <- function(x) {
  ifelse(x>=0, 1, 0.01)
}

#credit goes to David Selby for feedforward, backpropagate, and train (source: https://selbydavid.com/2018/01/09/neural-network/)
#I have made significant modifications to these functions to expand from binary classification to multiple classification.
feedforward <- function(x, w1, w2) {
  z1 <- cbind(1, x) %*% w1
  h <- lyrelu(z1)
  z2 <- cbind(1, h) %*% w2
  list(output = t(apply(z2, 1, softmax)), h = h)
}

backpropagate <- function(x, y, y_hat, w1, w2, h, learn_rate) {
  dw2 <- t(cbind(1, h)) %*% (y_hat - y)
  dh  <- (y_hat - y) %*% t(w2[-1, , drop = FALSE])
  dw1 <- t(cbind(1, x)) %*% (dlyrelu(h)*dh)
  
  w1 <- w1 - learn_rate * dw1
  w2 <- w2 - learn_rate * dw2
  
  list(w1 = w1, w2 = w2)
}

train <- function(x, y, hidden, learn_rate, iterations) {
  d <- ncol(x) + 1
  w1 <- matrix(rnorm(d * hidden), d, hidden)
  w2 <- matrix(rnorm((hidden + 1) * ncol(y)), hidden + 1, ncol(y))
  for (i in 1:iterations) {
    ff <- feedforward(x, w1, w2)
    bp <- backpropagate(x, y,
                        y_hat = ff$output,
                        w1, w2,
                        h = ff$h,
                        learn_rate = learn_rate)
    w1 <- bp$w1; w2 <- bp$w2
  }
  list(output = ff$output, w1 = w1, w2 = w2)
}

#Training the Softmax neural network with 10 hidden layers over 100,000 iterations. 
#Used only two features here for now, for visualization purposes. 
nnet10 <- train(x[,1:2], y, hidden=10, learn_rate =1e-4, iterations=1e5) 

#Model accuracy (roughly 83% on the entire data set with 2 features)
mean(apply(nnet10$output, 1, which.max) == apply(y, 1, which.max))

#We can do better by including all 4 features rather than just two.
