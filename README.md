# Neural-Network-Example-in-R
Softmax neural network example with leaky RELU hidden layer.

Credit for the following functions in this repository goes to David Selby (source: https://selbydavid.com/2018/01/09/neural-network/):

feedforward, backpropagate, train

I have made my own modifications to these functions to expand from binary classification to multiple classification.

This repository contains a multiple classification neural network created without the use of packages and only base R functions. I am using the built in Iris data set and have shown the neural network's performance when only using two features (Sepal length and width) out of the 4, for visualization purposes. In the repo you will find the R code, and two images showcasing the neural network's learned decision boundaries and where the actual target variables fall within them.

In a future update, I will show the NN model's accuracy with all 4 features.


