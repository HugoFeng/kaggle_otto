library(nnet)
library(e1071)
library('tree')
library('lazy')
set.seed(0)

using.tree <- function(trainX, trainY){
    print(paste("Using Decision tree."))
    trainXY.combined <- cbind(trainX, trainY)
    model <- tree(trainY~., trainXY.combined)
}

using.nnet <- function(trainX, trainY){
    hiddenLayer_size <- 10
    iterations <- 50
    print("Using Neural Network.")
    print(paste("Hidden Layer size: ", hiddenLayer_size))
    print(paste("Max iterations: ", iterations))
    
    trainY_matrix = class.ind(trainY)
    model <- nnet(x=trainX, y=trainY_matrix, size = hiddenLayer_size,  mmaxit = iterations, rang = 0.1, decay = 5e-4, MaxNWts=1000000, linout = FALSE, trace=FALSE)
}
# Load data set
inputData <- read.csv('../data/train.csv')
Num.train.totalSize <- dim(inputData)[1]
Num.train.totalCol <- dim(inputData)[2]
X <- inputData[, 2:(Num.train.totalCol-1)]
Y <- inputData[, Num.train.totalCol]
# Load test data set
testData <- read.csv('../data/test.csv')
testData <- testData[,-1]
Num.test.totalSize <- dim(testData)[1]
Num.test.totalCol <- dim(testData)[2]

###### model training and prediction
## train can be given as: using.nnet, using.tree
train <- using.tree
model <- train(X, Y)
print(paste("Predicting test data."))
result <- predict(model, testData)
maxindices <- max.col(result)
for(row in 1:Num.test.totalSize){
    result[row, ] <- rep(0, 9)
    result[row, maxindices[row]] <- 1
}

write.csv(result, file = "../data/result.csv")

