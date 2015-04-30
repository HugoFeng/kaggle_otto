# Load data set
inputData <- read.csv('../data/train.csv')
#head(inputData)
Num.totalSize <- dim(inputData)[1]
Num.totalCol <- dim(inputData)[2]
library(nnet)
library(e1071)
library('tree')
library('lazy')

set.seed(0)

Num.ts.size <- round(Num.totalSize/10)
Num.tr.size <- Num.totalSize-Num.ts.size

# Create a training and a test set
i.ts <- sample(1:Num.totalSize, Num.ts.size)
i.tr <- setdiff(1:Num.totalSize, i.ts)

X.ts <- inputData[i.ts, 2:(Num.totalCol-1)]
X.tr <- inputData[i.tr, 2:(Num.totalCol-1)]
Y.ts <- inputData[i.ts, Num.totalCol]
Y.tr <- inputData[i.tr, Num.totalCol]
Num.labels <- length(levels(Y.ts))
Y.tsi <- as.numeric(sub('Class_', '', Y.ts))
Y.tri <- as.numeric(sub('Class_', '', Y.tr))

# Shuffling training data
index <- sample(1:Num.tr.size)
X.tr <- X.tr[index, ]
Y.tr <- Y.tr[index]
Y.tri <- Y.tri[index]

using.tree <- function(trainX, trainY){
    trainXY.combined <- cbind(trainX, trainY)
    model <- tree(trainY~., trainXY.combined)
}

using.nnet <- function(trainX, trainY){
    hiddenLayer_size <- 8
    iterations <- 200
    print(paste("Hidden Layer size: ", hiddenLayer_size))
    print(paste("Max iterations: ", iterations))
    trainY_matrix = class.ind(trainY)
    model <- nnet(trainX, trainY_matrix, , size = hiddenLayer_size,  mmaxit = iterations, rang = 0.1, decay = 5e-4, MaxNWts=1000000, linout = FALSE, trace=FALSE)
}

BERate_list = NULL
mis_error_list = NULL
logloss_list = NULL
# for(hiddenLayer_size in 1:15)
# {
# Building a Neural Network


model <- using.nnet(X.tr, Y.tr)
result <- predict(model, X.ts)
result.label <- max.col(result)

# #Buiding SVM # TODO: still have problems
# model<-svm(Y.tri,X.tr)
# result.label <- predict(model, X.ts)

# # Building Tree
# model <- using.tree(X.tr, Y.tr)
# result<-predict(model, X.ts)
# result.label <- max.col(result)

# # Building KNN model with 'lazy' package # has problem
# model <- lazy(Y.tr~., trainXY.combined)
# result<- predict(model, X.ts)
# result.label <- max.col(result)

# # accuracy of classification
#accuracy <- mean(Y.tsi==result.label)
#print(paste("Classification accuracy: ", accuracy))

# mis-classification error
mis_error <- mean(Y.tsi!=result.label)
mis_error_list <- c(mis_error_list, mis_error)
print(paste("Mis-Classification error rate: ", mis_error))

# balanced error rate
## The index of column means the predicted results,
## while the index of row means the real class label.
BER_matrix <- matrix(0, nrow=Num.labels, ncol=Num.labels)
## For each sample and predicted results
for(sampleIndex in 1:Num.ts.size) {
    realLabel <- Y.tsi[sampleIndex]
    predictedLabel <- result.label[sampleIndex]
    BER_matrix[predictedLabel, realLabel] = BER_matrix[predictedLabel, realLabel] + 1
}
BERate <- 0
for(label in 1:Num.labels) {
    Tnum <- BER_matrix[label, label]
    TFnum <- sum(BER_matrix[,label])
    BERate <- BERate + Tnum/TFnum
}
BERate <- BERate/Num.labels
BERate_list <- c(BERate_list, BERate)
print(paste("balanced error rate: ", BERate))

# logloss
logloss <- 0
for(sampleIndex in Num.ts.size) {
    outputs <- result[sampleIndex, ]
    # Probability of being classified as one of the labels
    p_i <- outputs/sum(outputs)
    # avoid 0 probability
    predictedLabel <- result.label[sampleIndex]
    p_ij <- max(min(p_i[predictedLabel], 1-10^-15), 10^-15)
    log_p_ij <- log(p_ij)
    logloss <- logloss + log_p_ij
}
logloss <- -logloss/Num.ts.size
logloss_list <- c(logloss_list, logloss)
print(paste("logloss: ", logloss))

# } #line 34
# x <- 1:15
# par(mfrow=c(1,1))
# #plot(c(1, 15), c(0, 1), type="n")
# plot(x, BERate_list, xlim=c(1, 15), ylim=c(0, 1), type="l", col="red")
# lines(x, mis_error_list, col="blue")
# lines(x, logloss_list, col="black")