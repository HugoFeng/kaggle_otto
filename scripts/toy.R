library(nnet)
library(e1071)
library('tree')
library('randomForest')
set.seed(0)
par(mfrow=c(1,1))

using.tree <- function(trainX, trainY, ...){
    print("Training Decision tree.")
    trainXY.combined <- cbind(trainX, trainY)
    model <- tree(trainY~., data=trainXY.combined, ...)
}

using.randomForest <- function(trainX, trainY, ...){
    print("Training Forest.")
    trainXY.combined <- cbind(trainX, trainY)
    model <- randomForest(trainY~., data=trainXY.combined, ...)
}

using.svm <- function(trainX, trainY, ...){
    print("Training svm.")
    trainXY.combined <- cbind(trainX, trainY)
    model <- svm(trainY~., data=trainXY.combined, ...)
}

using.nnet <- function(trainX, trainY, ...){
    hiddenLayer_size <- 10
    iterations <- 50
    print("Training Neural Network.")
    print(paste("   Hidden Layer size:", hiddenLayer_size))
    print(paste("   Max iterations:", iterations))
    
    trainY_matrix = class.ind(trainY)
    model <- nnet(x=trainX, y=trainY_matrix, size = hiddenLayer_size, mmaxit = iterations, 
                  rang = 0.1, decay = 5e-4, MaxNWts=1000000, linout = FALSE, trace=FALSE, ...)
}

using.bagging <- function(train, trainX, trainY, testX, model.num=5, ...) {
    print(paste("Using Bagging with", model.num, "models."))
    train.num <- dim(trainX)[1]
    if(model.num>1)
        # If number of models is greater than 1, 
        # always take 1/3 more than the average part of training samples.
        train.sub.num <- floor(4./3 * train.num/model.num)
    else
        train.sub.num <- floor(train.num/model.num)
    test.num <- dim(testX)[1]
    result <- data.frame(matrix(0, nrow=test.num, ncol=9))
    for(model.index in 1:model.num){
        train.sub.indices <- sample(1:train.num, train.sub.num)
        train.sub.x <- trainX[train.sub.indices, ]
        train.sub.y <- trainY[train.sub.indices]
        model <- train(train.sub.x, train.sub.y, ...)
        result.tmp = predict(model, testX)
        if(is.factor(result.tmp)){
            result.label <- as.numeric(sub('Class_', '', result.tmp))
        }else{
            result.label <- max.col(result.tmp)
        }
        # for each test sample, accumulate the predicted label vote in the result data frame
        for(test.index in 1:test.num){
            label <- result.label[test.index]
            result[test.index, label] <- result[test.index, label] + 1
        }
    }
    result
}

pca.projecter <- function(pca, original, feature_size){
    projected <- predict(pca, original)
    subtract <- projected[,1:feature_size]
    subtract <- data.frame(subtract)
}

# Computing logloss for a predicted result
compute.logloss <- function(result.df, target.label, test.size, label.size=9){
    logloss <- 0
    # For some classifiers, the predicted results only contains predicted labels, 
    # so we need to construct a matrix just like class.ind() did.
    if(is.factor(result.df)){
        result.matrix <- matrix(0, nrow=label.size, ncol=label.size)
        result.label <- as.numeric(sub('Class_', '', result.df))
        for(sample.index in 1:test.size){
            predictedLabel <- result.label[sample.index]
            result.matrix[sample.index, predictedLabel] <- 1
        }
        result.df <- result.matrix
    } 
    for(sample.index in 1:test.size) {
        outputs <- result.df[sample.index, ]
        # Probability of being classified as one of the labels
        p.i <- outputs/sum(outputs)
        
        targetLabel <- target.label[sample.index]
        # avoid 0 probability
        p.ij <- max(min(p.i[targetLabel], 1-10^-15), 10^-15)
        log_p.ij <- log(p.ij)
        logloss <- logloss + log_p.ij
    }
    logloss <- -logloss/test.size
}

# Computing Balanced Error rate
compute.BERate <- function(result.label, target.label, test.size, label.size){
    ###### Balanced error rate
    ## The index of column means the predicted results,
    ## while the index of row means the real class label.
    BER.matrix <- matrix(0, nrow=label.size, ncol=label.size)
    ## For each sample and predicted results
    for(sample.index in 1:test.size) {
        realLabel <- target.label[sample.index]
        predictedLabel <- result.label[sample.index]
        BER.matrix[predictedLabel, realLabel] = BER.matrix[predictedLabel, realLabel] + 1
    }
    BERate <- 0
    for(label in 1:label.size) {
        Tnum <- BER.matrix[label, label]
        TFnum <- sum(BER.matrix[,label])
        BERate <- BERate + Tnum/TFnum
    }
    BERate <- BERate/label.size
}

# Prepare plot
plot(seq(1, 100), rep(0, 100), ylim = c(0, 1.0), axes=FALSE, main="Bagging with Decision tree", 
     type="n", ylab = "Misclassification Rate", xlab = "Feature Size")
axis(side=1, at=seq(0, 100, by=10))
axis(side=2, at=seq(0, 1, by=0.1))
box()

# Load data set
inputData <- read.csv('../data/train.csv')
Num.totalSize <- dim(inputData)[1]
Num.totalCol <- dim(inputData)[2]

# Shuffle the row order of the original data
shuffleIndeces <- sample(1:Num.totalSize)
inputData <- inputData[shuffleIndeces, ]

Num.folds <- 3 # K-fold cross-validation
Num.labels <- length(levels(inputData[, Num.totalCol]))
Num.test.size <- floor(Num.totalSize/Num.folds)
fold.size <- floor(Num.totalSize/Num.folds)
Num.train.size <- Num.totalSize-Num.test.size

all.x <- inputData[, 2:(Num.totalCol-1)]
all.y <- inputData[, Num.totalCol]

features.pca <- prcomp(all.x, center = TRUE, scale. = TRUE)
for (features.size in seq(10, 90, 10)){
    print("##############################")
    print(paste("##    PCA feature size", features.size))
    # Gather stats for all the folds
    BERate.list = NULL
    mis_error.list = NULL
    logloss.list = NULL
    for (fold.index in 1:Num.folds) {
        print(paste("---------- Fold #", fold.index, "----------"))

        # Create a training and a test set for this fold
        fold.test.indices <- (1+(fold.index-1)*fold.size) : (fold.index * fold.size)
        fold.train.indices <- setdiff(1:Num.totalSize, fold.test.indices)
        fold.test.size <- fold.size
        fold.train.size <- Num.totalSize-fold.test.size

        fold.test.x <- all.x[fold.test.indices, ]
        fold.train.x <- all.x[fold.train.indices, ]
        fold.test.y <- all.y[fold.test.indices]
        fold.train.y <- all.y[fold.train.indices]
        
        fold.test.yi <- as.numeric(sub('Class_', '', fold.test.y))
        fold.train.yi <- as.numeric(sub('Class_', '', fold.train.y))

        # using PCA to project features to another space
        fold.train.x <- pca.projecter(features.pca, fold.train.x, features.size)
        fold.test.x <- pca.projecter(features.pca, fold.test.x, features.size)

        ###### model training and prediction
        ## train can be given as: using.nnet, using.tree, using.randomForest, using.bagging
        train <- using.bagging
        if(all.equal(train, using.bagging)){
            # when calling using.bagging, need to pass the trainer to it as the first argument
            result <- using.bagging(using.tree, fold.train.x, fold.train.y, fold.test.x, model.num=5) 
        }else{
            model <- train(fold.train.x, fold.train.y)
            result <- predict(model, fold.test.x)  
        }
    
        if(is.factor(result)){
            result.label <- as.numeric(sub('Class_', '', result))
        }else{
            result.label <- max.col(result)
        }

        ###### mis-classification error
        mis_error <- mean(as.numeric(fold.test.yi!=result.label))
        mis_error.list <- c(mis_error.list, mis_error)
        print(paste("   Mis-Classification error rate:", mis_error))

        ###### Balanced error rate
        BERate <- compute.BERate(result.label, fold.test.yi, fold.test.size, Num.labels)
        BERate.list <- c(BERate.list, BERate)
        print(paste("   Balanced error rate:", BERate))

        ###### logloss
        logloss <- compute.logloss(result, fold.test.yi, fold.test.size)
        logloss.list <- c(logloss.list, logloss)
        print(paste("   Logloss:", logloss))
        
    }
    mis_error.allfolds <- mean(mis_error.list)
    BERate.allfolds <- mean(BERate.list)
    logloss.allfolds <- mean(logloss.list)
    print("===== all folds done, summary below =====")
    print(paste("Mis-Classification error rate: ", mis_error.allfolds))
    print(paste("Balanced error rate: ", BERate.allfolds))
    print(paste("Logloss: ", logloss.allfolds))
    print(paste("##    PCA feature size", features.size))
    print("##############################")
    points(features.size, mis_error.allfolds, pch = 16, cex = 0.6)
    text(features.size, mis_error.allfolds+0.1, round(mis_error.allfolds, 3), cex=0.8)
    sdev <- sd(mis_error.list)
    arrows(features.size, mis_error.allfolds-sdev, features.size, mis_error.allfolds+sdev, length=0.05, angle=90, code=3)
}
