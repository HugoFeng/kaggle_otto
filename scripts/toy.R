library(nnet)
library(e1071)
library('tree')
library('randomForest')
set.seed(0)

using.tree <- function(trainX, trainY){
    print("Using Decision tree.")
    trainXY.combined <- cbind(trainX, trainY)
    model <- tree(trainY~., data=trainXY.combined)
}

using.randomForest <- function(trainX, trainY){
    print("Using Random Forest.")
    trainXY.combined <- cbind(trainX, trainY)
    model <- randomForest(trainY~., data=trainXY.combined)
}

using.svm <- function(trainX, trainY){
    print("Using svm.")
    trainXY.combined <- cbind(trainX, trainY)
    model <- svm(trainY~., data=trainXY.combined)
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

pca.projecter <- function(pca, original, feature_size){
    projected <- predict(pca, original)
    subtract <- projected[,1:feature_size]
    subtract <- data.frame(subtract)
}

# Load data set
inputData <- read.csv('../data/train.csv')
Num.totalSize <- dim(inputData)[1]
Num.totalCol <- dim(inputData)[2]

# Shuffle the original data
shuffleIndeces <- sample(1:Num.totalSize)
inputData <- inputData[shuffleIndeces, ]

Num.folds <- 3
Num.labels <- length(levels(inputData[, Num.totalCol]))
Num.ts.size <- floor(Num.totalSize/Num.folds)
fold.size <- floor(Num.totalSize/Num.folds)
Num.tr.size <- Num.totalSize-Num.ts.size

all.x <- inputData[, 2:(Num.totalCol-1)]
all.y <- inputData[, Num.totalCol]
# project all features to log space, and 0s are avoided
all.x.projected <- log(all.x+10^-3)
features.pca <- prcomp(all.x.projected, center = TRUE, scale. = TRUE)
for (features.size in seq(10, 40, 10)){
    print("####################")
    print(paste("## PCA feature size ", features.size))
    # Gather stats for all the folds
    BERate_list = NULL
    mis_error_list = NULL
    logloss_list = NULL
    for (fold.index in 1:Num.folds) {
        print("------------------")
        print(paste("Working on fold #", fold.index))
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
        ## train can be given as: using.nnet, using.tree, using.randomForest
        train <- using.svm
        model <- train(fold.train.x, fold.train.y)
        result <- predict(model, fold.test.x)
        if(is.factor(result)){
            result.label <- as.numeric(sub('Class_', '', result))
        }else{
            result.label <- max.col(result)
        }

        

        ###### mis-classification error
        mis_error <- mean(as.numeric(fold.test.yi!=result.label))
        mis_error_list <- c(mis_error_list, mis_error)
        print(paste("   Mis-Classification error rate: ", mis_error))

        ###### Balanced error rate
        ## The index of column means the predicted results,
        ## while the index of row means the real class label.
        BER_matrix <- matrix(0, nrow=Num.labels, ncol=Num.labels)
        ## For each sample and predicted results
        for(sampleIndex in 1:Num.ts.size) {
            realLabel <- fold.test.yi[sampleIndex]
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
        print(paste("   balanced error rate: ", BERate))

        ###### logloss
        if(!is.factor(result)){
            logloss <- 0
            for(sampleIndex in Num.ts.size) {
                outputs <- result[sampleIndex, ]
                # Probability of being classified as one of the labels
                p_i <- outputs/sum(outputs)
                
                predictedLabel <- result.label[sampleIndex]
                # avoid 0 probability
                p_ij <- max(min(p_i[predictedLabel], 1-10^-15), 10^-15)
                log_p_ij <- log(p_ij)
                logloss <- logloss + log_p_ij
            }
            logloss <- -logloss/Num.ts.size
            logloss_list <- c(logloss_list, logloss)
            print(paste("   logloss: ", logloss))
        }
    }
    mis_error_allfolds <- mean(mis_error_list)
    BERate_allfolds <- mean(BERate_list)
    logloss_allfolds <- mean(logloss_list)
    print("=======^^^===all folds done===^^^======")
    print(paste("Mis-Classification error rate: ", mis_error_allfolds))
    print(paste("balanced error rate: ", BERate_allfolds))
    print(paste("logloss: ", logloss_allfolds))
    print("=======================================")
}

# x <- 1:15
# par(mfrow=c(1,1))
# #plot(c(1, 15), c(0, 1), type="n")
# plot(x, BERate_list, xlim=c(1, 15), ylim=c(0, 1), type="l", col="red")
# lines(x, mis_error_list, col="blue")
# lines(x, logloss_list, col="black")
