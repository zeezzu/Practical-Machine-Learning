library(rattle.data)
## Loading required package: RGtk2
## Rattle: A free graphical interface for data mining with R.
## Version 3.5.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
library(caret)
## Loading required package: lattice
## Loading required package: ggplot2
library(rpart)
library(rpart.plot)
library(corrplot)
library(randomForest)
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
library(RColorBrewer)
## Finally, load the same seed with the following line of code:
set.seed(56789)
## Download dataset
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./data/pml-training.csv"
testFile  <- "./data/pml-testing.csv"
if (!file.exists("./data")) {
  dir.create("./data")
}
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile = trainFile, method = "curl")
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile = testFile, method = "curl")
}
rm(trainUrl)
rm(testUrl)
trainRaw <- read.csv(trainFile)
testRaw <- read.csv(testFile)
dim(trainRaw)
dim(testRaw)
rm(trainFile)
rm(testFile)
## Clean the Near Zero Variance Variables
NZV <- nearZeroVar(trainRaw, saveMetrics = TRUE)
head(NZV, 20)
training01 <- trainRaw[, !NZV$nzv]
testing01 <- testRaw[, !NZV$nzv]
dim(training01)
dim(testing01)
rm(trainRaw)
rm(testRaw)
rm(NZV)
## Removing some columns of the dataset that do not contribute much to the accelerometer measurements
regex <- grepl("^X|timestamp|user_name", names(training01))
training <- training01[, !regex]
testing <- testing01[, !regex]
rm(regex)
rm(training01)
rm(testing01)
dim(training)
dim(testing)
## Removing columns that contain NA's
cond <- (colSums(is.na(training)) == 0)
training <- training[, cond]
testing <- testing[, cond]
rm(cond)
## Correlation Matrix of Columns in the Training Data set
corrplot(cor(training[, -length(names(training))]), method = "color", tl.cex = 0.5)
## Split the cleaned training set into a pure training data set (70%) and a validation data set (30%)
set.seed(56789) # For reproducibile purpose
inTrain <- createDataPartition(training$classe, p = 0.70, list = FALSE)
validation <- training[-inTrain, ]
training <- training[inTrain, ]
rm(inTrain)
## Fit a predictive model for activity recognition using Decision Tree algorithm
modelTree <- rpart(classe ~ ., data = training, method = "class")
prp(modelTree)
## Now, to estimate the performance of the model on the validation data set
predictTree <- predict(modelTree, validation, type = "class")
confusionMatrix(validation$classe, predictTree)
accuracy <- postResample(predictTree, validation$classe)
ose <- 1 - as.numeric(confusionMatrix(validation$classe, predictTree)$overall[1])
rm(predictTree)
rm(modelTree)
modelRF <- train(classe ~ ., data = training, method = "rf", trControl = trainControl(method = "cv", 5), ntree = 250)
modelRF
predictRF <- predict(modelRF, validation)
confusionMatrix(validation$classe, predictRF)
## Now, to estimate the performance of the model on the validation data set
accuracy <- postResample(predictRF, validation$classe)
ose <- 1 - as.numeric(confusionMatrix(validation$classe, predictRF)$overall[1])
## Now, to apply the Random Forest model to the original testing data set downloaded from the data source. 
## First, remove the problem_id column
rm(predictRF)
rm(accuracy)
rm(ose)
predict(modelRF, testing[, -length(names(testing))])
## Function to generate files with predictions to submit for assignment
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("./Assignment_Solutions/problem_id_",i,".txt")
    write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
}
## Generating the Files
pml_write_files(predict(modelRF, testing[, -length(names(testing))]))
rm(modelRF)
rm(training)
rm(testing)
rm(validation)
rm(pml_write_files)