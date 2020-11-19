library(caret)
library(dplyr)
library(keras)
library(tensorflow)
library(magrittr)

DATASET_FILENAME  <- "ScoreCard_cleaner_Dataset.csv"
OUTPUT_FIELD <- "md_earn_wne_p10"
setwd("D:\\DataScience\\SEM1\\PracticalBusinessAnalytics(COMM053)\\Coursework\\NNProject")

dataset <- read.csv(DATASET_FILENAME,encoding="UTF-8",stringsAsFactors = T)
dataset <- dataset[,c(3:6,17:34)]

#Step:1 Normalize the data
#Normalization
maxvalue <- apply(dataset,2,max)
minvalue <- apply(dataset,2,min)
dataset <- as.data.frame(scale(dataset, center = minvalue, scale = maxvalue-minvalue))

#Step2: Create Training and Test Data
set.seed(123)

# Determine sample size
splitdata <- sample(2, nrow(dataset), replace=TRUE, prob=c(0.7, 0.3))
# Split the Input Dataset
data.training <- dataset[splitdata==1, 1:ncol(dataset)]


data.test <- dataset[splitdata==2, 1:ncol(dataset)-1]
data.testtarget <- dataset[splitdata==2, ncol(dataset)]

folds <- createFolds(y=data.training[,ncol(dataset)], k =5, list=F)
data.training$folds <- folds

modellist <- list()
f=1
for (f in unique(data.training$folds)){
  print(paste("\n Fold: ", f))
  ind <- which(data.training$folds == f)
  train_df <- data.training[-ind,1:ncol(dataset)-1]
  train_target <- as.matrix(data.training[-ind,OUTPUT_FIELD])
  valid_df <- data.training[ind,1:ncol(dataset)-1]
  valid_target <- as.matrix(data.training[ind,OUTPUT_FIELD])
  
  #Create Model
  model <- keras_model_sequential()
  model %>% keras::layer_dense(units=10, activation = 'relu', input_shape = 21) %>%
    keras::layer_dropout(0.2) %>%
    keras::layer_dense(units=5,activation='relu') %>%
    keras::layer_dropout(0.2) %>%
    keras::layer_dense(unit=1, activation = "linear")
  
  #Inspection of model
  summary(model)
  
  #Compile and Fit Code
  model %>% keras::compile(loss='mse', optimizer = 'rmsprop', metrics = list('mae'))
  
  model_1 <- model %>% keras::fit(x=as.matrix(train_df), y=train_target, batch_size = 32,
             epochs = 50,validation_data = list(as.matrix(valid_df),valid_target))
  
  #Visualizing the model
  plot(model_1)
  # Plot the accuracy of the training data 
  plot(model_1$metrics$mae, main="Mean Absolute Error", xlab = "epoch", ylab="MAE", col="blue", type="l")
  
  #Evaluate Model
  # Evaluate on test data and labels
  measure <- model %>% evaluate(as.matrix(data.test), data.testtarget)
  modellist[[f]] <- measure
  
  # Print the score on testDatset
  print(measure)
  
  predicted <- model %>% predict(as.matrix(data.test))
  result <- caret::postResample(predicted, data.testtarget)
  plot(data.testtarget, predicted)
}

print(modellist)
