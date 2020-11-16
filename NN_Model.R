library(dplyr)
library(keras)
library(tensorflow)
library(magrittr)
DATASET_FILENAME  <- "clean_dataset.csv"
OUTPUT_FIELD <- "md_earn_wne_p10"
setwd("D:\\DataScience\\SEM1\\PracticalBusinessAnalytics(COMM053)\\Coursework\\Project")

dataset <- read.csv(DATASET_FILENAME,encoding="UTF-8",stringsAsFactors = T)
#Categorical Fields
dataset$CONTROL = factor(dataset$CONTROL, levels = 
                               c('Public','Private nonprofit','Private for-profit'), labels = c(1,2,3))
dataset$HIGHDEG = factor(dataset$HIGHDEG, 
                             levels = c('Non-degree-granting', 'Certificate degree', 
                                        'Associate degree', "Bachelor's degree", "Graduate degree"), 
                             labels = c(1,2,3,4,5))
dataset <- dataset[,2:10]

#Convert Factors to Numeric
dataset %<>% mutate_if(is.factor,as.numeric)

#Step:1 Normalize the data
#Normalization
maxvalue <- apply(dataset,2,max)
minvalue <- apply(dataset,2,min)
dataset <- as.data.frame(scale(dataset, center = minvalue, scale = maxvalue-minvalue))

#Convert Dataset into Matrix
dataset <- as.matrix(dataset)
dimnames(dataset) <- NULL


#Step2: Create Training and Test Data
set.seed(1234)

# Determine sample size
ind <- sample(2, nrow(dataset), replace=TRUE, prob=c(0.67, 0.33))
# Split the dataset
data.training <- dataset[ind==1, 1:8]
data.test <- dataset[ind==2, 1:8]

# Split the class attribute
data.trainingtarget <- dataset[ind==1, 9]
data.testtarget <- dataset[ind==2, 9]

#Create Model
model <- keras_model_sequential()
model %>% keras::layer_dense(units=8, activation = 'relu', input_shape = c(8)) %>%
  keras::layer_dropout(0.2) %>%
  keras::layer_dense(units=3,activation='relu') %>%
  keras::layer_dropout(0.2) %>%
  keras::layer_dense(unit=1, activation = "softmax")

#Inspection of model
summary(model)

#Compile and Fit Code
model %>% compile(loss='mse', optimizer = 'rmsprop', metrics = 'mae')

#Fit Model
mymodel <- model %>% fit(data.training, data.trainingtarget, shuffle=T, epoch = 50, batch_size = 10,
                         validation_split = 0.2, verbose=1,
                         callbacks = c(callback_early_stopping(monitor="val_loss")))

#Visualizing the model
plot(mymodel)
# Plot the accuracy of the training data 
plot(mymodel$metrics$mae, main="Mean Absolute Error", xlab = "epoch", ylab="MAE", col="blue", type="l")

#Predict earnings for the test data
predicted <- model %>% predict(data.test)

#Evaluate Model
# Evaluate on test data and labels
measure <- model %>% evaluate(data.test, data.testtarget)

# Print the score
print(measure)
mean((data.testtarget-predicted)^2)
plot(data.testtarget, predicted)
