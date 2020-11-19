library(dplyr)
library(magrittr)
library(keras)
library(tensorflow)

DATASET_FILENAME  <- "ScoreCard_cleaner_Dataset.csv"
OUTPUT_FIELD <- "md_earn_wne_p10"
setwd("D:\\DataScience\\SEM1\\PracticalBusinessAnalytics(COMM053)\\Coursework\\NNProject")

dataset <- read.csv(DATASET_FILENAME,encoding="UTF-8",stringsAsFactors = T)
dataset <- dataset[,c(3:6,17:34)]
str(dataset)

Salary_Band <- cut(dataset$md_earn_wne_p10, breaks = c(min(dataset$md_earn_wne_p10)-1, 
                   quantile(dataset$md_earn_wne_p10, probs=0.25),
                   median(dataset$md_earn_wne_p10), 
                   quantile(dataset$md_earn_wne_p10, probs=0.75),
                   max(dataset$md_earn_wne_p10)), labels = c("BandA", "BandB",
                  "BandC", "BandD"))

dataset <- dataset %>% mutate(SalaryBand = Salary_Band)

#Convert the Salary Class into numeric labels
dataset$SalaryBand <- factor(dataset$SalaryBand, levels = c("BandA", "BandB",
                                "BandC", "BandD"), labels = c(1,2,3,4))

#Convert Factors to Numeric
dataset %<>% mutate_if(is.factor,as.numeric)
summary(dataset)
dataset <- dataset[,c(1:21,23)]

#Convert to matrix
dataset <- as.matrix(dataset)
dimnames(dataset) <- NULL

#Normalize
maxvalue <- apply(dataset[,1:21],2,max)
minvalue <- apply(dataset[,1:21],2,min)
dataset[,1:21] <- scale(dataset[,1:21], center = minvalue, scale = maxvalue-minvalue)
dataset[,22] <- as.numeric(dataset[,22])-1

# Determine sample size
set.seed(123)
splitdata <- sample(2, nrow(dataset), replace=TRUE, prob=c(0.70, 0.30))
# Split the Input Dataset
data.training <- dataset[splitdata==1, 1:21]
data.test <- dataset[splitdata==2, 1:21]

data.trainingtarget <- dataset[splitdata==1, 22]
data.testtarget <- dataset[splitdata==2, 22]

#One hot Encoding
trainlabels <- keras::to_categorical(data.trainingtarget)
testlabels <- keras::to_categorical(data.testtarget)

#Create Model
model <- keras_model_sequential()
model %>% keras::layer_dense(units=64, activation = 'relu', input_shape = 21) %>%
  keras::layer_dropout(rate=0.1) %>%
  keras::layer_dense(units=32, activation = 'relu') %>%
  keras::layer_dropout(rate=0.2) %>%
  keras::layer_dense(units=16, activation = 'relu') %>%
  keras::layer_dropout(rate=0.3) %>%
  keras::layer_dense(unit=4, activation = "sigmoid")

#Compile and Fit Code
model %>% keras::compile(loss="categorical_crossentropy", optimizer = 'adam', metrics = "accuracy")

model_1 <- model %>% keras::fit(x=data.training, y=trainlabels, batch_size = 128,epochs = 100,
                                validation_split = 0.2)

#Visualizing the model
plot(model_1)

#Evaluate the model
model %>% evaluate(data.test, testlabels)

#Prediction & Confusion Matrix
predictedClass <- model %>% predict_classes(data.test)
table(Predicted = predictedClass, Actual = data.testtarget)

