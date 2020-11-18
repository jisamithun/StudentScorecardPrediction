# ************************************************
# Global Environment variables

FILE <- "ScoreCard_cleaner_Dataset.csv"
OUTPUTFIELD <- "md_earn_wne_p10"

MYLIBRARIES<-c("gbm",
               "carat",
               "formattable")


# ************************************************
mae <- function(actual, predict){
  return(mean(abs(actual-predict)))
}

rmse <- function(actual, predict){
  return(sqrt(mean((actual-predict)^2)))
}

r2 <- function(actual, predict){
  return(1-(sum((actual-predict)^2)/sum((actual-mean(actual))^2)))
}

cleanDataset <- function(dataset){
  drops <- c("","Salary_Class_After_10_years")
  cleanData <- dataset[ , !(names(dataset) %in% drops)]
  return(cleanData)
}

# ************************************************
gbm_model <- function(train_data, test_data){
  
  # GBM function parameters
  DISTRIBUTION <- "gaussian"#"multinomial"
  NTREES <- 10000
  FOLDS <- c(3,5,10)
  LEARNING_RATE <- 0.1
  
  input_fields <- paste(names(train_data)[which(names(train_data)!=OUTPUTFIELD)], collapse = "+")
  formula <- as.formula(paste(OUTPUTFIELD,"~",input_fields))
  
  maeList_training <- vector()
  rmseList_training <- vector()
  r2List_training <- vector()
  
  maeList_testing <- vector()
  rmseList_testing <- vector()
  r2List_testing <- vector()
  
  varList <- vector()
  
  
  for (var in FOLDS){
    
  
    # Train the model
    model <- gbm(
                formula = formula
                ,data = train_data
                ,distribution = DISTRIBUTION
                ,n.trees = NTREES
                ,shrinkage = LEARNING_RATE
                ,cv.folds = var
                )
    
    print("****************************************")
    title <- paste("result of number of folds =",var)
    print(title)
    print(model)
    #print(sqrt(min(model$train.error)))
    
    print("Printed influence of predictors.")
    importance = summary.gbm(model, plotit=TRUE)
    names(importance)[1]<-title
    print(formattable::formattable(importance))
    
    cv_ntrees = gbm.perf(model, method = "cv")
    
    #***************************************************************
    # Evaluate the model
    output <- predict.gbm(
                          object = model
                          ,newdata = train_data
                          ,n.trees = cv_ntrees
                          ,type = "response"
                          )
    
    #labels = colnames(output)[apply(output, 1, which.max)]
    actual <- train_data[,OUTPUTFIELD]
    predict <- output
    
    
    # Calculate MAE
    MAE <- round(mae(actual, predict),2)
    #print(paste("MAE of model:", MAE))
    maeList_training <- c(maeList_training, MAE)
    
    # Calculate RMSE
    RMSE <- round(rmse(actual, predict),2)
    #print(paste("RMSE of model:", RMSE))
    rmseList_training <- c(rmseList_training, RMSE)
    
    # Calculate rsquared
    R2 <- round(r2(actual, predict),2)
    #print(paste("R Squared of model:", R2))
    r2List_training <- c(r2List_training, R2)
  
    #***************************************************************
    
    # Evaluate the model by testing data
    output <- predict.gbm(
                          object = model
                          ,newdata = test_data
                          ,n.trees = cv_ntrees
                          ,type = "response"
                          )
    
    #labels = colnames(output)[apply(output, 1, which.max)]
    actual <- test_data[,OUTPUTFIELD]
    predict <- output
    
    
    # Calculate MAE
    MAE <- round(mae(actual, predict),2)
    print(paste("MAE of testing data:", MAE))
    maeList_testing <- c(maeList_testing, MAE)
    
    # Calculate RMSE
    RMSE <- round(rmse(actual, predict),2)
    print(paste("RMSE of testing data:", RMSE))
    rmseList_testing <- c(rmseList_testing, RMSE)
    
    # Calculate rsquared
    R2 <- round(r2(actual, predict),2)
    print(paste("R Squared of testing data:", R2))
    r2List_testing <- c(r2List_testing, R2)
    
    
    
    varList <- c(varList, var)
  }
  
  #train_measures <- data.frame(varList, maeList_training, rmseList_training, r2List_training)
  #print(formattable::formattable(train_measures))
  
  
  test_measures <- data.frame(varList, maeList_testing, rmseList_testing, r2List_testing)
  names(test_measures)[names(test_measures) == "varList"] <- "Number of folds"
  names(test_measures)[names(test_measures) == "maeList_testing"] <- "MAE"
  names(test_measures)[names(test_measures) == "rmseList_testing"] <- "RMSE"
  names(test_measures)[names(test_measures) == "r2List_testing"] <- "R Squared value"
  
  print(formattable::formattable(test_measures))
  
  
}

# ************************************************

main <- function(){
  print(Sys.time())
  
  
  dataset<-read.csv(FILE,encoding="UTF-8",stringsAsFactors = FALSE)
  
  dataset <- cleanDataset(dataset)
  
  dataset<-dataset[order(runif(nrow(dataset))),]
  
  # use ALL fields (columns)
  training_records<-round(nrow(dataset)*(70/100))
  training_data <- dataset[1:training_records,]
  testing_data = dataset[-(1:training_records),]
  
  gbm_model(training_data,testing_data)
  
  
  print(Sys.time())
}

# ************************************************

gc() # garbage collection to automatically release memory

# clear plots and other graphics
if(!is.null(dev.list())) dev.off()
graphics.off()

# This clears all warning messages
assign("last.warning", NULL, envir = baseenv())

# clears the console area
cat("\014")

print("START")

library(pacman)
pacman::p_load(char=MYLIBRARIES,install=TRUE,character.only=TRUE)


set.seed(123)

# ************************************************
main()

print("end")