library(keras)
library(tensorflow)


###Functions for MLP Neural Model
# ************************************************
# MLP_TrainClassifier()
## MLP NEURAL NETWORK
#
# INPUT:  Frame      - trainDF            - scaled [0.0,1.0], fields & rows
#         String     - fieldNameOutput    - Name of the field to predict
#         Int Vector - hidden             - Number of hidden layer neurons for each layer
#         boolean    - plot               - TRUE = output charts/results
#
# OUTPUT: object     - trained neural network
# ************************************************
MLP_TrainClassifier<- function(trainDF,
                                 trainingtarget,
                                 hidden,
                                 plot
                                 ){
  
  targetdata <- which(names(trainDF)==trainingtarget)
  #DataFrame with Input Fields
  train_inputs <- trainDF[-targetdata]
  
  #DataFrame with Target Field
  train_expected <- trainDF[,targetdata]
  
  #Convert Dataset into Matrix
  traindata <- as.matrix(trainDF)
  
  #Modelling
  mlpclassifier = keras_model_sequential()
  mlpclassifier %>% 
    keras::layer_dense(units=ncol(train_inputs), activation = 'relu', input_shape = ncol(train_inputs)) %>%
    keras::layer_dropout(0.2) %>%
    keras::layer_dense(units=hidden,activation='relu') %>%
    keras::layer_dropout(0.2) %>%
    keras::layer_dense(unit=1, activation = "softmax")
  
  #Loss function and Optimizer
  mlpclassifier %>% compile(loss='mse', optimizer = 'rmsprop', metrics = 'mae')
  
  #Fit Model
  mymodel <- mlpclassifier %>% fit(traindata, train_expected, epoch = 100, shuffle=T, batch_size = 5,
                                   validation_split = 0.2)
  
  #Plot the Neural Network error (loss)
  if(plot==T)
    print(plot(mymodel))
    # Plot the accuracy of the training data 
    plot(mymodel$metrics$mae, main="Mean Absolute Error", xlab = "epoch", ylab="MAE", col="blue", type="l")
  
  return(mlp_classifier)
}

# ************************************************
# EVALUATE_MLP() :
# ************************************************

  MLP_Evaluate<-function(test,testtarget,model,plot){
  
  predicted <- model %>% predict(testDF)
  
  print(mean((testtarget-predicted)^2))
  
  if (plot==TRUE)
    plot(testtarget, predicted)
}

