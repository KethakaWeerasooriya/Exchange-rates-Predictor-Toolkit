# Load necessary libraries
library(dplyr) # Package to manipulate data
library(readxl) # Package to read xlsx file
library(neuralnet) # Package to build neural networks
library(grid) # Package for data visualization
library(MASS) # Package for data analysis
library(MLmetrics) # Package for machine learning metrics
library(Metrics) # Package for metrics calculation
library(readxl) # Package to read excel files

#Define a function to evaluate the model
evaluation <- function(actual, predict) {
  rmse_mlp <- rmse(actual = actual, predicted = predict)
  mae_mlp <- mae(actual = actual, predicted = predict)
  mape_mlp <- MAPE(y_pred = predict, y_true = actual)
  #SMPE
  return(c(rmse_mlp, mae_mlp, mape_mlp))
}
#Define a function the reverse of normalized data – de-normalized
denormalize <- function(x, min, max) {
  return( (max - min) * x + min )
}
# Set working directory
setwd("C:/Users/hirun/Desktop/MLCW")


# Load dataset
USDvsEUR <- read_excel('ExchangeUSD (2).xlsx', col_names = TRUE)
#Display the summary and details of the data set
head(USDvsEUR)
summary(USDvsEUR)
colnames(USDvsEUR)
#Rename  columns names
names(USDvsEUR) <- c("date", "Wdy", "usdVSeur" )
head(USDvsEUR)




#partB
#Create a table with variables as columns
lagged_data <- bind_cols(target = lag(USDvsEUR$usdVSeur, 0), 
                         # Create lagged variables
                         t1 = lag(USDvsEUR$usdVSeur, 1),
                         
                         t2 = lag(USDvsEUR$usdVSeur, 2),
                         
                         t3 = lag(USDvsEUR$usdVSeur, 3),
                         
                         t4 = lag(USDvsEUR$usdVSeur, 4),
                         
                         t7 = lag(USDvsEUR$usdVSeur, 7)) 

#Display the existence of NA values due to that shifting
lagged_data 
# Remove rows with NA values
lagged_data <- lagged_data[complete.cases(lagged_data),]
#verify that if  NA values are removed
sum(is.na(lagged_data))
#make a I/O matrix for the training set
training_dataset <- as.matrix(lagged_data[1:380,])
# make I/O matrix for the testing set
testing_dataset <- as.matrix(lagged_data[381:nrow(lagged_data),])


#PartC
#Check if data needs to be normalized
summary(training_dataset)
summary(testing_dataset)
# Normalize data with min max normalization
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

#Use the normalization function across columns with a margin of  2
normalized_training_dataset <- apply(training_dataset, 2, normalize)
normalized_testing_Dataset <- apply(testing_dataset, 2, normalize)
#Show the normalized matrices summary 
summary(normalized_training_dataset)
summary(normalized_testing_Dataset)



#PartD
# Extract original output for denormalization training and testing desired output 

trained_output <- lagged_data$target
tested_output <- as.matrix(lagged_data[381:nrow(lagged_data), "target"]) 
#Identify the training sets maximum and minimum values for de-normalization.
train_min <- min(trained_output)
train_max <- max(trained_output)
#Define the model formula to be used in the NN
model_formula <- as.formula(target ~ t1 + t2 + t3 + t4 + t7)
#assign diffrent learning rates 
leariningRate1<-0.1
leariningRate2<-0.2

#assign different hidden layers
hiddenLayer1<-c(2,4)
hiddenLayer2<-c(4)
hiddenLayer3<-c(3,7)
hiddenLayer4<-c(6,3)
hiddenLayer5<-c(4,7)
hiddenLayer6<-c(8)
hiddenLayer7<-c(6)
hiddenLayer8<-c(3,6)
hiddenLayer9<-c(9)
hiddenLayer10<-c(5,7)
hiddenLayer11<-c(10)
hiddenLayer12<-c(4,8)
hiddenLayer13<-c(7,8)
hiddenLayer14<-c(4,7)
hiddenLayer15<-c(10,5)



# Define a function to describe model
model_Descri <- function(model, formula, hidden_layers = 1, epoch = 1, learningrate = 0.1, 
                         algorithm = "rprop+", act_fct = "logistic"){
  return(c(hidden_layers, formula, epoch, learningrate, algorithm, act_fct))
}
#Define a function that returns the model's output
modelResults <- function(model, dataNormarl, min, max) {
  #Plot the NN model
  plot(model)
  
  # Use the test data set to predict the result. 
  model_result <- neuralnet::compute(model, dataNormarl) 
  
  
  model_pred <- model_result$net.result
  
  #Reverse the predicted output's normalization.
  output_denorm <- denormalize(model_pred, min, max)
  
  # model Evaluate
  model_eval <- evaluation(tested_output, output_denorm)
  
  
  return(model_eval)
}


#PartF


#generating test and training data for every model
#1NN
#Using the training data set, create the neural network model (default model for 1 hidden layer).


#matrix1 # Build models
model_formula1 <- as.formula(target ~ t1 + t2 + t3 + t4 + t7)


model_1 <- neuralnet(formula = model_formula1,
                     data = normalized_training_dataset,
                     hidden = hiddenLayer1,
                     linear.output = TRUE)

#Save the evaluation's result values and the model's description.
model_1_Evalation = modelResults(model_1, normalized_testing_Dataset, train_min, train_max)
model_1_Description = model_Descri(model_1, model_formula1, hidden_layers = hiddenLayer1)

model_1_Summary = c(model_1_Description, model_1_Evalation)
#2NN
# Use the training data set to construct the neural network model (1 hidden layers).
model_2 <- neuralnet(formula = model_formula,
                     data = normalized_training_dataset,
                     hidden = hiddenLayer2,
                     linear.output = TRUE)
#Save the evaluation's result values and the model's description.
model_2_Evalation <- modelResults(model_2, normalized_testing_Dataset, train_min, train_max)
model_2_Description <- model_Descri(model_2, model_formula, hidden_layers = hiddenLayer2)
model_2_Summary <- c(model_2_Description, model_2_Evalation)
#3NN
# Use the training data set to construct the neural network model (3 hidden layers).
model_3 <- neuralnet(formula = model_formula,
                     data = normalized_training_dataset,
                     hidden = hiddenLayer3,
                     linear.output = TRUE)
#Save the evaluation's result values and the model's description.
model_3_Evalation <- modelResults(model_3, normalized_testing_Dataset, train_min, train_max)
model_3_Description <- model_Descri(model_3, model_formula, hidden_layers = hiddenLayer3)
#4NN# Use the training data set to construct the neural network model (4 hidden layers).
model_3_Summary <- c(model_3_Description, model_3_Evalation)

model_4 <- neuralnet(formula = model_formula,
                     data = normalized_training_dataset,
                     hidden = hiddenLayer4,
                     learningrate = leariningRate1,
                     linear.output = TRUE)

model_4_Evalation <- modelResults(model_4, normalized_testing_Dataset, train_min, train_max)
model_4_Description <- model_Descri(model_4, model_formula, hidden_layers = hiddenLayer4, 
                                    learningrate = leariningRate1)
model_4_summary <- c(model_4_Description, model_4_Evalation)
#5NN# Use the training data set to construct the neural network model (5 hidden layers).
model_5 <- neuralnet(formula = model_formula,
                     data = normalized_training_dataset,
                     hidden = hiddenLayer5,
                     linear.output = TRUE)
model_5_Evalation <- modelResults(model_5, normalized_testing_Dataset, train_min, train_max)
model_5_Description <- model_Descri(model_5, model_formula, hidden_layers = hiddenLayer5)
model_5_Summary <- c(model_5_Description, model_5_Evalation)

#6NN# Use the training data set to construct the neural network model (6 hidden layers).
model_6 <- neuralnet(formula = model_formula,
                     data = normalized_training_dataset,
                     hidden = c(8, 4, 2),
                     linear.output = TRUE)
model_6_Evalation <- modelResults(model_6, normalized_testing_Dataset, train_min, train_max)
model_6_Description <- model_Descri(model_6, model_formula, hidden_layers = c(8, 4, 2))
model_6_Summary <- c(model_6_Description, model_6_Evalation)
#7NN# Use the training data set to construct the neural network model (7 hidden layers).
model_7 <- neuralnet(formula = model_formula,
                     data = normalized_training_dataset,
                     hidden = hiddenLayer7,
                     linear.output = TRUE)
model_7_Evalation <- modelResults(model_7, normalized_testing_Dataset, train_min, train_max)
model_7_Description <- model_Descri(model_7, model_formula, hidden_layers = hiddenLayer7)
model_7_Summary <- c(model_7_Description, model_7_Evalation)
#8NN# Use the training data set to construct the neural network model (8 hidden layers).
model_8 <- neuralnet(formula = model_formula,
                     data = normalized_training_dataset,
                     hidden = hiddenLayer8,
                     linear.output = TRUE)

model_8_Evalation <- modelResults(model_8, normalized_testing_Dataset, train_min, train_max)
model_8_Description <- model_Descri(model_8, model_formula, hidden_layers = hiddenLayer8)
model_8_Summary <- c(model_8_Description, model_8_Evalation)
#9NN# Use the training data set to construct the neural network model (9 hidden layers).
model_9 <- neuralnet(formula = model_formula,
                     data = normalized_training_dataset,
                     hidden = hiddenLayer9,
                     linear.output = TRUE)
model_9_Evalation <- modelResults(model_9, normalized_testing_Dataset, train_min, train_max)
model_9_Description <- model_Descri(model_9, model_formula, hidden_layers = hiddenLayer9)
model_9_Summary <- c(model_9_Description, model_9_Evalation)
#10NN# Use the training data set to construct the neural network model (10 hidden layers).
model_10 <- neuralnet(formula = model_formula,
                      data = normalized_training_dataset,
                      hidden = hiddenLayer10,
                      linear.output = TRUE)
model_10_Evalation <- modelResults(model_10, normalized_testing_Dataset, train_min, train_max)
model_10_Description <- model_Descri(model_10, model_formula, hidden_layers = hiddenLayer10)
model_10_Summary <- c(model_10_Description, model_10_Evalation)
#11NN# Use the training data set to construct the neural network model (11 hidden layers).

model_11 <- neuralnet(formula = model_formula,
                      data = normalized_training_dataset,
                      hidden = hiddenLayer11,
                      linear.output = TRUE)
model_11_Evalation <- modelResults(model_11, normalized_testing_Dataset, train_min, train_max)
model_11_Description <- model_Descri(model_11, model_formula, hidden_layers = hiddenLayer11)
model_11_Summary <- c(model_11_Description, model_11_Evalation)
#12NN# Use the training data set to construct the neural network model (12 hidden layers).
model_12 <- neuralnet(formula = model_formula,
                      data = normalized_training_dataset,
                      hidden = hiddenLayer12,
                      linear.output = TRUE)
model_12_Evalation <- modelResults(model_12, normalized_testing_Dataset, train_min, train_max)
model_12_Description <- model_Descri(model_12, model_formula, hidden_layers = hiddenLayer12)
model_12_Summary <- c(model_12_Description, model_12_Evalation)
#13NN# Use the training data set to construct the neural network model (12 hidden layers).
model_13 <- neuralnet(formula = model_formula,
                      data = normalized_training_dataset,
                      hidden = hiddenLayer13,
                      linear.output = TRUE)
model_13_Evalation <- modelResults(model_13, normalized_testing_Dataset, train_min, train_max)
model_13_Description <- model_Descri(model_13, model_formula, hidden_layers = hiddenLayer13)
model_13_Summary <- c(model_13_Description, model_13_Evalation)
#14NN# Use the training data set to construct the neural network model (12 hidden layers).
model_14 <- neuralnet(formula = model_formula,
                      data = normalized_training_dataset,
                      hidden = hiddenLayer14,
                      linear.output = TRUE)
model_14_Evalation <- modelResults(model_14, normalized_testing_Dataset, train_min, train_max)
model_14_Description <- model_Descri(model_14, model_formula, hidden_layers = hiddenLayer14)
model_14_Summary <- c(model_14_Description, model_14_Evalation)
#15NN# Use the training data set to construct the neural network model (12 hidden layers).
model_15 <- neuralnet(formula = model_formula,
                      data = normalized_training_dataset,
                      hidden = hiddenLayer15,
                      linear.output = TRUE)
model_15_Evalation <- modelResults(model_15, normalized_testing_Dataset, train_min, train_max)
model_15_Description <- model_Descri(model_15, model_formula, hidden_layers = hiddenLayer15)
model_15_Summary <- c(model_15_Description, model_15_Evalation)
# Find the maximum length among the model summaries
max_length <- max(length(model_1_Summary), length(model_2_Summary), length(model_3_Summary), 
                  length(model_4_summary),
                  length(model_5_Summary), length(model_6_Summary), length(model_7_Summary), 
                  length(model_8_Summary),
                  length(model_9_Summary), length(model_10_Summary), length(model_11_Summary), 
                  length(model_12_Summary),length(model_13_Summary),length(model_14_Summary),
                  length(model_15_Summary))

# Function to pad summaries with NA values
pad_Summary <- function(summary, max_length) {
  length_diff <- max_length - length(summary)
  if (length_diff > 0) {
    summary <- c(summary, rep(NA, length_diff))
  }
  return(summary)
}
# Pad summaries with NA values
model_1_Summary <- pad_Summary(model_1_Summary, max_length)
model_2_Summary <- pad_Summary(model_2_Summary, max_length)
model_3_Summary <- pad_Summary(model_3_Summary, max_length)
model_4_summary <- pad_Summary(model_4_summary, max_length)
model_5_Summary <- pad_Summary(model_5_Summary, max_length)
model_6_Summary <- pad_Summary(model_6_Summary, max_length)
model_7_Summary <- pad_Summary(model_7_Summary, max_length)
model_8_Summary <- pad_Summary(model_8_Summary, max_length)
model_9_Summary <- pad_Summary(model_9_Summary, max_length)
model_10_Summary <- pad_Summary(model_10_Summary, max_length)
model_11_Summary <- pad_Summary(model_11_Summary, max_length)
model_12_Summary <- pad_Summary(model_12_Summary, max_length)
model_13_Summary <- pad_Summary(model_13_Summary, max_length)
model_14_Summary <- pad_Summary(model_14_Summary, max_length)
model_15_Summary <- pad_Summary(model_15_Summary, max_length)
#G)
# Combine all model summaries into a matrix
# Combine all model summaries into a matrix
comparison_table <- rbind(model_1_Summary, model_2_Summary, model_3_Summary, 
                          model_4_summary, model_5_Summary, model_6_Summary, 
                          model_7_Summary, model_8_Summary, model_9_Summary, 
                          model_10_Summary, model_11_Summary, model_12_Summary,
                          model_13_Summary, model_14_Summary, model_15_Summary)

# Verify the comparison table's dimensions.
dim(comparison_table)
# Set column names
length(colnames(comparison_table))

colnames(comparison_table) <- c("Hidden_Layers", "Formula", "Epoch", "Learning_Rate", "Algorithm", 
                                "Activation_Function",
                                "RMSE", "MAE", "MAPE", "Additional_Info1")

# Display comparison table
comparison_table
#length(model_13 Summary)
# Select the best model based on the lowest RMSE
best_model_index <- which.min(comparison_table[, "RMSE"])
best_model <- switch(best_model_index,
                     model_1 = model_1,
                     model_2 = model_2,
                     model_3 = model_3,
                     model_4 = model_4,
                     model_5 = model_5,
                     model_6 = model_6,
                     model_7 = model_7,
                     model_8 = model_8,
                     model_9 = model_9,
                     model_10 = model_10,
                     model_11 = model_11,
                     model_12 = model_12,
                     model_13 = model_13,
                     model_14 = model_14,
                     model_15 = model_15
                     
                     )
# Predict using the best model
best_model_result <- neuralnet::compute(best_model, normalized_testing_Dataset)
best_model_pred <- best_model_result$net.result
# De-normalize the predicted output
output_denorm <- denormalize(best_model_pred, train_min, train_max)


# Assuming you have your best MLP model trained and named as `best_model`
# normalized_testing_Dataset is your testing dataset and tested_output is the true values for the testing dataset

# Predict using the best MLP model
best_model_pred <- compute(best_model, normalized_testing_Dataset)$net.result

# De-normalize the predicted output
output_denorm <- denormalize(best_model_pred, train_min, train_max)

# Plotting
plot(tested_output, type = "l", col = "blue", lwd = 2, ylim = c(min(tested_output, output_denorm), max(tested_output, output_denorm)), 
     main = "Actual vs. Predicted", xlab = "Index", ylab = "Output Value")
lines(output_denorm, type = "l", col = "red", lwd = 2)
legend("topleft", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 1, lwd = 2)

# Statistical indices
mse <- mean((tested_output - output_denorm)^2)
r_squared <- cor(tested_output, output_denorm)^2

cat("Mean Squared Error (MSE):", mse, "\n")
cat("Coefficient of Determination (R²):", r_squared, "\n")


