#Source code for Neural Networks

#importing a library to read xlsx files
library(readxl)


library(ggplot2)
library(neuralnet)
library(dplyr)


setwd("C:/Users/hirun/Desktop/MLCW")


# Load dataset
USDvsEUR <- read_excel('ExchangeUSD (2).xlsx', col_names = TRUE)

colnames(USDvsEUR) <- c("date", "Wdy", "usdVSeur")
summary(USDvsEUR)

#consider USD/EUR data only
USDvsEUR <- USDvsEUR[, 3]

# FIRST day LAG (t-1)

lag_current <- lag(USDvsEUR, 1)
lag_output <- USDvsEUR
#combining the column
tLag1 <- bind_cols(lag_current, lag_output)
#removing null values
tLag1 <- tLag1[complete.cases(tLag1), ]

beforeUSDvsEUR_lag <- lag(USDvsEUR, 2)
lag_output <- USDvsEUR
#combining the column
tLag2 <- bind_cols(beforeUSDvsEUR_lag, lag_current, lag_output)
#removing Null values
tLag2 <- tLag2[complete.cases(tLag2), ]

beforeUSDvsEUR_lag2 <- lag(USDvsEUR, 3)
lag_output <- USDvsEUR
tLag3 <- bind_cols(beforeUSDvsEUR_lag2, beforeUSDvsEUR_lag, lag_current, lag_output)
tLag3 <- tLag3[complete.cases(tLag3), ]



beforeUSDvsEUR_lag3 <- lag(USDvsEUR, 4)
lag_output <- USDvsEUR
tLag4 <-
  bind_cols(beforeUSDvsEUR_lag3, beforeUSDvsEUR_lag2, beforeUSDvsEUR_lag, lag_current, lag_output)
tLag4 <- tLag4[complete.cases(tLag4), ]

beforeUSDvsEUR_lag4 <- lag(USDvsEUR, 5)
lag_output <- USDvsEUR
tLag5 <-
  bind_cols(beforeUSDvsEUR_lag4, beforeUSDvsEUR_lag3, beforeUSDvsEUR_lag2, beforeUSDvsEUR_lag, lag_current, lag_output)
tLag5 <- tLag5[complete.cases(tLag5), ]

beforeUSDvsEUR_lag5 <- lag(USDvsEUR, 6)
lag_output <- USDvsEUR
tLag6 <-
  bind_cols(beforeUSDvsEUR_lag5,
            beforeUSDvsEUR_lag4,
            beforeUSDvsEUR_lag3,
            beforeUSDvsEUR_lag2,
            beforeUSDvsEUR_lag,
            lag_current,
            lag_output)
tLag6 <- tLag6[complete.cases(tLag6), ]

#sixth day lag
beforeUSDvsEUR_lag6 <- lag(USDvsEUR, 7)
lag_output <- USDvsEUR
#tLag7 <- bind_cols(beforeUSDvsEUR_lag6,beforeUSDvsEUR_lag5,beforeUSDvsEUR_lag4,beforeUSDvsEUR_lag3,beforeUSDvsEUR_lag2,
#beforeUSDvsEUR_lag, lag_current, lag_output) ,,,, #G previous 5 ekk na
tLag7 <-
  bind_cols(beforeUSDvsEUR_lag6,
            beforeUSDvsEUR_lag5,
            beforeUSDvsEUR_lag4,
            beforeUSDvsEUR_lag3,
            beforeUSDvsEUR_lag2,
            beforeUSDvsEUR_lag,
            lag_current,
            lag_output)
tLag7 <- tLag7[complete.cases(tLag7), ]
#view the matrice
tLag1
tLag2
tLag3
tLag4
tLag7

#part c

#min and max values that used to denormalize
min_value1 <- min(tLag1)
max_value1 <- max(tLag1)
min_value2 <- min(tLag2)
max_value2 <- max(tLag2)
min_value3 <- min(tLag3)
max_value3 <- max(tLag3)
min_value4 <- min(tLag4)
max_value4 <- max(tLag4)
min_value4 <- min(tLag5)
max_value4 <- max(tLag5)
min_value4 <- min(tLag6)
max_value4 <- max(tLag6)
min_value7 <- min(tLag7)
max_value7 <- max(tLag7)
#normalizing
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# see side by side matrix normalization
#see the last 7 days of USDvsEUR rate

par(mfrow = c(1, 2))
#list of matrix to pass through the loop
matrix_List <- list(tLag1, tLag2, tLag3,
                    tLag4, tLag5, tLag6, tLag7)
# loop to get the box plots normalization ( before and after)
for (i in 1:length(matrix_List)) {
  boxplot(matrix_List[[i]])
  matrix_List[[i]] <-
    as.data.frame(apply(matrix_List[[i]], 2, normalize))
  print(paste0("Summary for tLag", i, ":"))
  print(summary(matrix_List[[i]]))
  boxplot(matrix_List[[i]])
}
# column names
colnames(tLag1) <- c("lag_current", "lag_output")
colnames(tLag2) <- c("beforeUSDvsEUR_lag", "lag_current", "lag_output")
colnames(tLag3) <-
  c("beforeUSDvsEUR_lag2", "beforeUSDvsEUR_lag", "lag_current", "lag_output")
colnames(tLag4) <-
  c("beforeUSDvsEUR_lag3", "beforeUSDvsEUR_lag2", "beforeUSDvsEUR_lag", "lag_current",
    "lag_output")
colnames(tLag5) <-
  c("beforeUSDvsEUR_lag4",
    "beforeUSDvsEUR_lag3",
    "beforeUSDvsEUR_lag2",
    "beforeUSDvsEUR_lag",
    "lag_current",
    "lag_output")
colnames(tLag6) <-
  c(
    "beforeUSDvsEUR_lag5",
    "beforeUSDvsEUR_lag4",
    "beforeUSDvsEUR_lag3",
    "beforeUSDvsEUR_lag2",
    "beforeUSDvsEUR_lag",
    "lag_current",
    "lag_output"
  )

colnames(tLag7) <-
  c(
    "beforeUSDvsEUR_lag6",
    "beforeUSDvsEUR_lag5",
    "beforeUSDvsEUR_lag4",
    "beforeUSDvsEUR_lag3",
    "beforeUSDvsEUR_lag2",
    "beforeUSDvsEUR_lag",
    "lag_current",
    "lag_output"
  )


#partD
#unnormalizing function
denormalize <- function(x, min, max) {
  return((max - min) * x + min)
}
#FOR A ONE INPUT:tLag1
#set seed for reproducibility
set.seed(123)
#get number of rows in dataset
n_rows <- nrow(tLag1)
#get index of rows for training and testing
train_index <- sample(seq_len(n_rows), size = 380, replace = FALSE)
test_index <- setdiff(seq_len(n_rows), train_index)
#split train-test dataset
train_data <- tLag1[train_index,]
test_data <- tLag1[test_index,]

#make nueral networks
#train the 1st neural network 
nn1 <- neuralnet(
  lag_output ~ lag_current,
  data = train_data,
  hidden = c(2, 4),
  act.fct = tanh,
  #85
  linear.output = TRUE
)
#plot for the above neural network
plot(nn1)
#train the 2nd neural network
nn2 <- neuralnet(
  lag_output ~ lag_current,
  data = train_data,
  hidden = 5,
  act.fct = "logistic",
  linear.output = FALSE
)
#plot gor the above NN
plot(nn2)
#train the 3rd neural network
nn3 <- neuralnet(
  lag_output ~ lag_current,
  data = train_data,
  hidden = c(3, 7),
  act.fct = tanh,
  linear.output = TRUE
)
#plot for the above neural network
plot(nn3)

#extract actual values from test data
output_denorm <- test_data$lag_output
output_denorm <-
  denormalize(output_denorm, min = min_value1, max = max_value1)

#list to store all the results
results <- list()
#loop through the three nns
for (i in 1:3) {
  nn_name <- paste0("nn", i)
  
  #prediction
  nn_predic <- predict(get(nn_name), test_data)
  nn_predic <-
    denormalize(nn_predic, min = min_value1, max = max_value1)
  
  # RMSE
  
  rmse <- sqrt(mean((nn_predic - output_denorm) ^ 2))
  
  # MAE
  
  mae <- mean(abs(nn_predic - output_denorm))
  
  
  # MAPE
  
  mape <-
    mean(abs((output_denorm - nn_predic) / output_denorm)) * 100
  
  # sMAPE
  
  smape <-
    mean(2 * abs(nn_predic - output_denorm) / (abs(nn_predic) +
                                                 abs(output_denorm))) * 100
  
  
  
  
  
  #saving the results
  results[[i]] <-
    list(
      nn = nn_name,
      rmse = rmse,
      mae = mae,
      mape = mape,
      smape = smape
    )
  
  #display the result
  cat("Results for neural network ", nn_name, "\n")
  cat(paste0("RMSE: ", round(rmse, 2), "\n"))
  cat(paste0("MAE: ", round(mae, 2), "\n"))
  cat(paste0("MAPE: ", round(mape, 2), "%\n"))
  cat(paste0("sMAPE: ", round(smape, 2), "%\n\n"))
}

#FOR 2 INPUT: tLag2
#set seed for reproducibility
set.seed(123)
#get number of rows in dataset
n_rows <- nrow(tLag2)
#get index of rows for training and testing
train_index <- sample(seq_len(n_rows), size = 380, replace = FALSE)
test_index <- setdiff(seq_len(n_rows), train_index)
#split to train and test data set
train_data <- tLag2[train_index,]
test_data <- tLag2[test_index,]

# NEURAL NETWORKS
set.seed(123)
#train the  4th neural network 
nn4 <- neuralnet(
  lag_output ~ lag_current + beforeUSDvsEUR_lag,
  data = train_data,
  hidden = c(6, 3),
  act.fct = tanh,
  
  linear.output = FALSE
)
#plot the  4th neural network 
plot(nn4)
#train the  5th neural network 
nn5 <- neuralnet(
  lag_output ~ beforeUSDvsEUR_lag + lag_current,
  data = train_data,
  hidden = c(4, 7) ,
  act.fct = "logistic",
  linear.output = TRUE
)
#plot the  5th neural network 
plot(nn5)
#train the  6th neural network 
nn6 <- neuralnet(
  lag_output ~ beforeUSDvsEUR_lag + lag_current,
  data = train_data,
  hidden = 8,
  act.fct = tanh,
  linear.output = TRUE
)

#plot the  6th neural network 
plot(nn6)
#extract actual values from test data
output_denorm <- test_data$lag_output
output_denorm <-
  denormalize(output_denorm, min = min_value2, max = max_value2)
#loop through the three neural networks
for (i in 4:6) {
  nn_name <- paste0("nn", i)
  
  #prediction
  nn_predic <- predict(get(nn_name), test_data)
  nn_predic <-
    denormalize(nn_predic, min = min_value2, max = max_value2)
  
  #calculate RMSE
  rmse <- sqrt(mean((nn_predic - output_denorm) ^ 2))
  
  #calculate MAE
  mae <- mean(abs(nn_predic - output_denorm))
  
  #calculate MAPE
  mape <-
    mean(abs((output_denorm - nn_predic) / output_denorm)) * 100
  
  #calculate sMAPE
  smape <-
    mean(2 * abs(nn_predic - output_denorm) / (abs(nn_predic) +
                                                 abs(output_denorm))) *
    100
  
  
  #save results
  results[[i]] <-
    list(
      nn = nn_name,
      rmse = rmse,
      mae = mae,
      mape = mape,
      smape = smape
    )
  
  #display results
  
  cat("Results for neural network ", nn_name, "\n")
  cat(paste0("RMSE: ", round(rmse, 2), "\n"))
  cat(paste0("MAE: ", round(mae, 2), "\n"))
  cat(paste0("MAPE: ", round(mape, 2), "%\n"))
  cat(paste0("sMAPE: ", round(smape, 2), "%\n\n"))
}



# FOR 3 INPUT: tLag3
#set seed for reproducibility
set.seed(123)
#getting the number of rows in dataset
n_rows <- nrow(tLag3)
#get index of rows for training and testing
train_index <- sample(seq_len(n_rows), size = 380, replace = FALSE)
test_index <- setdiff(seq_len(n_rows), train_index)
#train test data split
train_data <- tLag3[train_index,]
test_data <- tLag3[test_index,]

#convert lag_output to numeric
train_data$lag_output <- as.numeric(train_data$lag_output)
test_data$lag_output <- as.numeric(test_data$lag_output)
# NEURAL NETWORKS
#set seed for reproducibility
set.seed(123)

#train the neural network 7
nn7 <-
  neuralnet(
    lag_output ~ beforeUSDvsEUR_lag2 + beforeUSDvsEUR_lag + lag_current,
    data = train_data,
    hidden = 6,
    act.fct = tanh,
    linear.output = TRUE
  )
#plot the 7th neural network 
plot(nn7)
#train the 8th neural network 
nn8 <-
  neuralnet(
    lag_output ~ beforeUSDvsEUR_lag2 + beforeUSDvsEUR_lag + lag_current,
    data = train_data,
    hidden = c(3, 6),
    act.fct = "logistic",
    linear.output = TRUE
  )
#plot the  8th neural network 
plot(nn8)
#train the  9th neural network 

nn9 <-
  neuralnet(
    lag_output ~ beforeUSDvsEUR_lag2 + beforeUSDvsEUR_lag + lag_current,
    data = train_data,
    hidden = 9,
    act.fct = tanh,
    linear.output = FALSE
  )
#plot the  9th neural network 
plot(nn9)

#extract actual values from test data and unnormalizing
output_denorm <- test_data$lag_output
output_denorm <-
  denormalize(output_denorm, min = min_value3, max = max_value3)
#lloop through the three neural networks
for (i in 7:9) {
  nn_name <- paste0("nn", i)
  
  #predicting and unnormalizing
  nn_predic <- predict(get(nn_name), test_data)
  nn_predic <-
    denormalize(nn_predic, min = min_value3, max = max_value3)
  
  #calculate RMSE
  rmse <- sqrt(mean((nn_predic - output_denorm) ^ 2))
  
  #calculate MAE
  mae <- mean(abs(nn_predic - output_denorm))
  
  #calculate MAPE
  mape <-
    mean(abs((output_denorm - nn_predic) / output_denorm)) * 100
  
  #calculate sMAPE
  smape <-
    mean(2 * abs(nn_predic - output_denorm) / (abs(nn_predic) +
                                                 abs(output_denorm))) *
    100
  
  #save to list
  results[[i]] <-
    list(
      nn = nn_name,
      rmse = rmse,
      mae = mae,
      mape = mape,
      smape = smape
    )
  
  
  #display results
  cat("Results for neural network ", nn_name, "\n")
  cat(paste0("RMSE: ", round(rmse, 2), "\n"))
  cat(paste0("MAE: ", round(mae, 2), "\n"))
  cat(paste0("MAPE: ", round(mape, 2), "%\n"))
  cat(paste0("sMAPE: ", round(smape, 2), "%\n\n"))
}
#FOR 4 INPUT: tLag4

set.seed(123)

#get number of rows in dataset
n_rows <- nrow(tLag4)
#get index of rows for training and testing
train_index <- sample(seq_len(n_rows), size = 380, replace = FALSE)
test_index <- setdiff(seq_len(n_rows), train_index)
#train test data split
train_data <- tLag4[train_index,]
test_data <- tLag4[test_index,]
# NEURAL NETWORKS

set.seed(123)
#train the  10th neural network 1

nn10 <-
  neuralnet(
    lag_output ~ beforeUSDvsEUR_lag3 + beforeUSDvsEUR_lag2 + beforeUSDvsEUR_lag + lag_current,
    data = train_data,
    hidden = c(5, 7),
    act.fct = tanh,
    linear.output = TRUE
  )
#plot the  10th neural network 
plot(nn10)
#train the  11th neural network 
nn11 <-
  neuralnet(
    lag_output ~ beforeUSDvsEUR_lag3 + beforeUSDvsEUR_lag2 + beforeUSDvsEUR_lag + lag_current,
    data = train_data,
    hidden = 10,
    act.fct = "logistic",
    
    linear.output = TRUE
  )
#plot the  11th neural network 
plot(nn11)
#train the  12th neural network 
nn12 <-
  neuralnet(
    lag_output ~ beforeUSDvsEUR_lag3 + beforeUSDvsEUR_lag2 + beforeUSDvsEUR_lag + lag_current,
    data = train_data,
    hidden = c(4, 8),
    act.fct = tanh,
    linear.output = TRUE
  )
#plot the  12th neural network 
plot(nn12)
#extract real values and denormalize test data

output_denorm <- test_data$lag_output
output_denorm <-
  denormalize(output_denorm, min = min_value4, max = max_value4)
#loop through the three neural networks
for (i in 10:12) {
  nn_name <- paste0("nn", i)
  
  #predicting and unnormalizing
  nn_predic <- predict(get(nn_name), test_data)
  nn_predic <-
    denormalize(nn_predic, min = min_value4, max = max_value4)
  
  #calculate RMSE
  rmse <- sqrt(mean((nn_predic - output_denorm) ^ 2))
  
  
  #calculate MAE
  mae <- mean(abs(nn_predic - output_denorm))
  
  #calculate MAPE
  mape <-
    mean(abs((output_denorm - nn_predic) / output_denorm)) * 100
  
  #calculate sMAPE
  smape <-
    mean(2 * abs(nn_predic - output_denorm) / (abs(nn_predic) +
                                                 abs(output_denorm))) *
    100
  
  #saving to the list
  results[[i]] <-
    list(
      nn = nn_name,
      rmse = rmse,
      mae = mae,
      mape = mape,
      smape = smape
    )
  
  #display results
  cat("Results for neural network ", nn_name, "\n")
  
  cat(paste0("RMSE: ", round(rmse, 2), "\n"))
  cat(paste0("MAE: ", round(mae, 2), "\n"))
  cat(paste0("MAPE: ", round(mape, 2), "%\n"))
  cat(paste0("sMAPE: ", round(smape, 2), "%\n\n"))
}

#FOR 7 INPUT: tLag7
#set seed for reproducibility
set.seed(123)
#get number of rows in dataset
n_rows <- nrow(tLag7)
#get index of rows for training and testing
train_index <- sample(seq_len(n_rows), size = 380, replace = FALSE)
test_index <- setdiff(seq_len(n_rows), train_index)
#train-test split
train_data <- tLag7[train_index,]
test_data <- tLag7[test_index,]
# NEURAL NETWORKS
#set seed for reproducibility
set.seed(123)
#train the  13 thneural network 
nn13 <- neuralnet(
  lag_output ~
    beforeUSDvsEUR_lag6 + beforeUSDvsEUR_lag5 + beforeUSDvsEUR_lag4 + beforeUSDvsEUR_lag3 + beforeUSDvsEUR_lag2 +
    beforeUSDvsEUR_lag + lag_current,
  data =
    train_data,
  
  
  hidden = c(7, 8),
  act.fct = tanh,
  linear.output = TRUE
)
#plot the 13th neural network 
plot(nn13)
#train the  14th neural network 
nn14 <- neuralnet(
  lag_output ~
    beforeUSDvsEUR_lag6 + beforeUSDvsEUR_lag5 + beforeUSDvsEUR_lag4 + beforeUSDvsEUR_lag3 + beforeUSDvsEUR_lag2 +
    beforeUSDvsEUR_lag + lag_current,
  data =
    train_data,
  hidden = c(4, 7),
  act.fct = "logistic",
  linear.output = TRUE
)
#pplot the  14th neural network 
plot(nn14)
#train the  15th neural network 
nn15 <- neuralnet(
  lag_output ~
    beforeUSDvsEUR_lag6 + beforeUSDvsEUR_lag5 + beforeUSDvsEUR_lag4 + beforeUSDvsEUR_lag3 + beforeUSDvsEUR_lag2 +
    beforeUSDvsEUR_lag + lag_current,
  data =
    train_data,
  hidden = c(10, 5),
  act.fct = tanh,
  linear.output = TRUE
)
#plot the  15th neural network 
plot(nn15)

#extract actual values from test data and unnormalizing
output_denorm <- test_data$lag_output
output_denorm <-
  denormalize(output_denorm, min = min_value7, max = max_value7)
#loop through the three neural networks
for (i in 13:15) {
  nn_name <- paste0("nn", i)
  
  #predicting and unnormalizing
  nn_predic <- predict(get(nn_name), test_data)
  nn_predic <-
    denormalize(nn_predic, min = min_value7, max = max_value7)
  
  #calculate RMSE
  rmse <- sqrt(mean((nn_predic - output_denorm) ^ 2))
  
  #calculate MAE
  mae <- mean(abs(nn_predic - output_denorm))
  
  #calculate MAPE
  mape <-
    mean(abs((output_denorm - nn_predic) / output_denorm)) * 100
  
  
  #calculate sMAPE
  smape <-
    mean(2 * abs(nn_predic - output_denorm) / (abs(nn_predic) + abs(output_denorm))) * 100
  
  #store the results in the list
  results[[i]] <-
    list(
      nn = nn_name,
      rmse = rmse,
      mae = mae,
      mape = mape,
      smape = smape
    )
  
  #print the results
  cat("Results for neural network ", nn_name, "\n")
  cat(paste0("RMSE: ", round(rmse, 2), "\n"))
  cat(paste0("MAE: ", round(mae, 2), "\n"))
  cat(paste0("MAPE: ", round(mape, 2), "%\n"))
  cat(paste0("sMAPE: ", round(smape, 2), "%\n\n"))
}

#Part h
#Scatter Plot
# Take the RMSE values out of the list of results.
rmse_values <- sapply(results, function(res)
  res$rmse)

# Find which MLP network index has the lowest RMSE.
best_mlp_index <- which.min(rmse_values)

# Get the most effective MLP network
best_mlp <- get(paste0("nn", best_mlp_index))

# Predictions for the top MLP network
best_model_pred <- predict(best_mlp, test_data)
best_model_pred <-
  denormalize(best_model_pred, min = min_value1, max = max_value1)

# Actual values
output_denorm <-
  denormalize(test_data$lag_output, min = min_value1, max = max_value1)

# Create a scatter plot
plot(
  output_denorm,
  best_model_pred,
  col = "blue",
  pch = 16,
  xlab = "Actual Values",
  ylab = "Predicted Values",
  main = "Actual vs Predicted Values"
)

# Add a diagonal line for reference
abline(0, 1, col = "red")


legend("bottomright",
       legend = "Ideal Fit",
       col = "red",
       lty = 1)

# Add a regression line (optional)
# abline(lm(best_model_pred ~ output_denorm), col = "green")

#Line chart
# Extract RMSE values from the results list
rmse_values <- sapply(results, function(res)
  res$rmse)

# Find the index of the MLP network with the lowest RMSE
best_mlp_index <- which.min(rmse_values)

# Retrieve the best MLP network
best_mlp <- get(paste0("nn", best_mlp_index))

# Predictions of the best MLP network
best_model_pred <- predict(best_mlp, test_data)
best_model_pred <-
  denormalize(best_model_pred, min = min_value1, max = max_value1)

# Actual values
output_denorm <-
  denormalize(test_data$lag_output, min = min_value1, max = max_value1)

# Create a line chart
plot(
  output_denorm,
  type = "l",
  col = "blue",
  ylim = range(c(output_denorm, best_model_pred)),
  xlab = "Time",
  ylab = "USDvsEUR Rate",
  main = "Predictions vs. Actual Values"
)
lines(best_model_pred, col = "red")
legend(
  "topright",
  legend = c("Actual", "Predicted"),
  col = c("blue", "red"),
  lty = 1
)