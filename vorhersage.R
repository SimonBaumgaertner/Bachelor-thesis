library(telescope)


data <- read.table("data.csv", sep = ",", header = TRUE)
# Specify the attribute name as a string
attribute_name <- "demand"

# Specify the number of rows for training and validation
training_size <- 8640 # lets give it about 1 year of training data
validation_size <- 24 # 24 hour forecast
# Let's determine out starting indeces randomly but chose values that are multiples of 24 (starting at 00:00 each day)
min <- training_size  # Minimum value
max <- 30000  # Maximum value (limited by dataset)
amount_of_samples <- 200  # Number of random startingIndexes you want
# Calculate the start and end values that are divisible by validation_size
start_value <- ceiling(min / validation_size) * validation_size
end_value <- floor(max / validation_size) * validation_size

# Generate the sequence
seq_values <- seq(start_value, end_value, by = validation_size)
# Generate random startingIndexes
random_startingIndexes <- sample(seq_values, amount_of_samples)

# Initialize a vector to store the RMSE values and remember the run for printing
actuals <- c()
predictions <- c()
run <- 1
for (startingIndex in random_startingIndexes) {
  # Create the training set and validation set using the dynamic attribute name
  training_data <- data[(startingIndex-training_size):(startingIndex-1), ][, attribute_name]
  validation_data <- data[startingIndex:(startingIndex + validation_size - 1), ][, attribute_name]

  # Make predictions with 'telescope.forecast'
  prediction <- telescope.forecast(training_data, h = length(validation_data),natural = TRUE, plot = FALSE)

  prediction_values <- as.vector(prediction$mean)


  run <- run + 1
  # Append the RMSE value to the vector
  actuals <- c(actuals, validation_data)
  predictions <- c(predictions, prediction_values)
}
# Calculate RMSE
rmse <- sqrt(mean((predictions - actuals)^2))

# Calculate MAE
mae <- mean(abs(predictions - actuals))

# Calculate MAPE
mape <- mean(abs((actuals - predictions) / actuals)) * 100

# Calculate standard deviation of error
sd <- sd(abs(actuals - predictions))

# Print the RMSE, MAE, and MAPE
cat("Root Mean Squared Error (RMSE) for attribute '", attribute_name, "':", rmse, "\n")
cat("Mean Absolute Error (MAE) for attribute '", attribute_name, "':", mae, "\n")
cat("Mean Absolute Percentage Error (MAPE) for attribute '", attribute_name, "':", mape, "%\n")
cat("Standard Deviation of Error (SD) for attribute '", attribute_name, "':", sd, "\n")

