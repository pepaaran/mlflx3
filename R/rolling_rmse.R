# Function to get a rolling RMSE measure

rolling_rmse <- function(data, truth, estimate, window = 365){
  # Arguments:
  #   data: data.frame containing columns truth and estimate
  #   truth: column name indicating true, observed values
  #   estimate: column name for the estimated values
  #   window: length of the window over which the RMSE is computed
  
  # Get number of rows in data
  n <- nrow(data)
  
  # Compute differences between truth and estimate, squared
  dif2 <- (data[[truth]] - data[[estimate]])^2
  
  # Compute rolling RMSE values, stepping one observation at a time
  # and starting on the first row

  lapply(1:(n-window+1), function(i){
    dif2[i:(i+window-1)] |>
      mean(na.rm = TRUE) |>
      sqrt()
  }) |> unlist()
  
}
