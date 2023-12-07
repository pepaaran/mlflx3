# Function to get a rolling rsq measure

rolling_rsq <- function(data, truth, estimate, window = 365, traditional = FALSE){
  # Arguments:
  #   data: data.frame containing columns truth and estimate
  #   truth: column name indicating true, observed values
  #   estimate: column name for the estimated values
  #   window: length of the window over which the R2 is computed
  #   traditional: boolean indicating whether to compute the R2 as the
  #     traditional coefficient of determination or, if FALSE, as the
  #     correlation
  
  # Get number of rows in data
  n <- nrow(data)

  # Compute rolling R2 values, stepping one observation at a time
  # and starting on the first row
  if(traditional){
    lapply(1:(n-window+1), function(i){
      data[i:(i+window-1), ] |>
        rsq_trad(truth, estimate, na_rm = TRUE) |>
        pull(.estimate)
      }) |> unlist()
  }else{
  lapply(1:(n-window+1), function(i){
    data[i:(i+window-1), ] |>
      rsq(truth, estimate, na_rm = TRUE) |>
      pull(.estimate)
      }) |> unlist()
  }
}
