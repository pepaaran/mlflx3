library(tidyverse)
# library(patchwork)
# library(ggrepel)
library(yardstick)
# library(rbeni)    # get it by devtools::install_github("https://github.com/stineb/rbeni.git")
# library(cowplot)
# library(jcolors)
# library(ingestr)
# library(khroma)
# library(ggplot2)

source(here::here("R/rolling_rsq.R"))

# Leave-site-out predictions from LSTM model without categorical covariates
lstm_lso <- read_csv("model/preds/lstm_lso_epochs_150_patience_20_hdim_256_conditional_0.csv")

# Leave-site-out predictions from LSTM model with less parameters
lstm_lso_128 <- read_csv("model/preds/lstm_lso_epochs_150_patience_20_hdim_128_conditional_0.csv")

# Leave-site-out predictions from DNN
dnn_lso <- read_csv("model/preds/dnn_lso_epochs_150_patience_20.csv")

# Leave-vegetation-out predictions from LSTM


# Leave-continent-out predictions from LSTM


# P-model simulations
pmodel <- read_csv("data/external/pmodel_outputs.csv")

# Input data frame containing sites metadata
df_metadata <- read_csv("data/processed/df_imputed.csv") |>
  dplyr::group_by(sitename) |>
  dplyr::summarise(ai = first(ai),
                   koeppen_code = first(koeppen_code),
                   classid = first(classid))

# Get LSTM and DNN rolling R2 scores
sites <- df_metadata$sitename
n_sites <- length(sites)

# Set window to compute rolling R2
window <- 365

# Initialize object with results for first site
rolling_r2_sites <- data.frame(
  sitename = sites[1],
  lstm = lstm_lso |>
    filter(sitename == sites[1]) |>
    rolling_rsq(truth = 'GPP_NT_VUT_REF', estimate = "gpp_lstm"),
  dnn = dnn_lso |>
    filter(sitename == sites[1]) |>
    rolling_rsq(truth = 'GPP_NT_VUT_REF', estimate = 'gpp_dnn'))

rolling_r2_sites <- cbind(rolling_r2_sites,
                          lstm_lso |>
                            filter(sitename == sites[1]) |>
                            select(date) |>
                            slice_tail(n = nrow(rolling_r2_sites)))

# Continue computing results for other sites
for(site in sites[-1]){
  print(paste("Computing rolling mean for:", site))
  
  r2_rolling <- data.frame(
    sitename = site,
    lstm = lstm_lso |>
      filter(sitename == site) |>
      rolling_rsq(truth = 'GPP_NT_VUT_REF', estimate = "gpp_lstm"),
    dnn = dnn_lso |>
      filter(sitename == site) |>
      rolling_rsq(truth = 'GPP_NT_VUT_REF', estimate = 'gpp_dnn')
  )
  
  r2_rolling <- cbind(r2_rolling, 
                      lstm_lso |>
                        filter(sitename == site) |>
                        select(date) |>
                        slice_tail(n = nrow(r2_rolling)))
  
  # Append to results from other sites
  rolling_r2_sites <- rbind(rolling_r2_sites,
                            r2_rolling)
}

# Write results
saveRDS(here::here("notebooks/rolling_r2_sites.rda"))

grouped <- rolling_r2_sites |>
  select(sitename, lstm, dnn) |>
  group_by(sitename)

# Aggregate the R2 scores, for the same "day since start of measurements"
rolling_r2_aggregated <- lapply(1:2193,     # length of the longest time series
                                function(i){ summarise(grouped, lstm = nth(lstm, i),
                                                       dnn = nth(dnn, i)) |>
                                    select(lstm, dnn) |> 
                                    apply(MARGIN = 2, FUN = mean, na.rm = TRUE)}) |>
  bind_rows() 

# Save aggregated values
saveRDS(rolling_r2_aggregated, file = here::here("notebooks/rolling_r2_aggregated.rda"))

