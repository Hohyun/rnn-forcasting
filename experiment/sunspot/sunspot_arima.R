require(datasets)
require(forecast)
library(e1071)  # kurtosis
library(dplyr)
library(tidyr)
library(ggplot2)

setwd("~/Documents/KNOU/rnn-forcasting/experiment/sunspot")

# Helper functions --------------------------------------------------
## decompose gaussian distribution and residual
my.kurtosis <- function(x) {
  m4 <- sum((x - mean(x))^4) / length(x)
  s2 <- var(x)^2
  (m4/s2) - 3
}

## find best moving average filter value fit kurtosis 3
find_best_m <- function(x) {
  best_m <- 100
  best_k <- 100
  
  for (m in 2:100) {
    ma.x <- ma(x, m) 
    k <- kurtosis(ma.x, na.rm = TRUE)
    
    if (abs(k) < abs(best_k)) {
      best_m <- m
      best_k <- k
    }
    
    #cat(m, k, "\n")
  }
  cat("Best: m", best_m, " k", best_k)  ## best_m = 37
  return(best_m)
}

# Prepare dataset -----------------------------------------------------
## prepare dataset for NN 
x <- sunspot.year
sunspot <- data.frame(x=x)
write.csv(sunspot, file = 'sunspot_data.csv', row.names = TRUE)

## prepare dataset B (decomposed) 
x <- sunspot.year
ma_filter <- find_best_m(x[1:length(x)-1])  #37
x_ma <- ma(x, ma_filter)
x_residual <- x - x_ma
sunspot_b <- data.frame(x = x, x_ma=round(x_ma,2), x_residual=round(x_residual,2))
write.csv(sunspot_b, file = 'sunspot_data_b.csv', row.names = TRUE)


# Experiment: sunspot AR(9) ------------------------------------------
dataset <- sunspot.year[1:288] 
n.train <- 221  # 1700~1920
n.test  <- 67   # 1921~1987

fit_ar900 <- arima(dataset[1:n.train], order = c(9,0,0))

my.predict1 <- function(fit, dataset, step, ar.order = 9) {
  coef <- fit$coef[1:ar.order]
  intercept <- fit$coef[ar.order+1]
  val <- intercept
  for (i in seq(1:ar.order)) {
    val <- val + coef[i] * (dataset[step-i] - intercept)
  }
  names(val) = c()
  val
}

do_experiment_ar900 = function(from.idx = 222, to.idx = 288) {
  mse = 0;  mae = 0;  mape = 0;  t = 1
  df = data.frame()
  
  for (i in from.idx:to.idx) {
    y = dataset[i]
    y_ = my.predict1(fit_ar900, dataset, i)
    df = bind_rows(df, data.frame(time_step=t, actual=y, predict=y_))
    #df = bind_rows(df, data.frame(time_step=t, category="actual", value = y))
    #df = bind_rows(df, data.frame(time_step=t, category="predict", value = y_))
    # print(c(y, y_))
    mse = mse + (y - y_)^2
    mae = mae + abs(y - y_)
    mape = mape + abs(y - y_) / abs(y)
    t = t + 1
  }
  mse <- mse / length(from.idx:to.idx)
  mae <- mae / length(from.idx:to.idx)
  mape <- mape / length(from.idx:to.idx)   
  cat("MAE:", mae, "MSE:", mse, "MAPE:", mape)
  return (df)
}

df_ar900 <- do_experiment_ar900()
df_temp <- gather(df_ar900, 'category', value, -time_step)
p <- ggplot(data = df_ar900, aes(x=time_step, y=value, color=category)) 
p + geom_line() + ggtitle("ARIMA") # + theme(legend.position="bottom")

# Experiment: sunspot ARIMA-ma ---------------------------------------


x <- sunspot.year
ma_filter <- find_best_m(x[1:length(x)-1])  #37
x_ma <- ma(x, ma_filter)   # NA : 1-18, 271-289
x_residual <- x - x_ma

n_na <-18
auto.arima(x_ma[19:221])  # ARIMA(2,1,0)
fit_ar210 <- arima(x_ma[19:221], order = c(2,1,0))

my.predict2 <- function(fit, dataset, step, ar.order=2) {
  ma_diff = diff(dataset)
  coef <- fit$coef[1:ar.order]
  #intercept <- fit$coef[ar.order+1]
  val <- 0
  for (i in seq(1:ar.order)) {
    val <- val + coef[i] * dataset[step-i-1]
  }
  val <- sum(ma_diff[1:step-1]) + val
  names(val) = c()
  val
}

do_experiment_ar210 = function(from.idx = 222-18, to.idx = 269) {
  mse = 0;  mae = 0;  mape = 0;  t = 1
  df = data.frame()
  
  for (i in from.idx:to.idx) {
    y = dataset[i]
    y_ = my.predict1(fit_ar900, x_ma[19:269], i, 18)
    df = bind_rows(df, data.frame(time_step=t, actual=y, predict=y_))
    #df = bind_rows(df, data.frame(time_step=t, category="actual", value = y))
    #df = bind_rows(df, data.frame(time_step=t, category="predict", value = y_))
    # print(c(y, y_))
    mse = mse + (y - y_)^2
    mae = mae + abs(y - y_)
    mape = mape + abs(y - y_) / abs(y)
    t = t + 1
  }
  mse <- mse / length(from.idx:to.idx)
  mae <- mae / length(from.idx:to.idx)
  mape <- mape / length(from.idx:to.idx)   
  cat("MAE:", mae, "MSE:", mse, "MAPE:", mape)
  return (df)
}

df_ar900 <- do_experiment_ar900()
df_temp <- gather(df_ar900, 'category', value, -time_step)
p <- ggplot(data = df_ar900, aes(x=time_step, y=value, color=category)) 
p + geom_line() + ggtitle("ARIMA") # + theme(legend.position="bottom")
