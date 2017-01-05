require(datasets)
require(forecast)
library(e1071)  # kurtosis
library(dplyr)

setwd("~/Documents/KNOU/Paper_project/experiment/sunspot")

dataset <- sunspot.year[1:288] 
n.train <- 221  # 1700~1920
n.test  <- 67   # 1921~1987


## AR(9)
fit <- arima(dataset[1:n.train], order = c(9,0,0))

my.predict <- function(fit, dataset, step, ar.order = 9) {
  coef <- fit$coef[1:ar.order]
  intercept <- fit$coef[ar.order+1]
  val <- intercept
  for (i in seq(1:ar.order)) {
    val <- val + coef[i] * (dataset[step-i] - intercept)
  }
  names(val) = c()
  val
}

## experiment ----------------------------------------
mse = 0
mae = 0
mape = 0

from.idx = 222
to.idx = 288

for (i in from.idx:to.idx) {
  y = dataset[i]
  y_ = my.predict(fit, dataset, i)
  print(c(y, y_))
  mse = mse + (y - y_)^2
  mae = mae + abs(y - y_)
  mape = mape + abs(y - y_) / y
}
mse <- mse / length(from.idx:to.idx)
mae <- mae / length(from.idx:to.idx)
mape <- mape / length(from.idx:to.idx) 

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
    
    cat(m, k, "\n")
  }
  cat("Best: m", best_m, " k", best_k)  ## best_m = 37
  return(best_m)
}


## prepare dataset for NN ----------------------------------------------
x <- sunspot.year
sunspot <- data.frame(x=x)
write.csv(sunspot, file = 'sunspot_data.csv', row.names = TRUE)

## prepare dataset B (decomposed) -------------------------------------
x <- sunspot.year
ma_filter <- find_best_m(x[1:length(x)-1])  #37
x_ma <- ma(x, ma_filter)
x_residual <- x - x_ma
sunspot_b <- data.frame(x = x, x_ma=round(x_ma,2), x_residual=round(x_residual,2))
write.csv(sunspot_b, file = 'sunspot_data_b.csv', row.names = TRUE)
