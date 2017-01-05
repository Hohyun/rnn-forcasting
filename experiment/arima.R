str(AirPassengers)

# 1. check data is stationary. if not, transform into 
plot(AirPassengers)
plot(stl(AirPassengers, s.window = "periodic"))

library(tseries)
adf.test(diff(log(AirPassengers)), alternative = "stationary", k = 0)

# Search best parameter using ACF/PACF or auto.arima
library(forecast)
auto.arima(diff(log(AirPassengers)))
tsdiag(auto.arima(diff(log(AirPassengers))))

# modeling arima
fit <- arima(log(AirPassengers), c(1,0,1),
             seasonal = list(order=c(0,1,1), periods = 12))

# prediction
pred <- predict(fit, n.ahead = 10*12)
ts.plot(AirPassengers, 2.718^pred$pred, log="y", lty=c(1,3))
