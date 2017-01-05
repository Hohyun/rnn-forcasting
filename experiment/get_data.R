library(quantmod)
library(plotly)
library(dygraphs)
setwd("/Users/hohkim/Documents/KNOU/Paper_project")

getSymbols("^KS11")
getSymbols("^GSPC")
getSymbols("^N225")
getSymbols("000001.SS")
#getSymbols("^DAX")
#getSymbols("^FTSE")

getSymbols("^VIX")
getFX("USD/KRW", from="2011-01-01")
dygraph(USDKRW)
save(USDKRW, file = "exchange_usdkrw.rda")
#load("exchange_usdkrw.rda")

saveSymbols(c(KS11, GSPC, N225, `000001.SS`, VIX), file.path = "/Users/hohkim/Documents/KNOU/Paper_project/stock_idx.csv")
