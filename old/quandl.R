# Quandl package must be installed
library(Quandl)

# Get your API key from quandl.com
quandl_api = "vkS5iE4CbLvky5oneS9F"
Quandl.api_key(quandl_api)

quandl_get <-
  function(sym, start_date="2012-01-01",end_date = "2017-05-31") {
    require(devtools)
    require(Quandl)
    # create a vector with all lines
    tryCatch(Quandl(c(
      paste0("WIKI/GOLD_DAILY_USD"),  #  Adj. Gold Daily USD
      start_date = start_date,
      end_date = end_date,
      type = "zoo"
    )))
  }
print(typeof(quandl_get))