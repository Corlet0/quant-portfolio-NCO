def prices_download(tickers: str,
                    start_date: str,
                    end_date: str,
                    interval: str,
                    max_ffill_days: int = 0
                   ):
  """Imports Adj Prices from yahoo finance into a df.
    Parameters:
            tickers: str, list
                List of tickers to download ('AAPL', 'MSFT', 'AMAZN')

            start_date: str
                Download start date string (YYYY-MM-DD) E.g. for start=”2020-01-01”, the first data point will be on “2020-01-01”

            end_date: str
                Download end date string (YYYY-MM-DD) E.g. for end=”2023-01-01”, the last data point will be on “2022-12-31”

            interval: str
                Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

            max_ffill_days : int, default = 0
                Maximum limit of consecutive days to propagate prices for holidays

  """
  #Downloads prices
  df_prices = yf.download(tickers,
                          start = start_date,
                          end = end_date,
                          interval = interval,
                          auto_adjust= True,
                          progress = False)['Close']

  # Ensure the output is strictly a 2D DataFrame even if a single ticker was passed
  if isinstance(df_prices, pd.Series):
      df_prices = df_prices.to_frame()

  # Cleans data if there is missing values

    #  Drop assets with no observations
  df_prices = df_prices.dropna(how="all", axis=1)

    # Bounded forward-fill to bridge exchange holidays across different markets
  if max_ffill_days > 0:
      df_prices = df_prices.ffill(limit=max_ffill_days)

    # Drop any remaining missing observations to prevent look-ahead bias and artificial data
  df_prices = df_prices.dropna(how="any", axis=0)

  return df_prices

#------------------------------------------------------------------------------------------------#

def returns_download(
    tickers,
    start_date,
    end_date,
    interval="1d",
    max_ffill_days=0,
    type_returns ="simple"
):
    """
    Downloads prices and calculates 
      if type_returns = "simple" (default):
        simple arithmetic daily returns:
          R_t = (P_t - P_{t-1}) / P_{t-1}

      if type_returns = "log"
        log daily returns:
           R_t = log(P_t) - log(P_{t-1})

    Returns
      pd.DataFrame
          Clean T x N matrix of simple returns without missing values.
    """
    df_prices = prices_download(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        max_ffill_days=max_ffill_days,
    )

    # Simple arithmetic returns
    if type_returns == "simple":
      df_returns = df_prices.pct_change().dropna(how="any")
    if type_returns == "log":
      df_returns = np.log(df_prices).diff().dropna()

    return df_returns
