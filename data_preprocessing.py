import pandas as pd
import numpy as np

import pandas_datareader.data as web
import pandas_ta as ta
import datetime



def preprocess_data(ticker):
    start_date = datetime.datetime(2015,1,1)
    # end_date = datetime.datetime(2022,3,18)
    end_date = datetime.datetime.today() - datetime.timedelta(days=2) # up to yesterday's date

    ## stock data
    ## bug fix: https://stackoverflow.com/questions/69500226/on-running-this-python-code-in-google-colab-it-showing-me-error-can-anyone-ple
    price = web.DataReader(ticker, 'yahoo', start_date, end_date)
    div = web.DataReader(ticker, 'yahoo-dividends', start_date, end_date)  ## Ex-Dividend date

    ## market data
    DJI = web.DataReader('^DJI', 'yahoo', start_date, end_date)
    NASDAQCom = web.DataReader('^IXIC', 'yahoo', start_date, end_date)
    NASDAQ100 = web.DataReader('^NDX', 'yahoo', start_date, end_date)
    SP500 = web.DataReader('^GSPC', 'yahoo', start_date, end_date)

    ## ref: https://www.alpharithms.com/calculate-macd-python-272222/
    price.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    price['EMA_5'] = price.ta.ema(close='Close', timeperiod=5, append=False)
    price['EMA_10'] = price.ta.ema(close='Close', timeperiod=10, append=False)
    price['EMA_25'] = price.ta.ema(close='Close', timeperiod=25, append=False)
    price['RSI_14'] = price.ta.rsi(close='Close', length=14)
    
    # Merge data
    ## stock data
    price = price.add_prefix('stock_')
    div = div.add_prefix('stock_')

    ## market data
    DJI = DJI.add_prefix('DJ_')
    NASDAQ = NASDAQCom.add_prefix('ND_')
    SP500 = SP500.add_prefix('SP_')


    df = price.join(div)
    df = df.join(DJI)
    df = df.join(NASDAQ)
    df = df.join(SP500)
    df = df.drop(columns='stock_action')
    df['stock_value'] = df['stock_value'].fillna(0)
    df['stock_allvalue'] = df['stock_Close'] + df['stock_value']

    # Macro data
    macro = web.DataReader(['DEXCHUS', 'DEXUSUK', 'DTWEXBGS', 'DEXUSEU',
                            'DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2',
                            'DGS3', 'DGS5', 'DFII5', 'DGS7', 'DFII7', 'DGS10',
                            'DFII10', 'DGS20', 'DFII20', 'DGS30', 'DFII30',
                            'FEDFUNDS',
                            'BOGMBASE', 'M1SL', 'M2SL',
                            'CPIAUCSL', 'PCEPI', 'PCEPILFE',
                            'PAYEMS', 'UNRATE',
                            'BOGZ1FL663067003Q', 'GDPC1', 'GDP', 'A939RX0Q048SBEA'], 'fred', start_date, end_date)
    macro = macro.ffill(axis=0)
    df = df.join(macro)
    
    # Label decision
    df['decision'] = np.nan

    for i in range(df.shape[0] - 5):
        if df.stock_allvalue.iloc[i + 5] / df.stock_Close.iloc[i] - 1 >= 0.03:  # next 5 business days = 1 week
            df.decision.iloc[i] = 0 # sell
        elif df.stock_allvalue.iloc[i + 5] / df.stock_Close.iloc[i] - 1 <= -0.03:
            df.decision.iloc[i] = 1 # buy
        else:
            df.decision.iloc[i] = 2 # hold
    df = df.dropna()
    # print(df.info())

    return df

# print(preprocess_data('AAPL'))