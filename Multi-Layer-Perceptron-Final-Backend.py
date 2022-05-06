from talib import MA_Type,RSI, MACD, STOCH
import numpy as np
import datetime
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas_datareader import data, wb
import pandas_datareader.data as web
import seaborn as sns
import statsmodels.api as sm
import lime.lime_tabular
import tensorflow as tf
import warnings
from sklearn import metrics
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
plt.style.use('default')

train_start_date = datetime.datetime(2019,2,12)
train_end_date = datetime.datetime(2021,2,12)
test_start_date = datetime.datetime(2021,2,12)
test_end_date = datetime.datetime(2022,2,12)
lookAheadPeriod = 1
# Length of maximum indicator
cutOff = 17
macdCutOff = 33
FB = data.DataReader("FB", 'yahoo', train_start_date, test_end_date)

AAPL = data.DataReader("AAPL", 'yahoo', train_start_date, test_end_date)

AMZN = data.DataReader("AMZN", 'yahoo', train_start_date, test_end_date)

NFLX = data.DataReader("NFLX", 'yahoo', train_start_date, test_end_date)

GOOG = data.DataReader("GOOG", 'yahoo', train_start_date, test_end_date)

MSFT = data.DataReader("MSFT", 'yahoo', train_start_date, test_end_date)

QQQ = data.DataReader("QQQ", 'yahoo', train_start_date, test_end_date)

XLK = data.DataReader("XLK", 'yahoo', train_start_date, test_end_date)

def make_dfs(df,n):
    df_ind = pd.DataFrame()
    a = 0
    for i in range(n,len(df),n):
        df_ind = pd.concat([df_ind,pd.DataFrame(df[a:i].reshape(1,n))])
        a = i
    df_ind = df_ind.reset_index(drop=True)
    return df_ind

def data_cleaning_for_all_indicators(ticker_symbol, n):
    dur = 5
    ticker_symbol.index = [i.date() for i in ticker_symbol.index]
    closePrice = ticker_symbol['Close'].to_numpy()
    dateTimePrice = ticker_symbol.index.values
    highList = ticker_symbol['High'].to_numpy()
    lowList = ticker_symbol['Low'].to_numpy()
    
    shiftDateTime = dateTimePrice[:-lookAheadPeriod]
    shiftClosePrice = closePrice[lookAheadPeriod:]
    highList = highList[:-lookAheadPeriod]
    lowList = lowList[:-lookAheadPeriod]

    closePrice = closePrice[:-lookAheadPeriod]
    
    RSI_14 = RSI(closePrice, timeperiod=14)
    RSI_14 = RSI_14[cutOff:]
    
    STOCH14K, STOCH14D = STOCH(
            highList, lowList, closePrice, fastk_period=14, slowk_period=3, slowd_period=3)
    STOCH14K = STOCH14K[cutOff:]
    STOCH14D = STOCH14D[cutOff:]
    
    closeDiff = shiftClosePrice - closePrice
    closeDiffLength = len(closeDiff)
    
    longOP = np.zeros(closeDiffLength)
    longOP[closeDiff >= 0] = 1

    # Sell if closing price is lesser in the end
    shortOP = np.zeros(closeDiffLength)
    shortOP[closeDiff < 0] = 1
    
    RSI_with_period = make_dfs(RSI_14,n)
    STOCH14K_with_period = make_dfs(STOCH14K,n)
    STOCH14D_with_period = make_dfs(STOCH14D,n)
    
    for i in range(n):
        RSI_with_period = RSI_with_period.rename(columns={i:'rsi'+str(i)})
        STOCH14K_with_period = STOCH14K_with_period.rename(columns={i:'stochk'+str(i)})
        STOCH14D_with_period = STOCH14D_with_period.rename(columns={i:'stochd'+str(i)})
    
    final_df_all_inds = pd.concat([RSI_with_period, STOCH14K_with_period, STOCH14D_with_period],axis=1)

    newClosePrice = closePrice[17:]
    store_diff = []
    first_index = 0
    for i in range(n,len(newClosePrice) - dur,n):
        store_diff.append(newClosePrice[i + dur] - newClosePrice[first_index + 1])
        first_index = i
    newCloseDiffLength = len(store_diff)
    newLongOP = np.zeros(newCloseDiffLength)
    newLongOP[np.array(store_diff) >= 0] = 1
    final_df = pd.concat([final_df_all_inds, pd.DataFrame(newLongOP,columns=['long_or_short'])],axis=1)
    
    return final_df

def data_cleaning_for_todays_data(ticker_symbol, n):
    dur = 5
    lookAheadPeriod = 1
    ticker_symbol.index = [i.date() for i in ticker_symbol.index]
    if type(ticker_symbol['Close'].to_numpy()[0]) != 'numpy.float64':
        closePrice = ticker_symbol['Close'].to_numpy(np.float64)
        highList = ticker_symbol['High'].to_numpy(np.float64)
        lowList = ticker_symbol['Low'].to_numpy(np.float64)
    else:
        closePrice = ticker_symbol['Close'].to_numpy()
        highList = ticker_symbol['High'].to_numpy()
        lowList = ticker_symbol['Low'].to_numpy()
    dateTimePrice = ticker_symbol.index.values
    highList = highList[:-lookAheadPeriod]
    lowList = lowList[:-lookAheadPeriod]

    closePrice = closePrice[:-lookAheadPeriod]
    
    RSI_14 = RSI(closePrice, timeperiod=14)
    RSI_14 = RSI_14[-n:]
    
    STOCH14K, STOCH14D = STOCH(
            highList, lowList, closePrice, fastk_period=14, slowk_period=3, slowd_period=3)
    STOCH14K = STOCH14K[-n:]
    STOCH14D = STOCH14D[-n:]
    return np.concatenate((RSI_14, STOCH14K, STOCH14D))


def send_predictions():
    todays_date = datetime.datetime.today()
    days_to_calculate_RSI = datetime.datetime.today() - datetime.timedelta(days=60)
    AAPL = data.DataReader("AAPL", 'yahoo', days_to_calculate_RSI, todays_date)
    MSFT = data.DataReader("MSFT", 'yahoo', days_to_calculate_RSI, todays_date)

    GOOG = data.DataReader("GOOG", 'yahoo', days_to_calculate_RSI, todays_date)

    AMZN = data.DataReader("AMZN", 'yahoo', days_to_calculate_RSI, todays_date)

    UNH = data.DataReader("UNH", 'yahoo', days_to_calculate_RSI, todays_date)

    WMT = data.DataReader("WMT", 'yahoo', days_to_calculate_RSI, todays_date)

    JNJ = data.DataReader("JNJ", 'yahoo', days_to_calculate_RSI, todays_date)

    BRK_A = data.DataReader("BRK-A", 'yahoo', days_to_calculate_RSI, todays_date)
    ticker_symbols = [AAPL,MSFT,GOOG,AMZN,UNH,WMT,JNJ,BRK_A]
    model = load_model('ALL_model.h5')
    all_stocks_predictions = []
    long_or_short = []
    df_all =[]
    for ticker in ticker_symbols:
        df_today = data_cleaning_for_todays_data(ticker, 5)
        all_stocks_predictions.append(model.predict(df_today.reshape(1,15))[0][0])
        long_or_short.append(np.where(model.predict(df_today.reshape(1,15)) > threshold, 1,0))
        df_all.append(df_today)
    all_stocks_predictions = np.array(all_stocks_predictions).reshape(1,8) * 100
    all_stocks_predictions = all_stocks_predictions.round(2)
    df_today_preds = pd.DataFrame(all_stocks_predictions,columns=['AAPL','MSFT','GOOG','AMZN','UNH','WMT','JNJ','BRK-A'])
    df_today_preds.to_csv('Predict.csv',index=False)


if __name__ == '__main__':
    ticker_symbols = [FB,AAPL,AMZN,NFLX,GOOG,MSFT,QQQ,XLK]
    df_all_tickers_ind = pd.DataFrame()
    for ticker in ticker_symbols:
        single_ticker_df = data_cleaning_for_all_indicators(ticker, 5)
        df_all_tickers_ind = pd.concat([df_all_tickers_ind,single_ticker_df])
    X, y = df_all_tickers_ind.values[:,:-1],df_all_tickers_ind.values[:,-1]
    X = X.astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    model = Sequential()
    model.add(Dense(40, activation='tanh', input_shape=(n_features,)))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=10)
    # fit the model
    model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1, validation_data=(X_val,y_val),callbacks=[early_stop, reduce_lr])
    model.save("ALL_model.h5")
    send_predictions()
    