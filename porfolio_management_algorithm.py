import pandas as pd
import numpy as np

import pandas_datareader.data as web
import datetime


def SafePortfolio(dcs=[1, 1, 1, 1, 1, 1, 1, 1],
                  sd=datetime.datetime(1981, 1, 1),
                  ed=datetime.datetime(2022, 4, 21)):
    if sum(dcs) == 0:
        return ('DANGER!!! Cash Out Now')

    ## prep data
    first_or_not = 0
    stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'BRK-A', 'UNH', 'JNJ', 'WMT']
    for i in range(8):
        if dcs[i] == 1:
            price = web.DataReader(stocks[i], 'yahoo', sd, ed)
            price = price[['Close']].add_suffix('_' + stocks[i])
            if first_or_not == 0:
                fulldata = price
                first_or_not += 1
            else:
                fulldata = fulldata.join(price)

    ## calculate return
    numit = fulldata.shape[1]
    for j in range(numit):
        colname = 'return' + str(j)
        fulldata[colname] = np.log(fulldata.iloc[:, j]) - np.log(fulldata.iloc[:, j].shift(5))

    ## calculate %
    fulldata = fulldata.iloc[:, numit:]
    C = fulldata.cov().to_numpy()
    C_inv = np.linalg.inv(C)
    OC_inv = np.matmul(np.array([1] * numit), C_inv)
    OC_invOt = np.matmul(OC_inv, np.array([[1]] * numit))
    res = OC_inv / OC_invOt

    ## deal with negative case
    ## In practice, a negative weight value mean taking a net short position
    ## For simplicity, let cut them out

    for l in range(numit):
        if res[l] < 0:
            res[l] = 0
    res = res / sum(res)

    res_fill = []
    res_pos = 0
    for k in range(8):
        if dcs[k] == 1:
            res_fill.append(res[res_pos])
            res_pos += 1
        else:
            res_fill.append(0)

    return res_fill

def InterpretPortfolio(dcs, pcport):
    stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'BRK-A', 'UNH', 'JNJ', 'WMT']
    explanation = list()
    for i in range(8):
        if dcs[i] == 1 and pcport[i] > 0:
            print('We recommend to have ' + stocks[i] + ' ' + str(round(pcport[i] * 100, 2)) + '% of your portfolio.')
            explanation.append('We recommend to have ' + stocks[i] + ' ' + str(round(pcport[i] * 100, 2)) + '% of your portfolio.')
        elif dcs[i] == 1 and pcport[i] == 0:
            print('To minimize the risk, we do not recommend buying ' + stocks[i] + '.')
            explanation.append('To minimize the risk, we do not recommend buying ' + stocks[i] + '.')
        else:
            print('According to our model, we do not recommend buying ' + stocks[i] + '.')
            explanation.append('According to our model, we do not recommend buying ' + stocks[i] + '.')

    return explanation

# decision = [0, 0, 0, 0, 1, 1, 1, 0]
# start_date = datetime.datetime(2021, 4, 19)
# end_date = datetime.datetime(2022, 4, 19)
#
# result = SafePortfolio(decision, start_date, end_date)
# print(result)
# [0, 0, 0, 0, 0.4381285216148676, 0.11931025780641576, 0.44256122057871666, 0]

# print(InterpretPortfolio(decision, result))