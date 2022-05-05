import os
import time
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import datetime

from flask import Flask, render_template, request
import matplotlib.pyplot as plt

from data_preprocessing import preprocess_data
from porfolio_management_algorithm import SafePortfolio, InterpretPortfolio

# initialising the flask app

app = Flask(__name__)

@app.route("/")
def home():
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'BRK-A', 'UNH', 'JNJ', 'WMT']
    trading_decision, portfolio = dict(), dict()
    trading_decision['Five-Days'] = [0, 0, 0, 0, 0, 1, 1, 1]

    # Suggested Portfolio Plot
    img = BytesIO()
    # portfolio = np.array([20, 8, 20, 4, 12, 16, 10, 10])
    # portfolio_percentage = [0, 0, 0, 0, 0.4381, 0.1193, 0.4426, 0]
    start_date = datetime.datetime(2021, 4, 21)
    end_date = datetime.datetime(2022, 4, 21)
    portfolio_percentage = np.array(SafePortfolio(dcs=trading_decision['Five-Days'], sd=start_date, ed=end_date))
    explanation = InterpretPortfolio(dcs=trading_decision['Five-Days'], pcport=portfolio_percentage)

    def autopct_more_than_1(pct):
        return ('%.2f%%' % pct) if pct > 1 else ''

    labeled_tickers = list()
    labeled_portfolio = list()
    for ticker, percentage in zip(tickers, portfolio_percentage):
        if percentage > 0:
            labeled_tickers.append(ticker)
            labeled_portfolio.append(percentage)
    plt.pie(labeled_portfolio, labels=labeled_tickers, startangle=90,
            autopct=autopct_more_than_1)
    plt.title('Suggested Portfolio')
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    portfolio_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    for ticker, percentage in zip(tickers, portfolio_percentage):
        if percentage > 0.0: portfolio[ticker] = percentage

    return render_template("home.html", tickers=tickers, portfolio=portfolio,
                           portfolio_plot_url=portfolio_plot_url)

@app.route("/view_model", methods=['POST'])
def view_model():
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'BRK-A', 'UNH', 'JNJ', 'WMT']
    if request.method == "POST":
        stock = request.form.get("stock")
        print(str(stock))
        df = preprocess_data(stock)

        decisions_df_one_day = pd.read_csv('data/Predict_1_day.csv')
        decisions_df_three_days = pd.read_csv('data/Predict_3_day.csv')
        decisions_df_five_days = pd.read_csv('data/Predict_5_day.csv')
        print(decisions_df_five_days[str(stock)])
        confidence_level = dict()
        confidence_level['One-Day'], confidence_level['Three-Days'], confidence_level['Five-Days'] = list(), list(), list()
        for ticker in tickers:
            confidence_level['One-Day'].append(decisions_df_one_day[ticker].to_numpy()[0])
            confidence_level['Three-Days'].append(decisions_df_three_days[ticker].to_numpy()[0])
            confidence_level['Five-Days'].append(decisions_df_five_days[ticker].to_numpy()[0])
        print(confidence_level)
        trading_decision = dict() # short if less than 50% and long if greater than 50%
        trading_decision['One-Day'] = [1, 1, 1, 0, 1, 1, 1, 1]
        trading_decision['Three-Days'] = [0, 0, 1, 1, 1, 1, 0, 0]
        trading_decision['Five-Days'] = [0, 0, 0, 0, 0, 1, 1, 1]
        print(trading_decision)

        trading_strategy = ''
        i = 0
        for ticker, decision in zip(tickers, trading_decision['Five-Days']):
            if ticker == str(stock):
                if trading_decision['Five-Days'][i] == 1: trading_strategy = 'Long/Buy'
                else: trading_strategy = 'Short/Sell'
                break
            i += 1

        stock_info = dict()
        for ticker, decision, percentage in zip(tickers, trading_decision['One-Day'], confidence_level['One-Day']):
            if ticker == str(stock):
                stock_info['One-Day'] = [decision, percentage]
                break
        for ticker, decision, percentage in zip(tickers, trading_decision['Three-Days'], confidence_level['Three-Days']):
            if ticker == str(stock):
                stock_info['Three-Days'] = [decision, percentage]
                break
        for ticker, decision, percentage in zip(tickers, trading_decision['Five-Days'], confidence_level['Five-Days']):
            if ticker == str(stock):
                stock_info['Five-Days'] = [decision, percentage]
                break
        print(stock_info)

        # Stock Plot
        img = BytesIO()
        plt.plot(df['stock_Close'])
        plt.title(str(stock) + ' Stock Price Information')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        stock_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Trading Decision Plot
        img = BytesIO()

        df_decision = pd.DataFrame({'Confidence Level': [stock_info['Five-Days'][1], stock_info['Three-Days'][1], stock_info['One-Day'][1]]},
                          index = ['5-Days', '3-Days', '1-Day'])
        category_colors = ['#FF4933', '#FFF533', '#72FF33']
        def show_color(confidence_level):
            color = list()
            for percentage in confidence_level:
                if percentage < 50: color.append(category_colors[0])
                else: color.append(category_colors[2])
            print(color)
            return color
        # colors = show_color(df_decision['Confidence Level'])
        # colors = ['green', 'green', 'red']
        # df_decision.plot.barh(color=colors)
        df_decision.plot.barh()
        for index, value in enumerate(df_decision['Confidence Level']):
            plt.text(value, index, str(value))
        plt.title(str(stock) + ' Future Trading Decisions')
        plt.xlabel('Confidence Level (0: short, 50: hold, 100: long)')
        plt.xticks(np.arange(0, 101, step=25))
        plt.ylabel('Decision')
        # plt.legend(loc='upper right')
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        decision_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Market Indicators Plot
        img = BytesIO()
        plt.plot(df['DJ_Close'], label='Dow Jones')
        plt.plot(df['SP_Close'], label='S&P 500')
        plt.plot(df['ND_Close'], label='Nasdaq')
        plt.title('Market Indicators')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        market_indicator_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Suggested Portfolio Plot
        img = BytesIO()
        # portfolio = np.array([20, 8, 20, 4, 12, 16, 10, 10])
        tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'BRK-A', 'UNH', 'JNJ', 'WMT']
        # portfolio_percentage = [0, 0, 0, 0, 0.4381, 0.1193, 0.4426, 0]
        start_date = datetime.datetime(2021, 4, 21)
        end_date = datetime.datetime(2022, 4, 21)
        portfolio_percentage = np.array(SafePortfolio(dcs=trading_decision['Five-Days'], sd=start_date, ed=end_date))
        explanation = InterpretPortfolio(dcs=trading_decision['Five-Days'], pcport=portfolio_percentage)

        def autopct_more_than_1(pct):
            return ('%.2f%%' % pct) if pct > 1 else ''

        labeled_tickers = list()
        labeled_portfolio = list()
        for ticker, percentage in zip(tickers, portfolio_percentage):
            if percentage > 0:
                labeled_tickers.append(ticker)
                labeled_portfolio.append(percentage)
        plt.pie(labeled_portfolio, labels=labeled_tickers, startangle=90, autopct=autopct_more_than_1)

        # plt.pie(portfolio_percentage, labels=tickers, startangle=90, autopct=lambda x: '{:.4f}'.format(x*portfolio_percentage.sum()/100))
        plt.title('Suggested Portfolio')
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        portfolio_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template("view_model.html", stock=stock, stock_plot_url=stock_plot_url,
                               decision_plot_url=decision_plot_url, market_indicator_plot_url=market_indicator_plot_url,
                               portfolio_plot_url=portfolio_plot_url, df=df, portfolio_explanation=explanation,
                               trading_strategy=trading_strategy)


if __name__ == "__main__":
    app.run(debug=True)
