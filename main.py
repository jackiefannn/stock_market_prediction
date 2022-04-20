import os
import time
import base64
from io import BytesIO
import numpy as np

from flask import Flask, render_template, request, url_for
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

from data_preprocessing import preprocess_data

# initialising the flask app

app = Flask(__name__)

@app.route("/")
def home():
    tickers = ['AAPL', 'FB', 'AMZN', 'NFLX', 'GOOG', 'MSFT', 'QQQ', 'XLK']
    return render_template("home.html", tickers=tickers)

@app.route("/view_model", methods=['POST'])
def view_model():
    if request.method == "POST":
        stock = request.form.get("stock")
        print(str(stock))
        df = preprocess_data(stock)

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
        trading_decisions = ['Sell', 'Hold', 'Buy']
        weighted_decisions = dict()
        if df['decision'].iloc[-2] == int(0): weighted_decisions['Two-Days'] = [70, 25, 5]
        elif df['decision'].iloc[-2] == int(1): weighted_decisions['Two-Days'] = [5, 25, 70]
        elif df['decision'].iloc[-2] == int(2): weighted_decisions['Two-Days'] = [10, 80, 10]
        if df['decision'].iloc[-3] == int(0): weighted_decisions['Three-Days'] = [70, 25, 5]
        elif df['decision'].iloc[-3] == int(1): weighted_decisions['Three-Days'] = [5, 25, 70]
        elif df['decision'].iloc[-3] == int(2): weighted_decisions['Three-Days'] = [10, 80, 10]
        if df['decision'].iloc[-5] == int(0): weighted_decisions['Five-Days'] = [70, 25, 5]
        elif df['decision'].iloc[-5] == int(1): weighted_decisions['Five-Days'] = [5, 25, 70]
        elif df['decision'].iloc[-5] == int(2): weighted_decisions['Five-Days'] = [10, 80, 10]


        labels = list(weighted_decisions.keys())
        data = np.array(list(weighted_decisions.values()))
        data_cum = data.cumsum(axis=1)
        category_colors = ['#FF4933', '#FFF533', '#72FF33']
        # category_colors = [[255, 73, 51], [255, 245, 51] , [114, 255, 51 ]]
        # category_colors = plt.colormaps['RdYlGn'](np.linspace(0.15, 0.85, data.shape[1]))
        fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.invert_yaxis()
        # ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(trading_decisions, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(labels, widths, left=starts, height=0.5,
                            label=colname, color=color)

            # r, g, b = color
            # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            # ax.bar_label(rects, label_type='center', color=text_color)
            ax.bar_label(rects, label_type='center', color='black')
        ax.legend(ncol=len(trading_decisions), bbox_to_anchor=(0, 1),
                  loc='upper right', fontsize='small')
        plt.title(str(stock) + ' Future Trading Decisions')
        plt.xlabel('Confidence Level')
        plt.ylabel('Decision')
        plt.legend(loc='upper right')
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        decision_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Macro Plot
        img = BytesIO()
        plt.plot(df['DJ_Close'], label='Dow Jones')
        plt.plot(df['SP_Close'], label='S&P 500')
        plt.plot(df['ND_Close'], label='Nasdaq')
        plt.title('Macro Data Information')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        macro_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Suggested Portfolio Plot
        img = BytesIO()
        portfolio = np.array([20, 8, 20, 4, 12, 16, 10, 10])
        tickers = ['AAPL', 'FB', 'AMZN', 'NFLX', 'GOOG', 'MSFT', 'QQQ', 'XLK']
        plt.pie(portfolio, labels=tickers, startangle=90, autopct=lambda x: '{:.0f}'.format(x*portfolio.sum()/100))
        plt.title('Suggested Portfolio')
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        portfolio_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template("view_model.html", stock=stock, stock_plot_url=stock_plot_url,
                               decision_plot_url=decision_plot_url, macro_plot_url=macro_plot_url,
                               portfolio_plot_url=portfolio_plot_url, df=df)


if __name__ == "__main__":
    app.run(debug=True)
