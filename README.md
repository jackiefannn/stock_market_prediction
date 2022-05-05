# Machine Learning for Predicting Investment Decision and Portfolio Management
This web application gives a list of stocks as well as a trading decision (sell, hold, buy) for the stocks. 
It will show the confidence level of the trading decision based on a percentage (sell: 0%, hold: 50%, buy: 100%).

This web application will also show portfolio management strategy of which stocks to have along with the percentage 
of how much should be invested in a specific stock. 

## How to Run the Web Application
### Initializing the Application
After cloning the repository, start the application by running the 'main.py' program. This will initialize the Flask app
and the web application can be viewed on localhost (http://127.0.0.1:5000/). In this example URL, 5000 is the port 
number that the web application is running on and the specific port number can be found in the output console when 
running the program. 

### Navigating the Application
On the home screen, it will display the suggested portfolio along with the percentages of total capital to hold of 
the stocks for the current day. At the top bar, there are a list of buttons that link to specific stock information.
Once a button is clicked, the web application will navigate to another page with the following features:
- Suggested portfolio for all stocks
- Trading decision for the stock
- Bar chart showing the confidence level of the trading decision for 1, 3, and 5 days in the future
- Line graph showing the stock price 
- Line graph showing market indicators including Dow Jones, S&P 500, and NASDAQ

