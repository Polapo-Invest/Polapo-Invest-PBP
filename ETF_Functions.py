import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
import yahooquery as yq
import json

# Function to get ETF price data 
def get_etf_price_data(tickers: list, startyear: str):
    # tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOG', 'XLK']
    etf = yf.Tickers(tickers)
    data = etf.history(start=startyear, actions=False)
    data.drop(['Open', 'High', 'Low', 'Volume'], inplace=True, axis=1)
    data = data.droplevel(0, axis=1)
    data.ffill(inplace=True)
    df = data.resample('W').last()
    return df

def get_ticker_by_company_name(company_name):
    search = yq.search(company_name)
    quotes = search.get('quotes', [])
    
    print(quotes)
    if not quotes:
        result = json.dumps({'success':False, 'message':f"Wrong Company Name: {company_name} (Excluded from calculation)"})
        print(result)
        return result
    
    # Returning the first result
    ticker = quotes[0].get('symbol')
    print(ticker)
    result = json.dumps({'success': True, 'ticker':ticker})
    print(result)
    return result