from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import quantstats as qs
from scipy.optimize import minimize
from scipy.stats import norm
from io import BytesIO
import base64

app = Flask(__name__)

# Function to get ETF price data
def get_etf_price_data():
    tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOG', 'XLK']
    etf = yf.Tickers(tickers)
    data = etf.history(start='2010-01-01', actions=False)
    data.drop(['Open', 'High', 'Low', 'Volume'], inplace=True, axis=1)
    data = data.droplevel(0, axis=1)
    data.ffill(inplace=True)
    df = data.resample('W').last()
    return df

df = get_etf_price_data()

# Portfolio Backtesting Engine Class
class GEMTU772:
    # Initialization Function
    def __init__(self, price, param=52):
        self.param = param
        self.rets = price.pct_change().dropna()
        self.er = np.array(self.rets * self.param)
        self.vol = np.array(self.rets.rolling(self.param).std() * np.sqrt(self.param))
        cov = self.rets.rolling(self.param).cov().dropna() * self.param
        self.cov = cov.values.reshape(int(cov.shape[0]/cov.shape[1]), cov.shape[1], cov.shape[1])
        self.cost = 0.0005

    class CrossSectional:
        def ew(self, er):
            noa = er.shape[0]
            weights = np.ones_like(er) * (1/noa)
            return weights

        def msr(self, er, cov):
            noa = er.shape[0]
            init_guess = np.repeat(1/noa, noa)
            bounds = ((0.0, 1.0), ) * noa
            weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

            def neg_sharpe(weights, er, cov):
                r = weights.T @ er
                vol = np.sqrt(weights.T @ cov @ weights)
                return - r / vol

            weights = minimize(neg_sharpe, init_guess, args=(er, cov), method='SLSQP', constraints=(weights_sum_to_1,), bounds=bounds)
            return weights.x

        def gmv(self, cov):
            noa = cov.shape[0]
            init_guess = np.repeat(1/noa, noa)
            bounds = ((0.0, 1.0), ) * noa
            weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

            def port_vol(weights, cov):
                vol = np.sqrt(weights.T @ cov @ weights)
                return vol

            weights = minimize(port_vol, init_guess, args=(cov,), method='SLSQP', constraints=(weights_sum_to_1,), bounds=bounds)
            return weights.x

        def mdp(self, vol, cov):
            noa = vol.shape[0]
            init_guess = np.repeat(1/noa, noa)
            bounds = ((0.0, 1.0), ) * noa
            weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

            def neg_div_ratio(weights, vol, cov):
                weighted_vol = weights.T @ vol
                port_vol = np.sqrt(weights.T @ cov @ weights)
                return - weighted_vol / port_vol

            weights = minimize(neg_div_ratio, init_guess, args=(vol, cov), method='SLSQP', constraints=(weights_sum_to_1,), bounds=bounds)
            return weights.x

        def rp(self, cov):
            noa = cov.shape[0]
            init_guess = np.repeat(1/noa, noa)
            bounds = ((0.0, 1.0), ) * noa
            target_risk = np.repeat(1/noa, noa)
            weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

            def msd_risk(weights, target_risk, cov):
                port_var = weights.T @ cov @ weights
                marginal_contribs = cov @ weights
                risk_contribs = np.multiply(marginal_contribs, weights.T) / port_var
                w_contribs = risk_contribs
                return ((w_contribs - target_risk)**2).sum()

            weights = minimize(msd_risk, init_guess, args=(target_risk, cov), method='SLSQP', constraints=(weights_sum_to_1,), bounds=bounds)
            return weights.x

        def emv(self, vol):
            inv_vol = 1 / vol
            weights = inv_vol / inv_vol.sum()
            return weights

    class TimeSeries:
        def vt(self, port_rets, param, vol_target=0.1):
            vol = port_rets.rolling(param).std().fillna(0) * np.sqrt(param)
            weights = (vol_target / vol).replace([np.inf, -np.inf], 0).shift(1).fillna(0)
            weights[weights > 1] = 1
            return weights

        def cvt(self, port_rets, param, delta=0.01, cvar_target=0.05):
            def calculate_CVaR(rets, delta=0.01):
                VaR = rets.quantile(delta)
                return rets[rets <= VaR].mean()

            rolling_CVaR = -port_rets.rolling(param).apply(calculate_CVaR, args=(delta,)).fillna(0)
            weights = (cvar_target / rolling_CVaR).replace([np.inf, -np.inf], 0).shift(1).fillna(0)
            weights[weights > 1] = 1
            return weights

        def kl(self, port_rets, param):
            sharpe_ratio = (port_rets.rolling(param).mean() * np.sqrt(param) / port_rets.rolling(param).std())
            weights = pd.Series(2 * norm.cdf(sharpe_ratio) - 1, index=port_rets.index).fillna(0)
            weights[weights < 0] = 0
            weights = weights.shift(1).fillna(0)
            return weights

        def cppi(self, port_rets, m=3, floor=0.7, init_val=1):
            n_steps = len(port_rets)
            port_value = init_val
            floor_value = init_val * floor
            peak = init_val

            port_history = pd.Series(dtype=np.float64).reindex_like(port_rets)
            weight_history = pd.Series(dtype=np.float64).reindex_like(port_rets)
            floor_history = pd.Series(dtype=np.float64).reindex_like(port_rets)

            for step in range(n_steps):
                peak = np.maximum(peak, port_value)
                floor_value = peak * floor

                cushion = (port_value - floor_value) / port_value
                weight = m * cushion

                risky_alloc = port_value * weight
                safe_alloc = port_value * (1 - weight)
                port_value = risky_alloc * (1 + port_rets.iloc[step]) + safe_alloc

                port_history.iloc[step] = port_value
                weight_history.iloc[step] = weight
                floor_history.iloc[step] = floor_value

            return weight_history.shift(1).fillna(0)

    def transaction_cost(self, weights_df, rets_df, cost=0.0005):
        prev_weights_df = (weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:,:])) \
            .div((weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:,:])).sum(axis=1), axis=0)

        cost_df = abs(weights_df - prev_weights_df) * cost
        cost_df.fillna(0, inplace=True)
        return cost_df

    def run(self, cs_model, ts_model, cost):
        backtest_dict = {}
        rets = self.rets

        for i, index in enumerate(rets.index[self.param-1:]):
            if cs_model == 'EW':
                backtest_dict[index] = self.CrossSectional().ew(self.er[i])
            elif cs_model == 'MSR':
                backtest_dict[index] = self.CrossSectional().msr(self.er[i], self.cov[i])
            elif cs_model == 'GMV':
                backtest_dict[index] = self.CrossSectional().gmv(self.cov[i])
            elif cs_model == 'MDP':
                backtest_dict[index] = self.CrossSectional().mdp(self.vol[i], self.cov[i])
            elif cs_model == 'EMV':
                backtest_dict[index] = self.CrossSectional().emv(self.vol[i])
            elif cs_model == 'RP':
                backtest_dict[index] = self.CrossSectional().rp(self.cov[i])

        cs_weights = pd.DataFrame(list(backtest_dict.values()), index=backtest_dict.keys(), columns=rets.columns)
        cs_weights.fillna(0, inplace=True)
        cs_rets = cs_weights.shift(1) * rets.iloc[self.param-1:,:]
        cs_port_rets = cs_rets.sum(axis=1)

        if ts_model == 'VT':
            ts_weights = self.TimeSeries().vt(cs_port_rets, self.param)
        elif ts_model == 'CVT':
            ts_weights = self.TimeSeries().cvt(cs_port_rets, self.param)
        elif ts_model == 'KL':
            ts_weights = self.TimeSeries().kl(cs_port_rets, self.param)
        elif ts_model == 'CPPI':
            ts_weights = self.TimeSeries().cppi(cs_port_rets)
        elif ts_model == None:
            ts_weights = 1

        port_weights = cs_weights.multiply(ts_weights, axis=0)
        cost = self.transaction_cost(port_weights, rets)
        port_asset_rets = port_weights.shift() * rets - cost
        port_rets = port_asset_rets.sum(axis=1)
        port_rets.index = pd.to_datetime(port_rets.index).strftime("%Y-%m-%d")

        return port_weights, port_asset_rets, port_rets

    def performance_analytics(self, port_weights, port_asset_rets, port_rets):
        img = BytesIO()
        
        plt.figure(figsize=(12, 7))
        port_weights['Cash'] = 1 - port_weights.sum(axis=1)
        plt.stackplot(port_weights.index, port_weights.T, labels=port_weights.columns)
        plt.title('Portfolio Weights')
        plt.xlabel('Date')
        plt.ylabel('Weights')
        plt.legend(loc='upper left')
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url1 = base64.b64encode(img.getvalue()).decode('utf8')

        img = BytesIO()
        plt.figure(figsize=(12, 7))
        plt.plot((1 + port_asset_rets).cumprod() - 1)
        plt.title('Underlying Asset Performance')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.legend(port_asset_rets.columns, loc='upper left')
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')

        img = BytesIO()
        plt.figure(figsize=(12, 7))
        plt.plot((1 + port_rets).cumprod() - 1)
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url3 = base64.b64encode(img.getvalue()).decode('utf8')

        return plot_url1, plot_url2, plot_url3

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        cs_model = request.form.get('cs_model')
        ts_model = request.form.get('ts_model')
        engine = GEMTU772(df)
        res = engine.run(cs_model=cs_model, ts_model=ts_model, cost=0.0005)
        port_weights = res[0]
        port_asset_rets = res[1]
        port_rets = res[2]
        plot_url1, plot_url2, plot_url3 = engine.performance_analytics(port_weights, port_asset_rets, port_rets)
        return render_template('index.html', plot_url1=plot_url1, plot_url2=plot_url2, plot_url3=plot_url3)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
