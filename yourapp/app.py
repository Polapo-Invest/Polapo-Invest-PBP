import os

import google.generativeai as genai

from flask import Flask, render_template, request, Response, jsonify, stream_with_context
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import quantstats as qs
from scipy.optimize import minimize
from scipy.stats import norm
from io import BytesIO
import base64

load_dotenv()

genai.configure(api_key = os.environ.get('GEMINI_API_KEY'))

app = Flask(__name__)



# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
  safety_settings = safety_settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("INSERT_INPUT_HERE")

print(response.text)




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
    
        # Annualization Parameter    
        self.param = param
    
        # Intraday Return Rate 
        self.rets = price.pct_change().dropna()
      
        # Expected Rate of Return        
        self.er = np.array(self.rets * self.param)
      
        # Volatility        
        self.vol = np.array(self.rets.rolling(self.param).std() * np.sqrt(self.param))
      
        # Covariance Matrix   
        cov = self.rets.rolling(self.param).cov().dropna() * self.param
      
        # Transaction Cost per Unit 
        self.cov = cov.values.reshape(int(cov.shape[0]/cov.shape[1]), cov.shape[1], cov.shape[1])
        
        self.cost = 0.0005
   
    # Cross-Sectional Risk Models Class 
    class CrossSectional:
        #EW
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
                r = weights.T @ er # @ means multiplication
                vol = np.sqrt(weights.T @ cov @ weights)
                return - r / vol

            weights = minimize(neg_sharpe, init_guess, args=(er, cov), method='SLSQP', constraints=(weights_sum_to_1,), bounds=bounds)
            return weights.x
        
        #GMV
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
        #MDP
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
        #RP
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
        #EMV
        def emv(self, vol):
            inv_vol = 1 / vol
            weights = inv_vol / inv_vol.sum()
            return weights
   
    # Time-Series Risk Models Class
    class TimeSeries:
        #VT
        def vt(self, port_rets, param, vol_target=0.1):
            vol = port_rets.rolling(param).std().fillna(0) * np.sqrt(param)
            weights = (vol_target / vol).replace([np.inf, -np.inf], 0).shift(1).fillna(0)
            weights[weights > 1] = 1
            return weights
        #CVT
        def cvt(self, port_rets, param, delta=0.01, cvar_target=0.05):
            def calculate_CVaR(rets, delta=0.01):
                VaR = rets.quantile(delta)
                return rets[rets <= VaR].mean()

            rolling_CVaR = -port_rets.rolling(param).apply(calculate_CVaR, args=(delta,)).fillna(0)
            weights = (cvar_target / rolling_CVaR).replace([np.inf, -np.inf], 0).shift(1).fillna(0)
            weights[weights > 1] = 1
            return weights
        #KL
        def kl(self, port_rets, param):
            sharpe_ratio = (port_rets.rolling(param).mean() * np.sqrt(param) / port_rets.rolling(param).std())
            weights = pd.Series(2 * norm.cdf(sharpe_ratio) - 1, index=port_rets.index).fillna(0)
            weights[weights < 0] = 0
            weights = weights.shift(1).fillna(0)
            return weights
        #CPPI
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
   
    # Transaction Cost Function (Compound rate of return method assuming reinvestment)
    def transaction_cost(self, weights_df, rets_df, cost=0.0005):
        prev_weights_df = (weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:,:])) \
            .div((weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:,:])).sum(axis=1), axis=0)
       
        # Investment Weight of Previous Period (The backslash ('\') in Python is used as a line continuation character.)
        cost_df = abs(weights_df - prev_weights_df) * cost
        cost_df.fillna(0, inplace=True)
        return cost_df
  
    # Backtesting Execution Function
    def run(self, cs_model, ts_model, cost):
        
        # Empty Dictionary   
        backtest_dict = {}
        
        # Intraday Return Rate DataFrame
        rets = self.rets
      
        # Select and Run Cross-Sectional Risk Models
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
     
        # Cross-Sectional Weights DataFrame
        cs_weights = pd.DataFrame(list(backtest_dict.values()), index=backtest_dict.keys(), columns=rets.columns)   
        cs_weights.fillna(0, inplace=True)
        
         # Cross-Sectional Risk Models Return on Assets
        cs_rets = cs_weights.shift(1) * rets.iloc[self.param-1:,:]
       
        # Cross-Sectional Risk Models Portfolio Return
        cs_port_rets = cs_rets.sum(axis=1)
        
        # Select and Run Time-Series Risk Models
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
            
        # Final Portfolio Investment Weights
        port_weights = cs_weights.multiply(ts_weights, axis=0)
        
        # Transaction Cost DataFrame
        cost = self.transaction_cost(port_weights, rets)
        
        # Final Portfolio Return by Assets
        port_asset_rets = port_weights.shift() * rets - cost
        
        # Final Portfolio Return
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

#라우트 함수 
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        cs_model = request.form.get('cs_model') #cs 모델 선택
        ts_model = request.form.get('ts_model') #ts 모델 선택
        engine = GEMTU772(df)  #벡테이스팅 수행
        res = engine.run(cs_model=cs_model, ts_model=ts_model, cost=0.0005) #run메서드
        port_weights = res[0]
        port_asset_rets = res[1]
        port_rets = res[2]
        plot_url1, plot_url2, plot_url3 = engine.performance_analytics(port_weights, port_asset_rets, port_rets)
        return render_template('index.html', plot_url1=plot_url1, plot_url2=plot_url2, plot_url3=plot_url3) #렌더링
    return render_template('index.html')
    


@app.route("/generate_text", methods=["GET", "POST"])
def generate_text():
    if request.method == "POST":
        input_data = request.get_json()
        prompt = input_data["prompt"]
        model = genai.GenerativeModel(model_name="gemini-pro",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)
        text_result = model.generate_content(prompt)
        return jsonify({
            "status": {
                "code": 200,
                "message": "Success generate text",
            },
            "data": {
                "result": text_result.text,
            }
        }), 200
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None
        }), 405


@app.route("/generate_text_stream", methods=["GET", "POST"])
def generate_text_stream():
    if request.method == "POST":
        input_data = request.get_json()
        prompt = input_data["prompt"]
        model = genai.GenerativeModel(model_name="gemini-pro",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        def generate_stream():
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                print(chunk.text)
                yield chunk.text + "\n"

        return Response(stream_with_context(generate_stream()), mimetype="text/plain")
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None
        }), 405



if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
