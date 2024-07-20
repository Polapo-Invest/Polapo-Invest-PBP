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
import matplotlib
from PIL import Image
import tempfile


matplotlib.use('Agg') # Engine reset issue solution code (TkAgg->Agg)

load_dotenv()

genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

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

            rolling_CVaR = -port_rets.rolling(param).apply(calculate_CVaR, args=(delta,))
            weights = (cvar_target / rolling_CVaR).replace([np.inf, -np.inf], 0).shift(1).fillna(0)
            weights[weights > 1] = 1
            return weights
        #KL
        def kl(self, port_rets, param):
            rolling_mean = port_rets.rolling(param).mean()
            rolling_vol = port_rets.rolling(param).std()
            kelly_weights = rolling_mean / rolling_vol**2
            kelly_weights = kelly_weights.replace([np.inf, -np.inf], 0).shift(1).fillna(0)
            kelly_weights[kelly_weights < 0] = 0
            kelly_weights[kelly_weights > 1] = 1
            return kelly_weights
        #CPPI
        def cppi(self, port_rets, floor_value=1, cushion=0.1):
            port_value = (1 + port_rets).cumprod()
            weights = (port_value - floor_value) / port_value
            weights = weights.clip(lower=0).shift(1).fillna(0)
            return weights

    # Portfolio Value Calculation
    def portfolio_value(self, cs_model, ts_model, cost=0.0005, initial_value=1):
        weight_history = pd.DataFrame(index=self.rets.index, columns=self.rets.columns)
        port_history = pd.Series(index=self.rets.index)
        floor_history = pd.Series(index=self.rets.index)

        # Calculate Portfolio Values
        port_value = initial_value
        floor_value = initial_value
        for step in self.rets.index[self.param-1:]:
            # Get Weight and Portfolio Return
            weight = self.run(cs_model, ts_model, cost)[0].loc[step]
            port_rets = self.rets.loc[step]
            risky_alloc = weight.sum()
            safe_alloc = port_value - risky_alloc

            # Calculate Portfolio Value
            port_value = risky_alloc * (1 + port_rets) + safe_alloc

            # Store Values
            port_history.loc[step] = port_value
            weight_history.loc[step] = weight
            floor_history.loc[step] = floor_value

        return weight_history.shift(1).fillna(0)
   
    # Transaction Cost Function (Compound rate of return method assuming reinvestment)
    def transaction_cost(self, weights_df, rets_df, cost=0.0005):
        prev_weights_df = weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:,:])
        sum_weights = (weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:,:])).sum(axis=1)
        sum_weights_replaced = sum_weights.replace(0, np.nan)
        normalized_weights_df = prev_weights_df.div(sum_weights_replaced, axis=0)
        
        # Investment Weight of Previous Period (The backslash ('\') in Python is used as a line continuation character.)
        cost_df = abs(weights_df - normalized_weights_df) * cost
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
    

    def performance_analytics(self, port_rets):
        if not isinstance(port_rets.index, pd.DatetimeIndex):
            port_rets.index = pd.to_datetime(port_rets.index)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
            qs.reports.html(port_rets, output=tmp_file.name)
            tmp_file.seek(0)
            report_html = tmp_file.read().decode('utf-8')

        return report_html
    
    
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
    
#Route function
@app.route('/', methods=["GET", "POST"])
def report():
    if request.method == "POST":
        cs_model = request.form.get('cs_model') # cs model selection
        ts_model = request.form.get('ts_model') # ts model selection
        engine = GEMTU772(df) # Run backtesting
        res = engine.run(cs_model=cs_model, ts_model=ts_model, cost=0.0005)
        port_weights, port_asset_rets, port_rets = res

        report_html = engine.performance_analytics(port_rets)

        return render_template('report_viewer.html', report_html=report_html)

    return render_template('index.html')



@app.route("/generate_text", methods=["GET", "POST"])
def generate_text():
    if request.method == "POST":
        input_data = request.get_json()
        prompt = input_data["prompt"]
        model = genai.GenerativeModel(model_name="gemini-1.5-pro",
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

def process_image_data(img_data):
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes))
    return img


@app.route("/generate_text_stream", methods=["GET", "POST"])
def generate_text_stream():
    if request.method == "POST":
        input_data = request.get_json()
        prompt = input_data["prompt"]
        image_data = input_data.get("images", [])
        images = [process_image_data(img) for img in image_data]
        model = genai.GenerativeModel(model_name="gemini-1.5-pro",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        def generate_stream():
            response = model.generate_content([prompt] + images, stream=True)
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
