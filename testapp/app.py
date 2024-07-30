import os
import google.generativeai as genai
from flask import Flask, render_template, request, Response, jsonify, stream_with_context
from dotenv import load_dotenv
from io import BytesIO
import base64
import matplotlib
from PIL import Image

from Backtesting_Engine import GEMTU772
from ETF_Functions import *

matplotlib.use('Agg') # Engine reset issue solution code (TkAgg->Agg)

load_dotenv()

genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

app = Flask(__name__)

df = get_etf_price_data()
    
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
    app.run(debug=True)