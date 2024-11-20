# Polapo Invest Portfolio Backtesting Platform

## 🏆 Ranked Top 30 in the Upstage 2024 Global AI Week Hackathon

<a href="https://ibb.co/rvK61fS"><img src="https://i.ibb.co/9GmTC3L/Global-AI-Week-Polapo-Invest.png" alt="Global-AI-Week-Polapo-Invest" border="0"></a>

## 🔑 INSTALLATION GUIDES
1. Clone the repository:
```bash
git clone https://github.com/Polapo-Invest/Polapo-Invest-PBP.git
```
2. Navigate to folder:
```bash
cd OPT-WEP
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create `.env` file in the `OPT-WEP` directory for storing API keys and store all necessary key values
```
GEMINI_API_KEY_SECRET='INPUT_YOUR_GEMINI_API_KEY_VALUE_HERE'
PREDIBASE_API_KEY='INPUT_YOUR_PREDIBASE_API_KEY_VALUE_HERE'
PINECONE_API_KEY='INPUT_YOUR_PINECONE_API_KEY_VALUE_HERE'
```
5. Run application:
```bash
python main.py
```
## 📌 Project Overview
### 1. Background (Problem we're solving)
The Polapo Invest PBP is an abbreviation of our project name, 'Polapo Invest Portfolio Backtesting Platform'.

We believe that people working in finance, such as portfolio managers, quant researchers, and quant traders, need a platform where they can easily and efficiently access to not only plain financial data like stock prices, but intuitive investment portfolio insights.

Also, many individual beginner stock investors have high possibilities to be exposed to the risk of panic trading. We think this is due to a lack of investment experience and an unsystematic investment strategy.

### 2. Solution
To solve the problem stated above, we've worked on developing a platform with features of a portfolio backtesting engine based on asset allocation modeling and LLM-based chatbots for the users to gain investment insights and establish investment strategies based on figures and data.

### 3. Core technology
• Asset allocation modeling: Building cross-sectional models such as equal weight (EW) and global minimum variance (GMV) and time-series models such as volatility targeting(VT) and CVaR targeting to derive the optimal asset allocation ratio

• Portfolio Backtesting Engine: An engine that analyzes the profitability and performance of a user's investment portfolio based on asset allocation modeling

• LLM-based chatbots: Utilized Predibase and LlamaIndex for fine-tuning Solar LLM (solar-1-mini-chat-240612) and setting up a RAG system with SEC data (Section 1A - Risk Factors, Section 7 - Management’s Discussion and Analysis of Financial Condition and Results of Operations). Gemini 1.5 pro model was also used for general chatting and financial graph and chart analysis.

• Web App: Implemented above features in a Flask web application in a user-friendly UI

### 4. Customer audience
• People working in the finance field such as portfolio managers, quant researchers, and quant traders, who want more efficiency in their workflow.

• From individual beginner investors exposed to the risk of panic trading to advanced investors who want easy access to investment portfolio insights.

## 💻 Usage

You can watch the video by clicking the image below.

[<img src="https://i.ibb.co/6XKYp6X/image.png" alt="Polap_Invest"/>](https://youtu.be/CAy_eMLIP4Y)

## ⚙️ System Configuration and Architecture
![image](https://i.ibb.co/LYM1Vds/image.png)
![image](https://i.ibb.co/Rg9DYmQ/image.png)
![image](https://i.ibb.co/g38s0By/image.png)
