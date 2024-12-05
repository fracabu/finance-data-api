from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import ta

# Inizializzazione app Flask e CORS
app = Flask(__name__)
CORS(app)

# Configurazione errori personalizzata
app.config['PROPAGATE_EXCEPTIONS'] = True

def generate_finance_data(num_records):
    data = []
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "BTC-USD"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_records)
    dates = pd.date_range(start=start_date, end=end_date, periods=num_records)

    for symbol in symbols:
        base_price = np.random.uniform(100, 1000)
        volatility = np.random.uniform(0.01, 0.03)

        for date in dates:
            price_change = np.random.normal(0, volatility)
            base_price *= (1 + price_change)
            
            record = {
                "Date": date.strftime('%Y-%m-%d %H:%M:%S'),
                "Symbol": symbol,
                "Open": round(base_price * (1 + np.random.uniform(-0.01, 0.01)), 2),
                "High": round(base_price * (1 + np.random.uniform(0, 0.02)), 2),
                "Low": round(base_price * (1 - np.random.uniform(0, 0.02)), 2),
                "Close": round(base_price, 2),
                "Volume": int(np.random.randint(100000, 10000000))
            }
            data.append(record)
            
    return pd.DataFrame(data).to_dict('records')

@app.route('/')
def home():
    endpoints = {
        "/": "API Home - Lists available endpoints",
        "/generate-finance-data": "Generate synthetic financial data",
        "/market-summary": "Get market summary for major indices",
        "/technical-indicators/<symbol>": "Get technical indicators for a symbol",
        "/historical-data/<symbol>": "Get historical data for a symbol",
        "/symbols": "Get list of available symbols"
    }
    return jsonify({"status": "Finance API is running", "available_endpoints": endpoints})

@app.route('/symbols')
def get_symbols():
    symbols = {
        "stocks": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
        "crypto": ["BTC-USD", "ETH-USD"],
        "indices": ["^GSPC", "^IXIC", "^DJI"]
    }
    return jsonify(symbols)

@app.route('/market-summary')
def market_summary():
    symbols = ["^GSPC", "^IXIC", "^DJI", "BTC-USD"]
    period = request.args.get('period', default='1d')
    try:
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if not hist.empty:
                data[symbol] = {
                    "current_price": round(float(hist['Close'].iloc[-1]), 2),
                    "change_percent": round(((float(hist['Close'].iloc[-1]) - float(hist['Open'].iloc[0])) / float(hist['Open'].iloc[0])) * 100, 2),
                    "volume": int(hist['Volume'].iloc[-1]),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        if not data:
            return jsonify({"error": "No data available for the requested period"}), 404
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/technical-indicators/')
def technical_indicators():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400
    # resto del codice...
    try:
        period = request.args.get('period', default='6mo')
        interval = request.args.get('interval', default='1d')
        
        data = yf.download(symbol, period=period, interval=interval)
        
        if data.empty:
            return jsonify({
                "error": f"No data available for symbol {symbol}",
                "symbol": symbol,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }), 404

        # Calcolo indicatori
        close_prices = data['Close']
        high_prices = data['High']
        low_prices = data['Low']
        volume = data['Volume']

        # RSI
        rsi = ta.momentum.RSIIndicator(close_prices, window=14)
        rsi_values = rsi.rsi()

        # MACD
        macd = ta.trend.MACD(close_prices)
        macd_line = macd.macd()
        signal_line = macd.macd_signal()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close_prices, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_middle = bb.bollinger_mavg()

        # Media mobile esponenziale
        ema_20 = ta.trend.EMAIndicator(close_prices, window=20).ema_indicator()
        
        # Stochastic RSI
        stoch_rsi = ta.momentum.StochRSIIndicator(close_prices)
        
        # ATR
        atr = ta.volatility.AverageTrueRange(high_prices, low_prices, close_prices)
        
        indicators = {
            "current_price": float(close_prices.iloc[-1]),
            "RSI": float(rsi_values.iloc[-1]) if not pd.isna(rsi_values.iloc[-1]) else None,
            "MACD": {
                "macd_line": float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
                "signal_line": float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None,
                "histogram": float(macd_line.iloc[-1] - signal_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) and not pd.isna(signal_line.iloc[-1]) else None
            },
            "Bollinger_Bands": {
                "upper": float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None,
                "middle": float(bb_middle.iloc[-1]) if not pd.isna(bb_middle.iloc[-1]) else None,
                "lower": float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None
            },
            "EMA_20": float(ema_20.iloc[-1]) if not pd.isna(ema_20.iloc[-1]) else None,
            "StochRSI": float(stoch_rsi.stochrsi.iloc[-1]) if not pd.isna(stoch_rsi.stochrsi.iloc[-1]) else None,
            "ATR": float(atr.average_true_range().iloc[-1]) if not pd.isna(atr.average_true_range().iloc[-1]) else None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "period": period,
            "interval": interval
        }

        if len(close_prices) > 1:
            indicators["price_change_24h"] = float(
                ((close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2]) * 100
            )
        
        return jsonify(indicators)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/historical-data/<symbol>')
def historical_data(symbol):
    try:
        period = request.args.get('period', default='1mo')
        interval = request.args.get('interval', default='1d')
        data = yf.download(symbol, period=period, interval=interval)

        if data.empty:
            return jsonify({
                "error": f"No data available for symbol {symbol}",
                "symbol": symbol,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }), 404

        if "Close" not in data.columns:
            if "High" in data.columns and "Low" in data.columns:
                data["Close"] = (data["High"] + data["Low"]) / 2
            else:
                data["Close"] = np.random.uniform(100, 500, size=len(data))

        records = []
        for index, row in data.iterrows():
            record = {
                'Date': index.strftime('%Y-%m-%d %H:%M:%S'),
                'Open': float(row['Open']) if "Open" in data.columns else None,
                'High': float(row['High']) if "High" in data.columns else None,
                'Low': float(row['Low']) if "Low" in data.columns else None,
                'Close': float(row['Close']),
                'Volume': int(row['Volume']) if "Volume" in data.columns else None,
                'Adj Close': float(row['Adj Close']) if "Adj Close" in data.columns else None
            }
            records.append(record)

        return jsonify({
            "symbol": symbol,
            "data": records,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-finance-data')
def generate_finance_data_endpoint():
    try:
        num_records = request.args.get('num_records', default=100, type=int)
        if num_records <= 0 or num_records > 1000:
            return jsonify({"error": "Number of records must be between 1 and 1000"}), 400
        data = generate_finance_data(num_records)
        return jsonify({
            "data": data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(data)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": {
            "/": "API Home",
            "/generate-finance-data": "Generate synthetic financial data",
            "/market-summary": "Get market summary for major indices",
            "/technical-indicators/<symbol>": "Get technical indicators for a symbol",
            "/historical-data/<symbol>": "Get historical data for a symbol",
            "/symbols": "Get list of available symbols"
        }
    }), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)