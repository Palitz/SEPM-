import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
import yfinance as yf
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import json
from functools import wraps
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
CORS(app)

app.secret_key = "supersecretkey"

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

TOP_STOCKS = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "META", "NVDA", "NFLX"]

def init_db():
    """Initialize the SQLite database and create tables."""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Watchlist table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, symbol)
        )
    """)
    
    # Portfolio table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            shares INTEGER NOT NULL,
            purchase_price REAL NOT NULL,
            purchase_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    conn.commit()
    conn.close()

init_db()

@app.context_processor
def override_static():
    """Force Flask to load the latest CSS by appending a random query string."""
    return dict(url_for=lambda endpoint, **kwargs: url_for(endpoint, **kwargs) + "?v=" + str(os.urandom(8).hex()))

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                           (username, email, hashed_password))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Username or Email already exists")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session["user"] = username
            session["user_id"] = user[0]
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["user"])

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    session.pop("user_id", None)
    return jsonify({"status": "success"})

@app.route("/stock-data")
@login_required
def get_stock_data():
    try:
        # List of top stocks to track
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
        stock_data = {}
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                
                stock_data[symbol] = {
                    'name': info.get('longName', 'N/A'),
                    'price': info.get('currentPrice', 0),
                    'change': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('regularMarketVolume', 0),
                    'market_cap': info.get('marketCap', 0)
                }
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return jsonify(stock_data)
    except Exception as e:
        print(f"Error in get_stock_data: {str(e)}")
        return jsonify({'error': 'Failed to fetch stock data'}), 500

@app.route('/api/stock-history/<symbol>')
@login_required
def stock_history(symbol):
    try:
        print(f"Fetching stock history for {symbol}")
        # Get timeRange from query parameters
        timeRange = request.args.get('timeRange', '1Y')
        
        # Map timeRange to yfinance period
        period_map = {
            '1D': '1d',
            '1W': '1wk',
            '1M': '1mo',
            '3M': '3mo',
            '1Y': '1y'
        }
        period = period_map.get(timeRange, '1y')
        
        # Fetch historical data
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            print(f"No data available for {symbol}")
            return jsonify({"error": "No data available for this symbol"}), 404
            
        print(f"Retrieved {len(hist)} days of historical data")
        
        # Calculate price changes
        hist['Change'] = hist['Close'].pct_change() * 100
        
        # Prepare historical data
        historical_data = []
        for index, row in hist.iterrows():
            historical_data.append({
                'date': index.strftime('%Y-%m-%d'),
                'price': float(row['Close']),
                'change': float(row['Change']) if not pd.isna(row['Change']) else 0.0
            })
            
        print("Historical data prepared")
            
        # Get predictions
        try:
            print("Starting prediction process")
            # Create a copy of the historical data for predictions
            hist_copy = hist.copy()
            hist_copy.name = symbol  # Add symbol name to the DataFrame
            predictions = predict_stock_prices(hist_copy, symbol)
            
            print(f"Received predictions: {predictions}")
            
            # Format predictions data
            formatted_predictions = []
            for pred in predictions:
                if pred['predicted_price'] != "N/A":
                    formatted_predictions.append({
                        'date': pred['date'],
                        'predicted_price': float(pred['predicted_price']),
                        'change': float(pred['change']),
                        'confidence': float(pred['confidence'])
                    })
            
            print(f"Formatted {len(formatted_predictions)} predictions")
            
        except Exception as e:
            print(f"Prediction error in stock_history: {str(e)}")
            import traceback
            print(traceback.format_exc())
            formatted_predictions = []
            
        response_data = {
            'historical': historical_data,
            'predictions': formatted_predictions
        }
        
        print(f"Returning response with {len(historical_data)} historical points and {len(formatted_predictions)} predictions")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Stock history error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/watchlist", methods=["GET", "POST", "DELETE"])
@login_required
def watchlist():
    if request.method == "GET":
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        user_id = session.get('user_id')
        
        cursor.execute("SELECT symbol FROM watchlist WHERE user_id = ?", (user_id,))
        symbols = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return jsonify({"symbols": symbols})
    
    elif request.method == "POST":
        data = request.get_json()
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({"error": "No symbol provided"}), 400
            
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        user_id = session.get('user_id')
        
        cursor.execute("INSERT INTO watchlist (user_id, symbol) VALUES (?, ?)",
                      (user_id, symbol))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})
    
    elif request.method == "DELETE":
        data = request.get_json()
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({"error": "No symbol provided"}), 400
            
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        user_id = session.get('user_id')
        
        cursor.execute("DELETE FROM watchlist WHERE user_id = ? AND symbol = ?",
                      (user_id, symbol))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})

@app.route("/portfolio", methods=["GET", "POST", "DELETE"])
def portfolio():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session["user_id"]
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    if request.method == "GET":
        cursor.execute("""
            SELECT symbol, shares, purchase_price, purchase_date 
            FROM portfolio WHERE user_id = ?
        """, (user_id,))
        holdings = cursor.fetchall()
        
        portfolio_data = []
        for holding in holdings:
            symbol, shares, purchase_price, purchase_date = holding
            stock = yf.Ticker(symbol)
            current_price = stock.info.get("regularMarketPrice", 0)
            
            portfolio_data.append({
                "symbol": symbol,
                "shares": shares,
                "purchase_price": purchase_price,
                "purchase_date": purchase_date,
                "current_price": current_price,
                "total_value": shares * current_price,
                "profit_loss": (current_price - purchase_price) * shares,
                "profit_loss_percentage": ((current_price - purchase_price) / purchase_price * 100)
            })
        
        conn.close()
        return jsonify({"holdings": portfolio_data})

    elif request.method == "POST":
        data = request.get_json()
        symbol = data.get("symbol", "").strip().upper()
        shares = int(data.get("shares", 0))
        purchase_price = float(data.get("purchase_price", 0))
        
        if not all([symbol, shares, purchase_price]):
            return jsonify({"error": "Missing required fields"}), 400

        try:
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO portfolio (user_id, symbol, shares, purchase_price)
                VALUES (?, ?, ?, ?)
            """, (user_id, symbol, shares, purchase_price))
            conn.commit()
            conn.close()
            return jsonify({"status": "success"})
        except Exception as e:
            conn.close()
            return jsonify({"error": str(e)}), 400

    elif request.method == "DELETE":
        data = request.get_json()
        symbol = data.get("symbol", "").strip().upper()
        
        if not symbol:
            return jsonify({"error": "No symbol provided"}), 400

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM portfolio WHERE user_id = ? AND symbol = ?",
                      (user_id, symbol))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})

@app.route("/portfolio-predictions")
def portfolio_predictions():
    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        user_id = session.get('user_id')
        
        cursor.execute("SELECT symbol, shares, purchase_price FROM portfolio WHERE user_id = ?", (user_id,))
        holdings = cursor.fetchall()
        conn.close()

        predictions = {}
        total_current_value = 0
        total_predicted_value = 0

        for holding in holdings:
            try:
                symbol, shares, purchase_price = holding
                stock = yf.Ticker(symbol)
                
                # Get current price
                current_price = stock.info.get("regularMarketPrice", 0)
                if not current_price:
                    continue
                    
                current_value = shares * current_price
                total_current_value += current_value

                # Get historical data for prediction
                hist = stock.history(period="1y")
                if not hist.empty and "Close" in hist.columns:
                    # Get price predictions
                    price_predictions = predict_stock_prices(hist, symbol)
                    
                    # Calculate predicted values
                    predicted_values = []
                    for pred in price_predictions:
                        if pred["predicted_price"] != "N/A":
                            predicted_value = shares * pred["predicted_price"]
                            predicted_values.append({
                                "date": pred["date"],
                                "value": round(predicted_value, 2),
                                "price": pred["predicted_price"],
                                "change": round(predicted_value - current_value, 2),
                                "change_percentage": round((predicted_value - current_value) / current_value * 100, 2)
                            })
                            if pred["date"] == price_predictions[-1]["date"]:
                                total_predicted_value += predicted_value

                    predictions[symbol] = {
                        "current_price": current_price,
                        "current_value": round(current_value, 2),
                        "predicted_values": predicted_values
                    }
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue

        if total_current_value == 0:
            return jsonify({
                "predictions": {},
                "total_current_value": 0,
                "total_predicted_value": 0,
                "total_change": 0,
                "total_change_percentage": 0
            })

        return jsonify({
            "predictions": predictions,
            "total_current_value": round(total_current_value, 2),
            "total_predicted_value": round(total_predicted_value, 2),
            "total_change": round(total_predicted_value - total_current_value, 2),
            "total_change_percentage": round((total_predicted_value - total_current_value) / total_current_value * 100, 2)
        })
    except Exception as e:
        print(f"Portfolio predictions error: {e}")
        return jsonify({
            "predictions": {},
            "total_current_value": 0,
            "total_predicted_value": 0,
            "total_change": 0,
            "total_change_percentage": 0
        }), 500

def analyze_sentiment(text):
    """
    Simple sentiment analysis function.
    Returns a score between -1 (negative) and 1 (positive).
    """
    # Basic sentiment words
    positive_words = {'up', 'rise', 'gain', 'positive', 'growth', 'bullish', 'strong', 'profit', 'success'}
    negative_words = {'down', 'fall', 'loss', 'negative', 'decline', 'bearish', 'weak', 'loss', 'fail'}
    
    # Convert text to lowercase and split into words
    words = set(text.lower().split())
    
    # Count positive and negative words
    positive_count = len(words.intersection(positive_words))
    negative_count = len(words.intersection(negative_words))
    
    # Calculate sentiment score
    total = positive_count + negative_count
    if total == 0:
        return 0
    
    return (positive_count - negative_count) / total

def predict_stock_prices(history, symbol):
    """Predict stock prices for the next 7 days using multiple models."""
    try:
        print("Starting prediction process...")
        print(f"Processing data for symbol: {symbol}")
        
        # Convert index to datetime if it's not already
        if not isinstance(history.index, pd.DatetimeIndex):
            history.index = pd.to_datetime(history.index)
        
        # Calculate technical indicators
        print("Calculating technical indicators...")
        
        # Simple Moving Average (SMA)
        history['SMA_5'] = history['Close'].rolling(window=5, min_periods=1).mean()
        history['SMA_20'] = history['Close'].rolling(window=20, min_periods=1).mean()
        
        # Relative Strength Index (RSI)
        delta = history['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        history['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = history['Close'].ewm(span=12, adjust=False).mean()
        exp2 = history['Close'].ewm(span=26, adjust=False).mean()
        history['MACD'] = exp1 - exp2
        history['Signal_Line'] = history['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        history['BB_middle'] = history['Close'].rolling(window=20, min_periods=1).mean()
        history['BB_std'] = history['Close'].rolling(window=20, min_periods=1).std()
        history['BB_upper'] = history['BB_middle'] + (history['BB_std'] * 2)
        history['BB_lower'] = history['BB_middle'] - (history['BB_std'] * 2)
        
        # Volume indicators
        history['Volume_MA'] = history['Volume'].rolling(window=20, min_periods=1).mean()
        history['Volume_Ratio'] = history['Volume'] / history['Volume_MA']
        
        # Fill NaN values
        history = history.fillna(method='ffill').fillna(method='bfill')
        history['RSI'] = history['RSI'].fillna(50)
        history['Volume_Ratio'] = history['Volume_Ratio'].fillna(1)
        
        # Prepare features for prediction
        print("Preparing features...")
        features = pd.DataFrame()
        features['Close'] = history['Close']
        features['SMA_5'] = history['SMA_5']
        features['SMA_20'] = history['SMA_20']
        features['RSI'] = history['RSI']
        features['MACD'] = history['MACD']
        features['Signal_Line'] = history['Signal_Line']
        features['BB_upper'] = history['BB_upper']
        features['BB_lower'] = history['BB_lower']
        features['Volume_Ratio'] = history['Volume_Ratio']
        
        # Fill any remaining NaN values
        features = features.fillna(0)
        
        print(f"Data points available: {len(features)}")
        
        # Check if we have enough data points (reduced from 30 to 10)
        if len(features) < 10:
            print(f"Not enough data points: {len(features)}")
            return [{'date': 'N/A', 'predicted_price': 'N/A', 'change': 0, 'confidence': 0}]
        
        # Prepare target variable (next day's price)
        target = history['Close'].shift(-1).dropna()
        features = features[:-1]  # Remove the last row since we don't have the next day's price
        
        # Split data into training and testing sets
        train_size = int(len(features) * 0.8)
        X_train = features[:train_size]
        y_train = target[:train_size]
        X_test = features[train_size:]
        y_test = target[train_size:]
        
        # Train multiple models
        print("Training models...")
        
        # ARIMA model
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(history['Close'], order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=7)
            arima_predictions = forecast.values
            print(f"ARIMA predictions: {arima_predictions}")
        except Exception as e:
            print(f"ARIMA model error: {e}")
            arima_predictions = None
        
        # Linear Regression
        try:
            from sklearn.linear_model import LinearRegression
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_predictions = lr_model.predict(X_test[-7:])
            print(f"Linear Regression predictions: {lr_predictions}")
        except Exception as e:
            print(f"Linear Regression error: {e}")
            lr_predictions = None
        
        # Random Forest
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_predictions = rf_model.predict(X_test[-7:])
            print(f"Random Forest predictions: {rf_predictions}")
        except Exception as e:
            print(f"Random Forest error: {e}")
            rf_predictions = None
        
        # Combine predictions (ensemble)
        print("Combining predictions...")
        predictions = []
        
        # Get the last date from the historical data
        last_date = history.index[-1]
        
        # Generate future dates for predictions
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(7)]
        
        # Calculate weighted average of predictions
        for i in range(7):
            pred_values = []
            weights = []
            
            if arima_predictions is not None and i < len(arima_predictions):
                pred_values.append(arima_predictions[i])
                weights.append(0.4)  # 40% weight for ARIMA
            
            if lr_predictions is not None and i < len(lr_predictions):
                pred_values.append(lr_predictions[i])
                weights.append(0.3)  # 30% weight for Linear Regression
            
            if rf_predictions is not None and i < len(rf_predictions):
                pred_values.append(rf_predictions[i])
                weights.append(0.3)  # 30% weight for Random Forest
            
            if pred_values:
                # Calculate weighted average
                weighted_pred = sum(p * w for p, w in zip(pred_values, weights)) / sum(weights)
                
                # Calculate confidence based on model agreement
                confidence = min(100, len(pred_values) * 25)  # 25% per model, max 100%
                
                # Calculate percentage change from last known price
                last_price = history['Close'].iloc[-1]
                change = ((weighted_pred - last_price) / last_price) * 100
                
                predictions.append({
                    'date': future_dates[i].strftime('%Y-%m-%d'),
                    'predicted_price': round(weighted_pred, 2),
                    'change': round(change, 2),
                    'confidence': round(confidence, 1)
                })
        
        print(f"Generated {len(predictions)} predictions")
        return predictions

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return [{'date': 'N/A', 'predicted_price': 'N/A', 'change': 0, 'confidence': 0}]

def fetch_stock_data_for_prediction(symbol, period="1y"):
    """
    Fetch comprehensive stock data for prediction including technical indicators
    and market sentiment.
    """
    try:
        stock = yf.Ticker(symbol)
        
        # Fetch historical data
        hist = stock.history(period=period)
        
        # Calculate technical indicators
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])
        hist['MACD'], hist['MACD_Signal'], hist['MACD_Hist'] = calculate_macd(hist['Close'])
        hist['BB_Upper'], hist['BB_Middle'], hist['BB_Lower'] = calculate_bollinger_bands(hist['Close'])
        
        # Calculate volume indicators
        hist['Volume_SMA'] = hist['Volume'].rolling(window=20).mean()
        hist['Volume_Ratio'] = hist['Volume'] / hist['Volume_SMA']
        
        # Calculate price momentum
        hist['Price_Momentum'] = hist['Close'].pct_change(periods=5)
        
        # Get market sentiment data
        try:
            news = stock.news
            sentiment_scores = [analyze_sentiment(item['title']) for item in news]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        except:
            avg_sentiment = 0
            
        # Get market sector performance
        try:
            sector = stock.info.get('sector', '')
            sector_performance = get_sector_performance(sector)
        except:
            sector_performance = 0
            
        return {
            'historical_data': hist,
            'sentiment_score': avg_sentiment,
            'sector_performance': sector_performance,
            'market_cap': stock.info.get('marketCap', 0),
            'pe_ratio': stock.info.get('trailingPE', 0),
            'beta': stock.info.get('beta', 0)
        }
    except Exception as e:
        print(f"Error fetching prediction data for {symbol}: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal Line, and MACD Histogram"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    return upper_band, middle_band, lower_band

def get_sector_performance(sector):
    """
    Get the performance of the stock's sector.
    In production, this should fetch real sector performance data.
    """
    # Placeholder - should be replaced with actual sector performance data
    return 0

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.errorhandler(Exception)
def handle_error(error):
    print(f"Global error handler caught: {str(error)}")
    import traceback
    print(traceback.format_exc())
    return jsonify({"error": str(error)}), 500

@app.route('/top-stocks')
@login_required
def top_stocks():
    return render_template('top_stocks.html', username=session['user'])

@app.route('/add-to-watchlist', methods=['POST'])
@login_required
def add_to_watchlist():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
            
        # Add to user's watchlist in database
        # This is a placeholder - implement your database logic here
        
        return jsonify({'status': 'success', 'message': f'{symbol} added to watchlist'})
    except Exception as e:
        print(f"Error in add_to_watchlist: {str(e)}")
        return jsonify({'error': 'Failed to add stock to watchlist'}), 500

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
