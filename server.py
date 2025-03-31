import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
import yfinance as yf
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

app.secret_key = "supersecretkey"

TOP_STOCKS = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "META", "NVDA", "NFLX"]

def init_db():
    """Initialize the SQLite database and create tables."""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

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
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[0], password):
            session["user"] = username
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
    return jsonify({"status": "success"})

@app.route("/stock-data")
def get_stock_data():
    if "user" not in session:
        return jsonify({"message": "Unauthorized", "status": "fail"}), 401

    query = request.args.get("query", "").strip().upper()
    stocks = [query] if query else TOP_STOCKS

    stock_data = {}
    for symbol in stocks:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")

            if not hist.empty and "Close" in hist.columns:
                stock_data[symbol] = {
                    "price": round(float(hist["Close"].iloc[-1]), 2),
                    "high": round(float(hist["High"].max()), 2) if "High" in hist.columns else "N/A",
                    "low": round(float(hist["Low"].min()), 2) if "Low" in hist.columns else "N/A"
                }
            else:
                stock_data[symbol] = {"price": "N/A", "high": "N/A", "low": "N/A"}

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            stock_data[symbol] = {"price": "N/A", "high": "N/A", "low": "N/A"}

    return jsonify(stock_data)

@app.route("/stock-history")
def stock_history():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    symbol = request.args.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "No stock symbol provided"}), 400

    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5y")

        if hist.empty or "Close" not in hist.columns:
            return jsonify({"error": "No data available"}), 404

        # Ensure datetime index is properly formatted
        hist.index = pd.to_datetime(hist.index)
        hist.index = hist.index.to_period("D")  # Fix frequency issue for ARIMA

        historical_data = [
            {"date": str(date), "price": round(price, 2)}
            for date, price in zip(hist.index, hist["Close"])
        ]

        # Train ARIMA Model and Predict Future Prices
        predictions = predict_stock_prices(hist)

        return jsonify({"historical_data": historical_data, "predictions": predictions})
    
    except Exception as e:
        print(f"Error fetching history for {symbol}: {e}")
        return jsonify({"error": "Failed to fetch stock history"}), 500

def predict_stock_prices(history):
    try:
        prices = history["Close"].dropna().values  # Drop NaN values

        if len(prices) < 10:
            return [{"date": "N/A", "predicted_price": "N/A"}]

        model = ARIMA(prices, order=(1, 1, 1))  
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=4)  # Predict next 4 quarters
        future_dates = pd.date_range(start=history.index[-1].to_timestamp(), periods=4, freq="Q")

        predictions = [
            {"date": str(date.date()), "predicted_price": round(price, 2)}
            for date, price in zip(future_dates, forecast)
        ]
        return predictions

    except Exception as e:
        print(f"Prediction error: {e}")
        return [{"date": "N/A", "predicted_price": "N/A"}]

if __name__ == '__main__':
    app.run(debug=True)
