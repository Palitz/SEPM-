from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
import yfinance as yf

app = Flask(__name__)
CORS(app)

# Secret key for session management
app.secret_key = "supersecretkey"

# Mock user database (Replace with actual database in production)
USERS = {
    "user1": "password123",
    "admin": "adminpass"
}

# Stocks to fetch
STOCKS = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "META", "NVDA", "NFLX"]


# ------------------ ROUTES ------------------ #

@app.route("/")
def home():
    """Redirects to login page"""
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Handles user login"""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in USERS and USERS[username] == password:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    """Displays the dashboard only if logged in"""
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["user"])


@app.route("/logout")
def logout():
    """Logs out the user"""
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/stock-data")
def get_stock_data():
    """Fetches stock data from Yahoo Finance"""
    if "user" not in session:
        return jsonify({"message": "Unauthorized", "status": "fail"}), 401

    stock_data = {}
    for symbol in STOCKS:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")  # Fetch 1-year data
            
            if not hist.empty:
                latest_price = hist["Close"].iloc[-1] if "Close" in hist.columns else None
                high_price = hist["High"].max() if "High" in hist.columns else None
                low_price = hist["Low"].min() if "Low" in hist.columns else None

                stock_data[symbol] = {
                    "price": round(float(latest_price), 2) if latest_price else "N/A",
                    "high": round(float(high_price), 2) if high_price else "N/A",
                    "low": round(float(low_price), 2) if low_price else "N/A"
                }
            else:
                stock_data[symbol] = {"price": "N/A", "high": "N/A", "low": "N/A"}

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            stock_data[symbol] = {"price": "N/A", "high": "N/A", "low": "N/A"}

    return jsonify(stock_data)


if __name__ == '__main__':
    app.run(debug=True)
