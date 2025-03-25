import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
import yfinance as yf
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

# Secret key for session management
app.secret_key = "supersecretkey"

# Default top stocks
TOP_STOCKS = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "META", "NVDA", "NFLX"]

# ------------------ DATABASE SETUP ------------------ #

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

init_db()  # Run once when the server starts

# ------------------ ROUTES ------------------ #

@app.route("/")
def home():
    """Redirects to login page"""
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """Handles user registration"""
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)  # Hash password

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
    """Handles user login"""
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

    query = request.args.get("query", "").strip().upper()

    # If no query, return top stocks
    stocks = [query] if query else TOP_STOCKS

    stock_data = {}
    for symbol in stocks:
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
