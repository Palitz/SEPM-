document.addEventListener("DOMContentLoaded", function () {
    fetchStockData(); // Load top stocks by default

    const stockSearch = document.getElementById("stockSearch");
    stockSearch.addEventListener("input", function () {
        const query = stockSearch.value.trim().toUpperCase();
        if (query) {
            fetchStockData(query, true);
        } else {
            document.getElementById("searchResults").innerHTML = ""; // Clear search results
            fetchStockData(); // Reload top stocks when search is cleared
        }
    });

    document.getElementById("logout").addEventListener("click", async function () {
        try {
            const response = await fetch("/logout", { method: "POST" });
            if (response.ok) {
                window.location.href = "/"; // Redirect to login page
            }
        } catch (error) {
            console.error("Logout failed:", error);
        }
    });
});

let stockChart = null; // Store the chart instance globally

// Fetch stock data (top stocks or search results)
async function fetchStockData(query = "", isSearch = false) {
    try {
        const response = await fetch(`/stock-data?query=${query}`);
        if (!response.ok) throw new Error("Failed to fetch stock data");

        const stockData = await response.json();
        
        resetChartAndPredictions(); // Reset chart & predictions when searching

        if (isSearch && Object.keys(stockData).length === 0) {
            document.getElementById("searchResults").innerHTML = "<tr><td colspan='4'>No results found</td></tr>";
        } else {
            updateStockTable(stockData, isSearch);
            if (!isSearch) plotStockGraph(stockData); // Only update chart for top stocks

            // Auto-fetch stock history for first result in search
            const firstStockSymbol = Object.keys(stockData)[0];
            if (firstStockSymbol) fetchStockHistory(firstStockSymbol);
        }
    } catch (error) {
        console.error("Error fetching stock data:", error);
    }
}

// Populate stock table
function updateStockTable(stockData, isSearch) {
    const tableBody = isSearch ? document.getElementById("searchResults") : document.getElementById("stockTableBody");
    tableBody.innerHTML = ""; // Clear old data

    Object.entries(stockData).forEach(([symbol, data]) => {
        const row = document.createElement("tr");

        const stockCell = document.createElement("td");
        stockCell.innerHTML = `<a href="#" class="stock-link" data-symbol="${symbol}">${symbol}</a>`;
        stockCell.style.cursor = "pointer";

        const priceCell = document.createElement("td");
        priceCell.textContent = data.price !== "N/A" ? `$${data.price}` : "N/A";

        const highCell = document.createElement("td");
        highCell.textContent = data.high !== "N/A" ? `$${data.high}` : "N/A";

        const lowCell = document.createElement("td");
        lowCell.textContent = data.low !== "N/A" ? `$${data.low}` : "N/A";

        row.appendChild(stockCell);
        row.appendChild(priceCell);
        row.appendChild(highCell);
        row.appendChild(lowCell);

        tableBody.appendChild(row);
    });

    // Attach click event to fetch stock history
    document.querySelectorAll(".stock-link").forEach(link => {
        link.addEventListener("click", function (event) {
            event.preventDefault();
            fetchStockHistory(this.dataset.symbol);
        });
    });
}

// Fetch stock history and predictions
async function fetchStockHistory(symbol) {
    try {
        const response = await fetch(`/stock-history?symbol=${symbol}`);
        const data = await response.json();

        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }

        updateChart(symbol, data.historical_data, data.predictions);
        displayPredictions(symbol, data.predictions);
    } catch (error) {
        console.error("Failed to fetch stock history:", error);
    }
}

// Plot stock graph using Chart.js
function plotStockGraph(stockData) {
    const labels = Object.keys(stockData);
    const prices = Object.values(stockData).map(data => data.price !== "N/A" ? data.price : null);

    const ctx = document.getElementById("stockChart").getContext("2d");

    if (stockChart) {
        stockChart.destroy(); // Remove existing chart to prevent overlap
    }

    stockChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Stock Prices",
                data: prices,
                backgroundColor: "rgba(54, 162, 235, 0.6)",
                borderColor: "rgba(54, 162, 235, 1)",
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: false }
            }
        }
    });
}

// Update line chart with historical and predicted stock data
function updateChart(symbol, historicalData, predictions) {
    const chartContainer = document.getElementById("chartContainer");
    chartContainer.innerHTML = '<canvas id="stockChart"></canvas>'; // Reset canvas
    chartContainer.style.display = "block"; // Show chart container

    const ctx = document.getElementById("stockChart").getContext("2d");

    const labels = historicalData.map(entry => entry.date).concat(predictions.map(entry => entry.date));
    const prices = historicalData.map(entry => entry.price);
    const predictedPrices = predictions.map(entry => entry.predicted_price);

    if (stockChart) {
        stockChart.destroy();
    }

    stockChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: `${symbol} Stock Price History`,
                    data: prices.concat(Array(predictedPrices.length).fill(null)), // Add nulls to align data
                    borderColor: "#007bff",
                    backgroundColor: "rgba(0, 123, 255, 0.2)",
                    borderWidth: 2
                },
                {
                    label: `${symbol} Predicted Prices`,
                    data: Array(prices.length).fill(null).concat(predictedPrices), // Shift predictions to the right
                    borderColor: "#ff5733",
                    backgroundColor: "rgba(255, 87, 51, 0.2)",
                    borderWidth: 2,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: "Date" } },
                y: { title: { display: true, text: "Price (USD)" } }
            }
        }
    });
}

// Display predictions in a table
function displayPredictions(symbol, predictions) {
    const predictionContainer = document.getElementById("predictionContainer");
    predictionContainer.innerHTML = `<h3>${symbol} Predicted Prices (Next 4 Periods)</h3>`;

    if (!predictions || predictions.length === 0 || predictions.every(pred => !pred.predicted_price)) {
        predictionContainer.innerHTML += "<p>No predictions available.</p>";
        return;
    }

    let html = "<table><tr><th>Date</th><th>Predicted Price</th></tr>";
    predictions.forEach(pred => {
        html += `<tr><td>${pred.date}</td><td>$${pred.predicted_price}</td></tr>`;
    });
    html += "</table>";

    predictionContainer.innerHTML += html;
}

// Reset chart and predictions when searching
function resetChartAndPredictions() {
    document.getElementById("chartContainer").innerHTML = "";
    document.getElementById("predictionContainer").innerHTML = "";
}
