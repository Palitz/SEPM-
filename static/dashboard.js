document.addEventListener("DOMContentLoaded", function () {
    fetchStockData();

    let stockChart = null;

    async function fetchStockData(query = "") {
        let url = query ? `/stock-data?query=${query}` : "/stock-data";
        try {
            const response = await fetch(url);
            const result = await response.json();

            if (result.status === "fail") {
                alert("Session expired. Please log in again.");
                window.location.href = "/";
                return;
            }

            const stockTableBody = document.getElementById("stockTableBody");
            stockTableBody.innerHTML = ""; // Clear existing data

            // ❗ Clear the chart & predictions if a new stock is searched
            resetChartAndPredictions();

            for (const [symbol, data] of Object.entries(result)) {
                const row = document.createElement("tr");

                const stockCell = document.createElement("td");
                stockCell.innerHTML = `
                    <img src="/static/logos/${symbol}.png" class="stock-logo" alt="${symbol} Logo">
                    <a href="#" class="stock-link" data-symbol="${symbol}">${symbol}</a>
                `;
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

                stockTableBody.appendChild(row);
            }

            document.querySelectorAll(".stock-link").forEach(link => {
                link.addEventListener("click", function (event) {
                    event.preventDefault();
                    fetchStockHistory(this.dataset.symbol);
                });
            });
        } catch (error) {
            console.error("Error fetching stock data:", error);
        }
    }

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

    function updateChart(symbol, historicalData, predictions) {
        const chartContainer = document.getElementById("chartContainer");
        chartContainer.innerHTML = '<canvas id="stockChart"></canvas>'; // Reset canvas
        chartContainer.style.display = "block"; // Show chart container

        const ctx = document.getElementById("stockChart").getContext("2d");

        // Combine historical and predicted data
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
                maintainAspectRatio: false, // Allow custom sizing
                scales: {
                    x: { title: { display: true, text: "Date" } },
                    y: { title: { display: true, text: "Price (USD)" } }
                },
                layout: {
                    padding: {
                        left: 20,
                        right: 20,
                        top: 10,
                        bottom: 10
                    }
                }
            }
        });

        // Adjust the size dynamically
        document.getElementById("stockChart").style.width = "900px";
        document.getElementById("stockChart").style.height = "500px";
    }

    function displayPredictions(symbol, predictions) {
        const predictionContainer = document.getElementById("predictionContainer");
        predictionContainer.innerHTML = `<h3>${symbol} Predicted Prices (Next 4 Periods)</h3>`;

        if (!predictions || predictions.length === 0 || predictions[0].predicted_price === "N/A") {
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

    function resetChartAndPredictions() {
        const chartContainer = document.getElementById("chartContainer");
        chartContainer.innerHTML = ""; // Remove old chart

        const predictionContainer = document.getElementById("predictionContainer");
        predictionContainer.innerHTML = ""; // Clear predictions
    }

    document.getElementById("searchBtn").addEventListener("click", function () {
        const query = document.getElementById("stockSearch").value.trim();
        if (query) {
            fetchStockData(query);
            resetChartAndPredictions(); // ❗ Ensure old data is cleared
        }
    });

    document.getElementById("stockSearch").addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            document.getElementById("searchBtn").click();
        }
    });

    document.getElementById("logoutBtn").addEventListener("click", async function () {
        try {
            const response = await fetch("/logout", { method: "POST" });
            const result = await response.json();

            if (result.status === "success") {
                window.location.href = "/"; // Redirect to login page
            }
        } catch (error) {
            console.error("Logout failed:", error);
        }
    });
});
