document.addEventListener("DOMContentLoaded", function () {
    fetchStockData(); // Load top stocks by default

    const stockSearch = document.getElementById("stockSearch");
    stockSearch.addEventListener("input", function () {
        const query = stockSearch.value.trim().toUpperCase();
        if (query) {
            fetchStockData(query, true);
        } else {
            document.getElementById("searchResults").innerHTML = ""; // Clear search results
        }
    });

    document.getElementById("logout").addEventListener("click", function () {
        window.location.href = "/logout";
    });
});

let stockChart = null; // Store the chart instance globally

// Fetch stock data (top stocks or search results)
async function fetchStockData(query = "", isSearch = false) {
    try {
        const response = await fetch(`/stock-data?query=${query}`);
        if (!response.ok) throw new Error("Failed to fetch stock data");

        const stockData = await response.json();
        
        if (isSearch && Object.keys(stockData).length === 0) {
            document.getElementById("searchResults").innerHTML = "<tr><td colspan='4'>No results found</td></tr>";
        } else {
            updateStockTable(stockData, isSearch);
            if (!isSearch) plotStockGraph(stockData); // Only update chart for top stocks
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
        const row = `<tr>
                        <td>${symbol}</td>
                        <td>${data.price !== "N/A" ? `$${data.price}` : "N/A"}</td>
                        <td>${data.high !== "N/A" ? `$${data.high}` : "N/A"}</td>
                        <td>${data.low !== "N/A" ? `$${data.low}` : "N/A"}</td>
                    </tr>`;
        tableBody.innerHTML += row;
    });
}

// Plot stock data using Chart.js
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
