document.addEventListener("DOMContentLoaded", function () {
    // Handle login form submission
    const loginForm = document.getElementById("loginForm");
    if (loginForm) {
        loginForm.addEventListener("submit", async function (event) {
            event.preventDefault();
            const formData = new FormData(loginForm);

            const response = await fetch("/login", {
                method: "POST",
                body: formData
            });

            if (response.redirected) {
                window.location.href = response.url;
            } else {
                const data = await response.text();
                document.getElementById("error").innerHTML = data;
            }
        });
    }

    // Fetch stock data if on dashboard
    if (window.location.pathname === "/dashboard") {
        fetchStockData();
    }

    // Logout button functionality
    const logoutButton = document.getElementById("logout");
    if (logoutButton) {
        logoutButton.addEventListener("click", function () {
            window.location.href = "/logout";
        });
    }
});

// Fetch stock data and display it
async function fetchStockData() {
    try {
        const response = await fetch("/stock-data");
        if (!response.ok) {
            throw new Error("Failed to fetch stock data");
        }

        const stockData = await response.json();
        updateStockTable(stockData);
        plotStockGraph(stockData);
    } catch (error) {
        console.error("Error fetching stock data:", error);
    }
}

// Populate stock data table
function updateStockTable(stockData) {
    const stockTableBody = document.getElementById("stockTableBody");
    stockTableBody.innerHTML = ""; // Clear existing data

    Object.entries(stockData).forEach(([symbol, data]) => {
        const row = `<tr>
                        <td>${symbol}</td>
                        <td>${data.price !== "N/A" ? `$${data.price}` : "N/A"}</td>
                        <td>${data.high !== "N/A" ? `$${data.high}` : "N/A"}</td>
                        <td>${data.low !== "N/A" ? `$${data.low}` : "N/A"}</td>
                    </tr>`;
        stockTableBody.innerHTML += row;
    });
}

// Plot stock data using Chart.js
function plotStockGraph(stockData) {
    const labels = Object.keys(stockData);
    const prices = Object.values(stockData).map((data) => data.price);

    const ctx = document.getElementById("stockChart").getContext("2d");
    new Chart(ctx, {
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
