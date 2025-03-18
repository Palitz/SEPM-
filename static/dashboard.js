document.addEventListener("DOMContentLoaded", function () {
    fetchStockData();

    async function fetchStockData() {
        const response = await fetch("/stock-data");
        const result = await response.json();

        if (result.status === "fail") {
            alert("Session expired. Please log in again.");
            window.location.href = "/";
            return;
        }

        const stockTable = document.getElementById("stockTable");
        stockTable.innerHTML = "";  // Clear existing data

        for (const [symbol, data] of Object.entries(result)) {
            const row = `
                <tr>
                    <td>${symbol}</td>
                    <td>${data.price !== "N/A" ? `$${data.price}` : "N/A"}</td>
                    <td>${data.high !== "N/A" ? `$${data.high}` : "N/A"}</td>
                    <td>${data.low !== "N/A" ? `$${data.low}` : "N/A"}</td>
                </tr>
            `;
            stockTable.innerHTML += row;
        }
    }

    document.getElementById("logoutBtn").addEventListener("click", async function () {
        const response = await fetch("/logout", { method: "POST" });
        const result = await response.json();

        if (result.status === "success") {
            window.location.href = "/";  // Redirect to login page
        }
    });
});
