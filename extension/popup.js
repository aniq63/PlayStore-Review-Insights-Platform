const analyzeBtn = document.getElementById("analyze")
const btnLoader = document.getElementById("btnLoader")
const btnText = document.getElementById("btnText")
const statusContainer = document.getElementById("statusContainer")
const statusDiv = document.getElementById("status")
const percentText = document.getElementById("percent")
const progressFill = document.getElementById("progressFill")
const resultsDiv = document.getElementById("results")
const emptyState = document.getElementById("emptyState")

let sentimentChart
let topicChart
let negativeChart

// Premium Color Palettes
const colors = {
    positive: '#10b981',
    neutral: '#6366f1',
    negative: '#f43f5e',
    chart: ['#6366f1', '#8b5cf6', '#d946ef', '#ec4899', '#f43f5e', '#f97316', '#eab308']
}

function updateStatus(message, percent) {
    statusDiv.textContent = message
    percentText.textContent = `${percent}%`
    progressFill.style.width = `${percent}%`
}

analyzeBtn.addEventListener("click", async () => {
    // Reset UI
    resultsDiv.style.display = 'none'
    emptyState.style.display = 'none'
    statusContainer.style.display = 'block'
    btnLoader.style.display = 'inline-block'
    btnText.textContent = "Analyzing..."
    analyzeBtn.disabled = true

    updateStatus("Collecting reviews...", 20)

    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })

    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: scrapeReviews
    },
        async (result) => {
            const reviews = result[0].result
            const appId = extractAppId(tab.url)

            if (reviews.length === 0 && appId === "unknown") {
                updateStatus("Failed: App ID unknown", 0)
                btnLoader.style.display = 'none'
                btnText.textContent = "Analyze Reviews"
                analyzeBtn.disabled = false
                emptyState.style.display = 'block'
                emptyState.innerHTML = '<p style="color: #f43f5e;">Error: Could not determine App ID.</p>'
                return
            }

            const statusMsg = reviews.length > 0 ? "Analyzing collected reviews..." : "Fetching reviews via SerpApi..."
            updateStatus(statusMsg, 50)

            try {
                const response = await fetch("http://localhost:8000/analyze", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        app_id: appId,
                        reviews: reviews
                    })
                })

                if (!response.ok) throw new Error("Backend error")

                const data = await response.json()

                updateStatus("Processing insights...", 90)

                setTimeout(() => {
                    statusContainer.style.display = 'none'
                    resultsDiv.style.display = 'block'
                    btnLoader.style.display = 'none'
                    btnText.textContent = "Analyze Again"
                    analyzeBtn.disabled = false

                    renderSentimentChart(data.sentiment_distribution)
                    renderTopicChart(data.topics)
                    renderNegativeChart(data.insights.top_negative_topics)
                }, 500)

            }
            catch (err) {
                updateStatus(`Error: ${err.message}`, 0)
                btnLoader.style.display = 'none'
                btnText.textContent = "Try Again"
                analyzeBtn.disabled = false
                console.error(err)
            }
        })
})

function scrapeReviews() {
    let reviews = []
    const nodes = document.querySelectorAll(".h3YV2d")
    nodes.forEach(n => reviews.push(n.innerText))
    return reviews.slice(0, 50)
}

function extractAppId(url) {
    const match = url.match(/id=([^&]+)/)
    return match ? match[1] : "unknown"
}

const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: false
        }
    },
    animation: {
        duration: 2000,
        easing: 'easeOutQuart'
    }
}

function renderSentimentChart(data) {
    const ctx = document.getElementById("sentimentChart").getContext('2d')
    if (sentimentChart) sentimentChart.destroy()

    sentimentChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Positive", "Neutral", "Negative"],
            datasets: [{
                data: [data.Positive || 0, data.Neutral || 0, data.Negative || 0],
                backgroundColor: [colors.positive, colors.neutral, colors.negative],
                borderWidth: 0,
                hoverOffset: 10
            }]
        },
        options: {
            ...chartDefaults,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: { color: '#94a3b8', font: { family: 'Outfit' }, padding: 20 }
                }
            },
            cutout: '70%'
        }
    })
}

function renderTopicChart(topics) {
    const ctx = document.getElementById("topicChart").getContext('2d')
    if (topicChart) topicChart.destroy()

    topicChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: topics.map(t => t.topic),
            datasets: [{
                label: "Reviews",
                data: topics.map(t => t.count),
                backgroundColor: colors.chart,
                borderRadius: 6,
                barThickness: 20
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                x: { ticks: { color: '#94a3b8', font: { family: 'Outfit' } }, grid: { display: false } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    })
}

function renderNegativeChart(items) {
    const ctx = document.getElementById("negativeChart").getContext('2d')
    if (negativeChart) negativeChart.destroy()

    negativeChart = new Chart(ctx, {
        type: "bar",
        indexAxis: 'y',
        data: {
            labels: items.map(i => i.topic),
            datasets: [{
                label: "Negative Reviews",
                data: items.map(i => i.negative_reviews),
                backgroundColor: 'rgba(244, 63, 94, 0.8)',
                borderRadius: 4,
                barThickness: 15
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: '#94a3b8', font: { family: 'Outfit' } }, grid: { display: false } }
            }
        }
    })
}
