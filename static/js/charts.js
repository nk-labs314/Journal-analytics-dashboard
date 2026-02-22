// Mood Line Chart
const moodCtx = document.getElementById('moodChart')?.getContext('2d');

if (moodCtx) {
    new Chart(moodCtx, {
        type: 'line',
        data: {
            labels: window.moodDates,
            datasets: [
                {
                    label: 'Mood Score',
                    data: window.moodValues,
                    borderColor: 'blue',
                    backgroundColor: 'blue',
                    borderWidth: 2,
                    tension: 0.3
                },
                {
                    label: '7-Entry Rolling Average',
                    data: window.rollingValues,
                    borderColor: 'red',
                    backgroundColor: 'red',
                    borderWidth: 2,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

// Scatter Plot
const scatterCtx = document.getElementById('scatterChart')?.getContext('2d');

if (scatterCtx && window.sleepValues && window.moodValuesScatter) {
    new Chart(scatterCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Sleep vs Mood',
                data: window.sleepValues.map((sleep, index) => ({
                    x: sleep,
                    y: window.moodValuesScatter[index]
                })),
                backgroundColor: 'purple'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Sleep Hours'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Mood Score'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}