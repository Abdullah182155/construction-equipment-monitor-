/**
 * Charts Module
 * Chart.js visualizations for the equipment monitoring dashboard.
 */

class DashboardCharts {
    constructor() {
        this.activityChart = null;
        this.timelineChart = null;
        this.gaugeChart = null;

        // Data buffers
        this.timelineData = [];
        this.maxTimelinePoints = 60;
        this.activityCounts = {
            'Digging': 0,
            'Swinging/Loading': 0,
            'Dumping': 0,
            'Hauling': 0,
            'Transporting': 0,
            'Waiting': 0
        };

        // Colors
        this.colors = {
            'Digging':          { bg: 'rgba(34, 197, 94, 0.7)',   border: '#22c55e' },
            'Swinging/Loading': { bg: 'rgba(56, 189, 248, 0.7)',  border: '#38bdf8' },
            'Dumping':          { bg: 'rgba(250, 204, 21, 0.7)',  border: '#facc15' },
            'Hauling':          { bg: 'rgba(251, 146, 60, 0.7)',  border: '#fb923c' },
            'Transporting':     { bg: 'rgba(99, 102, 241, 0.7)',  border: '#6366f1' },
            'Waiting':          { bg: 'rgba(148, 163, 184, 0.5)', border: '#94a3b8' }
        };

        this._initCharts();
    }

    _initCharts() {
        // Chart.js global defaults for dark theme
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = 'rgba(148, 163, 184, 0.06)';
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.font.size = 11;

        this._initActivityChart();
        this._initTimelineChart();
        this._initGaugeChart();
    }

    _initActivityChart() {
        const ctx = document.getElementById('activity-chart');
        if (!ctx) return;

        this.activityChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(this.activityCounts),
                datasets: [{
                    data: Object.values(this.activityCounts),
                    backgroundColor: Object.values(this.colors).map(c => c.bg),
                    borderColor: Object.values(this.colors).map(c => c.border),
                    borderWidth: 2,
                    hoverOffset: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 16,
                            usePointStyle: true,
                            pointStyleWidth: 10,
                            font: { size: 11 }
                        }
                    }
                }
            }
        });
    }

    _initTimelineChart() {
        const ctx = document.getElementById('timeline-chart');
        if (!ctx) return;

        this.timelineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Utilization %',
                    data: [],
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.08)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointHoverBackgroundColor: '#6366f1'
                }, {
                    label: 'Motion Score',
                    data: [],
                    borderColor: '#22d3ee',
                    backgroundColor: 'rgba(34, 211, 238, 0.05)',
                    borderWidth: 1.5,
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    borderDash: [4, 4]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                scales: {
                    x: {
                        display: true,
                        grid: { display: false },
                        ticks: { maxTicksLimit: 10, font: { size: 10 } }
                    },
                    y: {
                        display: true,
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(148, 163, 184, 0.04)' },
                        ticks: {
                            callback: v => v + '%',
                            font: { size: 10 }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            padding: 12,
                            usePointStyle: true,
                            font: { size: 10 }
                        }
                    }
                }
            }
        });
    }

    _initGaugeChart() {
        const ctx = document.getElementById('utilization-gauge');
        if (!ctx) return;

        this.gaugeChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Utilization', 'Remaining'],
                datasets: [{
                    data: [0, 100],
                    backgroundColor: [
                        'rgba(99, 102, 241, 0.8)',
                        'rgba(148, 163, 184, 0.06)'
                    ],
                    borderColor: ['#6366f1', 'transparent'],
                    borderWidth: [2, 0],
                    hoverOffset: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                circumference: 180,
                rotation: -90,
                cutout: '78%',
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                }
            },
            plugins: [{
                id: 'gaugeText',
                afterDraw: (chart) => {
                    const { ctx, chartArea } = chart;
                    const val = chart.data.datasets[0].data[0];
                    ctx.save();
                    ctx.textAlign = 'center';
                    ctx.fillStyle = '#f1f5f9';
                    ctx.font = '800 28px Inter';
                    ctx.fillText(
                        Math.round(val) + '%',
                        (chartArea.left + chartArea.right) / 2,
                        chartArea.bottom - 10
                    );
                    ctx.font = '400 10px Inter';
                    ctx.fillStyle = '#64748b';
                    ctx.fillText(
                        'FLEET UTIL',
                        (chartArea.left + chartArea.right) / 2,
                        chartArea.bottom + 8
                    );
                    ctx.restore();
                }
            }]
        });
    }

    /**
     * Update activity distribution chart.
     * @param {string} activity - Activity label
     */
    updateActivity(activity) {
        if (activity in this.activityCounts) {
            this.activityCounts[activity]++;
        }
        if (this.activityChart) {
            this.activityChart.data.datasets[0].data = Object.values(this.activityCounts);
            this.activityChart.update('none');
        }
    }

    /**
     * Add a data point to the utilization timeline.
     * @param {number} utilization - 0 to 1
     * @param {number} motionScore - raw motion score
     */
    addTimelinePoint(utilization, motionScore) {
        const now = new Date();
        const label = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        this.timelineData.push({ label, utilization: utilization * 100, motionScore: Math.min(motionScore * 10, 100) });

        if (this.timelineData.length > this.maxTimelinePoints) {
            this.timelineData.shift();
        }

        if (this.timelineChart) {
            this.timelineChart.data.labels = this.timelineData.map(d => d.label);
            this.timelineChart.data.datasets[0].data = this.timelineData.map(d => d.utilization);
            this.timelineChart.data.datasets[1].data = this.timelineData.map(d => d.motionScore);
            this.timelineChart.update('none');
        }
    }

    /**
     * Update the utilization gauge.
     * @param {number} utilization - 0 to 1
     */
    updateGauge(utilization) {
        if (this.gaugeChart) {
            const pct = utilization * 100;
            this.gaugeChart.data.datasets[0].data = [pct, 100 - pct];

            // Color based on utilization level
            let color;
            if (pct >= 70) color = '#4ade80';
            else if (pct >= 40) color = '#fb923c';
            else color = '#f87171';

            this.gaugeChart.data.datasets[0].backgroundColor[0] = color + 'cc';
            this.gaugeChart.data.datasets[0].borderColor[0] = color;
            this.gaugeChart.update('none');
        }
    }
}

window.DashboardCharts = DashboardCharts;
