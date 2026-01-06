let chart;

export function initPerformanceChart() {
    const options = {
        series: [{
            name: 'Account Value',
            data: []
        }],
        chart: {
            type: 'area',
            height: 250,
            background: '#161b22',
            toolbar: {
                show: true,
                tools: {
                    download: false,
                    selection: true,
                    zoom: true,
                    zoomin: true,
                    zoomout: true,
                    pan: true,
                    reset: true
                },
                autoSelected: 'zoom'
            },
            zoom: {
                enabled: true,
                type: 'x',
                autoScaleYaxis: true
            },
            animations: {
                enabled: true,
                easing: 'easeinout',
                speed: 500
            }
        },
        colors: ['#00e396'],
        dataLabels: { enabled: false },
        stroke: {
            curve: 'smooth',
            width: 2
        },
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.7,
                opacityTo: 0.1,
                stops: [0, 90, 100]
            }
        },
        theme: { mode: 'dark' },
        xaxis: {
            type: 'datetime',
            labels: {
                datetimeUTC: false,
                format: 'dd MMM HH:mm',
                style: { colors: '#8b949e', fontSize: '10px' }
            },
            axisBorder: { show: false },
            axisTicks: { show: true, color: '#30363d' }
        },
        yaxis: {
            labels: {
                formatter: (value) => "$" + value.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0}),
                style: { colors: '#8b949e', fontSize: '10px' }
            }
        },
        tooltip: {
            theme: 'dark',
            x: { format: 'dd MMM yyyy HH:mm' },
            y: {
                formatter: (value) => "$" + value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})
            }
        },
        grid: {
            borderColor: '#30363d',
            strokeDashArray: 4,
            yaxis: { lines: { show: true } },
            padding: { bottom: 10 }
        },
        noData: {
            text: 'No trades yet',
            style: { color: '#8b949e', fontSize: '14px' }
        },
        annotations: {
            points: []
        }
    };

    chart = new ApexCharts(document.querySelector("#performance-chart"), options);
    chart.render();
    window.performanceChart = chart;
}

export async function updatePerformanceData() {
    if (!chart) return;
    
    try {
        // Fetch history for chart
        const historyResponse = await fetch('/api/performance/history');
        const historyData = await historyResponse.json();
        
        let seriesData = [];
        let annotations = [];
        
        if (historyData.history && Array.isArray(historyData.history) && historyData.history.length > 0) {
            historyData.history.forEach((point, index) => {
                const time = new Date(point.time).getTime();
                seriesData.push({ x: time, y: point.value });
                
                // Add trade markers
                if (point.action) {
                    const isBuy = point.action.includes('BUY');
                    const isClose = point.action.includes('CLOSE');
                    
                    if (isBuy || isClose) {
                        annotations.push({
                            x: time,
                            y: point.value,
                            marker: {
                                size: 6,
                                fillColor: isBuy ? '#238636' : '#f85149',
                                strokeColor: '#fff',
                                strokeWidth: 1
                            },
                            label: {
                                text: isBuy ? 'BUY' : 'CLOSE',
                                style: {
                                    color: '#fff',
                                    background: isBuy ? '#238636' : '#f85149',
                                    fontSize: '9px',
                                    padding: { left: 4, right: 4, top: 2, bottom: 2 }
                                }
                            }
                        });
                    }
                }
            });
        }
        
        if (seriesData.length > 0) {
            chart.updateSeries([{ data: seriesData }]);
            if (annotations.length > 0) {
                chart.updateOptions({ annotations: { points: annotations } });
            }
        }
        
        // Fetch stats separately
        const statsResponse = await fetch('/api/performance/stats');
        const stats = await statsResponse.json();
        
        const statsEl = document.getElementById('stats-summary');
        if (statsEl && stats) {
            const pnlColor = (stats.total_pnl_pct || 0) >= 0 ? '#238636' : '#f85149';
            const winColor = (stats.win_rate || 0) >= 50 ? '#238636' : '#f85149';
            
            statsEl.innerHTML = `
                <span><strong>Trades:</strong> ${stats.total_trades || 0}</span>
                <span><strong>Win Rate:</strong> <span style="color: ${winColor}">${(stats.win_rate || 0).toFixed(1)}%</span></span>
                <span><strong>P&L:</strong> <span style="color: ${pnlColor}">${(stats.total_pnl_pct || 0) >= 0 ? '+' : ''}${(stats.total_pnl_pct || 0).toFixed(2)}%</span></span>
                <span><strong>Capital:</strong> $${(stats.current_capital || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>
            `;
        }
        
    } catch (e) {
        console.error("Failed to update performance chart", e);
    }
}
