let chart;

export function initPerformanceChart() {
    const options = {
        series: [{
            name: 'Account Value',
            data: []
        }],
        chart: {
            type: 'area',
            height: 280,
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
                style: { colors: '#c9d1d9', fontSize: '12px' },
                rotate: -45,
                rotateAlways: false
            },
            axisBorder: { show: false },
            axisTicks: { show: true, color: '#30363d' }
        },
        yaxis: {
            labels: {
                formatter: (value) => "$" + value.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0}),
                style: { colors: '#c9d1d9', fontSize: '12px' }
            },
            tickAmount: 5
        },
        tooltip: {
            theme: 'dark',
            x: { format: 'dd MMM yyyy HH:mm' },
            custom: function({series, seriesIndex, dataPointIndex, w}) {
                // robustness check
                if (!w.config.series[seriesIndex] || !w.config.series[seriesIndex].data[dataPointIndex]) {
                    return '';
                }

                const data = w.config.series[seriesIndex].data[dataPointIndex];
                const value = series[seriesIndex][dataPointIndex];
                const valueFormatted = value != null ? value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : 'N/A';
                const date = new Date(data.x);
                const dateFormatted = date.toLocaleString('en-GB', { 
                    day: '2-digit', month: 'short', year: 'numeric', 
                    hour: '2-digit', minute: '2-digit' 
                });

                let actionHtml = '';
                // Check if extra data exists and has an action
                if (data.extra && data.extra.action) {
                    const action = data.extra.action;
                    let color = '#8b949e';
                    if (action === 'BUY') color = '#238636';
                    else if (action === 'SELL') color = '#1f6feb';
                    else if (action.includes('CLOSE')) color = '#a371f7';
                    
                    const price = data.extra.price ? `$${data.extra.price.toLocaleString()}` : 'N/A';

                    actionHtml = `
                        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #444;">
                            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ${color}; margin-right: 6px;"></span>
                                <span style="color: #eee; font-weight: 600; font-size: 13px;">${action}</span>
                            </div>
                            <div style="color: #aaa; font-size: 12px; margin-left: 14px;">Price: ${price}</div>
                        </div>
                    `;
                }

                return `
                    <div style="background: #161b22; padding: 10px; border: 1px solid #30363d; border-radius: 4px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); min-width: 150px;">
                        <div style="color: #8b949e; font-size: 11px; margin-bottom: 4px;">${dateFormatted}</div>
                        <div style="font-size: 16px; font-weight: 700; color: #fff; margin-bottom: ${actionHtml ? '4px' : '0'};">
                            $${valueFormatted}
                        </div>
                        ${actionHtml}
                    </div>
                `;
            }
        },
        grid: {
            borderColor: '#30363d',
            strokeDashArray: 4,
            yaxis: { lines: { show: true } },
            padding: { bottom: 10, top: 50, right: 30, left: 20 }
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
                seriesData.push({ 
                    x: time, 
                    y: point.value,
                    extra: point
                });
                
                // Add trade markers
                if (point.action) {
                    const action = point.action;
                    const isBuy = action === 'BUY';
                    const isSell = action === 'SELL';
                    const isCloseLong = action === 'CLOSE_LONG';
                    const isCloseShort = action === 'CLOSE_SHORT';
                    const isGenericClose = action.includes('CLOSE') && !isCloseLong && !isCloseShort;

                    // Only add markers for trade entry/exit actions
                    if (isBuy || isSell || isCloseLong || isCloseShort || isGenericClose) {
                        let color = '#8b949e';
                        let symbol = 'circle';
                        let markerSize = 6;
                        let labelText = '';

                        if (isBuy) {
                            color = '#00ff9d'; // Vibrant entry green
                            symbol = 'rect'; // Triangle effectively via shape
                            markerSize = 8;
                        } else if (isSell) {
                            color = '#1f6feb'; // Entry blue
                            symbol = 'rect';
                            markerSize = 8;
                        } else {
                            color = '#f85149'; // Exit red
                            symbol = 'circle';
                            markerSize = 5;
                        }

                        annotations.push({
                            x: time,
                            y: point.value,
                            marker: {
                                size: markerSize,
                                fillColor: color,
                                strokeColor: '#fff',
                                strokeWidth: 1.5,
                                shape: symbol === 'rect' ? 'square' : 'circle' 
                            },
                            label: {
                                text: '', // Removing bulky text labels
                                borderWidth: 0,
                                style: { background: 'transparent' }
                            }
                        });
                    }
                }
            });
        }
        
        if (seriesData.length > 0) {
            // FORCE full replace of series to ensure name and data are tied
            chart.updateSeries([{ 
                name: 'Account Value',
                data: seriesData 
            }]);
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

// Expose refresh function specifically for fullscreen toggle or resize events
window.refreshPerformanceAnnotations = updatePerformanceData;
