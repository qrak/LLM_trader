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
                    const action = point.action;
                    const isBuy = action === 'BUY';
                    const isSell = action === 'SELL';
                    const isCloseLong = action === 'CLOSE_LONG';
                    const isCloseShort = action === 'CLOSE_SHORT';
                    const isGenericClose = action.includes('CLOSE') && !isCloseLong && !isCloseShort;

                    // Only add markers for trade entry/exit actions
                    if (isBuy || isSell || isCloseLong || isCloseShort || isGenericClose) {
                        let color = '#8b949e';
                        let label = action;

                        // Check current fullscreen state for scaling
                        const isFullscreen = document.getElementById('fullscreen-modal')?.classList.contains('active');
                        const scale = isFullscreen ? 1.3 : 1.0;
                        const fontSize = isFullscreen ? '15px' : '13px';
                        const borderRadius = isFullscreen ? 6 : 4;
                        const markerSize = isFullscreen ? 10 : 7;
                        
                        if (isBuy) {
                            color = '#238636';  // Green - open LONG
                            label = '▲ LONG';
                        } else if (isSell) {
                            color = '#1f6feb';  // Blue - open SHORT
                            label = '▼ SHORT';
                        } else if (isCloseLong) {
                            color = '#f85149';  // Red - close LONG
                            label = '✕ CLOSE';
                        } else if (isCloseShort) {
                            color = '#a371f7';  // Purple - close SHORT
                            label = '✓ CLOSE';
                        } else if (isGenericClose) {
                            // Legacy compatibility
                            color = '#f85149';
                            label = '✕ CLOSE';
                        }
                        
                        // Smart Collision Detection Logic
                        // Instead of just toggling, we check multiple vertical levels
                        // Base offline levels relative to point: [Top-Close, Top-Far, Bottom-Close, Bottom-Far, Top-VeryFar]
                        // Offsets are scaled by the current view scale
                        const baseOffsets = [-25, -55, 35, 65, -85]; 
                        const levels = baseOffsets.map(o => o * scale);
                        let chosenLevel = 0;
                        
                        // Simple grid-based collision check
                        // We map time to "slots" to check for horizontal proximity
                        // Slot width is roughly 2 hours in ms
                        const SLOT_WIDTH_MS = 2 * 60 * 60 * 1000;
                        const timeSlot = Math.floor(time / SLOT_WIDTH_MS);
                        
                        // Check which level is free in this time slot and adjacent slots
                        // We look at occupied levels in [slot-1, slot, slot+1]
                        const occupiedLevels = new Set();
                        
                        // Check recent annotations
                        annotations.forEach(ann => {
                            const annTime = ann.x;
                            const annSlot = Math.floor(annTime / SLOT_WIDTH_MS);
                            if (Math.abs(annSlot - timeSlot) <= 1) {
                                // This annotation is close horizontally, mark its level as occupied
                                if (ann._level !== undefined) {
                                    occupiedLevels.add(ann._level);
                                }
                            }
                        });
                        
                        // Find first free level
                        for (let i = 0; i < levels.length; i++) {
                            if (!occupiedLevels.has(i)) {
                                chosenLevel = i;
                                break;
                            }
                            // If all levels full, default to last level (highest stack)
                            if (i === levels.length - 1) chosenLevel = i;
                        }
                        
                        const yOffset = levels[chosenLevel];

                        annotations.push({
                            x: time,
                            y: point.value,
                            // Store internal level for collision detection of subsequent points
                            _level: chosenLevel, 
                            marker: {
                                size: markerSize,
                                fillColor: color,
                                strokeColor: '#fff',
                                strokeWidth: 2
                            },
                            label: {
                                text: label,
                                style: {
                                    color: '#fff',
                                    background: color,
                                    fontSize: fontSize,
                                    fontWeight: '700',
                                    padding: { 
                                        left: 8 * scale, 
                                        right: 8 * scale, 
                                        top: 5 * scale, 
                                        bottom: 5 * scale 
                                    },
                                    fontFamily: 'Inter, sans-serif'
                                },
                                offsetY: yOffset,
                                borderRadius: borderRadius,
                                textAnchor: 'middle'
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

// Expose refresh function specifically for fullscreen toggle or resize events
window.refreshPerformanceAnnotations = updatePerformanceData;
