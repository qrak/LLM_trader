// CACHE BUSTER 2.2
let network;
let nodes;
let edges;

export function initSynapseNetwork() {
    const container = document.getElementById('synapse-network');
    
    nodes = new vis.DataSet([]);
    edges = new vis.DataSet([]);
    
    const data = { nodes, edges };
    const options = {
        nodes: {
            shape: 'dot',
            size: 14,
            font: { size: 11, color: '#c9d1d9', face: 'Inter, sans-serif' },
            borderWidth: 2,
            shadow: true,
            color: {
                background: '#58a6ff',
                border: '#30363d',
                highlight: { background: '#79c0ff', border: '#ffffff' }
            }
        },
        edges: {
            width: 2,
            color: { color: '#58a6ff', opacity: 0.8 },
            smooth: { type: 'cubicBezier', roundness: 0.5 },
            arrows: { to: { enabled: true, scaleFactor: 0.5 } }
        },
        physics: {
            enabled: true,
            stabilization: { 
                enabled: true,
                iterations: 200,
                fit: true,
                updateInterval: 25
            },
            barnesHut: {
                gravitationalConstant: -8000,
                centralGravity: 0.1,
                springConstant: 0.04,
                springLength: 250,
                damping: 0.4,
                avoidOverlap: 0.5
            }
        },
        layout: {
            improvedLayout: true,
            hierarchical: false
        },
        interaction: {
            tooltipDelay: 100,
            hideEdgesOnDrag: false,
            hover: true,
            zoomView: true,
            dragView: true
        }
    };
    network = new vis.Network(container, data, options);
    window.synapseNetwork = network;
    network.once('stabilizationIterationsDone', () => {
        network.setOptions({ physics: { enabled: false } });
        network.fit({ animation: false });
    });
    window.fitSynapseNetwork = fitNetwork;
}

export function fitNetwork() {
    if (network) {
        network.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
    }
}

export async function updateSynapses() {
    if (!nodes) return;
    
    try {
        const response = await fetch('/api/brain/memory?limit=50');
        const data = await response.json();
        
        // Only update if there's new data
        const currentCount = nodes.length;
        const newCount = data.trades ? data.trades.length : 0;
        
        if (newCount === currentCount && currentCount > 0) {
            return; // No change
        }
        
        // Clear existing nodes
        nodes.clear();
        edges.clear();
        
        // Add trades as nodes
        if (data.trades && data.trades.length > 0) {
            const newNodes = data.trades.map((trade, index) => {
                // Color based on action
                let color = '#58a6ff'; // Default blue
                let size = 12;
                
                if (trade.action === 'BUY') {
                    color = '#238636';
                    size = 16;
                } else if (trade.action === 'CLOSE_LONG') {
                    color = '#00e396';
                    size = 16;
                } else if (trade.action === 'SELL') {
                    color = '#f85149';
                    size = 16;
                } else if (trade.action === 'CLOSE_SHORT') {
                    color = '#ff6b6b';
                    size = 16;
                } else if (trade.action === 'UPDATE') {
                    color = '#8b949e';
                    size = 8;
                }
                
                const timestamp = trade.timestamp ? new Date(trade.timestamp).toLocaleString() : '';
                const priceStr = trade.price ? `$${parseFloat(trade.price).toLocaleString()}` : '';
                
                return {
                    id: trade.id || index,
                    label: trade.action,
                    title: `${timestamp}\n${priceStr}\n${trade.reasoning || ''}`,
                    color: { background: color, border: '#30363d' },
                    size: size
                };
            });
            
            nodes.add(newNodes);
            
            // Create edges: connect sequential trades WITHIN the same trade sequence
            // A trade sequence is BUY -> (UPDATE)* -> CLOSE
            let lastBuyIndex = -1;
            let lastNodeIndex = -1;
            
            for (let i = 0; i < newNodes.length; i++) {
                const action = data.trades[i].action;
                
                if (action === 'BUY' || action === 'SELL') {
                    // Start of new sequence - no back-link from previous CLOSE
                    lastBuyIndex = i;
                    lastNodeIndex = i;
                } else if (lastNodeIndex >= 0) {
                    // UPDATE or CLOSE - connect to previous node in sequence
                    edges.add({
                        from: newNodes[lastNodeIndex].id,
                        to: newNodes[i].id
                    });
                    lastNodeIndex = i;
                    
                    // If this is a CLOSE, reset for next sequence
                    if (action.includes('CLOSE')) {
                        lastNodeIndex = -1;
                    }
                }
            }
            
            // Re-enable physics briefly to layout, then freeze
            if (network) {
                network.setOptions({ physics: { enabled: true } });
                setTimeout(() => {
                    network.setOptions({ physics: { enabled: false } });
                }, 2000);
            }
        }
        
        // Show experience count
        if (data.experience_count !== undefined) {
            const countEl = document.getElementById('experience-count');
            if (countEl) {
                countEl.textContent = `${data.experience_count} learned experiences`;
            }
        }
        
    } catch (e) {
        console.error("Failed to update synapses", e);
    }
}
