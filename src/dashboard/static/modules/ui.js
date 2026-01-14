
import { fitNetwork } from './synapse_viewer.js';

/**
 * UI Controller for Sidebar and Tabs
 */

export function initUI() {
    setupSidebarNavigation();
    setupMobileMenu();
}

function setupSidebarNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from all
            navItems.forEach(nav => nav.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            
            // Add active class to clicked
            item.classList.add('active');
            
            // Show target tab
            const targetId = item.getAttribute('data-target');
            const targetTab = document.getElementById(targetId);
            if (targetTab) {
                targetTab.classList.add('active');
                
                // Trigger resize event for charts/graphs
                window.dispatchEvent(new Event('resize'));
                
                // Specific fix for vis-network graph in Brain Activity tab
                if (targetId === 'tab-brain') {
                    // Small delay to ensure the div is fully visible/rendered
                    setTimeout(() => fitNetwork(), 50);
                }
            }
        });
    });
}

function setupMobileMenu() {
    // Optional: Add hamburger menu logic here if needed later
}
