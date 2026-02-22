
import { initSynapseNetwork } from './synapse_viewer.js';

/**
 * UI Controller for Sidebar and Tabs
 */

export function initUI() {
    setupSidebarNavigation();
    setupMobileMenu();
}

function setupSidebarNavigation() {
    const navItems = document.querySelectorAll('.tab-btn');
    const navList = document.querySelector('.sidebar-nav');
    if (navList) {
        navList.setAttribute('role', 'tablist');
        navList.setAttribute('aria-orientation', 'vertical');
    }
    
    navItems.forEach((item, index) => {
        // Accessibility Setup
        const targetId = item.getAttribute('data-target');
        const isActive = item.classList.contains('active');
        item.setAttribute('role', 'tab');
        item.setAttribute('aria-selected', isActive);
        item.setAttribute('aria-controls', targetId);
        item.setAttribute('id', `nav-${targetId}`);
        item.setAttribute('tabindex', isActive ? '0' : '-1');

        const targetPanel = document.getElementById(targetId);
        if (targetPanel) {
            targetPanel.setAttribute('role', 'tabpanel');
            targetPanel.setAttribute('aria-labelledby', `nav-${targetId}`);
        }

        item.addEventListener('keydown', (e) => {
            let nextIndex = null;
            if (e.key === 'ArrowDown' || e.key === 'ArrowRight') nextIndex = (index + 1) % navItems.length;
            if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') nextIndex = (index - 1 + navItems.length) % navItems.length;
            if (nextIndex !== null) {
                e.preventDefault();
                navItems[nextIndex].focus();
                navItems[nextIndex].click();
            }
        });

        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from all
            navItems.forEach(nav => {
                nav.classList.remove('active');
                nav.setAttribute('aria-selected', 'false');
                nav.setAttribute('tabindex', '-1');
            });
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            
            // Add active class to clicked
            item.classList.add('active');
            item.setAttribute('aria-selected', 'true');
            item.setAttribute('tabindex', '0');
            
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
                    setTimeout(() => {
                        if (window.fitSynapseNetwork) window.fitSynapseNetwork();
                    }, 50);
                }
            }
        });
    });
}

function setupMobileMenu() {
    // Optional: Add hamburger menu logic here if needed later
}
