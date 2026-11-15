


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
            } else if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                item.click();
            }
        });

        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            navItems.forEach(nav => {
                nav.classList.remove('active');
                nav.setAttribute('aria-selected', 'false');
                nav.setAttribute('tabindex', '-1');
            });
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            
            item.classList.add('active');
            item.setAttribute('aria-selected', 'true');
            item.setAttribute('tabindex', '0');
            
            const targetId = item.getAttribute('data-target');
            const targetTab = document.getElementById(targetId);
            if (targetTab) {
                targetTab.classList.add('active');
                
                window.dispatchEvent(new Event('resize'));
                
                if (targetId === 'tab-brain') {
                    if (window.updateDecisionPanel) {
                        window.updateDecisionPanel().catch(() => {});
                    }
                    if (window.updateRulesPanel) {
                        window.updateRulesPanel().catch(() => {});
                    }
                }
            }

            if (window.closeMobileMenu) {
                window.closeMobileMenu();
            }
        });
    });
}

function setupMobileMenu() {
    const toggleBtn = document.getElementById('mobile-menu-toggle');
    const closeBtn = document.getElementById('mobile-menu-close');
    const sidebar = document.getElementById('sidebar');

    if (!toggleBtn || !closeBtn || !sidebar) return;

    function openMenu() {
        sidebar.classList.add('mobile-open');
        toggleBtn.setAttribute('aria-expanded', 'true');
        toggleBtn.setAttribute('aria-label', 'Close menu');
        closeBtn.setAttribute('aria-expanded', 'true');
        closeBtn.setAttribute('aria-label', 'Close menu');
        closeBtn.focus();
    }

    function closeMenu() {
        sidebar.classList.remove('mobile-open');
        toggleBtn.setAttribute('aria-expanded', 'false');
        toggleBtn.setAttribute('aria-label', 'Open menu');
        closeBtn.setAttribute('aria-expanded', 'false');
        toggleBtn.focus();
    }

    toggleBtn.addEventListener('click', openMenu);
    closeBtn.addEventListener('click', closeMenu);

    document.addEventListener('click', (e) => {
        if (sidebar.classList.contains('mobile-open') &&
            !sidebar.contains(e.target) &&
            !toggleBtn.contains(e.target)) {
            closeMenu();
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && sidebar.classList.contains('mobile-open')) {
            closeMenu();
            toggleBtn.focus();
        }
    });

    window.closeMobileMenu = closeMenu;
}
