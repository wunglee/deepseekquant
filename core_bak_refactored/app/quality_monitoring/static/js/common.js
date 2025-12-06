/**
 * DeepSeekQuant 应用层通用 JavaScript 工具库
 */

// ========== 全局配置 ==========
const AppConfig = {
    apiBaseUrl: '/api/v1',
    refreshInterval: 5000,
    chartColors: {
        primary: '#2563eb',
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        info: '#3b82f6'
    }
};

// ========== 导航激活状态 ==========
function setActiveNav() {
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

// ========== ECharts 空状态渲染 ==========
function showEmptyChart(chart, text = '暂无数据') {
    chart.setOption({
        xAxis: { show: false },
        yAxis: { show: false },
        series: [],
        graphic: [{
            type: 'text',
            left: 'center',
            top: 'center',
            style: {
                text: text,
                fontSize: 16,
                fill: '#999'
            }
        }]
    }, true);
}

// ========== API 请求封装 ==========
async function apiRequest(endpoint, options = {}) {
    const url = endpoint.startsWith('http') ? endpoint : `${AppConfig.apiBaseUrl}${endpoint}`;
    
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API请求失败:', error);
        showToast('请求失败: ' + error.message, 'danger');
        throw error;
    }
}

// ========== 轻量级提示框 ==========
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        padding: 12px 20px;
        background: ${type === 'success' ? '#10b981' : type === 'danger' ? '#ef4444' : '#3b82f6'};
        color: white;
        border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        z-index: 9999;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ========== 日期格式化 ==========
function formatDate(dateString, format = 'YYYY-MM-DD HH:mm:ss') {
    const date = new Date(dateString);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    
    return format
        .replace('YYYY', year)
        .replace('MM', month)
        .replace('DD', day)
        .replace('HH', hours)
        .replace('mm', minutes)
        .replace('ss', seconds);
}

// ========== 数字格式化 ==========
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return '-';
    return Number(num).toFixed(decimals);
}

function formatPercent(num, decimals = 2) {
    if (num === null || num === undefined) return '-';
    return (Number(num) * 100).toFixed(decimals) + '%';
}

// ========== 安全获取嵌套属性 ==========
function getNestedValue(obj, path, defaultValue = null) {
    return path.split('.').reduce((acc, part) => acc?.[part], obj) ?? defaultValue;
}

// ========== 防抖函数 ==========
function debounce(func, wait = 300) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ========== 节流函数 ==========
function throttle(func, limit = 300) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ========== 加载状态管理 ==========
function showLoading(element) {
    if (typeof element === 'string') {
        element = document.querySelector(element);
    }
    if (element) {
        element.innerHTML = '<div class="loading"></div>';
    }
}

function hideLoading(element, content = '') {
    if (typeof element === 'string') {
        element = document.querySelector(element);
    }
    if (element) {
        element.innerHTML = content;
    }
}

// ========== 表格渲染工具 ==========
function renderTable(tableId, data, columns) {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    const tbody = table.querySelector('tbody');
    if (!tbody) return;
    
    if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="' + columns.length + '" class="text-center text-muted">暂无数据</td></tr>';
        return;
    }
    
    tbody.innerHTML = data.map(row => {
        return '<tr>' + columns.map(col => {
            const value = typeof col.field === 'function' 
                ? col.field(row) 
                : getNestedValue(row, col.field);
            return '<td>' + (col.format ? col.format(value, row) : value || '-') + '</td>';
        }).join('') + '</tr>';
    }).join('');
}

// ========== 系统状态检查 ==========
async function checkSystemHealth() {
    try {
        const response = await apiRequest('/health');
        const indicator = document.querySelector('.status-indicator');
        if (indicator) {
            indicator.style.background = response.status === 'healthy' ? '#10b981' : '#ef4444';
        }
        return response.status === 'healthy';
    } catch (error) {
        const indicator = document.querySelector('.status-indicator');
        if (indicator) {
            indicator.style.background = '#ef4444';
        }
        return false;
    }
}

// ========== 页面初始化 ==========
document.addEventListener('DOMContentLoaded', function() {
    // 设置导航激活状态
    setActiveNav();
    
    // 定期检查系统健康状态
    checkSystemHealth();
    setInterval(checkSystemHealth, 30000);
    
    // 添加动画样式
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
});

// ========== 导出工具函数 ==========
window.AppUtils = {
    setActiveNav,
    apiRequest,
    showToast,
    showEmptyChart,
    formatDate,
    formatNumber,
    formatPercent,
    getNestedValue,
    debounce,
    throttle,
    showLoading,
    hideLoading,
    renderTable,
    checkSystemHealth
};
