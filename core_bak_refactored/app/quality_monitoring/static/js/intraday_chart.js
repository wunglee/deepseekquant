/**
 * 分时图模块 - 完全独立的分时图管理
 * 职责：管理分时图的布局、数据加载、渲染
 */

// ==================== 全局状态 ====================
let intradayPriceChart = null
let intradayVolumeChart = null
let intradayData = null
let intradayUpdateTimer = null
let lastIntradayBatchIndex = 0
let lastIntradayRequestTime = 0
let intradayDataMode = 'mock'  // 'mock' 或 'real'
let virtualIntradayTime = 0  // 🎮 虚拟交易时间（秒），用于模拟模式

// ==================== 核心函数：彻底重建分时图布局 ====================

/**
 * 🔥 彻底重建分时图布局（从最外层容器开始重建）
 * @param {boolean} isStock - 是否是股票（true=股票，左右布局; false=指数，单列布局）
 */
function rebuildIntradayLayout(isStock) {
    const container = document.getElementById('intradayContainer')
    if (!container) {
        console.error('❌ 找不到分时图容器')
        return
    }
    
    // 🔧 1. 停止定时器
    if (intradayUpdateTimer) {
        clearInterval(intradayUpdateTimer)
        intradayUpdateTimer = null
    }
    
    // 🔧 2. 销毁旧的图表实例
    if (intradayPriceChart) {
        try {
            intradayPriceChart.dispose()
        } catch(e) {
            console.warn('销毁价格图失败:', e)
        }
        intradayPriceChart = null
    }
    if (intradayVolumeChart) {
        try {
            intradayVolumeChart.dispose()
        } catch(e) {
            console.warn('销毁成交量图失败:', e)
        }
        intradayVolumeChart = null
    }
    
    // 🔧 3. 清空全局数据
    intradayData = null
    lastIntradayBatchIndex = 0
    lastIntradayRequestTime = 0
    
    // 🔧 4. 清空容器（连根拔起）
    container.innerHTML = ''
    
    // 🔧 5. 根据类型重建布局
    if (isStock) {
        // 股票：左右布局（左侧图表 + 右侧盘口/成交）
        container.innerHTML = `
            <div style="display:grid; grid-template-columns: 2fr 1fr; gap:12px; width:100%;">
                <!-- 左侧：分时曲线+成交量容器 -->
                <div style="min-width:0; overflow:hidden; display:flex; flex-direction:column;">
                    <div id="intradayPriceChart" style="height:360px; width:100%;"></div>
                    <div id="intradayVolumeChart" style="height:180px; width:100%; margin-top:8px;"></div>
                </div>
                <!-- 右侧：挂单+成交明细 -->
                <div style="min-width:0; overflow:hidden;">
                    <div style="height:280px; border:1px solid #e5e7eb; border-radius:6px; padding:8px; overflow-y:auto;">
                        <div style="font-size:12px; font-weight:600; margin-bottom:8px; color:#374151;">买卖盘口</div>
                        <table style="width:100%; font-size:11px;">
                            <thead style="background:#f9fafb;">
                                <tr>
                                    <th style="padding:4px; text-align:left; width:25%;"></th>
                                    <th style="padding:4px; text-align:center; width:40%;">价格</th>
                                    <th style="padding:4px; text-align:right; width:35%;">数量</th>
                                </tr>
                            </thead>
                            <tbody id="orderBookBody"></tbody>
                        </table>
                    </div>
                    <div style="height:140px; margin-top:8px; border:1px solid #e5e7eb; border-radius:6px; padding:8px; overflow-y:auto;">
                        <div style="font-size:12px; font-weight:600; margin-bottom:8px; color:#374151;">成交明细</div>
                        <div id="tickerList" style="font-size:11px;"></div>
                    </div>
                </div>
            </div>
        `
    } else {
        // 指数：单列布局（只有图表）
        container.innerHTML = `
            <div style="display:flex; flex-direction:column;">
                <div id="intradayPriceChart" style="height:360px; width:100%;"></div>
                <div id="intradayVolumeChart" style="height:180px; width:100%; margin-top:8px;"></div>
            </div>
        `
    }
    
    // 🔧 6. 重新初始化图表实例
    const priceChartDom = document.getElementById('intradayPriceChart')
    const volumeChartDom = document.getElementById('intradayVolumeChart')
    
    if (priceChartDom && volumeChartDom) {
        intradayPriceChart = echarts.init(priceChartDom)
        intradayVolumeChart = echarts.init(volumeChartDom)
        
        // 连接两个图表，确保 tooltip 同步
        echarts.connect([intradayPriceChart, intradayVolumeChart])
        
        showIntradayLoading(true)
    } else {
        console.error('❌ 无法找到图表DOM元素')
    }
}

/**
 * 显示/隐藏分时图加载状态
 */
function showIntradayLoading(show) {
    if (!intradayPriceChart || !intradayVolumeChart) return
    
    if (show) {
        intradayPriceChart.setOption({
            graphic: [{
                type: 'text',
                left: 'center',
                top: 'center',
                style: { text: '加载中...', fontSize: 16, fill: '#999' }
            }]
        }, true)
        
        intradayVolumeChart.setOption({
            graphic: [{
                type: 'text',
                left: 'center',
                top: 'center',
                style: { text: '加载中...', fontSize: 16, fill: '#999' }
            }]
        }, true)
    } else {
        intradayPriceChart.setOption({ graphic: [] })
        intradayVolumeChart.setOption({ graphic: [] })
    }
}

// 🔧 删除：isTradingTime() - 前端不再判断交易时段，完全依赖后端 should_poll
// 原因：所有控制前端行为的参数必须来自后端

// ==================== 导出接口 ====================
window.IntradayChart = {
    // 状态访问
    getCharts: () => ({ price: intradayPriceChart, volume: intradayVolumeChart }),
    getData: () => intradayData,
    setData: (data) => { intradayData = data },
    getMode: () => intradayDataMode,
    setMode: (mode) => { intradayDataMode = mode },
    getTimer: () => intradayUpdateTimer,
    setTimer: (timer) => { intradayUpdateTimer = timer },
    getBatchIndex: () => lastIntradayBatchIndex,
    setBatchIndex: (index) => { lastIntradayBatchIndex = index },
    getRequestTime: () => lastIntradayRequestTime,
    setRequestTime: (time) => { lastIntradayRequestTime = time },
    getVirtualTime: () => virtualIntradayTime,  // 🎮 获取虚拟时间
    setVirtualTime: (time) => { virtualIntradayTime = time },  // 🎮 设置虚拟时间
    
    // 核心功能
    rebuildLayout: rebuildIntradayLayout,
    showLoading: showIntradayLoading
    // 🔧 删除：isTradingTime - 前端不再提供交易时段判断功能
}
