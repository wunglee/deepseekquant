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

/**
 * 根据市场配置动态生成时间轴标签配置
 * @param {string} marketCode - 市场代码
 * @param {Array} fullTradingTimes - 完整的交易时间数组
 * @returns {Object} - ECharts x轴标签配置对象
 */
function generateTimeAxisLabelConfig(marketCode, fullTradingTimes) {
    // 获取市场配置
    const market = window.marketConfig?.[marketCode?.toUpperCase()] || 
                   window.marketsConfig?.find(m => m.code.toUpperCase() === marketCode?.toUpperCase());
    
    const tradingHours = market?.detailed_trading_hours || {};
    const { open, close, lunch_start, lunch_end } = tradingHours;
    
    // 解析时间
    const [openHour, openMin] = open?.split(':').map(Number) || [9, 30];
    const [closeHour, closeMin] = close?.split(':').map(Number) || [15, 0];
    const [lunchStartHour, lunchStartMin] = lunch_start?.split(':').map(Number) || [11, 30];
    const [lunchEndHour, lunchEndMin] = lunch_end?.split(':').map(Number) || [13, 0];
    
    // 🔧 动态生成半小时时间点
    const displayTimes = new Set();
    
    // 从开盘时间开始，每30分钟生成一个时间点，直到收盘
    let currentHour = openHour;
    let currentMinute = openMin;
    
    while (currentHour < closeHour || (currentHour === closeHour && currentMinute <= closeMin)) {
        const timeStr = `${String(currentHour).padStart(2, '0')}:${String(currentMinute).padStart(2, '0')}:00`;
        displayTimes.add(timeStr);
        
        // 记录当前时间，用于判断是否遇到午休开始
        const prevHour = currentHour;
        const prevMinute = currentMinute;
        
        // 添加30分钟
        currentMinute += 30;
        if (currentMinute >= 60) {
            currentMinute -= 60;
            currentHour += 1;
        }
        
        // 🔧 如果遇到午休开始时间，先添加午休开始时间，然后跳到午休结束时间
        if (prevHour === lunchStartHour && prevMinute === lunchStartMin) {
            // 确保午休开始时间已添加（已在上面的displayTimes.add中添加）
            // 跳到午休结束时间
            currentHour = lunchEndHour;
            currentMinute = lunchEndMin;
        }
    }
    
    // 确保收盘时间显示，但排除午休结束时间
    const closeTimeStr = `${String(closeHour).padStart(2, '0')}:${String(closeMin).padStart(2, '0')}:00`;
    displayTimes.add(closeTimeStr);
    
    // 从显示时间集合中移除午休结束时间（如果存在）
    const lunchEndTimeStr = `${String(lunchEndHour).padStart(2, '0')}:${String(lunchEndMin).padStart(2, '0')}:00`;
    displayTimes.delete(lunchEndTimeStr);
    
    console.log('🔍 动态生成的时间点:', Array.from(displayTimes).sort());
    
    // 🔧 创建自定义的 formatter，只显示指定的时间点
    return {
        interval: 0,  // 显示所有标签，由 formatter 控制哪些显示
        formatter: function(value, index) {
            // 只显示指定的时间点，其他时间返回空字符串
            if (displayTimes.has(value)) {
                return formatTimeAxisLabel(value, index);
            }
            return '';
        },
        showMinLabel: true,
        showMaxLabel: true
    };
}

/**
 * 显示图表加载状态（通用函数）
 */
function showChartLoading(chart, show, text = '加载中') {
    if (show) {
        chart.clear()
        chart.setOption({
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
        }, true)
    } else {
        // 清除graphic（实际上不需要，因为renderXXX会覆盖）
        chart.setOption({ graphic: [] })
    }
}

/**
 * 显示分时图错误信息
 */
function showIntradayError(message) {
    showIntradayLoading(false)

    // 在图表上显示错误
    const errorOption = {
        title: {
            text: '加载失败',
            subtext: message,
            left: 'center',
            top: 'center',
            textStyle: { fontSize: 16, color: '#ef4444' },
            subtextStyle: { fontSize: 12, color: '#9ca3af' }
        },
        xAxis: { show: false },
        yAxis: { show: false },
        series: []
    }

    intradayPriceChart.setOption(errorOption, true)
    intradayVolumeChart.setOption(errorOption, true)

    // 盘口和成交明细显示错误
    const tbody = document.getElementById('orderBookBody')
    if (tbody) {
        tbody.innerHTML = `<tr><td colspan="3" style="text-align:center; padding:20px; color:#ef4444;">${message}</td></tr>`
    }

    const tickerList = document.getElementById('tickerList')
    if (tickerList) {
        tickerList.innerHTML = `<div style="text-align:center; padding:20px; color:#ef4444;">${message}</div>`
    }
}

function generateTradingTimes(tradingTimes,start,end,stepSeconds = 5) {
    if (!tradingTimes) {
        tradingTimes = []
    }
    
    // 将开始和结束时间转换为总秒数（从当天00:00:00开始计算）
    const startTotalSeconds = start[0] * 3600 + start[1] * 60;
    const endTotalSeconds = end[0] * 3600 + end[1] * 60; // 结束时间是XX:XX:00
    
    // 循环从开始时间到结束时间，按stepSeconds步长递增
    for (let totalSeconds = startTotalSeconds; totalSeconds <= endTotalSeconds; totalSeconds += stepSeconds) {
        // 将总秒数转换回时、分、秒
        const hour = Math.floor(totalSeconds / 3600);
        const minute = Math.floor((totalSeconds % 3600) / 60);
        const second = totalSeconds % 60;
        
        // 检查是否超出结束时间
        if (hour > end[0] || (hour === end[0] && minute > end[1])) continue;
        
        const timeStr = `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}:${String(second).padStart(2, '0')}`;
        
        // 避免重复添加时间点
        if (!tradingTimes.includes(timeStr)) {
            tradingTimes.push(timeStr);
        }
    }
    
    return tradingTimes
}

/**
 * 初始化分时图时间轴
 */
function initializeIntradayTimeAxis() {
    if (!window.currentMarketCode) {
        console.warn('⚠️ 当前市场代码未设置，无法初始化时间轴')
        return
    }

    console.log('🔍 initializeIntradayTimeAxis - 当前市场:', window.currentMarketCode)

    // 生成当前市场的完整交易时间轴
    const timeAxisInfo = getFullTradingTimes(window.currentMarketCode)
    const fullTradingTimes = timeAxisInfo.tradingTimes
    const lunchBreakRange = timeAxisInfo.lunchBreakRange
    console.log('🔍 initializeIntradayTimeAxis - 时间轴长度:', fullTradingTimes.length, '午休范围:', lunchBreakRange)
    
    // 🔧 根据市场配置动态生成半小时时间点
    const axisLabelConfig = generateTimeAxisLabelConfig(window.currentMarketCode, fullTradingTimes)
    
    
    // 设置图表的基础配置（与具体股票无关的时间轴和分割线）
    const charts = IntradayChart.getCharts()
    console.log('🔍 initializeIntradayTimeAxis - 图表实例:', charts)

    if (charts.price && charts.volume) {
        // 准备基础markLine配置（用于午休分割线等）
        const markLineData = makeLunchBreakLine()
        console.log('🔍 initializeIntradayTimeAxis - markLineData:', markLineData)

        // 为价格图表设置完整的基础配置
        charts.price.setOption({
            // 标题配置占位（数据相关部分在renderIntradayCharts中设置）
            title: {
                text: '',
                left: 'center',
                textStyle: { fontSize: 14 },
                subtext: '',
                subtextStyle: { fontSize: 11 }
            },
            // tooltip配置
            tooltip: {
                trigger: 'axis',
                formatter: function(params) {
                    let result = params[0].axisValue + '<br/>'
                    params.forEach(item => {
                        // 处理null值
                        if (item.value === null || item.value === undefined) {
                            return
                        }
                        const value = parseFloat(item.value)
                        result += item.marker + item.seriesName + ': ' + value.toFixed(2) + '<br/>'
                    })
                    return result
                }
            },
            // 网格配置
            grid: { left: '50px', right: '50px', top: '60px', bottom: '30px' },
            // x轴配置（时间轴）
            xAxis: {
                type: 'category',
                data: fullTradingTimes,  // 使用全局初始化的时间轴
                boundaryGap: false,
                axisLabel: axisLabelConfig  // 🔧 使用动态计算的刻度配置
            },
            // y轴配置
            yAxis: {
                type: 'value',
                scale: true,
                splitLine: {
                    lineStyle: { type: 'dashed', color: '#e5e7eb' }
                },
                axisLine: { onZero: false }
            },
            // 系列配置（基础结构）
            series: [
                {
                    name: '价格',
                    type: 'line',
                    data: [],  // 初始为空，等待 renderIntradayCharts 填充
                    smooth: 0.6,
                    symbol: 'none',
                    showSymbol: false,
                    lineStyle: { width: 2 },
                    itemStyle: { color: '#2563eb' },
                    areaStyle: {
                        color: {
                            type: 'linear',
                            x: 0, y: 0, x2: 0, y2: 1,
                            colorStops: [
                                { offset: 0, color: 'rgba(37, 99, 235, 0.3)' },
                                { offset: 1, color: 'rgba(37, 99, 235, 0.05)' }
                            ]
                        }
                    },
                    connectNulls: true,
                    markLine: {
                        symbol: 'none',
                        silent: false,
                        animation: false,
                        data: markLineData  // 午休分割线
                    }
                },
                {
                    name: '均价',
                    type: 'line',
                    data: [],  // 初始为空，等待 renderIntradayCharts 填充
                    smooth: 0.6,
                    symbol: 'none',
                    showSymbol: false,
                    lineStyle: { width: 1.5, color: '#f59e0b', type: 'dashed' },
                    connectNulls: true
                }
            ]
        }, true)  // 关键：使用true完全替换，清除旧市场配置

        // 为成交量图表设置完整的基础配置
        charts.volume.setOption({
            // 标题配置
            title: { 
                text: '成交量', 
                left: 'center', 
                textStyle: { fontSize: 12 } 
            },
            // tooltip配置
            tooltip: {
                trigger: 'axis',
                formatter: function(params) {
                    if (!params || params.length === 0) return ''
                    const volume = params[0].value
                    // 处理null值
                    if (volume === null || volume === undefined) {
                        return params[0].axisValue + '<br/>' + params[0].marker + '成交量: 无数据'
                    }
                    return params[0].axisValue + '<br/>' + params[0].marker + '成交量: ' + volume.toLocaleString() + '手'
                }
            },
            // 网格配置
            grid: { left: '50px', right: '50px', top: '40px', bottom: '20px' },
            // x轴配置（时间轴）
            xAxis: {
                type: 'category',
                data: fullTradingTimes,
                show: false,
                axisLabel: axisLabelConfig  // 🔧 使用相同的刻度配置
            },
            // y轴配置
            yAxis: {
                type: 'value',
                splitLine: { lineStyle: { type: 'dashed', color: '#e5e7eb' } }
            },
            // 系列配置（基础结构）
            series: [{
                name: '成交量',
                type: 'bar',
                data: [],  // 初始为空，等待 renderIntradayCharts 填充
                barWidth: '80%',
                markLine: {
                    symbol: 'none',
                    silent: false,
                    animation: false,
                    data: markLineData  // 午休分割线
                }
            }]
        }, true)  // 关键：使用true完全替换，清除旧市场配置
    }
}

// 获取模式特定的配置
function getModeConfig(mode, isInitial) {
    const lastRequestTime = IntradayChart.getRequestTime();
    
    switch (mode) {
        case 'mock':
            const simulationMode = window.selectedTradingPhase || 'trading';
            return {
                pollInterval: 5000,
                setupInitialState: function() {
                    const virtualStartTime = '2024-12-14 09:30:00';
                    IntradayChart.setVirtualTime(virtualStartTime);
                },
                getCurrentTime: function(isInitial) {
                    if (isInitial) {
                        // 首次加载：使用已设置的虚拟时间 (09:30:00)
                        return IntradayChart.getVirtualTime();
                    } else {
                        // 增量更新：虚拟时间递增 1 分钟
                        const lastVirtualTime = IntradayChart.getVirtualTime();
                        // 🔧 直接字符串操作：解析时间并增加 1 分钟
                        const [date, time] = lastVirtualTime.split(' ');
                        const [hours, minutes, seconds] = time.split(':').map(Number);
                        const totalMinutes = hours * 60 + minutes + 1;  // +1分钟
                        const newHours = Math.floor(totalMinutes / 60);
                        const newMinutes = totalMinutes % 60;
                        const newTime = `${date} ${String(newHours).padStart(2, '0')}:${String(newMinutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
                        IntradayChart.setVirtualTime(newTime);
                        return newTime;
                    }
                },
                getUpdateTimeRange: function(lastRequestTime, currentTime) {
                    // 模拟模式：使用虚拟时间
                    return {
                        start: lastRequestTime,  // 从上次结束时间开始
                        end: currentTime
                    };
                },
                buildUrl: function(symbol, tickRange) {
                    let url = `/api/v1/intraday/mock?symbol=${encodeURIComponent(symbol)}&trading_phase=${simulationMode}`;

                    // 🔧 传递last_price（用于保证价格连续性）
                    const intradayData = IntradayChart.getData();
                    if (intradayData && intradayData.current_price) {
                        url += `&last_price=${intradayData.current_price}`;
                    }

                    if (tickRange) {
                        url += `&tick_range=${encodeURIComponent(JSON.stringify(tickRange))}`;
                    }
                    
                    return url;
                },
                shouldGenerateTickRange: simulationMode === 'trading', // 属性而不是方法
                shouldRecordFullTimestamp: false
            };
        case 'real':
        default:
            return {
                pollInterval: 1000,
                setupInitialState: function() {
                    // 真实模式下不需要特殊初始化
                },
                getCurrentTime: function(isInitial) {
                    // 📊 真实模式：使用系统实际时间
                    return Math.floor(Date.now() / 1000);  // 当前时间（秒）
                },
                getUpdateTimeRange: function(lastRequestTime, currentTime) {
                    // 🔧 真实模式：使用市场时间，而不是浏览器时间
                    // 应该从后端API获取准确的市场时间，暂时使用服务器时间
                    const marketDateTimeStr = AppUtils.formatToMarketDateTimeStr(new Date());
                    const dateStr = AppUtils.extractFromDateStr(marketDateTimeStr);  // YYYY-MM-DD
                    const timeStr = AppUtils.extractFromTimeStr(marketDateTimeStr); // HH:MM:SS
                    const newEndTime = `${dateStr} ${timeStr}`;

                    // 如果有lastRequestTime，使用它；否则使用开盘时间
                    let newStartTime;
                    if (lastRequestTime) {
                        newStartTime = lastRequestTime;
                    } else {
                        // 首次增量更新，从开盘到现在
                        // 根据当前市场代码获取正确的开盘时间
                        const currentMarketCode = window.currentMarketCode || 'CN'; // 默认为中国市场
                        const marketConfig = window.marketConfig || {};
                        const marketInfo = marketConfig[currentMarketCode];
                        
                        if (marketInfo && marketInfo.trading_hours) {
                            // 解析交易时间字符串，获取开盘时间
                            const tradingHours = AppUtils.parseTradingHoursString(marketInfo.trading_hours);
                            if (tradingHours && tradingHours.open) {
                                newStartTime = `${dateStr} ${tradingHours.open}`;
                            } else {
                                // 如果解析失败，抛出异常
                                console.error(`❌解析市场 ${currentMarketCode} 交易时间失败，无法获取开盘时间`);
                                throw new Error(`解析市场 ${currentMarketCode} 交易时间失败，无法获取开盘时间`);
                            }
                        } else {
                            // 如果没有市场配置，抛出异常
                            console.error(`❌无法获取市场 ${currentMarketCode} 的配置信息`);
                            throw new Error(`无法获取市场 ${currentMarketCode} 的配置信息`);
                        }
                    }
                    
                    return {
                        start: newStartTime,
                        end: newEndTime
                    };
                },
                buildUrl: function(symbol, tickRange) {
                    let url = `/api/v1/intraday/data?symbol=${encodeURIComponent(symbol)}`;

                    if (tickRange) {
                        url += `&tick_range=${encodeURIComponent(JSON.stringify(tickRange))}`;
                    }
                    
                    return url;
                },
                shouldGenerateTickRange: !isInitial && lastRequestTime, // 属性而不是方法
                shouldRecordFullTimestamp: true
            };
    }
}

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
    showLoading: showIntradayLoading,
    
    // 分时图相关功能
    generateTimeAxisLabelConfig: generateTimeAxisLabelConfig,
    showChartLoading: showChartLoading,
    showIntradayError: showIntradayError,
    generateTradingTimes: generateTradingTimes,
    initializeIntradayTimeAxis: initializeIntradayTimeAxis,
    getModeConfig: getModeConfig
    // 🔧 删除：isTradingTime - 前端不再提供交易时段判断功能
}
