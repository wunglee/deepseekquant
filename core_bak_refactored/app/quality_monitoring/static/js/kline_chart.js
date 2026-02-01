/**
 * K线图模块 - 独立的K线图管理
 * 职责：管理K线图的渲染、数据加载、实时更新
 */

// ==================== KlineChart 模块对象 ====================
// 使用块级作用域避免 IIFE，支持更好的调试体验
{
    // ==================== 私有状态 ====================
    let kline_chart = null
    let indicator_chart = null
    let dataZoomChart = null
    let allKlineData = []  // 所有K线数据
    let allEvents = []      // 所有事件数据
    let allIndicatorsData = {}  // 所有技术指标数据（后端API计算）
    let isLoadingNewStock = false  // 标记是否正在加载新股票（需要复位 dataZoom）
    let realtimeKlineTimer = null
    let currentRealtimeKline = null
    let mock_trading_phase = 'trading'  // 模拟控制：before_open, trading, after_close外部传入，
    let current_period = 'daily' // 当前周期（内部状态）
    let current_indicator = 'VOL' // 当前指标（内部状态）
    let current_market_code = 'CN'// 当前市场（内部状态）
    let current_zoom = null// 当前缩放（内部状态）
    let current_symbol = null// 当前股票（内部状态）
    // 无限滚动相关状态
    let isLoadingMore = false  // 加载状态标志
    let hasMoreData = true      // 是否还有更多数据
    let lastLoadPosition = -1   // 上次触发加载的位置（避免重复触发）
    let lastStartValue = -1     // 修复：初始值设为-1（无效值），第一次获取到真实值后才开始比较
    let userIsMoving = false    // 用户是否正在拖动
    let movingResetTimer = null // 拖动重置定时器
    let isAdjustingBySystem = false // 标记系统是否正在自动调整（避免误判为用户拖动）
    let infiniteScrollEnabled = false // 修复：标记是否启用无限滚动（防止首次加载时误触发）
    let initialLoadComplete = false  // 标记初始加载是否完成
    let use_mock_mode=false

    // ==================== 私有函数 ====================
    function getCharts() {
        return {
            kline: kline_chart,
            indicator: indicator_chart,
            dataZoom: dataZoomChart
        };
    }
    // ==================== 辅助函数 ====================

    function selectPeriod(period, element) {
        const container = document.getElementById('periodSelector');
        container.querySelectorAll('.btn-segment').forEach(b => b.classList.remove('active'));
        element.classList.add('active');
        current_period = period;
        loadData()
    }

    function selectIndicator(indicator, element) {
        const container = document.getElementById('indicatorSelector');
        container.querySelectorAll('.btn-segment').forEach(b => b.classList.remove('active'));
        element.classList.add('active');
        current_indicator = indicator;
        renderIndicator();
    }

    function updateChartsIncremental(newData) {
        // K线图增量更新逻辑
        if (newData && newData.length > 0) {
            // 将新数据追加到现有数据
            allKlineData = allKlineData.concat(newData);
            renderKline();
        }
    }
    
    function generateTradingTimesInternal(start, end, stepSeconds = 5) {
        let tradingTimes = []
    
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
    
    function initializeIntradayTimeAxis(marketCode) {
        // K线图不需要分时图的时间轴初始化
        console.log('K线图无需初始化分时图时间轴');
    }
    
    function getModeConfig(isInitial) {
        if (use_mock_mode) {
            return {
                pollInterval: 3000,
                buildUrl: function(tickRange) {
                    // Mock模式：使用mock端点并传递trading_phase
                    let url = `/api/v1/chart/data/mock?symbol=${encodeURIComponent(current_symbol.id || '')}&period=${current_period}&count=120&indicators=all&trading_phase=${mock_trading_phase}`;
                    return url;
                },
                buildRealtimeUrl: function() {
                    // Mock模式：使用mock端点获取实时K线
                    return `/api/v1/data/kline/realtime/mock?symbol=${encodeURIComponent(current_symbol.id)}&trading_phase=${mock_trading_phase}`;
                },
                buildHistoryUrl: function(beforeDate) {
                    // Mock模式：使用mock端点获取历史数据
                    return `/api/v1/chart/data/mock?symbol=${encodeURIComponent(current_symbol.id)}&period=${current_period}&count=60&before=${beforeDate}&indicators=all&trading_phase=${mock_trading_phase}`;
                },
            };
        } else {
            return {
                pollInterval: 3000,
                buildUrl: function(tickRange) {
                    // 真实模式：使用标准端点
                    let url = `/api/v1/chart/data?symbol=${encodeURIComponent(current_symbol.id || '')}&period=${current_period}&count=120&indicators=all`;
                    return url;
                },
                buildRealtimeUrl: function() {
                    // 真实模式：使用标准端点获取实时K线
                    return `/api/v1/data/kline/realtime?symbol=${encodeURIComponent(current_symbol.id)}&period=${current_period}`;
                },
                buildHistoryUrl: function(beforeDate) {
                    // 真实模式：使用标准端点获取历史数据
                    return `/api/v1/chart/data?symbol=${encodeURIComponent(current_symbol.id)}&period=${current_period}&count=60&before=${beforeDate}&indicators=all`;
                },
            };
        }
    }

    function getFullTradingTimes(marketCode) {
        // 对于K线图，返回空的时间段或者根据实际情况生成
        return { tradingTimes: [], lunchBreakRange: null };
    }

    function makeLunchBreakLine(marketCode) {
        // K线图的午休分割线
        const upperMarketCode = marketCode?.toUpperCase()
        const market = window.marketConfig?.[upperMarketCode] || {};
        const { lunch_start, lunch_end } = market.detailed_trading_hours || {};

        const markLineData = [];
        if (lunch_start && lunch_end) {
            const lunchStartTime = lunch_start + ':00'
            const lunchEndTime = lunch_end + ':00'

            markLineData.push(
                {
                    id: 'lunch-start-line',
                    xAxis: lunchStartTime,
                    label: { show: false },
                    lineStyle: {
                        color: '#000000',
                        type: 'dashed',
                        width: 1
                    }
                },
                {
                    id: 'lunch-end-line',
                    xAxis: lunchEndTime,
                    label: { show: false },
                    lineStyle: {
                        color: '#000000',
                        type: 'dashed',
                        width: 1
                    }
                }
            )
        }

        return markLineData;
    }

    // ==================== 工具函数 ====================

    /**
     * 计算移动平均线
     * @param {Array} data - K线数据
     * @returns {Array} MA数据
     */
    function calcMA(data) {
        const result = []
        for (let i = 0; i < data.length; i++) {
            if (i < current_period - 1) { result.push('-'); continue }
            let sum = 0
            for (let j = i - current_period + 1; j <= i; j++) sum += data[j].close
            result.push((sum / current_period).toFixed(2))
        }
        return result
    }
    function rebuildLayout() {
        const container = document.getElementById('klineContainer')
        if (!container) {
            console.error('❌ 找不到k线图容器')
            return
        }
         container.innerHTML = `
            <div style="display:flex; flex-direction:column;">
                    <!-- 周期切换（仅K线图显示） -->
                    <div id="periodSelector" class="segmented-control" style="margin-bottom:8px;">
                        <button class="btn btn-segment active" onclick="selectPeriod('daily',this)">日</button>
                        <button class="btn btn-segment" onclick="selectPeriod('weekly',this)">周</button>
                        <button class="btn btn-segment" onclick="selectPeriod('monthly',this)">月</button>
                    </div>
                    <!-- 中部：K线（含MA5/MA10/MA20 + 事件标注） -->
                    <div id="klineChart" style="height:420px;"></div>
                    <!-- 技术指标区域 -->
                    <div class="indicator-area" style="margin-top:8px;">
                        <div id="indicatorSelector" class="segmented-control" style="margin-bottom:8px;">
                            <button class="btn btn-segment active" onclick="selectIndicator('VOL',this)">VOL</button>
                            <button class="btn btn-segment" onclick="selectIndicator('MACD',this)">MACD</button>
                            <button class="btn btn-segment" onclick="selectIndicator('RSI',this)">RSI</button>
                            <button class="btn btn-segment" onclick="selectIndicator('KDJ',this)">KDJ</button>
                            <button class="btn btn-segment" onclick="selectIndicator('OBV',this)">OBV</button>
                        </div>
                        <div id="indicatorChart" style="height:200px;"></div>
                    </div>

                    <!-- 底部：数据窗口控制条 -->
                    <div id="dataZoomContainer" style="margin-top:12px; height:60px; position:relative;">
                        <!-- ECharts 的 slider dataZoom 将渲染到这里 -->
                    </div>
            </div>
        `
        const klineDom=window.document.getElementById('klineChart')
        const indicatorDom=window.document.getElementById('indicatorChart')
        const dataZoomDom=window.document.getElementById('dataZoomContainer')
        if (klineDom)
            kline_chart = echarts.init(klineDom)
        if (indicatorDom)
            indicator_chart = echarts.init(indicatorDom)
        if (dataZoomDom)
            dataZoomChart = echarts.init(dataZoomDom)

        // 连接三个图表，确保 tooltip 同步
        if (kline_chart && indicator_chart && dataZoomChart) {
            echarts.connect([kline_chart, indicator_chart, dataZoomChart])
        }
        setTimeout(() => {
            const charts = getCharts()
            if (charts.kline)
                charts.kline.resize()
            if (charts.indicator)
                charts.indicator.resize()
            if (charts.dataZoom)
                charts.dataZoom.resize()
        }, 50)
        // 显示K线相关元素
        document.getElementById('klineContainer').style.display = 'block'
        document.getElementById('intradayContainer').style.display = 'none'
    }
    // ==================== 图表渲染函数 ====================

    /**
     * 渲染技术指标图
     */
    function renderIndicator() {
        console.log('renderIndicator called with allKlineData:', allKlineData, 'current_zoom:', current_zoom, 'current_indicator:', current_indicator)
        console.log('📊 renderIndicator - allKlineData数量:', allKlineData ? allKlineData.length : 0)
        if (!allKlineData || !allKlineData.length) {
            showEmpty()
            return
        }
        console.log('📊 renderIndicator - processedData数量:', allIndicatorsData.length)
        if (indicator_chart) {
            // 🔧 优先使用传入的 current_zoom 参数（确保与K线图完全对齐）
            if (current_zoom) {
                // 使用传入的 zoom 参数（来自 renderKline）
                console.log('📊 技术指标使用current_zoom位置:', current_zoom)
            } else {
                // 降级：尝试从 K 线图获取（用于其他场景，如切换指标）
                current_zoom = { start: 75, end: 100 }  // 默认值
                try {
                    if (kline_chart) {
                        const klineOption = kline_chart.getOption()
                        if (klineOption && klineOption.dataZoom && klineOption.dataZoom[0]) {
                            current_zoom = {
                                start: klineOption.dataZoom[0].start || 75,
                                end: klineOption.dataZoom[0].end || 100
                            }
                            console.log('📊 技术指标从 K 线图读取 dataZoom 位置:', current_zoom)
                        }
                    }
                } catch(e) {
                    console.warn('无法获取 K 线图 dataZoom，使用默认值')
                }
            }
            indicator_chart.clear()
            const indicatorOption = getIndicatorOption()
            console.log('Setting indicator_chart option:', indicatorOption)
            indicator_chart.setOption(indicatorOption, true)  // 🔧 使用 true 强制清除旧配置
        }
    }
    function getIndicatorOption() {
        let indicator_option

        if (current_indicator === 'VOL') {
            const displayKlineData = toDisplayKlineData()
            indicator_option = {
                title: { text: '成交量（Volume）', left: 'center', textStyle: { fontSize: 12 } },
                tooltip: {
                    trigger: 'axis',
                    formatter: function(params) {
                        if (!params || params.length === 0) return ''
                        const volume = params[0].value
                        if (volume === null || volume === undefined) return params[0].axisValue
                        // 格式化成交量（亿手、万手等）
                        let displayValue = volume
                        let unit = '手'
                        if (volume >= 100000000) {
                            displayValue = (volume / 100000000).toFixed(2)
                            unit = '亿手'
                        } else if (volume >= 10000) {
                            displayValue = (volume / 10000).toFixed(2)
                            unit = '万手'
                        } else {
                            displayValue = volume.toFixed(0)
                        }
                        return params[0].axisValue + '<br/>' + params[0].marker + '成交量: ' + displayValue + unit
                    }
                },
                grid: { left: '8%', right: '8%', top: '15%', bottom: '15%' },
                xAxis: {
                    type: 'category',
                    data: displayKlineData.map(d => d.date),
                    axisLabel: {
                        show: true,
                        margin: 12
                    }
                },
                yAxis: {
                    type: 'value',
                    axisLabel: {
                        formatter: function(value) {
                            if (value >= 100000000) return (value / 100000000).toFixed(1) + '亿'
                            if (value >= 10000) return (value / 10000).toFixed(1) + '万'
                            return value.toFixed(0)
                        }
                    }
                },
                dataZoom: [
                    {
                        type: 'inside',
                        start: current_zoom.start,  // 🔧 同步 K 线图的位置
                        end: current_zoom.end,
                        zoomOnMouseWheel: true,
                        moveOnMouseMove: true,
                        moveOnMouseWheel: true,
                        throttle: 50
                    }
                ],
                series: [{
                    type: 'bar',
                    data: displayKlineData.map(d => d.volume),
                    itemStyle: {
                        color: (params) => {
                            // 根据当天涨跌上色（红涨绿跌）
                            const idx = params.dataIndex
                            if (idx >= displayKlineData.length) return '#64748b'
                            const data = displayKlineData[idx]
                            if (!data || data.close === null || data.open === null) return '#64748b'
                            return data.close >= data.open ? '#ef4444' : '#10b981'
                        }
                    },
                    barWidth: '60%'
                }]
            }
        } else if (current_indicator === 'MACD') {
            // 🔧 使用后端计算的MACD数据（中国标准：柱状图×2）
            console.log('🗒️ 切换到MACD指标，当前 allIndicatorsData:', allIndicatorsData)
            console.log('🗒️ allIndicatorsData 的键:', Object.keys(allIndicatorsData))

            const macdData = allIndicatorsData.macd || []
            console.log('🎯 MACD数据:', macdData.length, '条记录', macdData.slice(0, 3))

            if (!macdData || macdData.length === 0) {
                console.error('⚠️ MACD数据为空！')
                console.error('   - allIndicatorsData 对象:', allIndicatorsData)
                console.error('   - allIndicatorsData.macd:', allIndicatorsData.macd)
                console.error('   - 可能原因: 后端API未返回 macd 数据或字段名不匹配')
                indicator_option = {
                    title: { text: 'MACD（数据为空）', left: 'center', textStyle: { fontSize: 12, color: '#ef4444' } },
                    grid: { left: '8%', right: '8%', top: '15%', bottom: '15%' },
                    xAxis: { type: 'category', data: [] },
                    yAxis: { type: 'value' },
                    series: []
                }
            } else {
                const dates = macdData.map(d => d.date)
                const diff = macdData.map(d => d.macd)      // DIFF = EMA12 - EMA26
                const dea = macdData.map(d => d.signal)     // DEA = EMA(DIFF, 9)
                const macdBar = macdData.map(d => d.histogram)  // MACD柱 = (DIFF - DEA) × 2

                console.log('📊 MACD解析数据 - DIFF:', diff.filter(v => v !== null).length, 'DEA:', dea.filter(v => v !== null).length, 'MACD柱:', macdBar.filter(v => v !== null).length)

                indicator_option = {
                title: {
                    text: 'MACD（指数平滑异同移动平均线）',
                    left: 'center',
                    textStyle: { fontSize: 12 },
                    subtext: 'DIFF(12,26,9) DEA(9) MACD柱=(DIFF-DEA)×2',
                    subtextStyle: { fontSize: 10, color: '#999' }
                },
                tooltip: {
                    trigger: 'axis',
                    formatter: function(params) {
                        let result = params[0].axisValue + '<br/>'
                        params.forEach(item => {
                            if (item.value !== null && item.value !== undefined) {
                                const value = item.value.toFixed(4)
                                if (item.seriesName === 'DIFF') {
                                    result += item.marker + '<span style="color:#2563eb">DIFF: ' + value + '</span><br/>'
                                } else if (item.seriesName === 'DEA') {
                                    result += item.marker + '<span style="color:#ef4444">DEA: ' + value + '</span><br/>'
                                } else if (item.seriesName === 'MACD柱') {
                                    result += item.marker + 'MACD柱: ' + value + '<br/>'
                                }
                            }
                        })
                        return result
                    }
                },
                legend: { data: ['DIFF', 'DEA', 'MACD柱'], top: '5%', right: '8%' },
                grid: { left: '8%', right: '8%', top: '20%', bottom: '15%' },
                xAxis: {
                    type: 'category',
                    data: dates,
                    axisLabel: {
                        show: true,
                        margin: 12  // 增加横坐标与图表的间距
                    }
                },
                yAxis: {
                    type: 'value',
                    scale: true,  // 🔧 启用自动缩放
                    axisLabel: {
                        formatter: (value) => value.toFixed(3)
                    },
                    splitLine: {  // 🔧 添加0轴参考线
                        show: true,
                        lineStyle: {
                            color: '#e5e7eb',
                            type: 'dashed'
                        }
                    }
                },
                dataZoom: [
                    {
                        type: 'inside',
                        start: current_zoom.start,  // 🔧 同步 K 线图的位置
                        end: current_zoom.end,
                        zoomOnMouseWheel: true,
                        moveOnMouseMove: true,
                        moveOnMouseWheel: true,
                        throttle: 50
                    }
                ],
                series: [
                    {
                        name: 'DIFF',
                        type: 'line',
                        data: diff,
                        smooth: false,  // 🔧 不平滑，保持真实
                        symbol: 'none',
                        lineStyle: { width: 2 },
                        itemStyle: { color: '#2563eb' },
                        z: 10  // 置于柱状图上方
                    },
                    {
                        name: 'DEA',
                        type: 'line',
                        data: dea,
                        smooth: false,
                        symbol: 'none',
                        lineStyle: { width: 2 },
                        itemStyle: { color: '#ef4444' },
                        z: 10
                    },
                    {
                        name: 'MACD柱',
                        type: 'bar',
                        data: macdBar,
                        barWidth: '60%',
                        itemStyle: {
                            color: (params) => {
                                if (params.value === null || params.value === undefined) return '#cccccc'
                                // 🔧 MACD柱中国标准：红柱(多头)、绿柱(空头)
                                return params.value >= 0 ? '#ef4444' : '#10b981'
                            },
                            borderColor: (params) => {
                                if (params.value === null || params.value === undefined) return '#cccccc'
                                return params.value >= 0 ? '#dc2626' : '#059669'
                            },
                            borderWidth: 1
                        },
                        emphasis: {
                            itemStyle: {
                                opacity: 0.9,
                                shadowBlur: 10,
                                shadowColor: 'rgba(0,0,0,0.3)'
                            }
                        },
                        z: 5  // 置于线条下方
                    },
                    // 🔧 添加0轴线
                    {
                        type: 'line',
                        data: Array(dates.length).fill(0),
                        symbol: 'none',
                        lineStyle: {
                            color: '#6b7280',
                            width: 1,
                            type: 'solid'
                        },
                        silent: true,
                        z: 1
                    }
                ]}
            }
        } else if (current_indicator === 'RSI') {
            // 🔧 使用后端计算的RSI数据（更专业）
            const rsiData = allIndicatorsData.rsi || []
            const dates = rsiData.map(d => d.date)
            const rsi = rsiData.map(d => d.value)

            indicator_option = {
                title: { text: 'RSI（相对强弱指标）', left: 'center', textStyle: { fontSize: 12 } },
                tooltip: {
                    trigger: 'axis',
                    formatter: function(params) {
                        let result = params[0].axisValue + '<br/>'
                        const rsiItem = params.find(p => p.seriesName === 'RSI')
                        if (rsiItem && rsiItem.value !== null) {
                            result += rsiItem.marker + 'RSI: ' + rsiItem.value.toFixed(2) + '<br/>'
                            if (rsiItem.value >= 70) {
                                result += '<span style="color:#ef4444">● 超买区（可能回调）</span><br/>'
                            } else if (rsiItem.value <= 30) {
                                result += '<span style="color:#10b981">● 超卖区（可能反弹）</span><br/>'
                            }
                        }
                        return result
                    }
                },
                legend: { data: ['RSI', '超买线(70)', '超卖线(30)'], top: '5%', right: '8%' },
                grid: { left: '8%', right: '8%', top: '15%', bottom: '15%' },
                xAxis: {
                    type: 'category',
                    data: dates,
                    axisLabel: {
                        show: true,
                        margin: 12
                    }
                },
                yAxis: {
                    type: 'value',
                    min: 0,
                    max: 100,
                    interval: 20,
                    axisLabel: {
                        formatter: (value) => value.toFixed(0)
                    }
                },
                dataZoom: [
                    {
                        type: 'inside',
                        start: current_zoom.start,  // 🔧 同步 K 线图的位置
                        end: current_zoom.end,
                        zoomOnMouseWheel: true,
                        moveOnMouseMove: true,
                        moveOnMouseWheel: true,
                        throttle: 50
                    }
                ],
                series: [
                    {
                        name: 'RSI',
                        type: 'line',
                        data: rsi,
                        smooth: false,  // 🔧 不平滑
                        symbol: 'none',
                        lineStyle: { width: 2 },
                        itemStyle: { color: '#8b5cf6' },
                        areaStyle: {  // 🔧 添加渐变填充，更美观
                            color: {
                                type: 'linear',
                                x: 0, y: 0, x2: 0, y2: 1,
                                colorStops: [
                                    { offset: 0, color: 'rgba(139, 92, 246, 0.3)' },
                                    { offset: 1, color: 'rgba(139, 92, 246, 0.05)' }
                                ]
                            }
                        }
                    },
                    {
                        name: '超买线(70)',
                        type: 'line',
                        data: Array(dates.length).fill(70),
                        lineStyle: { type: 'dashed', color: '#ef4444', width: 1, opacity: 0.6 },
                        symbol: 'none',
                        silent: true
                    },
                    {
                        name: '超卖线(30)',
                        type: 'line',
                        data: Array(dates.length).fill(30),
                        lineStyle: { type: 'dashed', color: '#10b981', width: 1, opacity: 0.6 },
                        symbol: 'none',
                        silent: true
                    }
                ]
            }
        } else if (current_indicator === 'KDJ') {
            // 🔧 使用后端计算的KDJ数据（更专业）
            const kdjData = allIndicatorsData.kdj || []
            const dates = kdjData.map(d => d.date)
            const k = kdjData.map(d => d.k)
            const d = kdjData.map(d => d.d)
            const j = kdjData.map(d => d.j)

            // ⚠️ 计算实际数据范围，确保J线超界时也能正常显示
            const allValues = [...k, ...d, ...j].filter(v => v !== null && v !== undefined && !isNaN(v))
            const minVal = Math.min(...allValues, 0)  // 至少包含0
            const maxVal = Math.max(...allValues, 100)  // 至少包含100
            const yMin = Math.floor(minVal / 10) * 10  // 向下取整到10的倍数
            const yMax = Math.ceil(maxVal / 10) * 10   // 向上取整到10的倍数

            indicator_option = {
                title: {
                    text: 'KDJ（随机指标）',
                    left: 'center',
                    textStyle: { fontSize: 12 },
                    subtext: 'J线可超出0-100区间（技术特征）',
                    subtextStyle: { fontSize: 10, color: '#999' }
                },
                tooltip: {
                    trigger: 'axis',
                    formatter: function(params) {
                        let result = params[0].axisValue + '<br/>'
                        params.forEach(item => {
                            if (item.seriesName && ['K', 'D', 'J'].includes(item.seriesName)) {
                                const value = item.value === null || item.value === undefined ? 'N/A' : item.value.toFixed(2)
                                result += item.marker + item.seriesName + ': ' + value + '<br/>'
                            }
                        })
                        return result
                    }
                },
                legend: { data: ['K', 'D', 'J', '超买区', '超卖区'], top: '5%', right: '8%' },
                grid: { left: '8%', right: '8%', top: '20%', bottom: '15%' },
                xAxis: {
                    type: 'category',
                    data: dates,
                    axisLabel: {
                        show: true,
                        margin: 12
                    }
                },
                yAxis: {
                    type: 'value',
                    min: yMin,  // 🔧 动态范围，允许J线超界
                    max: yMax,
                    interval: 20,  // 每20个单位一个刻度
                    axisLabel: {
                        formatter: (value) => value.toFixed(0)
                    }
                },
                dataZoom: [
                    {
                        type: 'inside',
                        start: current_zoom.start,  // 🔧 同步 K 线图的位置
                        end: current_zoom.end,
                        zoomOnMouseWheel: true,
                        moveOnMouseMove: true,
                        moveOnMouseWheel: true,
                        throttle: 50
                    }
                ],
                series: [
                    {
                        name: 'K',
                        type: 'line',
                        data: k,
                        smooth: false,  // 🔧 不平滑，保持真实
                        symbol: 'none',
                        lineStyle: { width: 2 },
                        itemStyle: { color: '#2563eb' }
                    },
                    {
                        name: 'D',
                        type: 'line',
                        data: d,
                        smooth: false,
                        symbol: 'none',
                        lineStyle: { width: 2 },
                        itemStyle: { color: '#ef4444' }
                    },
                    {
                        name: 'J',
                        type: 'line',
                        data: j,
                        smooth: false,
                        symbol: 'none',
                        lineStyle: { width: 1.5, type: 'dashed' },  // 🔧 虚线，区分J线
                        itemStyle: { color: '#8b5cf6' }
                    },
                    {
                        name: '超买区',
                        type: 'line',
                        data: Array(dates.length).fill(80),
                        lineStyle: { type: 'dashed', color: '#ef4444', width: 1, opacity: 0.5 },
                        symbol: 'none',
                        silent: true  // 不响应鼠标事件
                    },
                    {
                        name: '超卖区',
                        type: 'line',
                        data: Array(dates.length).fill(20),
                        lineStyle: { type: 'dashed', color: '#10b981', width: 1, opacity: 0.5 },
                        symbol: 'none',
                        silent: true
                    }
                ]
            }
        } else if (current_indicator === 'OBV') {
            // 🔧 使用后端计算的OBV数据（更专业）
            const obvData = allIndicatorsData.obv || []
            const dates = obvData.map(d => d.date)
            const obv = obvData.map(d => d.value)

            indicator_option = {
                title: {
                    text: 'OBV（能量潮指标）',
                    left: 'center',
                    textStyle: { fontSize: 12 },
                    subtext: '累积方向性成交量',
                    subtextStyle: { fontSize: 10, color: '#999' }
                },
                tooltip: {
                    trigger: 'axis',
                    formatter: function(params) {
                        if (!params || params.length === 0) return ''
                        const value = params[0].value
                        if (value === null || value === undefined) return params[0].axisValue
                        // 格式化OBV值
                        let displayValue = value
                        let unit = ''
                        if (Math.abs(value) >= 100000000) {
                            displayValue = (value / 100000000).toFixed(2)
                            unit = '亿'
                        } else if (Math.abs(value) >= 10000) {
                            displayValue = (value / 10000).toFixed(2)
                            unit = '万'
                        } else {
                            displayValue = value.toFixed(0)
                        }
                        return params[0].axisValue + '<br/>' + params[0].marker + 'OBV: ' + displayValue + unit
                    }
                },
                grid: { left: '8%', right: '8%', top: '18%', bottom: '15%' },
                xAxis: {
                    type: 'category',
                    data: dates,
                    axisLabel: {
                        show: true,
                        margin: 12
                    }
                },
                yAxis: {
                    type: 'value',
                    axisLabel: {
                        formatter: function(value) {
                            if (Math.abs(value) >= 100000000) return (value / 100000000).toFixed(1) + '亿'
                            if (Math.abs(value) >= 10000) return (value / 10000).toFixed(1) + '万'
                            return value.toFixed(0)
                        }
                    }
                },
                dataZoom: [
                    {
                        type: 'inside',
                        start: current_zoom.start,  // 🔧 同步 K 线图的位置
                        end: current_zoom.end,
                        zoomOnMouseWheel: true,
                        moveOnMouseMove: true,
                        moveOnMouseWheel: true,
                        throttle: 50
                    }
                ],
                series: [{
                    type: 'line',
                    data: obv,
                    smooth: false,  // 🔧 不平滑
                    symbol: 'none',
                    lineStyle: { width: 2 },
                    itemStyle: { color: '#f59e0b' },
                    areaStyle: {  // 🔧 添加填充，显示累积趋势
                        color: {
                            type: 'linear',
                            x: 0, y: 0, x2: 0, y2: 1,
                            colorStops: [
                                { offset: 0, color: 'rgba(245, 158, 11, 0.3)' },
                                { offset: 1, color: 'rgba(245, 158, 11, 0.05)' }
                            ]
                        }
                    }
                }]
            }
        } else {
            indicator_option = {
                title: { text: current_indicator, left: 'center', textStyle: { fontSize: 12 } },
                tooltip: { trigger: 'axis' },
                grid: { left: '8%', right: '8%', top: '15%', bottom: '8%' },
                xAxis: { type:'category', data: allIndicatorsData.map(d=>d.date) },
                yAxis: { type:'value' },
                dataZoom: [
                    {
                        type: 'inside',
                        // 🔧 移除 start/end，让 echarts.connect 自动同步
                        zoomOnMouseWheel: true,
                        moveOnMouseMove: true,
                        moveOnMouseWheel: true,
                        throttle: 50
                    }
                ],
                series: [{ type:'line', data: allIndicatorsData.map(d=>d.close), smooth:true, itemStyle:{ color:'#94a3b8' } }]
            }
        }
        return indicator_option
    }
    /**
     * 渲染数据窗口控制条（统一放在页面底部）
     */
    function renderDataZoom() {
        const displayKlineData=toDisplayKlineData()
        const dates = displayKlineData.map(d => d.date)
        console.log('Processed dates:', dates)
        const option = {
            grid: {
                left: '8%',
                right: '8%',
                top: '0%',
                bottom: '0%',
                height: '100%'
            },
            xAxis: {
                type: 'category',
                data: dates,
                show: false  // 隐藏坐标轴，只显示 dataZoom 滑块
            },
            yAxis: {
                type: 'value',
                show: false
            },
            series: [
                {
                    type: 'line',
                    data: [],  // 空数据，只用于 dataZoom 的基础
                    show: false
                }
            ],
            dataZoom: [
                {
                    type: 'slider',
                    start: current_zoom.start,
                    end: current_zoom.end,
                    top: '50%',  // 垂直居中
                    height: '35px',
                    zoomLock: false,
                    realtime: true,
                    brushSelect: false,
                    handleIcon: 'path://M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
                    handleSize: '80%',
                    handleStyle: {
                        color: '#fff',
                        shadowBlur: 3,
                        shadowColor: 'rgba(0, 0, 0, 0.6)',
                        shadowOffsetX: 2,
                        shadowOffsetY: 2
                    },
                    textStyle: {
                        color: '#333'
                    },
                    borderColor: '#e5e7eb',
                    fillerColor: 'rgba(37, 99, 235, 0.2)',  // 选中区域颜色
                    dataBackground: {
                        lineStyle: {
                            color: '#cbd5e1'
                        },
                        areaStyle: {
                            color: '#f1f5f9'
                        }
                    }
                }
            ]
        }

        if (dataZoomChart) {
            dataZoomChart.setOption(option, false)
        }
        console.log('🎯 数据窗口控制条已渲染，start:', current_zoom.start, 'end:', current_zoom.end)
    }
    function toDisplayKlineData() {
            // 处理日期格式：从后端获取的时间字符串需要还原为Date
        const displayKlineData = allKlineData.map(d => {
            let dateStr = d.date
            if (typeof dateStr === 'string') {
                // 使用 AppUtils.extractFromMarketDateTimeStr 正确解析时间字符串为市场时区Date对象
                const marketTimezone = AppUtils.getMarketTimezone(current_market_code);
                const marketDate = AppUtils.extractFromMarketDateTimeStr(dateStr, marketTimezone)
                // 使用 AppUtils.formatToMarketDateTimeStr 格式化为标准日期字符串
                dateStr = AppUtils.extractFromDateStr(AppUtils.formatToMarketDateTimeStr(marketDate, marketTimezone), marketTimezone)
            }
            return { ...d, date: dateStr }
        })
        return displayKlineData
    }
    /**
     * 渲染K线图
     */
    function renderKline() {
        console.log('renderKline called with allKlineData:', allKlineData, 'allEvents:', allEvents)
        if (!allKlineData || !allKlineData.length) {
            showEmpty()
            return
        }

        // 💚 关键修复: 先保存 dataZoom 位置,再清空图表
        // 🔧 保存当前 dataZoom 位置（避免重新渲染时复位）
        // ⚠️ 关键：如果是加载新股票，使用默认位置；如果是无限滚动，保留当前位置

        if (!isLoadingNewStock) {
            // 仅在非新股票加载时才保留位置（如无限滚动、切换指标等）
            try {
                if (kline_chart) {
                    const currentOption = kline_chart.getOption()
                    if (currentOption && currentOption.dataZoom && currentOption.dataZoom[0]) {
                        current_zoom = {
                            start: currentOption.dataZoom[0].start || 0,
                            end: currentOption.dataZoom[0].end || 100
                        }
                        console.log('📍 保留当前 dataZoom 位置:', current_zoom)
                    }
                }
            } catch(e) {
                // 首次渲染时 getOption 可能失败，计算默认值
            }
        }

        // 如果没有保留的位置（新股票或首次加载），计算默认显示最近60天
        if (!current_zoom) {
            // 🔧 计算显示最近60天的dataZoom范围
            const totalDays = allKlineData.length
            const displayDays = 60  // 显示最近60天

            if (totalDays <= displayDays) {
                // 数据不足60天，显示全部
                current_zoom = { start: 0, end: 100 }
            } else {
                // 计算百分比：显示最后60天
                const startPercent = ((totalDays - displayDays) / totalDays) * 100
                current_zoom = { start: startPercent, end: 100 }
            }
            console.log(`🆕 加载新股票，显示最近${displayDays}天，dataZoom:`, current_zoom)
            isLoadingNewStock = false  // 重置标志
        }

        // 💚 现在清空图表(包括"加载中..."提示)
        if (kline_chart) {
            kline_chart.clear()
        }
        console.log('📊 renderKline - displayData数量:', allIndicatorsData.length)
        const option =getKlineOption()
        console.log('Setting kline_chart option:', option)
        // 💚 使用 true 完全替换配置,确保清除所有旧配置(包括graphic)
        if (kline_chart) {
            kline_chart.setOption(option, true)
        }
        renderIndicator()
        renderDataZoom()  // 🔧 渲染数据窗口控制条
    }
    function getKlineOption() {
        const displayKlineData = toDisplayKlineData()
        const ohlc = displayKlineData.map(d => [d.open, d.close, d.low, d.high])
        console.log('OHLC data:', ohlc)
        const dates = displayKlineData.map(d => d.date)
        console.log('Processed dates:', dates)
        const option ={
            title: { text: `${current_symbol.name || ''} K线图`, left: 'center', textStyle: { fontSize: 14 } },
            tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
            legend: { data: ['K线', 'MA5', 'MA10', 'MA20'], bottom: 0 },
            grid: { left: '8%', right: '8%', top: '15%', bottom: '12%' },
            xAxis: { type: 'category', data: dates, boundaryGap: true },
            yAxis: { scale: true },
            dataZoom: [
                {
                    type: 'inside',
                    start: current_zoom.start,  // 🔧 使用保留的位置
                    end: current_zoom.end,
                    zoomOnMouseWheel: true,
                    moveOnMouseMove: true,
                    moveOnMouseWheel: true,
                    throttle: 50
                }
                // 🔧 移除 slider 类型的 dataZoom，改为统一放在页面底部
            ],
            series: [
                {
                    name:'K线',
                    type:'candlestick',
                    data: ohlc,
                    itemStyle:{
                        color:'#ef4444',
                        color0:'#10b981',
                        borderColor:'#ef4444',
                        borderColor0:'#10b981'
                    }
                    // 🔧 暂时隐藏事件标记，让图表更清晰
                    // markPoint: {
                    //     data: (allEvents||[]).map(e => {
                    //         let evtDate = e.date
                    //         if (typeof evtDate === 'string' && evtDate.includes('GMT')) {
                    //             evtDate = new Date(evtDate).toISOString().split('T')[0]
                    //         }
                    //         return {
                    //             name: e.title,
                    //             xAxis: evtDate,
                    //             yAxis: e.price || 0,
                    //             symbolSize: e.severity==='critical'?60:40,
                    //             itemStyle: { color: e.impact === 'negative' ? '#ef4444' : '#10b981' }
                    //         }
                    //     })
                    // }
                },
                {
                    name:'MA5',
                    type:'line',
                    data: calcMA(allIndicatorsData,5),
                    smooth:true,
                    symbol: 'none',  // 🔧 移除圆点
                    lineStyle:{ opacity:0.6, color:'#f59e0b', width: 1.5 }
                },
                {
                    name:'MA10',
                    type:'line',
                    data: calcMA(allIndicatorsData,10),
                    smooth:true,
                    symbol: 'none',  // 🔧 移除圆点
                    lineStyle:{ opacity:0.6, color:'#6366f1', width: 1.5 }
                },
                {
                    name:'MA20',
                    type:'line',
                    data: calcMA(allIndicatorsData,20),
                    smooth:true,
                    symbol: 'none',  // 🔧 移除圆点
                    lineStyle:{ opacity:0.6, color:'#22c55e', width: 1.5 }
                }
            ]
         }
         return option
    }

    // ==================== 数据更新函数 ====================

    /**
     * 加载K线数据（主要数据加载函数）
     */
    function loadData() {
        console.log('🔍 开始加载K线数据:', { current_symbol, use_mock_mode })
        // 标记为加载新股票（会重置 dataZoom）
        isLoadingNewStock = true

        // 🔧 使用getModeConfig构建API URL
        const modeConfig = getModeConfig(true);
        const url = modeConfig.buildUrl();

        console.log('📡 请求URL:', url)

        // 显示加载状态
        showLoading(true)

        fetch(url)
            .then(response => {
                if (!response.ok) {
                    console.error(`❌ HTTP ${response.status}: ${response.statusText}`);
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json()
            })
            .then(result => {
                console.log('📥 收到API响应:', result)

                // 隐藏加载状态
                showLoading(false)

                if (result.status !== 'success') {
                    console.error('❌ API返回错误:', result.message)
                    showEmpty(result.message || '数据加载失败')
                    return
                }

                const chartData = result.data || {}
                allKlineData = chartData.kline || []
                allEvents = chartData.events || []
                allIndicatorsData = chartData.indicators || {}

                console.log(`📊 加载完成: ${allKlineData.length}条K线数据, ${allEvents.length}条事件, 指标类型:`, Object.keys(allIndicatorsData))

                // 渲染图表
                if (typeof renderKline === 'function') {
                    renderKline()

                    // 标记初始加载完成（启用无限滚动）
                    initialLoadComplete = true
                    infiniteScrollEnabled = true
                    console.log('✅ 初始加载完成，启用无限滚动')
                }
                // 启动实时K线轮询
                startInfiniteScrollDetection()
                // 启动实时K线轮询
                startRealtimeKlineUpdateTimer()
            })
            .catch(error => {
                console.error('❌ 加载K线数据失败:', error)
                showEmpty('数据加载失败，请稍后重试')
            })
    }

    /**
     * 仅更新图表数据（不重新渲染，避免视觉跳动）
     * @param {Array} data - K线数据
     */
    function updateChartData() {
        if (!kline_chart) {
            console.error('kline_chart 未初始化')
            return
        }

        try {
            const displayEvents = allEvents || []
            const displayKlineData = toDisplayKlineData()
            const ohlc = displayKlineData.map(d => [d.open, d.close, d.low, d.high])
            const dates = displayKlineData.map(d => d.date)

            // 🔧 关键优化：使用 lazyUpdate: true 延迟更新，不立即重绘

            kline_chart.setOption({
                xAxis: { data: dates },
                series: [
                    {
                        name: 'K线',
                        data: ohlc
                        // 🔧 暂时禁用事件标注气球
                        // markPoint: {
                        //     data: displayEvents.map(e => {
                        //         let evtDate = e.date
                        //         if (typeof evtDate === 'string' && evtDate.includes('GMT')) {
                        //             evtDate = new Date(evtDate).toISOString().split('T')[0]
                        //         }
                        //         return {
                        //             name: e.title,
                        //             xAxis: evtDate,
                        //             yAxis: e.price || 0,
                        //             symbolSize: e.severity === 'critical' ? 60 : 40,
                        //             itemStyle: { color: e.impact === 'negative' ? '#ef4444' : '#10b981' }
                        //         }
                        //     })
                        // }
                    },
                    { name: 'MA5', data: calcMA(allIndicatorsData, 5) },
                    { name: 'MA10', data: calcMA(allIndicatorsData, 10) },
                    { name: 'MA20', data: calcMA(allIndicatorsData, 20) }
                ]
            }, { notMerge: false, lazyUpdate: true })  // 🔧 lazyUpdate: true 延迟更新

            // 同步更新指标图
            updateIndicatorData()

            // 🔧 同步更新数据窗口控制条的 xAxis 数据
            if (dataZoomChart) {
                dataZoomChart.setOption({
                    xAxis: { data: dates }
                }, { notMerge: false, lazyUpdate: true })
            }

            console.log('🔄 已更新图表数据（延迟重绘），总计', allIndicatorsData.length, '条')
        } catch(e) {
            console.error('更新图表数据失败:', e)
        }
    }

    /**
     * 仅更新指标图数据
     */
    function updateIndicatorData() {
        if (!indicator_chart) {
            console.error('indicator_chart 未初始化')
            return
        }
        const displayKlineData = toDisplayKlineData()
        try {
            const dates = displayKlineData.map(d => d.date)

            if (current_indicator === 'VOL') {
                indicator_chart.setOption({
                    xAxis: { data: dates },
                    series: [{
                        data: displayKlineData.map(d => d.volume),
                        itemStyle: {
                            color: (params) => {
                                // 根据当天涨跌上色（红涨绿跌）
                                // 🔧 引用全局变量 allKlineData 而不是闭包捕获的局部变量
                                const idx = params.dataIndex
                                if (!allKlineData || idx >= allKlineData.length) return '#64748b'
                                const data = allKlineData[idx]
                                if (!data || data.close === null || data.open === null) return '#64748b'
                                return data.close >= data.open ? '#ef4444' : '#10b981'
                            }
                        }
                    }]
                }, { notMerge: false, lazyUpdate: true })
            } else if (current_indicator === 'MACD') {
                // 🔧 使用后端计算的MACD数据
                const macdData = allIndicatorsData.macd || []
                const macdDates = macdData.map(d => d.date)
                const macdLine = macdData.map(d => d.macd)
                const signalLine = macdData.map(d => d.signal)
                const histogram = macdData.map(d => d.histogram)
                indicator_chart.setOption({
                    xAxis: { data: macdDates },
                    series: [
                        { name: 'DIFF', data: macdLine },
                        { name: 'DEA', data: signalLine },
                        { name: 'MACD柱', data: histogram }
                    ]
                }, { notMerge: false, lazyUpdate: true })
            } else if (current_indicator === 'RSI') {
                // 🔧 使用后端计算的RSI数据
                const rsiData = allIndicatorsData.rsi || []
                const rsiDates = rsiData.map(d => d.date)
                const rsi = rsiData.map(d => d.value)
                indicator_chart.setOption({
                    xAxis: { data: rsiDates },
                    series: [{ data: rsi }]
                }, { notMerge: false, lazyUpdate: true })
            } else if (current_indicator === 'KDJ') {
                // 🔧 使用后端计算的KDJ数据
                const kdjData = allIndicatorsData.kdj || []
                const kdjDates = kdjData.map(d => d.date)
                const k = kdjData.map(d => d.k)
                const d = kdjData.map(d => d.d)
                const j = kdjData.map(d => d.j)
                indicator_chart.setOption({
                    xAxis: { data: kdjDates },
                    series: [
                        { name: 'K', data: k },
                        { name: 'D', data: d },
                        { name: 'J', data: j }
                    ]
                }, { notMerge: false, lazyUpdate: true })
            } else if (current_indicator === 'OBV') {
                // 🔧 使用后端计算的OBV数据
                const obvData = allIndicatorsData.obv || []
                const obvDates = obvData.map(d => d.date)
                const obv = obvData.map(d => d.value)
                indicator_chart.setOption({
                    xAxis: { data: obvDates },
                    series: [{ data: obv }]
                }, { notMerge: false, lazyUpdate: true })
            }
        } catch(e) {
            console.error('更新指标图数据失败:', e)
        }
    }

    /**
     * 调整 dataZoom 位置（在数据前置后保持视图不跳跃）
     * @param {number} oldLength - 原数据长度
     * @param {number} prependLength - 前置数据长度
     */
    function adjustDataZoomAfterPrepend(oldLength, prependLength) {
        if (!kline_chart) {
            console.error('kline_chart 未初始化')
            return
        }

        try {
            const option = kline_chart.getOption()
            if (!option.dataZoom || !option.dataZoom[0]) return

            const oldZoom = option.dataZoom[0]
            const newLength = oldLength + prependLength

            // 计算新的 start 和 end（百分比）
            // 原来的数据区间在新数组中的位置向后偏移
            const oldStartIndex = Math.floor((oldZoom.start || 0) / 100 * oldLength)
            const oldEndIndex = Math.floor((oldZoom.end || 100) / 100 * oldLength)

            const newStartIndex = oldStartIndex + prependLength
            const newEndIndex = oldEndIndex + prependLength

            const newStart = (newStartIndex / newLength) * 100
            const newEnd = (newEndIndex / newLength) * 100

            console.log('🎯 调整 dataZoom:', {
                old: { start: oldZoom.start?.toFixed(2) + '%', end: oldZoom.end?.toFixed(2) + '%', length: oldLength },
                new: { start: newStart.toFixed(2) + '%', end: newEnd.toFixed(2) + '%', length: newLength }
            })

            // 🔧 标记开始系统调整（避免被误判为用户拖动）
            if (typeof window.__dataZoomAdjusting === 'function') {
                window.__dataZoomAdjusting(true)
            }

            // 更新 dataZoom
            kline_chart.dispatchAction({
                type: 'dataZoom',
                dataZoomIndex: 0,
                start: newStart,
                end: newEnd
            })

            // 同步到指标图
            if (indicator_chart) {
                indicator_chart.dispatchAction({
                    type: 'dataZoom',
                    dataZoomIndex: 0,
                    start: newStart,
                    end: newEnd
                })
            }

            // 🔧 同步到数据窗口控制条
            if (dataZoomChart) {
                dataZoomChart.dispatchAction({
                    type: 'dataZoom',
                    dataZoomIndex: 0,
                    start: newStart,
                    end: newEnd
                })
            }

            // 🔧 150ms 后结束系统调整标记（给足够时间让轮询检测到）
            setTimeout(function() {
                if (typeof window.__dataZoomAdjusting === 'function') {
                    window.__dataZoomAdjusting(false)
                }
            }, 150)
        } catch(e) {
            console.error('调整 dataZoom 失败:', e)
        }
    }

    // ==================== 数据加载函数 ====================

    /**
     * 加载更多历史数据（真实API版本）
     * @param {Function} callback - 回调函数(success)
     */
    function loadMoreHistoryData(callback) {
        console.log('开始加载更多历史数据...', {current_symbol, use_mock_mode})

        // 检查是否有当前数据
        if (!allKlineData || allKlineData.length === 0) {
            console.warn('没有当前数据，无法加载更多')
            // 🔧 关键修复：失败时也要重置 lastLoadPosition
            lastLoadPosition = -1
            callback(false)
            return
        }

        // 获取最早的日期
        const earliestData = allKlineData[0]
        const beforeDate = earliestData.date  // 'YYYY-MM-DD' 格式

        // 🔧 使用getModeConfig构建API URL
        const modeConfig = getModeConfig(false);
        const url = modeConfig.buildHistoryUrl(beforeDate);
        console.log('📡 加载更多URL:', url)

        // 调用API
        fetch(url)
            .then(r => {
                // 检查HTTP状态码
                if (!r.ok) {
                    console.error(`❌ HTTP ${r.status}: ${r.statusText}`);
                    throw new Error(`HTTP ${r.status}: ${r.statusText}`);
                }
                return r.json()
            })
            .then(res => {
                if (res.status !== 'success') {
                    console.error('加载失败:', res.message)
                    callback(false)
                    return
                }

                const chartData = res.data || {}
                const newData = chartData.kline || []
                const newEvents = chartData.events || []
                const newIndicators = chartData.indicators || {}  // 🔧 新加载的指标数据

                // 🔧 详细日志：帮助诊断是否真到头
                console.log('📦 API返回数据:', {
                    status: res.status,
                    newDataLength: newData.length,
                    hasKline: !!chartData.kline,
                    beforeDate: beforeDate
                })

                // 检查是否有新数据
                if (newData.length === 0) {
                    console.warn('⚠️ API返回空数据！可能原因:')
                    console.warn('  1. 真的到达最早数据（数据源只有这么多）')
                    console.warn('  2. 缓存到头但未增量更新（需要清空缓存重试）')
                    console.warn('  3. API错误或数据格式问题')
                    console.warn('  当前 hasMoreData 将被设为 false，不会再触发请求')
                    console.warn('  如需重试，请刷新页面或切换周期')
                    // 🔧 关键修复：失败时也要重置 lastLoadPosition
                    lastLoadPosition = -1
                    callback(false)
                    return
                }

                // 合并数据（新数据插入到前面）
                const oldLength = allKlineData.length
                allKlineData = newData.concat(allKlineData)
                allEvents = newEvents.concat(allEvents)

                // 🔧 合并技术指标数据（将新指标插入到前面）
                for (const indicatorName in newIndicators) {
                    if (allIndicatorsData[indicatorName]) {
                        allIndicatorsData[indicatorName] = newIndicators[indicatorName].concat(allIndicatorsData[indicatorName])
                    } else {
                        allIndicatorsData[indicatorName] = newIndicators[indicatorName]
                    }
                }

                console.log('✅ 加载成功：新增', newData.length, '条数据，总计', allKlineData.length, '条')

                // 🔧 仅更新图表数据，不重新渲染（避免视觉跳动）
                updateChartData(allKlineData, allEvents, window.getMarketTimezone)

                // 调整 dataZoom 位置（保持视图不跳跃）
                adjustDataZoomAfterPrepend(oldLength, newData.length)

                // 🔧 关键修复：加载成功后重置加载位置标记，允许继续触发下一次加载
                lastLoadPosition = -1
                console.log('🔄 重置加载标记，允许下一次触发')

                callback(true)
            })
            .catch(err => {
                console.error('加载更多数据失败:', err)
                // 更详细的错误信息显示
                if (err instanceof TypeError && err.message.includes('fetch')) {
                    showEmpty('网络连接失败，请检查网络设置')
                }
                // 🔧 关键修复：异常时也要重置 lastLoadPosition
                lastLoadPosition = -1
                callback(false)
            })
    }

    // ==================== 实时K线功能 ====================

    /**
     * 获取实时K线数据（Mock和真实使用相同机制）
     */
    function fetchRealtimeKline() {
        if (!current_symbol) return

        // 🔧 使用getModeConfig构建实时K线URL
        const modeConfig = getModeConfig(false);
        const url = modeConfig.buildRealtimeUrl();
        console.log(`${use_mock_mode ? '🎭 Mock模式' : '🎯 真实模式'} - 获取实时K线 (period=${current_period})`)

        fetch(url)
            .then(r => r.json())
            .then(res => {
                if (res.status !== 'success') {
                    console.warn('获取实时K线失败:', res.message)
                    return
                }

                const realtimeData = res.data
                console.log('📊 实时K线数据:', realtimeData)

                // 更新实时K线
                currentRealtimeKline = realtimeData
                updateRealtimeKlineOnChart(realtimeData)

                // 根据should_poll决定是否继续轮询
                if (realtimeData.should_poll) {
                    // 继续轮询（盘前或盘中）
                    startRealtimeKlineUpdateTimer()
                } else {
                    // 停止轮询（盘后）
                    stopRealtimeKlineUpdateTimer()
                }
            })
            .catch(err => {
                console.error('获取实时K线失败:', err)
            })
    }

    /**
     * 更新实时K线到图表
     * 简化逻辑：前端不做业务判断，后端已处理好合并逻辑
     */
    function updateRealtimeKlineOnChart(realtimeData) {
        if (!allKlineData || !allKlineData.length) return
        if (realtimeData.trading_phase !== 'trading') return
        // 🔧 验证实时数据的完整性
        if (!realtimeData.date) {
            console.error('❌ 实时K线数据缺少date字段:', realtimeData)
            return
        }

        console.log('📊 更新实时K线:', {
            period: current_period,
            date: realtimeData.date,
            open: realtimeData.open,
            high: realtimeData.high,
            low: realtimeData.low,
            close: realtimeData.close,
            volume: realtimeData.volume
        })

        const realtimeDate = realtimeData.date

        // 简化逻辑：无论哪个周期，都查找是否已存在该日期的K柱
        const existingIndex = allKlineData.findIndex(d => {
            let dateStr = d.date
            if (typeof dateStr === 'string' && dateStr.includes('GMT')) {
                dateStr = new Date(dateStr).toISOString().split('T')[0]
            }
            return dateStr === realtimeDate
        })

        if (existingIndex >= 0) {
            // 更新已存在的K线（后端已完成合并逻辑）
            console.log(`🔄 ${current_period}线 - 更新K柱: ${realtimeDate}`)
            allKlineData[existingIndex] = {
                date: realtimeDate,
                open: realtimeData.open,
                high: realtimeData.high,
                low: realtimeData.low,
                close: realtimeData.close,
                volume: realtimeData.volume
            }
        } else {
            // 添加新K线（日线的新天、周线的新周、月线的新月）
            console.log(`🔄 ${current_period}线 - 添加新K柱: ${realtimeDate}`)
            allKlineData.push({
                date: realtimeDate,
                open: realtimeData.open,
                high: realtimeData.high,
                low: realtimeData.low,
                close: realtimeData.close,
                volume: realtimeData.volume
            })
        }

        // 更新图表数据（不重新渲染）
        updateChartData(allKlineData, allEvents, window.getMarketTimezone)
    }
    /**
     * 停止实时K线轮询
     */
    function stopRealtimeKlineUpdateTimer() {
        if (realtimeKlineTimer) {
            clearInterval(realtimeKlineTimer)
            realtimeKlineTimer = null
            console.log('⏸️ 停止实时K线轮询')
        }
    }

    /**
     * 启动实时K线（选择股票时调用）
     */
    function startRealtimeKlineUpdateTimer() {
        stopRealtimeKlineUpdateTimer()  // 先停止之前的轮询
        const modeConfig = getModeConfig(true);
        const pollInterval = modeConfig.pollInterval;
        if (!realtimeKlineTimer) {
            realtimeKlineTimer = window.setInterval(() => fetchRealtimeKline(), pollInterval)
            console.log('⏸️ 启动实时K线轮询')
        }
    }


    // 显示空状态
    function showEmpty(text='暂无数据') {
        if(kline_chart && indicator_chart){
            AppUtils.showChartEmpty(kline_chart, text)
            AppUtils.showChartEmpty(indicator_chart, text)
        }
    }

    // 显示加载状态
    function showLoading(show=true, text='加载中...') {
        if(kline_chart && indicator_chart){
            AppUtils.showChartLoading(kline_chart,show,  text)
            AppUtils.showChartLoading(indicator_chart, show, text)
        }
    }

    function clearChart(){
            // 🔧 停止分时图更新（使用独立模块）
             stopRealtimeKlineUpdateTimer()
            // 🔧清空全局数据
             kline_chart = null
             indicator_chart = null
             dataZoomChart = null
             allKlineData = []  // 所有K线数据
             allEvents = []      // 所有事件数据
             allIndicatorsData = {}  // 所有技术指标数据（后端API计算）
             isLoadingNewStock = false  // 标记是否正在加载新股票（需要复位 dataZoom）
             realtimeKlineTimer = null
             currentRealtimeKline = null
             mock_trading_phase = 'trading'  // 模拟控制：before_open, trading, after_close外部传入，不由kline_chart.js内部管理
             current_period = 'daily' // 当前周期（内部状态）
             current_indicator = 'VOL' // 当前指标（内部状态）
             current_market_code = 'CN'// 当前市场（内部状态）
             current_zoom = null// 当前缩放（内部状态）
             current_symbol = null// 当前股票（内部状态）
            // 无限滚动相关状态
             isLoadingMore = false  // 加载状态标志
             hasMoreData = true      // 是否还有更多数据
             lastLoadPosition = -1   // 上次触发加载的位置（避免重复触发）
             lastStartValue = -1     // 修复：初始值设为-1（无效值），第一次获取到真实值后才开始比较
             userIsMoving = false    // 用户是否正在拖动
             movingResetTimer = null // 拖动重置定时器
             isAdjustingBySystem = false // 标记系统是否正在自动调整（避免误判为用户拖动）
             infiniteScrollEnabled = false // 修复：标记是否启用无限滚动（防止首次加载时误触发）
             initialLoadComplete = false  // 标记初始加载是否完成
            // 🔧 1. 销毁旧的图表实例
            if (kline_chart) {
                try {
                    kline_chart.dispose()
                } catch(e) {
                    console.warn('销毁k线图失败:', e)
                }
                kline_chart = null
            }
            if (indicator_chart) {
                try {
                    indicator_chart.dispose()
                } catch(e) {
                    console.warn('销毁技术指标图失败:', e)
                }
                indicator_chart = null
            }
    }
    /**
     * 切换模拟时段
     */
    function setMockTradingPhase(phase) {
        console.log('🎭 切换模拟时段:', phase)

        // 更新按钮状态
        document.querySelectorAll('.mock-phase-btn').forEach(btn => {
            btn.classList.remove('active')
        })
        document.querySelector(`[data-phase="${phase}"]`)?.classList.add('active')

        // 使用KlineChart模块切换模拟时段
        mock_trading_phase=phase;

        // 重新加载K线数据（触发needs_realtime_kline判断）
        loadData()
    }
    // ==================== 无限滚动触发逻辑 ====================

    /**
     * 启动无限滚动检测
     */
    function startInfiniteScrollDetection() {
        console.log('🚀 启动无限滚动检测，当前状态:', {
            infiniteScrollEnabled,
            initialLoadComplete,
            hasMoreData,
            isLoadingMore
        })
        window.setInterval(function() {
            try {
                // 🔥 关键修复：初始加载未完成时，不进行任何检测
                if (!initialLoadComplete) {
                    return
                }

                // 检查图表是否存在
                if (!kline_chart || !indicator_chart || !dataZoomChart) {
                    return
                }

                // 🔧 优先从 K 线图获取 dataZoom 状态（主图表）
                let currentStart = 0
                let hasValidZoom = false

                // 尝试从 K 线图获取
                const kOption = kline_chart.getOption()
                if (kOption && kOption.dataZoom && kOption.dataZoom[0]) {
                    currentStart = kOption.dataZoom[0].start || 0
                    hasValidZoom = true
                }

                // 如果 K 线图没有，尝试从技术指标图获取
                if (!hasValidZoom) {
                    const iOption = indicator_chart.getOption()
                    if (iOption && iOption.dataZoom && iOption.dataZoom[0]) {
                        currentStart = iOption.dataZoom[0].start || 0
                        hasValidZoom = true
                    }
                }

                // 如果仍然没有，尝试从数据窗口控制条获取
                if (!hasValidZoom) {
                    const dOption = dataZoomChart.getOption()
                    if (dOption && dOption.dataZoom && dOption.dataZoom[0]) {
                        currentStart = dOption.dataZoom[0].start || 0
                        hasValidZoom = true
                    }
                }

                if (!hasValidZoom) return

                // 🔧 检测用户是否在向左拖动（start 值减小）
                // ⚠️ 排除系统自动调整导致的变化
                // 🔥 关键修复：首次获取到有效值时，不判断为拖动，只记录初始值
                if (lastStartValue === -1) {
                    // 首次获取到有效的 start 值，记录下来
                    lastStartValue = currentStart
                    console.log('📌 首次记录 dataZoom 位置:', currentStart.toFixed(2) + '%')
                } else if (Math.abs(currentStart - lastStartValue) > 0.5 && !isAdjustingBySystem) {
                    console.log('📍 检测到 dataZoom 变化:', {
                        lastStartValue: lastStartValue.toFixed(2),
                        currentStart: currentStart.toFixed(2),
                        delta: (currentStart - lastStartValue).toFixed(2),
                        isAdjustingBySystem,
                        infiniteScrollEnabled,
                        initialLoadComplete
                    })

                    userIsMoving = true

                    // 🔥 修复：首次检测到用户拖动时，启用无限滚动
                    if (!infiniteScrollEnabled && initialLoadComplete) {
                        infiniteScrollEnabled = true
                        console.log('✅ 检测到用户拖动，启用无限滚动')
                    }

                    // 清除之前的定时器
                    if (movingResetTimer) {
                        clearTimeout(movingResetTimer)
                    }

                    // 500ms 后认为用户停止拖动
                    movingResetTimer = setTimeout(function() {
                        userIsMoving = false
                        console.log('⏸️ 用户停止拖动，当前 start =', currentStart.toFixed(2) + '%')
                    }, 500)
                }

                lastStartValue = currentStart

                // 🎯 渐进式预加载机制（仅在用户正在拖动时才允许触发加载）
                // 🔧 优化：提前预加载阈值，避免数据延迟和窗口跳跃
                // 🔧 关键修复：在 start < 20 时不需要 userIsMoving 限制（避免拖不动）
                // 🔥 修复：只有在 infiniteScrollEnabled=true 时才允许触发
                if (infiniteScrollEnabled && hasMoreData && !isLoadingMore) {
                    var shouldLoad = false
                    var triggerReason = ''
                    var needsUserMoving = true  // 默认需要用户拖动

                    if (currentStart < 20 && lastLoadPosition !== 20) {
                        shouldLoad = true
                        triggerReason = '紧急加载(start < 20%)'
                        lastLoadPosition = 20
                        needsUserMoving = false  // 紧急情况，不需要 userIsMoving
                    } else if (currentStart < 40 && lastLoadPosition !== 40 && userIsMoving) {
                        shouldLoad = true
                        triggerReason = '预加载1(start < 40%)'
                        lastLoadPosition = 40
                    } else if (currentStart < 60 && currentStart >= 40 && lastLoadPosition !== 60 && userIsMoving) {
                        shouldLoad = true
                        triggerReason = '预加载2(start < 60%)'
                        lastLoadPosition = 60
                    }

                    if (shouldLoad) {
                        console.log('🚀 触发加载更多：' + triggerReason + '，当前状态:', {
                            currentStart: currentStart.toFixed(2) + '%',
                            infiniteScrollEnabled,
                            hasMoreData,
                            isLoadingMore,
                            userIsMoving,
                            lastLoadPosition
                        })
                        isLoadingMore = true
                        loadMoreHistoryData(function(success) {
                            isLoadingMore = false
                            if (!success) {
                                hasMoreData = false
                                console.log('✅ 已到达最早数据')
                            }
                        }, current_symbol)
                    }
                } else {
                    // 🔧 调试日志：输出为什么没有触发加载
                    if (currentStart < 60 && Math.abs(currentStart - lastStartValue) > 0.5) {
                        const reasons = []
                        if (!hasMoreData) reasons.push('hasMoreData=false')
                        if (isLoadingMore) reasons.push('isLoadingMore=true')
                        if (!userIsMoving) reasons.push('userIsMoving=false')
                        if (reasons.length > 0 && currentStart < 40) {
                            console.log('⚠️ 未触发加载 (start=' + currentStart.toFixed(2) + '%): ' + reasons.join(', '))
                        }
                    }
                }
            } catch(e) {
                // 忽略
            }
        }, 100)
    }

    // ==================== 公共接口 ====================
    window.KlineChart = {
        // 只导出data_explorer.html中使用的函数
        setCurrent: function(currentSymbol,marketCode,useMockMode,mockTradingPhase='trading')  {
            clearChart();
            current_symbol = currentSymbol;
            current_market_code = marketCode;
            use_mock_mode=useMockMode;
            mock_trading_phase=mockTradingPhase;
            rebuildLayout()
            loadData();
        },
        showEmpty:showEmpty,
        stopRealtimeKlineUpdateTimer:stopRealtimeKlineUpdateTimer
    };
} // End of KlineChart module block

