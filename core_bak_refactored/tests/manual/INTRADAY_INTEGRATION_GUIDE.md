# 分时数据前后端集成指南

## 📋 功能概览

分时数据获取和展示功能已完整集成，支持：
- ✅ 真实AKShare API调用（1分钟级别数据）
- ✅ 三层缓存机制（内存 → 数据库 → API）
- ✅ 前一交易日fallback
- ✅ 模拟数据兜底
- ✅ RESTful API端点
- ✅ 前端可视化展示

---

## 🔧 技术架构

### 后端数据流

```
API Request
    ↓
DataQualityAPIService (/api/v1/intraday/data)
    ↓
ChartDataAssembler (assemble_intraday_data)
    ↓
ProviderSelector (select_provider_for_symbol)
    ↓
AKShareDataProvider (get_intraday_data)
    ↓
三层缓存策略:
    1. 内存缓存（毫秒级）
    2. 数据库缓存（0.1-0.3秒）
    3. AKShare API（4-8秒）
    ↓
Fallback策略:
    → 前一交易日缓存
    → 模拟数据生成
```

### 核心组件

| 组件 | 路径 | 职责 |
|------|------|------|
| **API端点** | `app/quality_monitoring/api_service.py` | RESTful接口暴露 |
| **数据组装** | `app/quality_monitoring/api/chart_data.py` | 格式转换和组装 |
| **Provider选择** | `core/data/providers/provider_selector.py` | 配置驱动的数据源选择 |
| **AKShare实现** | `core/data/providers/akshare_provider.py` | 真实API调用 + 缓存 |
| **数据协议** | `core/data/providers/protocols.py` | `IntradayData`类型定义 |

---

## 🚀 快速开始

### 1. 启动API服务器

```bash
# 方法1: 使用测试脚本
python -m core_bak_refactored.tests.manual.test_intraday_api_manual

# 方法2: 使用主应用入口（如果有）
# python -m core_bak_refactored.app.main
```

服务器启动后访问：`http://localhost:5000`

### 2. 测试API端点

```bash
# 方法1: curl命令
curl "http://localhost:5000/api/v1/intraday/data?symbol=000300.SH"

# 方法2: Python测试脚本
python -m core_bak_refactored.tests.manual.test_intraday_api_manual test

# 方法3: 浏览器直接访问
# http://localhost:5000/api/v1/intraday/data?symbol=000300.SH&trade_date=2025-12-12
```

### 3. 前端可视化展示

在浏览器中打开：
```
core_bak_refactored/tests/manual/intraday_demo.html
```

**功能特性：**
- 实时分时走势图（ECharts）
- 买卖盘口数据（10档）
- 成交明细（最近20笔）
- 自动刷新（可选，每5秒）

---

## 📡 API接口文档

### 端点：`GET /api/v1/intraday/data`

#### 请求参数

| 参数 | 类型 | 必需 | 说明 | 示例 |
|------|------|------|------|------|
| `symbol` | string | ✅ | 证券代码 | `000300.SH` |
| `trade_date` | string | ❌ | 交易日期 | `2025-12-12` |

#### 响应格式

```json
{
  "status": "success",
  "data": {
    "symbol": "000300.SH",
    "name": "沪深300",
    "current_price": 3125.50,
    "yesterday_close": 3120.00,
    "change": 5.50,
    "change_percent": 0.18,
    "times": ["09:30", "09:31", "09:32", ...],
    "prices": [3121.0, 3122.5, 3123.0, ...],
    "volumes": [12000, 15000, 13000, ...],
    "avg_prices": [3121.0, 3121.75, 3122.0, ...],
    "order_book": {
      "bids": [
        {"price": 3125.49, "volume": 2000},
        {"price": 3125.48, "volume": 1800},
        ...
      ],
      "asks": [
        {"price": 3125.51, "volume": 1500},
        {"price": 3125.52, "volume": 2200},
        ...
      ]
    },
    "trade_records": [
      {"time": "14:59:50", "price": 3125.50, "volume": 100, "type": "buy"},
      {"time": "14:59:45", "price": 3125.48, "volume": 200, "type": "sell"},
      ...
    ]
  },
  "timestamp": "2025-12-12T10:24:29.573456"
}
```

#### 错误响应

```json
{
  "status": "error",
  "message": "缺少必需参数: symbol",
  "error_code": "MISSING_PARAMETER"
}
```

---

## 🧪 测试覆盖

### 单元测试（13个）

```bash
# 运行AKShare Provider单元测试
pytest core_bak_refactored/tests/units/core/data/providers/akshare_provider_test.py -v
```

**测试覆盖：**
- ✅ 初始化和配置
- ✅ API成功调用
- ✅ 内存缓存命中
- ✅ Fallback到前一交易日
- ✅ 模拟数据生成
- ✅ 数据格式转换
- ✅ 辅助方法（盘口、成交明细等）

### 集成测试（4个）

```bash
# 运行完整链路集成测试
pytest core_bak_refactored/tests/integration/test_intraday_integration.py -v
```

**测试覆盖：**
- ✅ 完整数据链路（Provider → Assembler → API）
- ✅ 缓存一致性验证
- ✅ API端点成功响应
- ✅ API端点错误处理

---

## 🔍 缓存机制详解

### 三层缓存架构

#### 第1层：内存缓存（毫秒级）

```python
# 缓存键格式
cache_key = f"intraday_{symbol}_{trade_date}"

# 缓存TTL
_cache_ttl = 300  # 5分钟
```

**优势：**
- 极速响应（< 1ms）
- 减少API调用
- 同一分钟内的多次请求共享数据

#### 第2层：数据库缓存（0.1-0.3秒）

```python
# 使用BaseProvider的数据库服务
db_service.get_cached_data(symbol, start_date, end_date, source='AKShareDataProvider')
```

**优势：**
- 跨进程共享
- 持久化存储
- 历史数据回溯

#### 第3层：外部API（4-8秒）

```python
# 调用AKShare真实API
df = akshare.stock_zh_a_hist_min_em(
    symbol=symbol,
    start_date=start_time,
    end_date=end_time,
    period='1',
    adjust=''
)
```

### Fallback策略

1. **前一交易日缓存**
   - 当天数据不可用时，返回前一交易日的缓存数据
   - 避免完全无数据的情况

2. **模拟数据生成**
   - 最后的兜底方案
   - 确保系统始终可用
   - 模拟数据会被缓存，保证一致性

---

## 📊 数据格式说明

### IntradayData对象

```python
@dataclass
class IntradayData:
    symbol: str                          # 证券代码
    name: str                            # 证券名称
    current_price: float                 # 当前价
    yesterday_close: float               # 昨收价
    change: float                        # 涨跌额
    change_percent: float                # 涨跌幅(%)
    ticks: List[IntradayTickRecord]      # 分时tick数据
    order_book_bids: List[OrderBookLevel]  # 买盘
    order_book_asks: List[OrderBookLevel]  # 卖盘
    trade_records: List[TickerRecord]          # 成交明细
    trade_date: str                      # 交易日期
```

### 支持的市场

根据`market_sources`配置驱动：

```yaml
market_sources:
  CN: akshare      # 中国A股
  HK: akshare      # 香港市场
  US: yahoo        # 美国市场
```

**测试代码示例：**
- `000300.SH` - 沪深300（CN市场 → AKShare）
- `000001.SH` - 上证指数
- `399001.SZ` - 深证成指
- `399006.SZ` - 创业板指

---

## 🐛 故障排查

### 常见问题

#### 1. API超时

**症状：**
```
ERROR: HTTPSConnectionPool: Read timed out (read timeout=15)
```

**解决方案：**
- 系统会自动fallback到缓存数据
- 检查网络连接
- 调整AKShare超时时间（如需要）

#### 2. 缓存未命中

**症状：**
- 日志显示"缓存未命中，调用外部API"
- 响应时间较长（4-8秒）

**原因：**
- 首次访问该symbol/日期组合
- 缓存已过期（超过TTL）

**验证：**
```bash
# 第二次访问应该命中缓存
curl "http://localhost:5000/api/v1/intraday/data?symbol=000300.SH"
# 响应时间 < 100ms
```

#### 3. 数据不一致

**症状：**
- 同一请求返回不同的数据

**检查点：**
- 是否使用模拟数据（会随机生成）
- 模拟数据是否正确缓存

---

## 🚦 性能指标

### 响应时间

| 场景 | 预期响应时间 | 数据来源 |
|------|-------------|---------|
| **内存缓存命中** | < 10ms | 内存 |
| **数据库缓存命中** | 100-300ms | 数据库 |
| **API首次调用** | 4-8秒 | AKShare |
| **Fallback到前一日** | < 10ms | 内存缓存 |
| **模拟数据生成** | < 50ms | 内存计算 |

### 缓存命中率

- **目标命中率**：> 90%（交易时间内）
- **监控方式**：查看日志中的缓存命中标记

---

## 📝 开发备注

### 代码规范遵循

✅ **架构原则：**
- 配置驱动（market_sources）
- 单一数据源（每个市场）
- 复用BaseProvider缓存机制
- 无冗余实现

✅ **测试规范：**
- 单元测试覆盖所有核心方法
- 集成测试验证完整链路
- Mock外部依赖

✅ **代码质量：**
- 完整的docstring
- 类型注解
- 异常处理
- 日志记录

### 未来优化方向

- [ ] 支持更多时间粒度（5分钟、15分钟）
- [ ] WebSocket实时推送
- [ ] 更智能的缓存策略（LRU）
- [ ] 数据质量监控和告警

---

## 📚 相关文档

- [API服务完整文档](../../app/quality_monitoring/api_service.py)
- [AKShare Provider实现](../../core/data/providers/akshare_provider.py)
- [数据协议定义](../../core/data/providers/protocols.py)
- [代码优化规范](../../../.qoder/rules/CODE_OPTIMIZATION_STRATEGY.md)
- [测试规范](../../../.qoder/rules/PECIFICATIONS.md)

---

**文档版本**: v1.0  
**更新时间**: 2025-12-12  
**状态**: ✅ 已完成集成  
