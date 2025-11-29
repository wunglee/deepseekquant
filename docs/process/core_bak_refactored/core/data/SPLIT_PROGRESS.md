# Data 模块拆分进度报告（第一阶段 - 已重置为正确策略）

## 目标
将 `data_fetcher.py`（6920行）按职责单一原则拆分为独立模块，并保持行为完全一致。

## 正确的拆分策略

1. **不可动的对照基准**：`data_fetcher_bak.py`（6920行）- 保留原始代码，不做任何修改
2. **可动的待拆解文件**：`data_fetcher.py`（6920行）- 逐步被掏空
3. **拆分模块**（初始328行）- 委派回 `data_fetcher.py`，逐步承接迁移的代码
4. **最终目标**：拆分模块 ≈7000行，`data_fetcher.py` ≈0行，`data_fetcher_bak.py` 保持6920行作为对照

## 当前状态（正确的初始状态）

### 文件清单
- `core_bak_refactored/core/data/data_fetcher_bak.py`：**6920行**（不可动，对照基准）
- `core_bak_refactored/core/data/data_fetcher.py`：**6920行**（可动，待拆解）
- 拆分模块合计：**328行**（委派版本）

**总计**：6920 + 328 = **7248行**

### 拆分模块明细（委派版本）
- `core/data/cache/`：18行
  - `key.py`：缓存键生成（8行）
  - `store.py`：缓存读写委派（18行，委派到 `DataFetcher._get_cached_data` 和 `_cache_data`）
  - `memory.py`、`lru.py`、`redis_adapter.py`：占位（14+16+18=48行）
- `core/data/fallback/`：11行
  - `orchestrator.py`：备用数据源委派（11行，委派到 `DataFetcher._try_fallback_sources`）
- `core/data/market/`：88行
  - `calendar.py`、`breadth.py`、`sector.py`、`status.py`：市场状态相关
- `core/data/analytics/`：12行
  - `volatility.py`：波动率计算
- `core/data/realtime/`：17行
  - `stream.py`、`iex.py`：实时数据占位
- `core/data/providers/`：50行
  - `yahoo.py`、`alpha_vantage.py`：数据源适配器
- `core/data/quality/metrics.py`：8行
- `core/data/credentials/manager.py`：13行
- `core/data/models.py`：7行
- `core/data/fetcher_orchestrator.py`：34行

### 测试覆盖
- ✅ `fetcher_orchestrator_test.py`：1个测试通过
- ⚠️ `cache/store_test.py`：2个测试失败（Mock配置问题，待修复）
- ✅ 其他测试：待运行

## 下一步行动（按正确策略）

### 第一批迁移（高优先级，~500行）
1. **缓存键生成**：从 `DataFetcher._generate_cache_key` 迁移到 `cache/key.py`（~10行）
2. **缓存读写**：从 `DataFetcher._get_cached_data` 和 `_cache_data` 迁移到 `cache/store.py`（~70行）
3. **备用数据源编排**：从 `DataFetcher._try_fallback_sources` 迁移到 `fallback/orchestrator.py`（~50行）
4. **市场日历**：从 `DataFetcher._is_market_open/_is_market_holiday` 迁移到 `market/calendar.py`（~40行）
5. **市场广度**：从 `DataFetcher._get_advance_decline` 迁移到 `market/breadth.py`（~45行）
6. **板块表现**：从 `DataFetcher._get_sector_performance` 迁移到 `market/sector.py`（~50行）
7. **波动率计算**：从 `DataFetcher._calculate_daily_volatility` 迁移到 `analytics/volatility.py`（~15行）
8. **HTTP客户端初始化**：从 `DataFetcher._setup_http_client` 迁移到 `http/client.py`（~30行）
9. **凭证管理**：从 `DataFetcher._setup_api_credentials` 迁移到 `credentials/manager.py`（~20行）
10. **缓存系统初始化**：从 `DataFetcher._setup_caching/_setup_redis_cache` 迁移到 `cache/`（~70行）

### 第二批迁移（中优先级，~1500行）
11. **Yahoo数据获取**：从 `DataFetcher._fetch_yahoo_data` 迁移到 `providers/yahoo.py`（~60行）
12. **Alpha Vantage数据获取**：从 `DataFetcher._fetch_alpha_vantage_data` 迁移到 `providers/alpha_vantage.py`（~90行）
13. **历史数据获取主流程**：从 `DataFetcher.get_historical_data` 迁移（~100行）
14. **实时数据获取**：从 `DataFetcher.get_real_time_data` 迁移到 `realtime/`（~50行）
15. **基本面数据**：从 `DataFetcher.get_fundamental_data` 相关方法迁移到 `fundamentals/`（~300行）
16. **市场状态获取**：从 `DataFetcher.get_market_status` 迁移（~80行）
17. **数据源初始化**：从 `DataFetcher._initialize_data_sources` 迁移（~40行）

### 第三批迁移（低优先级，~4000行）
18. **其他数据源实现**：IEX、Polygon、Twelve Data等（~2000行）
19. **数据质量指标**：`get_data_quality_metrics` 相关（~100行）
20. **WebSocket流式数据**：`stream_real_time_data` 相关（~200行）
21. **辅助方法**：各种私有辅助方法（~1000行）
22. **数据模型定义**：MarketData、枚举等（~100行）
23. **性能监控与统计**：性能指标更新逻辑（~50行）

## 行数变化预测

| 阶段 | data_fetcher.py | 拆分模块 | 总计 | 完成度 |
|------|----------------|---------|------|--------|
| 初始 | 6920 | 328 | 7248 | 0% |
| 第一批完成 | ~6400 | ~850 | ~7250 | ~7% |
| 第二批完成 | ~4900 | ~2350 | ~7250 | ~29% |
| 第三批完成 | ~0 | ~7100 | ~7100 | 100% |

*注：总计会略有变化（优化重构导致的±5%）*

## 验收标准
- ✅ 不可动对照文件 `data_fetcher_bak.py` 保持6920行不变
- ✅ 可动文件 `data_fetcher.py` 最终接近0行
- ✅ 拆分模块合计 ≈7000行（允许±10%优化重构误差）
- ✅ 所有测试保持通过
- ✅ 每次迁移后运行测试验证行为一致
