# 基础设施层 TODO

> **层级**：Infrastructure Layer  
> **职责**：提供纯数学/统计计算，不包含业务逻辑

---

## 📊 risk_metrics.py - StatisticalCalculator

**状态**：✅ 生产就绪（专家复审通过）  
**最后更新**：2024-11-09  
**测试覆盖**：21/21 通过

### 已完成
- [x] P0: 下行标准差公式修正（半方差）
- [x] P0: 新增 `calculate_simple_returns` 方法
- [x] P0: 简单收益率单元测试（边界条件）
- [x] P1: 价格有效性检查（日志警告）
- [x] P1: 协方差矩阵条件数检测

### 待办事项
- [ ] **P1**: 与标准统计库（scipy.stats）对比验证
  - 验证半方差计算与 scipy 一致性
  - 验证CVaR计算与标准实现对比
  - 生成对比测试报告

- [ ] **P2**: 定义数值误差常量
  ```python
  EPSILON = 1e-10  # 数值比较容差
  NEAR_ZERO_THRESHOLD = 1e-15  # 接近零判断阈值
  ```

- [ ] **P2**: 压力测试（万级资产规模）
  - 10,000资产协方差矩阵计算
  - 内存占用监控
  - 计算性能基准

---

## 📈 timeseries_calculator.py - TimeSeriesCalculator

**状态**：✅ 已完成（专家复审通过）  
**最后更新**：2024-11-08  
**测试覆盖**：24/24 通过

### 已完成
- [x] 技术指标基础计算（EMA, SMA, VWAP）
- [x] 滚动窗口计算
- [x] 24个单元测试

### 待办事项
- [ ] **P2**: 增加更多技术指标
  - MFI（资金流量指标）
  - Williams %R（反向随机指标）
  - SAR（抛物线转向指标）

- [ ] **P3**: 高级指标
  - Ichimoku云图
  - 卡尔曼滤波器

---

## 📉 technical_indicators.py

**状态**：✅ 已完成  
**最后更新**：2024-11-08

### 待办事项
- [ ] **P2**: 板块参数配置
  ```python
  SECTOR_PARAMS = {
      'main': {...},      # 主板参数
      'gem': {...},       # 创业板参数
      'star': {...},      # 科创板参数
  }
  ```

---

## ⚙️ cache_manager.py

**状态**：⚠️ 待优化（当前不存在，需创建）  
**优先级**：中  

### 待办事项
- [ ] **P1**: 创建缓存管理器
  ```python
  class CacheManager:
      """多级缓存管理"""
      
      def __init__(self, enable_memory=True, enable_redis=False):
          self.memory_cache = {}
          self.redis_client = None if not enable_redis else Redis()
      
      def get(self, key):
          # L1: 内存缓存
          # L2: Redis缓存（可选）
          pass
      
      def set(self, key, value, ttl=3600):
          pass
  ```

- [ ] **P1**: 指标计算缓存机制
  - 缓存最近N次计算结果
  - 基于数据哈希的智能缓存
  - TTL过期策略

- [ ] **P1**: 增量计算支持
  ```python
  def incremental_calculate(self, new_data, previous_state):
      """增量计算 - 避免全量重算"""
      pass
  ```

---

## 🗓️ 历史变更

- **2024-11-09**: 创建基础设施层独立TODO
- **2024-11-09**: risk_metrics.py 完成P0/P1修复
- **2024-11-08**: timeseries_calculator.py 专家复审通过
