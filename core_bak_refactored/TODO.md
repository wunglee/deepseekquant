# DeepSeekQuant - 系统级TODO（跨模块/未确定位置）

> **说明**：此文件仅包含跨越多个模块或未确定具体实施位置的任务
> 
> 各模块的具体TODO请查看：
> - `infrastructure/TODO.md` - 基础设施层
> - `core/signal/TODO.md` - 信号引擎模块
> - `core/risk/TODO.md` - 风险管理模块
> - `core/strategy/TODO.md` - 策略模块（待创建）

---

## 🔴 高优先级跨模块任务

### 端到端集成测试
**影响范围**：数据层 → 信号层 → 风险层 → 执行层  
**优先级**：高  
**目标**：验证完整交易工作流

**测试场景**：
```python
class EndToEndWorkflowTest(unittest.TestCase):
    def test_complete_trading_workflow(self):
        """完整交易工作流"""
        # 1. 数据获取 (core/data/)
        # 2. 信号生成 (core/signal/)
        # 3. 组合优化 (core/optimization/)
        # 4. 风险评估 (core/risk/)
        # 5. 订单执行模拟 (core/exec/)
        pass
```

**待办事项**：
- [ ] 设计端到端测试用例
- [ ] 准备测试数据集（包含极端场景）
- [ ] 实现测试脚本
- [ ] 验证各模块数据流转

---

### 多处理器协同测试
**影响范围**：`core/portfolio/` ↔ `core/risk/` ↔ `core/system_monitor.py`  
**优先级**：高  
**目标**：验证处理器间数据流与状态同步

**测试内容**：
- PortfolioProcessor → RiskProcessor 数据传递
- RiskMonitor 实时监控多个组合
- 并发场景下状态一致性

**待办事项**：
- [ ] 定义处理器间通信协议
- [ ] 实现数据流验证测试
- [ ] 并发压力测试

---

## 🟡 中优先级跨模块任务

### 系统性能优化
**影响范围**：全系统  
**优先级**：中  

**优化方向**：
1. **缓存机制**（infrastructure + core）
   - [ ] 指标计算结果缓存
   - [ ] 风险计算结果缓存
   - [ ] 多级缓存策略（内存 + Redis）

2. **增量计算**（infrastructure）
   - [ ] 技术指标增量更新
   - [ ] 风险指标增量计算
   - [ ] 状态快照机制

3. **向量化优化**（infrastructure）
   - [ ] numpy批量计算优化
   - [ ] numba JIT编译加速
   - [ ] 并行计算支持

---

## 🔵 低优先级跨模块任务

### 告警通知系统
**影响范围**：`core/risk/risk_monitor.py` + 新建通知服务  
**优先级**：低  

**功能点**：
- [ ] 邮件告警集成
- [ ] 短信告警集成
- [ ] 钉钉/企业微信集成
- [ ] 告警规则引擎
- [ ] 告警频率控制（去重、聚合）

**建议实施位置**：`infrastructure/notification_service.py`

---

## 📋 架构级决策待定

### 策略模块目录结构
**状态**：待决策  
**问题**：策略模块应该放在 `core/strategy/` 还是独立为 `strategies/`？

**选项A**：`core/strategy/`（推荐）
```
core/strategy/
├── base_strategy.py
├── trend_following.py
├── mean_reversion.py
└── breakout.py
```

**选项B**：独立 `strategies/`
```
strategies/
├── base/
├── trend/
├── mean_reversion/
└── custom/
```

**待办**：
- [ ] 评估策略数量和复杂度
- [ ] 确定目录结构
- [ ] 创建目录并记录TODO

---

## 🗓️ 历史变更

- **2024-11-09**: 重构TODO组织结构，拆分为模块级TODO
- **2024-11-08**: 原整体TODO.md（已拆分）
