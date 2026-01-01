---
trigger: glob
glob: .qoder/rules/CODE_OPTIMIZATION_STRATEGY.md
---

# DeepSeekQuant 代码优化策略规范

> **版本**: v1.0 | **更新**: 2025-11-28 | **类型**: 技术债务清理与代码质量提升

---

## 📋 目录

1. [代码重构规范](#代码重构规范)
2. [局部功能冗余碎片统一](#局部功能冗余碎片统一)
3. [技术债务清理流程](#技术债务清理流程)
4. [优化验收标准](#优化验收标准)

---

## 代码重构规范

### 核心原则
- **面向接口编程（Protocol/抽象接口）、依赖注入（DI）**：运行期装配具体实现（真实/Mock/加速版），保持真实与模拟的严格隔离
- **职责边界清晰**：函数/类的单一职责；跨模块依赖不越界
- **消除重复**：提取共性逻辑至共享组件或工具函数；避免冗余实现
- **接口稳定**：方法签名不变；保持向后兼容；必要时提供弃用标记与迁移说明
- **异常与类型**：补齐类型注解、异常分级与错误信息；日志语义一致
- **可读性与维护性**：命名规范、注释到位、内部文档同步
- **性能与鲁棒性**：向量化/边界防护/内存安全的合理优化（不引入新依赖）
- **统一使用全局导入**：避免在函数内部或方法内部重复导入相同模块，所有导入应在文件顶部完成，确保导入的一致性和可维护性

### 执行要求

- **验证要求**：完成重构后立即运行全量相关测试；仅在测试通过且文档同步后，方可进入"待验收(PENDING_ACCEPTANCE)"并生成本轮 ask.md
- **测试覆盖要求（强制）**：
  - ✅ **所有Bug修复必须新增对应的测试用例**，确保修复逻辑被测试覆盖
  - ✅ **所有新增功能必须编写单元测试**，覆盖正常流程和异常边界
  - ✅ **所有接口变更必须更新集成测试**，验证接口契约不被破坏
  - ✅ **测试必须在代码提交前通过**，不允许提交未测试的代码
  - ✅ **测试代码与生产代码同步提交**，测试文件路径遵循项目规范（如 `tests/units/` 对应 `core/`）
- **文档同步**：更新受影响的模块/接口设计文档与 SPRINT.md 状态；在 consultation.md 中记录重构后的关键点（于迭代收尾统一追加）

---

## 局部功能冗余碎片统一

### 定义

意图相同但实现不同的局部代码片段（可能不是完整方法），需在同一模块/层内统一为公共方法或工具函数。

### 多维度检测方法（强制执行全部7项）

#### 1. 语义相似度分析
- 代码向量相似度计算
- 注释/日志语义分析
- 方法名语义相似度

#### 2. 调用上下文比对
- 输入/输出形态一致性检查
- 参数传递模式相同性检查

#### 3. 模式识别（grep扫描）

常见计算模板库：
- **敞口求和**：`sum(alloc.weight for ...)`
- **比例影响**：`abs(loss) * factor * coefficient`
- **VaR放大**：`base_var * (multiplier - 1)`
- **集中度HHI**：`sum(w**2)`
- **分位数VaR**：`np.percentile(returns, (1-conf)*100)`
- **配置提取**：`config.get('key', default)`
- **数据检查**：`if len(data) < threshold`
- **类型转换**：`float(abs(...))`, `float(np.clip(...))`
- **异常处理**：`try-except-logger.error-return default`

#### 4. AST结构对比
表达式树等价但命名不同（使用ast模块比对）

#### 5. 重复代码块检测
连续5行以上的相似代码（使用diff算法）

#### 6. 类型检查模式
`isinstance` + 类型转换模式

#### 7. 控制流模式
相同的if-elif-else分支结构

---

### 系统化扫描流程（强制）

#### 阶段1：自动化扫描

1. 使用grep批量扫描常见模式（25个预定义正则模板）
2. 统计每种模式的出现次数与位置
3. 生成初步冗余报告（按影响范围排序）

#### 阶段2：人工分析验证

4. 逐一分析grep结果，确认是否为真正的冗余
5. 识别业务语义差异（如系数不同、边界条件不同）
6. 分类：可立即统一 / 需参数化 / 保留差异

#### 阶段3：分层优化执行

7. **第一优先级**：完全相同的代码片段（直接提取）
8. **第二优先级**：参数化后可统一的模式（提取+参数化）
9. **第三优先级**：需装饰器/基类的模式（如异常处理）

#### 阶段4：验证与文档

10. 运行全量测试验证无副作用
11. 更新设计文档记录所有新增公共方法
12. 生成优化报告（优化前后对比、代码减少量）

---

### 预定义grep扫描模板库（必须全部执行）

```bash
# 1. 异常处理模式
grep -rn "except Exception as e:" --include="*.py" <target_dir>

# 2. 数据长度检查
grep -rn "if len([^)]+) [<>=]" --include="*.py" <target_dir>

# 3. 配置提取模式
grep -rn "\.get('" --include="*.py" <target_dir>

# 4. float转换模式
grep -rn "return float(" --include="*.py" <target_dir>

# 5. np.percentile调用
grep -rn "np\.percentile\(" --include="*.py" <target_dir>

# 6. isinstance类型检查
grep -rn "isinstance(" --include="*.py" <target_dir>

# 7. logger调用模式
grep -rn "logger\.(debug|info|warning|error)" --include="*.py" <target_dir>

# 8. try语句起始
grep -rn "^[ \t]*try:" --include="*.py" <target_dir>

# 9. sum聚合模式
grep -rn "sum(" --include="*.py" <target_dir>

# 10. abs绝对值
grep -rn "abs(" --include="*.py" <target_dir>

# 11. float(abs(...))组合
grep -rn "float(abs(" --include="*.py" <target_dir>

# 12. np.clip调用
grep -rn "np\.clip\(" --include="*.py" <target_dir>

# 13. dict.values()遍历
grep -rn "\.values()" --include="*.py" <target_dir>

# 14. None检查
grep -rn "is None" --include="*.py" <target_dir>

# 15. 空列表检查
grep -rn "len([^)]+) == 0" --include="*.py" <target_dir>

# 16. max/min调用
grep -rn "\(max\|min\)(" --include="*.py" <target_dir>

# 17. pd.Series构造
grep -rn "pd\.Series\(" --include="*.py" <target_dir>

# 18. np.array构造
grep -rn "np\.array\(" --include="*.py" <target_dir>

# 19. 字符串格式化
grep -rn "f\"" --include="*.py" <target_dir>

# 20. getattr调用
grep -rn "getattr(" --include="*.py" <target_dir>

# 21. hasattr检查
grep -rn "hasattr(" --include="*.py" <target_dir>

# 22. enumerate使用
grep -rn "enumerate(" --include="*.py" <target_dir>

# 23. zip使用
grep -rn "zip(" --include="*.py" <target_dir>

# 24. list comprehension
grep -rn "\[.*for.*in.*\]" --include="*.py" <target_dir>

# 25. dict comprehension
grep -rn "{.*for.*in.*}" --include="*.py" <target_dir>
```

---

### 执行流程（强制步骤）

```
发现（多维度扫描）
  ↓
分类（立即/参数化/装饰器）
  ↓
提取公共方法
  ↓
替换调用
  ↓
回归测试
  ↓
文档同步
  ↓
生成报告
```

---

### 验收度量（必须全部满足）

- ✅ 已执行全部7项检测方法
- ✅ 已生成完整的冗余扫描报告（包含统计数据）
- ✅ 重复片段数量减少≥50%（或已达理论上限）
- ✅ 公共方法覆盖≥2处以上调用点
- ✅ 修改后测试全部通过，且无公开接口变更
- ✅ 设计文档已更新所有新增公共方法
- ✅ 优化报告已记录优化前后对比数据

---

### 风险控制（强制约束）

- ✅ 仅统一技术性计算逻辑，不改变业务口径/阈值/接口契约
- ✅ 如存在边界差异，保留系数参数化并加入文档注释
- ✅ 每次替换后立即运行相关测试
- ✅ 发现测试失败立即回滚并分析差异

---

### 输出要求（强制文档）

#### 1. 冗余扫描报告（必须）

- 扫描范围（目录、文件数）
- 各类模式的出现次数统计表
- 可优化项清单（按影响范围排序）
- 已优化项与待优化项分类

#### 2. 优化执行报告（必须）

- 新增公共方法清单（含签名、用途、替换范围）
- 优化前后对比（代码行数、重复片段数）
- 测试验证结果

#### 3. 设计文档更新（必须）

- 模块设计文档记录所有新增公共方法
- 接口设计文档记录方法签名与用途
- SPRINT.md记录优化前后对比数据

---

### 分层优化示例（risk模块实践）

#### 第一层：完全相同的代码片段

```python
# 直接提取
_calculate_total_exposure(portfolio_state)  # 8处重复
_calculate_proportional_impact(base_loss, factor, coef)  # 7处重复
_calculate_var_amplification(base_var, multiplier)  # 3处重复
calculate_hhi(weights)  # 1处重复，提取到infrastructure
calculate_historical_var(returns, conf_level)  # 8处重复
```

#### 第二层：参数化后可统一

```python
# 配置提取统一
ConfigExtractor.get_nested(config, 'market_configs.CN.alpha', default=0.4)
ConfigExtractor.get_market_config(config, 'CN', 'alpha', 0.4)

# 数据检查统一
@require_min_data(min_size=20, param_name='returns')
def calculate_metric(returns):
    ...

# 类型转换统一
SafeNumericConverter.to_bounded_float(value, min_val=0, max_val=1)
```

#### 第三层：装饰器/基类模式

```python
# 异常处理装饰器
@safe_execute(default_return=0.0, log_level='error')
def calculate_risk(data):
    return complex_calculation(data)

# 类型规范化
series = TypeValidator.ensure_series(data)
array = TypeValidator.ensure_numeric_array(data)

# 缓存模式
cache = LRUTTLCache(max_size=1000, ttl_seconds=3600)
```

---

### 强制检查清单（每次优化必须完成）

- [ ] 已执行25个grep扫描模板
- [ ] 已生成冗余扫描报告（含统计表）
- [ ] 已分类所有发现的冗余（立即/参数化/装饰器）
- [ ] 已按优先级执行优化（第一层→第二层→第三层）
- [ ] 每次替换后已运行测试验证
- [ ] 已更新设计文档（模块+接口）
- [ ] 已生成优化执行报告
- [ ] 重复率降低≥50%或已达理论上限

---

## 技术债务清理流程

### 识别技术债务

1. **代码扫描**：使用自动化工具识别重复代码、复杂度高的函数
2. **人工审查**：代码评审中发现的待改进项
3. **测试覆盖**：低覆盖率区域和测试缺失模块
4. **文档缺失**：缺少文档或文档过时的代码

### 优先级评估

**P0 - 立即处理**：
- 影响系统稳定性的关键bug
- 严重的性能瓶颈
- 重大的安全隐患

**P1 - 本迭代处理**：
- 影响开发效率的架构问题
- 高频使用模块的代码质量问题
- 重要功能的测试缺失

**P2 - 规划处理**：
- 次要模块的重构需求
- 可优化但不紧急的性能问题
- 非关键路径的文档完善

### 执行流程

```
识别债务 → 评估优先级 → 制定计划 → 执行清理 → 测试验证 → 文档更新 → 闭环确认
```

---

## 优化验收标准

### 代码质量指标

- **重复代码率**：≤5%（理想）或较优化前降低≥50%
- **函数复杂度**：圈复杂度≤10（单个函数）
- **测试覆盖率**：关键路径≥90%，整体≥80%
- **文档完整性**：所有公开API有docstring，设计文档已同步

### 性能指标

- **无性能倒退**：关键路径性能不劣于优化前
- **内存安全**：无内存泄漏，大对象及时释放
- **并发安全**：多线程场景无竞态条件

### 可维护性指标

- **命名规范**：遵循项目命名约定
- **注释充分**：复杂逻辑有注释说明
- **职责单一**：函数/类职责明确，不过度耦合
- **向后兼容**：公开API保持向后兼容或提供迁移指南

---

## 附录：优化实践案例

### 案例1：risk模块彻底优化（2025-11-28）

**背景**：
- risk模块存在~705处重复代码片段
- 配置提取、异常处理、类型转换等模式高度重复

**优化策略**：
1. 新增5个infrastructure共享工具模块（1,520行）
   - `error_handling.py` - 统一异常处理
   - `data_validators.py` - 统一数据验证
   - `config_utils.py` - 统一配置管理
   - `numeric_utils.py` - 统一数值处理
   - `risk_metrics.py` - 扩展统计计算

2. 分层提取（三层优化）
   - 第一层：直接提取完全相同的代码
   - 第二层：参数化统一
   - 第三层：装饰器/基类抽象

**成果**：
- 重复代码：705处 → 55处（↓92.2%）
- 代码总量：15,000行 → 13,190行（↓12.1%）
- 测试通过：210/210（100%）
- 新增可复用工具：5个模块供全项目使用

**详细报告**：
`docs/optimization/RISK_MODULE_COMPREHENSIVE_OPTIMIZATION_2025-11-28.md`

---

## 修订历史

| 版本 | 日期 | 修订内容 | 修订人 |
|------|------|----------|--------|
| v1.0 | 2025-11-28 | 初始版本，从主规范文件中提取优化策略 | Qoder AI |

---

**文档状态**：✅ 已生效  
**适用范围**：core_bak_refactored 所有模块  
**更新频率**：根据实践持续优化
