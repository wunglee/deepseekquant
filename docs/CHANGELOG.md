# 项目变更日志

## 索引规则

每次 ask → answer → 实施 为一个完整周期，统一索引追踪。

**格式说明**：
- 通过Git提交号(commit hash)追踪文档版本
- 查看历史版本：`git show <commit>:docs/ask.md`
- 恢复到某版本：`git checkout <commit> -- docs/ask.md`

---

## 2024-11 风险模块P1优化

### P1-3 智能限额管理（进行中-待验证）

**咨询阶段**：
- 📝 提问: `b78b094` - ask: P1-3智能阈值系统咨询
  - 文件快照: `git show b78b094:docs/ask.md`
  - 24个问题：智能阈值、组合优化、违规优先级、市场差异化
- 📥 回答: 待补充
  - 等待专家在 `docs/answer.md` 中填写

**实施阶段**：
- 🚧 初始实施: `4cbf07b` - feat(risk): P1-3 RiskLimitsManager智能化完整实现
  - ⚠️ **状态**: 未经专家确认，已标记为待验证
  - 新增文件: risk_limits_enhanced.py (682行)
  - 测试覆盖: 24/24通过
- 📋 标记待验证: `385a349` - docs: 标记P1-3代码为待验证状态

**当前状态**: ⏸️ 暂停，等待专家审核指导后重新实施

---

### P1-2 压力测试场景（已完成）

**咨询阶段**：
- 📝 提问: (历史提交) - ask: P1-2压力测试场景参数咨询
- 📥 回答: (历史提交) - answer: P1-2专家指导

**实施阶段**：
- ✅ 完整重做: `529eea4` - feat(risk): P1-2 StressTester完整重做
  - 6大核心场景 + 2个复合场景
  - 17个测试全部通过

**当前状态**: ✅ 已完成并通过测试

---

## 使用示例

### 查看P1-3的原始提问
```bash
git show b78b094:docs/ask.md
```

### 比较不同版本的计划文档
```bash
# 查看PLAN.md的演进历史
git log --oneline docs/PLAN.md

# 对比两个版本
git diff <commit1> <commit2> -- docs/PLAN.md
```

### 恢复到某个历史状态
```bash
# 恢复P1-3咨询时的ask.md
git checkout b78b094 -- docs/ask.md
```

---

## 文档管理约定

1. **ask.md**: 每次新咨询覆盖旧内容，历史通过Git查看
2. **answer.md**: 每次专家回答覆盖旧内容，历史通过Git查看
3. **PLAN.md**: 更新计划时直接修改，保持当前最新计划
4. **PROGRESS.md**: 更新进展时直接修改，保持当前最新进展
5. **CHANGELOG.md**: 持续追加索引，不删除历史记录

---

*最后更新: 2024-11-11*
