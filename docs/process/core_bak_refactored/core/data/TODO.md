# TODO（数据模块）

说明：本文件维护“未分配/待实施”的事项，按目标或特性组织；不直接落到具体代码文件维度。

## P1 — 数据质量评估服务（迁移与落地）
- [PENDING] 从 risk 模块迁移“数据质量前置检查”职责到 `core/data`
  - 内容：统一实现质量评估服务接口，业务模块仅消费质量标识
  - 交付：`DataQualityMonitor.validate(data)` 返回 `quality_report` 与 `quality_flags`
- [PENDING] 定义 `DataQualityPolicy`（规则集与阈值）
  - 内容：完整性/一致性/时效性/可靠性四类规则；各项阈值由专家提供
  - 交付：策略配置（JSON/YAML）与加载器；严禁默认硬编码
- [PENDING] 专家咨询与配置来源（ask/answer 流程）
  - 内容：在 `docs/ask.md` 中咨询阈值与规则细节；在 `docs/answer.md` 中记录答复与依据
  - 交付：将专家确认的参数固化为配置文件，供策略加载器使用
- [PENDING] 质量评估实现与测试
  - 内容：实现 `DataQualityMonitor.validate`；覆盖完整性/一致性/时效性检查
  - 交付：单元测试（覆盖典型缺失/类型错误/时效过期场景），通过率≥95%
- [PENDING] 输出集成与消费
  - 内容：在 `core/risk` 的入口仅消费 `quality_flags`，并外部化到 `report_snapshot`
  - 交付：集成测试验证 risk 模块不再内置规则/默认值
- [PENDING] 文档同步
  - 内容：在 `docs/design/core_bak_refactored/core/data/` 新增模块设计与接口文档；在 risk 模块文档中更新“外部依赖”说明
  - 交付：设计与接口文档首版，包含数据结构、接口签名、使用示例

## P1 — 质量评估配置加载与校验
- [PENDING] 策略配置加载器实现（支持 JSON/YAML）
- [PENDING] 配置校验器实现（必填项/范围/类型校验，错误提示清晰）
- [PENDING] 可选：热更新钩子设计（watcher/订阅机制）

## 提醒（规范约束）
- 数据质量规则与阈值必须由专家答复明确，不得自行设定默认值
- 业务模块（如 risk）不得内置数据质量规则，必须消费本模块提供的质量标识
