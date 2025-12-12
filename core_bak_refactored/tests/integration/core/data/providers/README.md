# Yahoo Finance Provider 集成测试

本目录包含 Yahoo Finance 数据提供者的集成测试，需要真实网络请求。

## 测试文件

### 1. `yahoo_proxy_config_test.py`
**功能**: 测试 Yahoo Finance 代理配置功能

**测试内容**:
- HTTP/2 支持
- 代理配置开关功能
- 配置文件独立控制每个数据源的代理

**运行方式**:
```bash
cd /path/to/deepseekquant
PYTHONPATH=. python core_bak_refactored/tests/integration/core/data/providers/yahoo_proxy_config_test.py
```

**配置要求**:
在 `core_bak_refactored/config/dev/system.yml` 中配置：
```yaml
data_providers:
  yahoo_finance:
    use_proxy: true/false  # 控制是否使用代理
```

---

### 2. `yahoo_rate_limit_test.py`
**功能**: 测试 Yahoo Finance 限速处理

**测试内容**:
- 验证每 2 秒请求 1 次（符合 Yahoo 官方限速）
- 测试连续 10 次请求的成功率
- 限速错误的处理

**运行方式**:
```bash
cd /path/to/deepseekquant
PYTHONPATH=. python core_bak_refactored/tests/integration/core/data/providers/yahoo_rate_limit_test.py
```

**预期结果**:
- 成功率 >= 80% (8/10)
- 总耗时约 20 秒（10 次请求，每次间隔 2 秒）

---

## 注意事项

1. **网络要求**: 这些是集成测试，需要真实的网络连接
2. **代理配置**: 如果在中国大陆运行，可能需要配置代理访问 Yahoo Finance
3. **限速**: 请勿频繁运行这些测试，以免触发 Yahoo Finance 的限速
4. **测试环境**: 建议在开发环境运行，不要在 CI/CD 中自动运行

---

## 与单元测试的区别

| 测试类型 | 位置 | 网络请求 | 用途 |
|---------|------|---------|------|
| 单元测试 | `tests/units/core/data/providers/yahoo_provider_test.py` | ❌ (使用 mock) | 测试逻辑正确性 |
| 集成测试 | `tests/integration/core/data/providers/yahoo_*_test.py` | ✅ (真实请求) | 测试实际功能 |

---

## 命名规范

根据 `.qoder/rules/PECIFICATIONS.md` 的要求：
- ✅ 使用 `*_test.py` 后缀格式
- ✅ 位于 `tests/integration/` 目录下
- ✅ 镜像源代码的目录结构
