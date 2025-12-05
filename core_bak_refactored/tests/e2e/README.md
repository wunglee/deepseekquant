# E2E 测试指南

## 概述

本目录包含应用层的端到端（E2E）测试，使用 Selenium WebDriver 测试完整的用户流程和界面交互。

## 测试范围

### 应用层页面测试 (`app/test_quality_monitoring_pages.py`)

#### 页面加载与导航测试
- ✅ Dashboard 页面加载
- ✅ 页面间导航（8个页面）
- ✅ 响应式布局测试（桌面/平板/移动）
- ✅ 系统状态指示器

#### 功能页面测试
1. **Data Explorer** - 数据浏览器交互
2. **Rules Manager** - 质量规则管理
3. **Scheduler Console** - 调度控制台
4. **Alerts Center** - 警报中心筛选
5. **Providers & Credentials** - 数据源与凭证管理
6. **Validation Reports** - 验证报告查看
7. **Realtime Monitor** - 实时监控

#### API CRUD 测试
- ✅ Providers CRUD 完整工作流
- ✅ Credentials CRUD 完整工作流
- ✅ 数据脱敏验证

## 环境准备

### 1. 安装 E2E 测试依赖

```bash
pip install -r tests/e2e/requirements-e2e.txt
```

### 2. 安装 Chrome 和 ChromeDriver

**macOS:**
```bash
brew install --cask google-chrome
brew install chromedriver
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install chromium-browser chromium-chromedriver

# 或使用 webdriver-manager 自动管理
pip install webdriver-manager
```

### 3. 启动应用服务

E2E 测试需要应用服务运行在 `http://localhost:5001`：

```bash
# 方式1：直接启动
python -m core_bak_refactored.app.main

# 方式2：使用示例启动
python -m core_bak_refactored.app.quality_monitoring.app_example
```

## 运行测试

### 运行所有 E2E 测试

```bash
# 从项目根目录运行
pytest core_bak_refactored/tests/e2e/ -v

# 仅运行应用层测试
pytest core_bak_refactored/tests/e2e/app/ -v
```

### 运行特定测试类

```bash
# 仅测试页面加载
pytest core_bak_refactored/tests/e2e/app/test_quality_monitoring_pages.py::TestQualityMonitoringPages -v

# 仅测试 API
pytest core_bak_refactored/tests/e2e/app/test_quality_monitoring_pages.py::TestProvidersCredentialsAPI -v
```

### 运行特定测试用例

```bash
# 测试 Dashboard 加载
pytest core_bak_refactored/tests/e2e/app/test_quality_monitoring_pages.py::TestQualityMonitoringPages::test_dashboard_page_loads -v

# 测试 Providers CRUD
pytest core_bak_refactored/tests/e2e/app/test_quality_monitoring_pages.py::TestProvidersCredentialsAPI::test_providers_crud_workflow -v
```

### 调试模式（显示浏览器）

编辑测试文件，注释掉无头模式：

```python
options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # 注释掉这行
```

## 测试覆盖率

当前测试覆盖：

- ✅ 8个页面的加载测试
- ✅ 页面间导航测试
- ✅ Providers 页面交互测试
- ✅ 响应式布局测试（3种视口）
- ✅ Providers API CRUD 测试
- ✅ Credentials API CRUD 测试
- ✅ 数据脱敏验证

## 故障排查

### 问题1：ChromeDriver 版本不匹配

**解决方案：**
```bash
pip install webdriver-manager
```

然后修改测试代码使用自动管理：

```python
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
```

### 问题2：服务器未运行

测试会自动跳过并提示：

```
SKIPPED [1] 服务器未运行，跳过E2E测试
```

确保在 5001 端口启动服务：

```bash
python -m core_bak_refactored.app.main --port 5001
```

### 问题3：元素未找到

增加等待时间：

```python
driver.implicitly_wait(20)  # 增加到20秒
```

或使用显式等待：

```python
wait = WebDriverWait(driver, 30)
element = wait.until(EC.presence_of_element_located((By.ID, "myElement")))
```

## 最佳实践

1. **测试隔离**：每个测试应独立运行，不依赖其他测试的状态
2. **等待策略**：使用显式等待而非 `time.sleep()`
3. **清理数据**：测试后清理创建的测试数据
4. **错误截图**：失败时保存截图便于调试
5. **Page Object**：对于复杂页面，使用 Page Object 模式

## 持续集成

在 CI/CD 中运行 E2E 测试：

```yaml
# .github/workflows/e2e.yml 示例
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/e2e/requirements-e2e.txt
      - name: Start app
        run: |
          python -m core_bak_refactored.app.main &
          sleep 10
      - name: Run E2E tests
        run: pytest core_bak_refactored/tests/e2e/ -v
```

## 参考资料

- [Selenium 官方文档](https://www.selenium.dev/documentation/)
- [pytest 官方文档](https://docs.pytest.org/)
- [WebDriver Manager](https://github.com/SergeyPirogov/webdriver_manager)
