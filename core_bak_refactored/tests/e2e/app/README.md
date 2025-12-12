# E2E 测试说明

## 概述

E2E（端到端）测试验证整个应用的功能，包括 UI 和 API。这些测试需要运行真实的 Web 服务器。

## 测试文件

### `test_quality_monitoring_pages.py`

包含两个测试类：

1. **TestQualityMonitoringPages** - UI 测试（需要浏览器）
2. **TestProvidersCredentialsAPI** - API 测试（HTTP 请求）

## 运行前准备

### 1. 安装依赖

```bash
pip install selenium requests pytest
```

### 2. 安装 Chrome 和 ChromeDriver

**macOS**:
```bash
brew install --cask google-chrome
brew install chromedriver
```

**Ubuntu**:
```bash
sudo apt-get install chromium-browser chromium-chromedriver
```

**Windows**:
- 下载 Chrome: https://www.google.com/chrome/
- 下载 ChromeDriver: https://chromedriver.chromium.org/

### 3. 启动 Web 服务器

```bash
cd core_bak_refactored/app/quality_monitoring
python app_example.py
```

服务器将启动在 `http://localhost:5001`

## 运行测试

### 运行所有 E2E 测试（需要服务器）

```bash
export RUN_E2E_TESTS=1  # Linux/macOS
# 或
set RUN_E2E_TESTS=1     # Windows

pytest core_bak_refactored/tests/e2e/app/test_quality_monitoring_pages.py -v
```

### 仅运行 UI 测试

```bash
export RUN_E2E_TESTS=1
pytest core_bak_refactored/tests/e2e/app/test_quality_monitoring_pages.py::TestQualityMonitoringPages -v
```

### 仅运行 API 测试（Mock 模式）

```bash
# 不需要真实服务器，使用 Mock
pytest core_bak_refactored/tests/e2e/app/test_quality_monitoring_pages.py::TestProvidersCredentialsAPI -v
```

### 使用真实服务器运行 API 测试

```bash
export USE_REAL_SERVER=1
pytest core_bak_refactored/tests/e2e/app/test_quality_monitoring_pages.py::TestProvidersCredentialsAPI -v
```

## 测试覆盖

### UI 测试

- ✅ Dashboard 页面加载
- ✅ 页面间导航
- ✅ Providers 页面功能
- ✅ Data Explorer 页面
- ✅ Rules Manager 页面
- ✅ Scheduler Console 页面
- ✅ Alerts Center 页面
- ✅ Validation Reports 页面
- ✅ Realtime Monitor 页面
- ✅ 响应式头部
- ✅ 系统状态指示器

### API 测试

- ✅ Providers CRUD 工作流
- ✅ Credentials CRUD 工作流

## 跳过测试的原因

默认情况下，E2E 测试会被跳过，因为：

1. **需要运行服务器** - 自动化测试通常不启动 Web 服务器
2. **需要浏览器环境** - CI/CD 环境可能没有安装 Chrome
3. **执行时间较长** - E2E 测试比单元测试慢得多
4. **环境依赖** - 需要特定的端口、数据库等

## CI/CD 集成

### 在 GitHub Actions 中运行

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: 安装 Chrome
        run: |
          sudo apt-get update
          sudo apt-get install -y chromium-browser chromium-chromedriver
      
      - name: 安装 Python 依赖
        run: |
          pip install -r requirements.txt
          pip install selenium pytest
      
      - name: 启动服务器
        run: |
          cd core_bak_refactored/app/quality_monitoring
          python app_example.py &
          sleep 10  # 等待服务器启动
      
      - name: 运行 E2E 测试
        env:
          RUN_E2E_TESTS: 1
        run: |
          pytest core_bak_refactored/tests/e2e/app/test_quality_monitoring_pages.py -v
```

## 故障排查

### 问题：chromedriver 版本不匹配

```bash
# 检查 Chrome 版本
google-chrome --version

# 下载匹配的 ChromeDriver
# https://chromedriver.chromium.org/downloads
```

### 问题：服务器未运行

```bash
# 检查服务器状态
curl http://localhost:5001/health

# 输出应该是 200 OK
```

### 问题：端口被占用

```bash
# 查找占用端口的进程
lsof -i :5001

# 杀死进程
kill -9 <PID>
```

## 最佳实践

1. **测试隔离** - 每个测试应该独立运行
2. **数据清理** - 测试后清理创建的数据
3. **等待策略** - 使用显式等待而不是 `time.sleep()`
4. **错误处理** - 捕获并记录详细的错误信息
5. **Mock 外部依赖** - API 测试应该 mock 外部服务

## 参考文档

- [Selenium Documentation](https://www.selenium.dev/documentation/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Flask Testing](https://flask.palletsprojects.com/en/2.0.x/testing/)
