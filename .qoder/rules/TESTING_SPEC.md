# DeepSeekQuant 测试规范

> **版本**: v1.0 | **更新**: 2025-11-26 | **范围**: 测试代码规范、测试文件命名、测试结构、测试最佳实践

---

## 📋 测试规范概述

本规范定义了 DeepSeekQuant 项目的测试代码编写标准，确保测试代码的一致性、可维护性和可读性。

---

## 🔧 测试文件命名规范（强制）

### 1. 测试文件命名规则

**核心规则**：
- ✅ **强制格式**：必须以 `*_test.py` 结尾（例如：`factor_model_test.py`）
- ❌ **严禁**：使用 `test_*.py` 前缀格式
- ✅ **一一对应原则**：一个源文件 `xxx.py` 必须且只能有一个对应测试文件 `xxx_test.py`
- ✅ **目录镜像原则**：`core_bak_refactored/tests/{test_type}/**` 必须镜像 `core_bak_refactored/core/**` 或 `infrastructure/**` 的目录结构

### 2. 命名示例

```
源文件位置                                    测试文件位置（必须唯一）
────────────────────────────────────────────────────────────────────
core_bak_refactored/core/risk/
  └── factor_model.py                      core_bak_refactored/tests/units/core/risk/
                                            └── factor_model_test.py ✅（唯一）

core_bak_refactored/infrastructure/
  └── cache_service.py                     core_bak_refactored/tests/infrastructure/
                                            └── cache_service_test.py ✅（唯一）

core_bak_refactored/core/data/_fragments/
  └── data_quality_checker.py              core_bak_refactored/tests/units/core/data/_fragments/
                                            └── data_quality_checker_test.py ✅
```

### 3. 禁止的命名模式

```
❌ test_factor_model.py          # 禁止test_前缀
❌ factor_model_unittest.py      # 禁止其他后缀
❌ test_factor_model_test.py     # 禁止混合格式
❌ factor_model_integration.py   # 集成测试应分离到专门目录
❌ factor_model_perf_test.py     # 性能测试应分离到专门目录
```

---

## 🏗️ 测试类型分离规范

### 1. 单元测试（与源文件一一对应）
- **位置**：`tests/units/` 镜像 `core/` 或 `infrastructure/`
- **命名**：`{source_name}_test.py`（必须唯一）
- **范围**：测试单个模块的功能
- **示例**：`tests/units/core/risk/factor_model_test.py`

### 2. 集成测试（独立目录）
- **位置**：`tests/integration/`
- **命名**：`{feature}_integration_test.py`
- **范围**：测试多个模块协作
- **示例**：`tests/integration/risk_calculator_integration_test.py`

### 3. 性能测试（独立目录）
- **位置**：`tests/performance/` 或 `tests/benchmarks/`
- **命名**：`{feature}_benchmark.py` 或 `{feature}_perf_test.py`
- **范围**：性能基准测试
- **示例**：`tests/performance/portfolio_risk_benchmark.py`

### 4. 端到端测试（独立目录）
- **位置**：`tests/e2e/`
- **命名**：`{scenario}_e2e_test.py`
- **范围**：完整业务流程
- **示例**：`tests/e2e/backtest_workflow_e2e_test.py`

---

## 📝 测试代码结构规范

### 1. 测试类结构模板

```python
import unittest
from unittest.mock import Mock, patch
from core_bak_refactored.core.data.providers.yahoo_provider import YahooFinanceDataProvider


class YahooFinanceDataProviderTest(unittest.TestCase):
    """YahooFinanceDataProvider 单元测试"""
    
    def setUp(self):
        """测试前置设置"""
        self.provider = YahooFinanceDataProvider()
        self.symbol = 'AAPL'
        self.start_date = '2023-01-01'
        self.end_date = '2023-12-31'
    
    def tearDown(self):
        """测试后置清理"""
        self.provider = None
    
    def test_fetch_history_prices_normal_case(self):
        """测试正常情况下的历史价格获取"""
        # 准备测试数据
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # 执行测试
        result = self.provider.fetch_history_prices(
            self.symbol, self.start_date, self.end_date
        )
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(col in result.columns for col in expected_columns))
    
    def test_fetch_history_prices_invalid_symbol(self):
        """测试无效股票代码的处理"""
        with self.assertRaises(ValueError):
            self.provider.fetch_history_prices('INVALID', self.start_date, self.end_date)
    
    def test_fetch_history_prices_network_error(self):
        """测试网络错误处理"""
        with patch('yfinance.download') as mock_download:
            mock_download.side_effect = ConnectionError("网络连接失败")
            
            with self.assertRaises(ConnectionError):
                self.provider.fetch_history_prices(
                    self.symbol, self.start_date, self.end_date
                )
```

### 2. 公共验证方法规范

```python
class YahooFinanceDataProviderPatchTest(unittest.TestCase):
    """补丁相关测试的公共验证方法"""
    
    def _verify_patch_applied(self, mock_patch):
        """验证补丁是否已应用"""
        self.assertTrue(mock_patch.called)
        self.assertEqual(mock_patch.call_count, 1)
    
    def _verify_dataframe_type(self, result):
        """验证返回数据类型"""
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
    
    def _test_with_patch_verification(self, test_func, *args, **kwargs):
        """带补丁验证的测试模板方法"""
        with patch('yfinance.Ticker') as mock_ticker:
            # 准备Mock返回值
            mock_instance = Mock()
            mock_instance.history.return_value = pd.DataFrame({
                'Open': [150.0], 'High': [155.0], 'Low': [149.0], 
                'Close': [152.0], 'Volume': [1000000]
            })
            mock_ticker.return_value = mock_instance
            
            # 执行测试函数
            result = test_func(*args, **kwargs)
            
            # 验证补丁应用
            self._verify_patch_applied(mock_ticker)
            
            return result
```

---

## 🚨 测试异常处理规范

### 1. 网络测试异常处理

```python
def test_network_related_scenarios(self):
    """网络相关场景测试"""
    
    # 网络超时
    with patch('requests.get', side_effect=TimeoutError("请求超时")):
        with self.assertRaises(TimeoutError):
            self.provider.fetch_data()
    
    # 连接错误
    with patch('requests.get', side_effect=ConnectionError("连接失败")):
        with self.assertRaises(ConnectionError):
            self.provider.fetch_data()
    
    # HTTP错误
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        with self.assertRaises(HTTPError):
            self.provider.fetch_data()
```

### 2. 补丁测试异常处理

```python
def test_patch_exception_handling(self):
    """补丁异常处理测试"""
    
    # 补丁方法抛出异常
    with patch('some_module.some_function') as mock_func:
        mock_func.side_effect = ValueError("模拟异常")
        
        with self.assertRaises(ValueError):
            self.provider.some_operation()
        
        # 验证异常被正确捕获和处理
        self.assertEqual(mock_func.call_count, 1)
```

### 3. 测试跳过规范

```python
@unittest.skipIf(not HAS_YFINANCE, "yfinance库未安装")
def test_yfinance_integration(self):
    """需要yfinance的集成测试"""
    # 测试逻辑...

@unittest.skipIf(not NETWORK_AVAILABLE, "网络不可用")
def test_network_operations(self):
    """需要网络的测试"""
    # 测试逻辑...
```

---

## 📊 测试日志规范

### 1. 日志级别使用

```python
import logging

class TestWithLogging(unittest.TestCase):
    """带日志的测试类"""
    
    def setUp(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def test_with_detailed_logging(self):
        """详细日志记录的测试"""
        self.logger.info("开始执行测试: %s", self._testMethodName)
        
        try:
            # 测试逻辑
            result = self.provider.some_operation()
            self.logger.debug("操作结果: %s", result)
            
            # 断言验证
            self.assertIsNotNone(result)
            self.logger.info("测试通过")
            
        except Exception as e:
            self.logger.error("测试失败: %s", str(e))
            raise
```

### 2. 测试结果报告

```python
def test_performance_with_metrics(self):
    """带性能指标的测试"""
    import time
    
    start_time = time.time()
    
    # 执行测试操作
    result = self.provider.complex_operation()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 记录性能指标
    self.logger.info("操作执行时间: %.3f秒", execution_time)
    
    # 性能断言
    self.assertLess(execution_time, 5.0, "操作应在5秒内完成")
    self.assertIsNotNone(result)
```

---

## 🔧 Mock和补丁使用规范

### 1. 补丁测试示例

```python
def test_with_complex_patching(self):
    """复杂补丁测试"""
    
    with patch('module1.function1') as mock_func1, \
         patch('module2.function2') as mock_func2, \
         patch('module3.Class3') as mock_class3:
        
        # 设置Mock返回值
        mock_func1.return_value = "mock_result1"
        mock_func2.return_value = "mock_result2"
        
        mock_instance = Mock()
        mock_instance.method.return_value = "mock_instance_result"
        mock_class3.return_value = mock_instance
        
        # 执行测试
        result = self.provider.complex_operation()
        
        # 验证调用
        mock_func1.assert_called_once()
        mock_func2.assert_called_once()
        mock_class3.assert_called_once()
        mock_instance.method.assert_called_once()
        
        # 验证结果
        self.assertEqual(result, "expected_result")
```

### 2. Mock使用原则

- ✅ **隔离测试**：使用Mock隔离外部依赖
- ✅ **行为验证**：验证Mock方法的调用情况
- ✅ **返回值控制**：精确控制Mock的返回值
- ✅ **异常模拟**：使用side_effect模拟异常
- ❌ **过度Mock**：避免Mock过多内部逻辑

---

## 📋 测试数据准备规范

### 1. 测试数据分类

```python
class TestDataPreparation:
    """测试数据准备示例"""
    
    @classmethod
    def setUpClass(cls):
        """类级别数据准备"""
        cls.sample_stock_data = pd.DataFrame({
            'Open': [150.0, 151.0, 152.0],
            'High': [155.0, 156.0, 157.0],
            'Low': [149.0, 150.0, 151.0],
            'Close': [152.0, 153.0, 154.0],
            'Volume': [1000000, 1200000, 1100000]
        })
    
    def setUp(self):
        """测试级别数据准备"""
        self.test_symbols = ['AAPL', 'GOOGL', 'MSFT']
        self.test_dates = ['2023-01-01', '2023-01-02', '2023-01-03']
```

### 2. 数据准备方法

```python
def create_test_dataframe(rows=10, columns=None):
    """创建测试DataFrame"""
    if columns is None:
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    data = {}
    for col in columns:
        if col == 'Volume':
            data[col] = np.random.randint(1000000, 5000000, rows)
        else:
            data[col] = np.random.uniform(100, 200, rows)
    
    return pd.DataFrame(data)
```

---

## ✅ 测试执行与验证规范

### 1. 测试执行命令

```bash
# 运行单个测试文件
PYTHONPATH=. python -m pytest core_bak_refactored/tests/units/core/data/providers/yahoo_provider_test.py -v

# 运行特定测试类
PYTHONPATH=. python -m pytest core_bak_refactored/tests/units/core/data/providers/yahoo_provider_test.py::YahooFinanceDataProviderTest -v

# 运行特定测试方法
PYTHONPATH=. python -m pytest core_bak_refactored/tests/units/core/data/providers/yahoo_provider_test.py::YahooFinanceDataProviderTest::test_fetch_history_prices_normal_case -v

# 运行所有测试
PYTHONPATH=. python -m pytest core_bak_refactored/tests/ -v
```

### 2. 测试验证标准

- ✅ **所有测试通过**：退出码为0
- ✅ **无严重警告**：仅允许可接受的警告（如库弃用警告）
- ✅ **测试覆盖率**：新增代码测试覆盖率≥80%
- ✅ **性能要求**：单个测试执行时间≤30秒
- ✅ **内存使用**：无内存泄漏

---

## 🔄 测试代码重构规范

### 1. 消除重复代码

```python
# ❌ 重复代码（重构前）
def test_method1(self):
    with patch('module.function') as mock_func:
        mock_func.return_value = "result1"
        result = self.provider.method1()
        mock_func.assert_called_once()
        self.assertEqual(result, "expected1")

def test_method2(self):
    with patch('module.function') as mock_func:
        mock_func.return_value = "result2"
        result = self.provider.method2()
        mock_func.assert_called_once()
        self.assertEqual(result, "expected2")

# ✅ 重构后（使用公共方法）
def _test_with_patch_template(self, test_method, expected_result, mock_return_value):
    """补丁测试模板方法"""
    with patch('module.function') as mock_func:
        mock_func.return_value = mock_return_value
        result = test_method()
        mock_func.assert_called_once()
        self.assertEqual(result, expected_result)

def test_method1(self):
    self._test_with_patch_template(
        self.provider.method1, "expected1", "result1"
    )

def test_method2(self):
    self._test_with_patch_template(
        self.provider.method2, "expected2", "result2"
    )
```

### 2. 重构验证步骤

1. **识别重复模式**：查找相似的测试代码结构
2. **提取公共方法**：将重复逻辑提取为模板方法
3. **参数化测试**：使用参数化减少重复测试方法
4. **验证重构效果**：运行测试确保功能不变
5. **代码质量检查**：确保重构后代码更清晰、更易维护

---

## 📖 相关文档

- **项目规范**: `.qoder/rules/PECIFICATIONS.md`
- **代码优化策略**: `.qoder/rules/CODE_OPTIMIZATION_STRATEGY.md`
- **ASK规范**: `.qoder/rules/ASK_SPEC.md`

---

**维护者**: DeepSeekQuant 开发团队  
**版本**: v1.0 (2025-11-26) - 初始版本，包含完整的测试规范体系  
**状态**: ✅ 生效中