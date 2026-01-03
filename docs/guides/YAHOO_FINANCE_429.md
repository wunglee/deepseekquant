# Yahoo Finance 数据提供者组件说明

## 组件结构

本目录包含两个主要组件：

1. `yahoo_provider.py` - Yahoo Finance 数据提供者主类
2. `yfinance_http2_patch.py` - yfinance 库的补丁文件

## 职责划分

### yahoo_provider.py
- **主要职责**：
  - 实现 `HistoricalDataProvider` 接口
  - 管理会话和代理配置
  - 处理数据标准化和格式转换
  - 提供统一的数据获取接口

- **不负责**：
  - 反爬虫逻辑
  - 请求限流
  - 浏览器模拟头
  - User-Agent 轮换

### yfinance_http2_patch.py
- **主要职责**：
  - 修复 yfinance 的 "Too Many Requests" (429) 问题
  - 实现完整的反爬虫机制
  - 处理请求限流
  - 管理浏览器模拟头
  - 实现 User-Agent 轮换
  - 处理 crumb 认证
  - 管理会话和 cookies

- **通过 monkey patch 机制**：
  - 重写 `yfinance.data.YfData.get` 方法
  - 所有通过 yfinance 的请求都会经过补丁处理

## 避免重复逻辑

为避免逻辑冗余和责任边界模糊：

1. **反爬虫逻辑**：仅在 `yfinance_http2_patch.py` 中实现
2. **请求限流**：仅在 `yfinance_http2_patch.py` 中实现
3. **浏览器模拟**：仅在 `yfinance_http2_patch.py` 中实现
4. **代理配置**：在 `yahoo_provider.py` 中设置，通过参数传递给补丁
5. **会话管理**：主要在 `yfinance_http2_patch.py` 中处理

## 使用方式

1. `yahoo_provider.py` 初始化时会调用 `patch_yfinance()`
2. 所有数据请求通过 `yahoo_provider.py` 接口发起
3. 实际的网络请求由 `yfinance_http2_patch.py` 处理
4. 代理配置在 `yahoo_provider.py` 中设置并传递给补丁

## 维护注意事项

- 修改反爬虫逻辑时，请在 `yfinance_http2_patch.py` 中进行
- 修改数据获取接口时，请在 `yahoo_provider.py` 中进行
- 避免在两个文件中重复实现相同的逻辑
- 保持清晰的职责边界，便于维护和调试