# Yahoo Finance 速率限制问题解决方案

## 问题现象

```
429 Too Many Requests. Rate limited. Try after a while.
```

即使第一次调用也会出现 429 错误。

## 根本原因

1. **Yahoo Finance 的速率限制非常严格**
   - 免费 API 限制：2000 次/小时
   - 同一 IP 短时间内多次请求会被限流
   - **国内 IP 更容易被限流**（可能被列入黑名单）

2. **HTTP/1.1 vs HTTP/2**
   - Yahoo Finance API 要求 HTTP/2
   - 使用 HTTP/1.1 会被拒绝或限流

3. **代理配置问题**
   - 代理服务未运行
   - 代理端口配置错误
   - 代理本身也被限流

## 解决方案

### 方案 1：使用代理（推荐用于真实项目）

#### 1.1 确保代理服务运行

```bash
# 检查代理端口是否开启
lsof -i :8002

# 常见代理工具
- v2ray: https://www.v2ray.com/
- clash: https://github.com/Dreamacro/clash
- shadowsocks: https://shadowsocks.org/
```

#### 1.2 配置文件

**config/dev/system.yml**:
```yaml
proxies:
  http: http://127.0.0.1:8002  # 替换为你的代理地址
  socks5: socks5://127.0.0.1:1081
```

**config/dev/data.yml**:
```yaml
data_providers:
  yahoo_finance:
    use_proxy: true  # 启用代理
```

#### 1.3 验证代理

```bash
# 运行诊断脚本
python core_bak_refactored/tests/manual/test_yahoo_proxy.py
```

应该看到：
```
✅ 代理连接成功
✅ 代理访问成功
```

### 方案 2：切换到 AKShare（推荐用于国内）

**优势**：
- ✅ 完全免费
- ✅ 无速率限制
- ✅ 国内可直连
- ✅ 支持 A股、港股、美股

**配置**：

**config/dev/data.yml**:
```yaml
market_sources:
  CN: akshare  # 中国市场
  US: akshare  # 美股也可以用 AKShare
  HK: akshare  # 港股

data_providers:
  akshare:
    use_proxy: false  # 国内源不需要代理
```

**安装**：
```bash
pip install akshare
```

**使用**：
```python
from core_bak_refactored.core.data.providers.factory import get_global_factory

factory = get_global_factory()
provider = factory.get('akshare')

# 获取标普500数据（AKShare 支持）
price_data = provider.get_index_prices('SPX', '2020-01-01', '2020-12-31')
```

### 方案 3：等待后重试

如果临时被限流：

```python
# 等待 60 秒后重试
import time
time.sleep(60)

# 或者增加重试次数和间隔
config/dev/data.yml:
max_retries: 5  # 从 3 增加到 5
```

### 方案 4：使用其他数据源

#### Tushare Pro
- 优势：数据质量高，支持 A股和港股
- 劣势：需要注册，免费用户限制 200 次/天
- 网站：https://tushare.pro/

#### Finnhub
- 优势：全球市场，支持新闻和情绪分析
- 劣势：需要 API Key，免费版限制 60 次/分钟
- 网站：https://finnhub.io/

## 诊断工具

运行诊断脚本检查配置：

```bash
python core_bak_refactored/tests/manual/test_yahoo_proxy.py
```

输出示例：
```
╔==========================================================╗
║           Yahoo Finance 代理配置诊断工具                  ║
╚==========================================================╝

步骤 1: 测试代理连接
✅ 代理配置: http://127.0.0.1:8002
✅ 代理连接成功
   出口 IP: xxx.xxx.xxx.xxx

步骤 2: 测试直连 Yahoo Finance API
❌ 429 Too Many Requests（速率限制）
   建议: 使用代理访问

步骤 3: 测试通过代理访问 Yahoo Finance API
✅ 代理访问成功
   获取到 1 个数据集

步骤 4: 测试 YahooFinanceDataProvider
✅ 成功获取数据
   Symbol: ^GSPC
   记录数: 5
```

## 推荐配置

### 开发环境（国内）

```yaml
# config/dev/data.yml
market_sources:
  CN: akshare
  US: akshare  # 或者 yahoo（需要代理）
  HK: akshare

data_providers:
  yahoo_finance:
    use_proxy: true  # 如果使用 Yahoo
  akshare:
    use_proxy: false
```

### 生产环境（海外）

```yaml
# config/prod/data.yml
market_sources:
  CN: yahoo
  US: yahoo
  HK: yahoo

data_providers:
  yahoo_finance:
    use_proxy: false  # 海外环境不需要代理
```

## 常见问题

### Q1: 代理配置了但还是 429？

**A**: 可能原因：
1. 代理服务未运行 - 检查 `lsof -i :8002`
2. 代理 IP 也被限流 - 更换代理节点
3. 代理类型不匹配 - 确保使用 HTTP 代理而不是 SOCKS5

### Q2: 为什么第一次调用就 429？

**A**: Yahoo Finance 对国内 IP 特别严格，即使第一次调用也可能被拒绝。建议：
1. 使用代理
2. 或切换到 AKShare

### Q3: 如何知道代理是否生效？

**A**: 运行诊断脚本：
```bash
python core_bak_refactored/tests/manual/test_yahoo_proxy.py
```

查看输出中的 IP 地址是否是代理的出口 IP。

### Q4: AKShare 和 Yahoo Finance 有什么区别？

**A**:
| 特性 | AKShare | Yahoo Finance |
|------|---------|---------------|
| 速率限制 | 无 | 2000次/小时 |
| 国内访问 | 直连 | 需要代理 |
| 数据覆盖 | 中国为主 | 全球市场 |
| 数据质量 | 高 | 高 |
| 成本 | 免费 | 免费（有限制） |

## 最佳实践

1. **开发环境使用 AKShare** - 无限制，快速开发
2. **生产环境使用混合策略**：
   - 中国市场：AKShare
   - 美国市场：Yahoo Finance（海外服务器）或 AKShare
3. **启用缓存** - 减少 API 调用次数
4. **实现降级策略** - Yahoo 失败时自动切换到 AKShare

## 参考链接

- AKShare 文档: https://akshare.akfamily.xyz/
- Yahoo Finance API: https://query2.finance.yahoo.com/
- httpx HTTP/2 支持: https://www.python-httpx.org/http2/
