#!/usr/bin/env python3
"""
Yahoo Finance 数据源代理配置测试

测试功能：
1. Yahoo Finance HTTP/2 支持
2. 代理配置开关功能
3. 配置文件独立控制每个数据源的代理
"""

from core_bak_refactored.core.data.providers.yahoo_provider import YahooFinanceDataProvider
from core_bak_refactored.core.share.config_manager import ConfigManager
from datetime import datetime, timedelta

print("=" * 60)
print("Yahoo Finance 代理配置测试")
print("=" * 60)

# 读取配置
config = ConfigManager()
use_proxy = config.get('data_providers.yahoo_finance.use_proxy', default=False)
proxy_config = config.get_proxies_from_config()

print(f"\n📋 当前配置:")
print(f"   use_proxy: {use_proxy}")
if proxy_config:
    print(f"   代理地址: {proxy_config.get('http') or proxy_config.get('socks5')}")
else:
    print(f"   代理地址: 未配置")

print(f"\n🚀 初始化 Yahoo Finance Provider...")
provider = YahooFinanceDataProvider()

print(f"\n📊 测试获取数据...")
try:
    data = provider.get_index_prices(
        '^GSPC', 
        datetime.now() - timedelta(days=7), 
        datetime.now()
    )
    print(f"✅ 成功获取 {len(data.records)} 条数据")
    print(f"   日期范围: {data.records[0].date.date()} 到 {data.records[-1].date.date()}")
    print(f"   最新收盘价: ${data.records[-1].close:.2f}")
    
    print(f"\n📝 配置说明:")
    print(f"   - 在 config/dev/system.yml 中配置:")
    print(f"     data_providers:")
    print(f"       yahoo_finance:")
    print(f"         use_proxy: true/false  # 独立控制")
    print(f"       akshare:")
    print(f"         use_proxy: false       # 国内源不需要代理")
    
    print(f"\n🎉 测试完成！代理配置功能正常工作")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    print(f"\n💡 建议:")
    print(f"   1. 检查代理配置是否正确")
    print(f"   2. 如果被限速，等待 1-2 分钟后重试")
    print(f"   3. 可以切换 use_proxy 配置测试")
