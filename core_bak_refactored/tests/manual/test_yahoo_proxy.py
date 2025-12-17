"""
Yahoo Finance 代理配置测试脚本

用途：
1. 验证代理服务是否运行
2. 测试 Yahoo Finance API 是否可通过代理访问
3. 诊断 429 (Too Many Requests) 问题

运行方式：
    python core_bak_refactored/tests/manual/test_yahoo_proxy.py
"""

import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_proxy_connection():
    """测试代理连接"""
    print("\n" + "=" * 60)
    print("步骤 1: 测试代理连接")
    print("=" * 60)
    
    from core_bak_refactored.core.share.config_manager import ConfigManager
    
    config_manager = ConfigManager()
    proxy_config = config_manager.get_proxies_from_config()
    
    if not proxy_config:
        print("❌ 未配置代理")
        print("   请在 config/dev/system.yml 中配置 proxies.http")
        return False
    
    proxy_url = proxy_config.get('http')
    print(f"✅ 代理配置: {proxy_url}")
    
    # 测试代理
    try:
        import httpx
        print(f"   正在测试代理连接...")
        client = httpx.Client(proxy=proxy_url, timeout=10.0)
        try:
            response = client.get('https://httpbin.org/ip')
            if response.status_code == 200:
                ip_info = response.json()
                print(f"✅ 代理连接成功")
                print(f"   出口 IP: {ip_info.get('origin', 'unknown')}")
                return True
            else:
                print(f"❌ 代理返回错误状态码: {response.status_code}")
                return False
        finally:
            client.close()
    except ImportError:
        print("❌ httpx 未安装")
        print("   运行: pip install 'httpx[http2]'")
        return False
    except Exception as e:
        print(f"❌ 代理连接失败: {e}")
        print(f"   建议: 检查代理服务是否运行 (如 v2ray, clash)")
        return False


def test_yahoo_api_direct():
    """测试直连 Yahoo API"""
    print("\n" + "=" * 60)
    print("步骤 2: 测试直连 Yahoo Finance API")
    print("=" * 60)
    
    try:
        import httpx
        print("   正在直连访问 Yahoo Finance...")
        client = httpx.Client(http2=True, timeout=10.0)
        try:
            response = client.get('https://query2.finance.yahoo.com/v8/finance/chart/^GSPC?interval=1d&range=5d')
            print(f"   HTTP 状态码: {response.status_code}")
            print(f"   HTTP 版本: {response.http_version}")
            
            if response.status_code == 200:
                print("✅ 直连成功（你的网络环境可以访问 Yahoo Finance）")
                return True
            elif response.status_code == 429:
                print("❌ 429 Too Many Requests（速率限制）")
                print("   建议: 使用代理访问")
                return False
            else:
                print(f"⚠️ 返回状态码 {response.status_code}")
                return False
        finally:
            client.close()
    except ImportError:
        print("❌ httpx 未安装")
        return False
    except Exception as e:
        print(f"❌ 直连失败: {e}")
        return False


def test_yahoo_api_via_proxy():
    """测试通过代理访问 Yahoo API"""
    print("\n" + "=" * 60)
    print("步骤 3: 测试通过代理访问 Yahoo Finance API")
    print("=" * 60)
    
    from core_bak_refactored.core.share.config_manager import ConfigManager
    
    config_manager = ConfigManager()
    proxy_config = config_manager.get_proxies_from_config()
    
    if not proxy_config:
        print("⏭️ 跳过（未配置代理）")
        return False
    
    proxy_url = proxy_config.get('http')
    
    try:
        import httpx
        print(f"   正在通过代理 {proxy_url} 访问 Yahoo Finance...")
        client = httpx.Client(http2=True, proxy=proxy_url, timeout=10.0)
        try:
            response = client.get('https://query2.finance.yahoo.com/v8/finance/chart/^GSPC?interval=1d&range=5d')
            print(f"   HTTP 状态码: {response.status_code}")
            print(f"   HTTP 版本: {response.http_version}")
            
            if response.status_code == 200:
                print("✅ 代理访问成功")
                data = response.json()
                if 'chart' in data and 'result' in data['chart']:
                    result_count = len(data['chart']['result'])
                    print(f"   获取到 {result_count} 个数据集")
                return True
            elif response.status_code == 429:
                print("❌ 429 Too Many Requests（即使通过代理也被限流）")
                print("   可能原因: 1) 代理 IP 被限流  2) 需要更换代理")
                return False
            else:
                print(f"⚠️ 返回状态码 {response.status_code}")
                return False
        finally:
            client.close()
    except Exception as e:
        print(f"❌ 代理访问失败: {e}")
        return False


def test_yfinance_provider():
    """测试 Yahoo Finance Provider"""
    print("\n" + "=" * 60)
    print("步骤 4: 测试 YahooFinanceDataProvider")
    print("=" * 60)
    
    try:
        from core_bak_refactored.core.data.providers.yahoo_provider import YahooFinanceDataProvider
        
        print("   正在初始化 YahooFinanceDataProvider...")
        provider = YahooFinanceDataProvider()
        
        print(f"   代理配置: {provider.proxy or '无（直连）'}")
        
        print("   正在获取 ^GSPC 数据 (最近 5 天)...")
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        
        price_data = provider.get_index_prices(
            '^GSPC',
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            datetime.now()
        )
        
        print(f"✅ 成功获取数据")
        print(f"   Symbol: {price_data.symbol}")
        print(f"   记录数: {len(price_data.records)}")
        print(f"   时间范围: {price_data.start_date} 到 {price_data.end_date}")
        
        return True
        
    except Exception as e:
        print(f"❌ Provider 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " Yahoo Finance 代理配置诊断工具 ".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = []
    
    # 测试 1: 代理连接
    results.append(("代理连接", test_proxy_connection()))
    
    # 测试 2: 直连 Yahoo API
    results.append(("直连 Yahoo API", test_yahoo_api_direct()))
    
    # 测试 3: 代理访问 Yahoo API
    results.append(("代理访问 Yahoo API", test_yahoo_api_via_proxy()))
    
    # 测试 4: Provider 集成测试
    results.append(("Provider 集成测试", test_yfinance_provider()))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:.<30} {status}")
    
    # 诊断建议
    print("\n" + "=" * 60)
    print("诊断建议")
    print("=" * 60)
    
    proxy_ok = results[0][1]
    direct_ok = results[1][1]
    proxy_api_ok = results[2][1]
    provider_ok = results[3][1]
    
    if provider_ok:
        print("✅ 一切正常！Yahoo Finance Provider 工作正常")
    elif not proxy_ok:
        print("⚠️ 代理未配置或不可用")
        print("   1. 检查代理服务是否运行 (如 v2ray, clash)")
        print("   2. 检查 config/dev/system.yml 中的代理地址")
        print("   3. 或在 config/dev/data_provider.yml 中设置 yahoo_finance.use_proxy: false")
    elif not proxy_api_ok:
        print("⚠️ 代理可用但无法访问 Yahoo Finance")
        print("   1. 代理可能被 Yahoo 限流")
        print("   2. 尝试更换代理节点")
        print("   3. 等待一段时间后重试")
    elif direct_ok and not provider_ok:
        print("⚠️ 直连可用但 Provider 失败")
        print("   1. 检查 yfinance 是否正确安装")
        print("   2. 检查 httpx[http2] 是否安装")
        print("   3. 查看上面的错误日志")
    else:
        print("⚠️ 网络环境问题")
        print("   1. 国内访问 Yahoo Finance 通常需要代理")
        print("   2. 配置代理后重新运行此脚本")
    
    print()


if __name__ == '__main__':
    main()
