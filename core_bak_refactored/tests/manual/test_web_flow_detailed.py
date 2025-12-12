#!/usr/bin/env python3
"""
详细的 Web 调用链路追踪

模拟从 Web 页面点击测试按钮的完整调用流程，带详细的链路追踪日志
"""
import sys
import os
import logging
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def print_section(title, level=1):
    """打印分节标题"""
    if level == 1:
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)
    else:
        print("\n" + "-" * 80)
        print(f"  {title}")
        print("-" * 80)

def print_step(step_num, description):
    """打印步骤"""
    print(f"\n📍 步骤 {step_num}: {description}")
    print("   " + "·" * 70)

def main():
    print_section("🔍 Web 调用链路完整追踪", level=1)
    
    start_time = time.time()
    
    # ============================================================
    # 1. 模拟前端请求
    # ============================================================
    print_step(1, "前端：用户点击测试按钮")
    print("   触发事件: testProvider('yahoo')")
    print("   发送请求: POST /api/v1/providers/yahoo/test")
    print("   请求体: { credentials: {}, proxy: {} }")
    
    # ============================================================
    # 2. API 层处理
    # ============================================================
    print_step(2, "API 层：路由到测试端点")
    print("   路由: @app.route('/api/v1/providers/<provider_id>/test')")
    print("   函数: test_provider_connection(provider_id='yahoo')")
    
    # ============================================================
    # 3. 获取 Provider Factory
    # ============================================================
    print_step(3, "领域层：获取 Provider Factory")
    from core_bak_refactored.core.data.providers.factory import get_global_factory
    
    factory = get_global_factory()
    print(f"   ✅ Factory 实例: {factory}")
    
    # 获取已注册的 Providers
    registered_providers = []
    for provider_name in ['mock', 'yahoo', 'akshare', 'tushare']:
        if factory.is_registered(provider_name):
            registered_providers.append(provider_name)
    print(f"   已注册 Providers: {registered_providers}")
    
    # ============================================================
    # 4. 获取 Yahoo Provider 实例
    # ============================================================
    print_step(4, "领域层：创建 Yahoo Provider 实例")
    
    try:
        provider = factory.get('yahoo')
        print(f"   ✅ Provider 类型: {type(provider).__name__}")
        print(f"   代理配置: {getattr(provider, 'proxy', 'None')}")
    except Exception as e:
        print(f"   ❌ 获取 Provider 失败: {e}")
        return
    
    # ============================================================
    # 5. 调用静态方法 test_provider
    # ============================================================
    print_step(5, "领域层：调用 test_provider 静态方法")
    print("   方法: BaseDataProvider.test_provider()")
    print("   参数: provider_id='yahoo', credential=''")
    
    from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider
    
    test_start = time.time()
    result = BaseDataProvider.test_provider('yahoo', credential='')
    test_duration = time.time() - test_start
    
    # ============================================================
    # 6. 测试结果分析
    # ============================================================
    print_step(6, "结果分析")
    
    print(f"\n   测试耗时: {test_duration:.2f} 秒")
    print(f"   状态: {result.get('status')}")
    print(f"   测试结果: {result.get('test_result')}")
    print(f"   可用性: {result.get('available')}")
    print(f"   消息: {result.get('message')}")
    
    if 'details' in result:
        print(f"\n   详细信息:")
        for key, value in result['details'].items():
            print(f"     • {key}: {value}")
    
    # ============================================================
    # 7. 调用链路总结
    # ============================================================
    print_section("📊 调用链路总结", level=1)
    
    total_duration = time.time() - start_time
    
    print("""
   调用链路流程：
   
   1️⃣  Web 前端 (providers.html)
      ↓ onclick="testProvider('yahoo')"
      
   2️⃣  JavaScript 函数
      ↓ fetch('/api/v1/providers/yahoo/test', {method: 'POST'})
      
   3️⃣  Flask API 端点 (api_service.py)
      ↓ @app.route('/api/v1/providers/<provider_id>/test')
      ↓ test_provider_connection(provider_id)
      
   4️⃣  Provider Factory
      ↓ get_global_factory().get('yahoo')
      
   5️⃣  Yahoo Provider 类 (yahoo_provider.py)
      ↓ BaseDataProvider.test_provider(provider_id, credential)
      
   6️⃣  HTTP/2 补丁层 (yfinance_http2_patch.py)
      ↓ patch_yfinance(proxy_url)
      ↓ patched_get() - User-Agent 轮换
      
   7️⃣  yfinance 库
      ↓ Ticker(symbol).history(start, end)
      
   8️⃣  Yahoo Finance API
      ↓ GET https://query2.finance.yahoo.com/v8/finance/chart/...
   """)
    
    print(f"\n   总耗时: {total_duration:.2f} 秒")
    print(f"   最终结果: {'✅ 成功' if result.get('status') == 'success' else '❌ 失败'}")
    
    # ============================================================
    # 8. 性能分析
    # ============================================================
    if result.get('status') == 'success':
        print_section("⚡ 性能分析", level=1)
        
        latency_ms = result.get('details', {}).get('latency_ms', 0)
        data_count = result.get('details', {}).get('data_count', 0)
        
        print(f"\n   API 响应时间: {latency_ms} ms ({latency_ms/1000:.2f} 秒)")
        print(f"   数据量: {data_count} 条记录")
        
        if latency_ms > 10000:
            print(f"\n   ⚠️  响应时间较长 (>{latency_ms/1000:.0f}秒)")
            print(f"   原因分析:")
            print(f"     • 可能经历了多次重试（429 错误）")
            print(f"     • 指数退避策略：5s → 10s → 20s → 40s")
            print(f"   优化建议:")
            print(f"     • 启用代理以避免 IP 限流")
            print(f"     • 或切换到 AKShare（无限流）")
        else:
            print(f"\n   ✅ 响应时间正常")
    
    # ============================================================
    # 9. 代理状态检查
    # ============================================================
    print_section("🔌 代理状态检查", level=1)
    
    from core_bak_refactored.core.share.config_manager import ConfigManager
    
    config_manager = ConfigManager()
    use_proxy = config_manager.get('data.data_providers.yahoo_finance.use_proxy', default=False)
    proxy_config = config_manager.get_proxies_from_config()
    
    print(f"\n   配置的代理开关: {use_proxy}")
    print(f"   配置的代理地址: {proxy_config}")
    print(f"   实际使用代理: {getattr(provider, 'proxy', 'None')}")
    
    if use_proxy and getattr(provider, 'proxy', None):
        print(f"\n   ✅ 代理已启用并生效")
    elif use_proxy and not getattr(provider, 'proxy', None):
        print(f"\n   ⚠️  代理配置了但未生效（代理服务可能未运行）")
    elif not use_proxy:
        print(f"\n   ℹ️  未启用代理（直连模式）")
    
    # ============================================================
    # 10. 最终建议
    # ============================================================
    print_section("💡 建议", level=1)
    
    if result.get('status') == 'success':
        print("""
   ✅ 测试通过！但有以下建议：
   
   1. 如果响应时间较长（>10秒）：
      • 考虑启用代理以避免 IP 限流
      • 或切换到 AKShare（无限流，响应更快）
   
   2. 如果配置了代理但未生效：
      • 检查代理服务是否运行：curl -x http://127.0.0.1:8002 https://httpbin.org/ip
      • 启动代理：open -a V2rayU 或 open -a "ClashX Pro"
   
   3. 生产环境建议：
      • 使用 AKShare（更稳定、无限流）
      • 或配置企业级代理池
        """)
    else:
        print("""
   ❌ 测试失败！建议：
   
   1. 立即方案：启用代理
      • 启动代理服务：open -a V2rayU
      • 验证代理：python core_bak_refactored/tests/manual/test_yahoo_proxy.py
   
   2. 简单方案：切换到 AKShare
      • 修改 config/dev/data.yml
      • market_sources.US: akshare
        """)
    
    print("\n" + "=" * 80 + "\n")

if __name__ == '__main__':
    main()
