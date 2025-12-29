#!/usr/bin/env python3
"""
完整回归测试 - 验证所有重构功能
"""

import sys
import os
import traceback

# 测试结果统计
results = {'passed': 0, 'failed': 0, 'tests': []}

def test(name, condition, error_msg=None):
    """记录测试结果"""
    if condition:
        results['passed'] += 1
        print(f'✅ {name}')
        results['tests'].append((name, True, None))
    else:
        results['failed'] += 1
        print(f'❌ {name}')
        if error_msg:
            print(f'   错误: {error_msg}')
        results['tests'].append((name, False, error_msg))
    return condition

print('='*60)
print('完整回归测试开始')
print('='*60)

# ==================== 测试1: ConfigManager 迁移 ====================
print('\n[测试组1] ConfigManager 迁移到 core/share')
try:
    from core_bak_refactored.core.share.config_manager import ConfigManager
    test('ConfigManager 从 core.share 导入', True)
    
    cm = ConfigManager(environment='dev')
    test('ConfigManager 实例化', True)
    
    # 测试单例模式
    cm2 = ConfigManager(environment='dev')
    test('ConfigManager 单例模式', cm is cm2)
    
    # 测试配置加载
    config_keys = list(cm._config.keys())
    test('配置文件加载', len(config_keys) > 0, f'配置键: {config_keys[:3]}')
    
    # 测试必要的配置文件
    for key in ['data_provider', 'market', 'system', 'cache']:
        test(f'  - {key} 配置加载', key in config_keys)
    
except Exception as e:
    test('ConfigManager 导入和初始化', False, str(e))
    print(f'详细错误: {traceback.format_exc()}')

# ==================== 测试2: update() 方法 ====================
print('\n[测试组2] ConfigManager.update() 方法')
try:
    cm = ConfigManager(environment='dev')
    
    # 测试 update 方法存在
    test('update() 方法存在', hasattr(cm, 'update'))
    
    # 测试 update 功能
    cm.update({'test_key': {'nested': 'value'}})
    result = cm.get('test_key.nested')
    test('update() 方法功能', result == 'value')
    
except Exception as e:
    test('update() 方法', False, str(e))

# ==================== 测试3: ProvidersConfig ====================
print('\n[测试组3] ProvidersConfig 重构')
try:
    cm = ConfigManager(environment='dev')
    pc = cm.get_provider_config()
    
    # 测试必要字段
    test('ProvidersConfig.default_index', hasattr(pc, 'default_index'))
    test('ProvidersConfig.cache_ttl', hasattr(pc, 'cache_ttl'))
    test('ProvidersConfig.providers', hasattr(pc, 'providers'))
    
    # 测试 data_providers 已移除
    import inspect
    fields = [f.name for f in pc.__dataclass_fields__.values()]
    test('data_providers 字段已移除', 'data_providers' not in fields)
    
    # 测试 use_proxy 融合到 providers
    test('providers 数量正确', len(pc.providers) >= 5)
    
    # 测试 get_provider_proxy_config 方法
    test('get_provider_proxy_config 方法存在', 
         hasattr(pc, 'get_provider_proxy_config'))
    
    # 测试代理配置
    finnhub_proxy = pc.get_provider_proxy_config('finnhub')
    test('finnhub use_proxy=True', finnhub_proxy == True)
    
    akshare_proxy = pc.get_provider_proxy_config('akshare')
    test('akshare use_proxy=False', akshare_proxy == False)
    
except Exception as e:
    test('ProvidersConfig', False, str(e))
    print(f'详细错误: {traceback.format_exc()}')

# ==================== 测试4: MarketConfig ====================
print('\n[测试组4] MarketConfig')
try:
    cm = ConfigManager(environment='dev')
    mc = cm.get_market_config()
    
    test('MarketConfig 加载', mc is not None)
    test('market_sources 存在', hasattr(mc, 'market_sources'))
    test('market_registry 存在', hasattr(mc, 'market_registry'))
    
    # 测试市场数量
    test('market_sources 数量', len(mc.market_sources) >= 6)
    test('market_registry 数量', len(mc.market_registry) >= 6)
    
except Exception as e:
    test('MarketConfig', False, str(e))

# ==================== 测试5: market_display 整合 ====================
print('\n[测试组5] market_display 整合到 market_registry')
try:
    import yaml
    
    # 检查配置文件
    with open('core_bak_refactored/config/dev/market.yml', 'r', encoding='utf-8') as f:
        market_data = yaml.safe_load(f)
    
    # market_display 应该不存在
    test('market_display 已移除', 'market_display' not in market_data)
    
    # market_registry 应包含 UI 字段
    market_registry = market_data.get('market_registry', {})
    cn_market = market_registry.get('CN', {})
    test('market_registry 包含 display_name', 'display_name' in cn_market)
    test('market_registry 包含 icon', 'icon' in cn_market)
    
    # 验证所有市场都有 UI 字段
    all_have_ui = all(
        'display_name' in info and 'icon' in info 
        for info in market_registry.values()
    )
    test('所有市场都有 UI 字段', all_have_ui)
    
except Exception as e:
    test('market_display 整合', False, str(e))

# ==================== 测试6: api_service.py 修复 ====================
print('\n[测试组6] api_service.py 修复')
try:
    # 检查语法
    import ast
    with open('core_bak_refactored/app/quality_monitoring/api_service.py', 'r') as f:
        code = f.read()
    
    ast.parse(code)
    test('api_service.py 语法正确', True)
    
    # 检查 _create_provider_instance 方法
    test('_create_provider_instance 方法存在', 
         'def _create_provider_instance' in code)
    
    # 检查 socketio_request
    test('socketio_request 导入正确', 
         'from flask_socketio import SocketIO, emit, request as socketio_request' in code)
    test('socketio_request.sid 使用正确', 
         'socketio_request.sid' in code)
    
    # 检查硬编码已移除
    test('硬编码市场列表已移除', 
         "{'code': 'CN', 'name': '中国 A股'" not in code)
    
    # 检查从配置读取
    test('从 market_registry 读取市场', 
         'market_registry' in code and 'display_name' in code)
    
except Exception as e:
    test('api_service.py', False, str(e))

# ==================== 测试7: DataProviderFactory ====================
print('\n[测试组7] DataProviderFactory')
try:
    from core_bak_refactored.core.data.providers.factory import DataProviderFactory, get_global_factory
    
    test('DataProviderFactory 导入', True)
    
    factory = DataProviderFactory()
    test('DataProviderFactory 实例化', True)
    
    # 测试全局工厂
    global_factory = get_global_factory()
    test('get_global_factory', global_factory is not None)
    
    # 测试列出 providers
    providers = factory.list_providers()
    test('list_providers', len(providers) > 0, f'providers: {providers[:3]}')
    
    # 测试检查 provider
    test('is_registered(akshare)', factory.is_registered('akshare'))
    
except Exception as e:
    test('DataProviderFactory', False, str(e))
    print(f'详细错误: {traceback.format_exc()}')

# ==================== 测试8: 配置文件完整性 ====================
print('\n[测试组8] 配置文件完整性')
try:
    import yaml
    
    # data_provider.yml
    with open('core_bak_refactored/config/dev/data_provider.yml', 'r') as f:
        dp_config = yaml.safe_load(f)
    
    test('data_provider.yml 加载', dp_config is not None)
    test('providers 列表存在', 'providers' in dp_config)
    
    # 检查每个 provider 都有 use_proxy
    providers = dp_config.get('providers', [])
    all_have_proxy = all('use_proxy' in p for p in providers)
    test('所有 provider 都有 use_proxy', all_have_proxy)
    
    # market.yml
    with open('core_bak_refactored/config/dev/market.yml', 'r') as f:
        market_config = yaml.safe_load(f)
    
    test('market.yml 加载', market_config is not None)
    test('market_sources 存在', 'market_sources' in market_config)
    test('market_registry 存在', 'market_registry' in market_config)
    
except Exception as e:
    test('配置文件完整性', False, str(e))

# ==================== 测试9: 路径计算修复 ====================
print('\n[测试组9] ConfigManager 路径计算')
try:
    cm = ConfigManager(environment='dev')
    
    # 测试 get_config_path
    market_path = cm.get_config_path('market')
    test('get_config_path("market")', market_path.endswith('market.yml'))
    
    # 验证文件存在
    test('配置文件存在', os.path.exists(market_path))
    
    # 测试不带后缀
    path1 = cm.get_config_path('system')
    test('自动添加 .yml 后缀', path1.endswith('system.yml'))
    
except Exception as e:
    test('路径计算', False, str(e))

# ==================== 测试10: 集成测试 ====================
print('\n[测试组10] 集成测试')
try:
    # 测试完整的配置链路
    cm = ConfigManager(environment='dev')
    
    # 获取所有配置
    monitoring = cm.get_monitoring_config()
    test('MonitoringConfig', monitoring is not None)
    
    system = cm.get_system_config()
    test('SystemConfig', system is not None)
    
    cache = cm.get_cache_config()
    test('CacheConfig', cache is not None)
    
    market = cm.get_market_config()
    test('MarketConfig', market is not None)
    
    provider = cm.get_provider_config()
    test('ProvidersConfig', provider is not None)
    
    # 测试配置互操作
    market_sources = market.market_sources
    test('market_sources 可访问', market_sources is not None)
    
    providers_list = provider.providers
    test('providers 列表可访问', providers_list is not None)
    
except Exception as e:
    test('集成测试', False, str(e))
    print(f'详细错误: {traceback.format_exc()}')

# ==================== 总结 ====================
print('\n' + '='*60)
print('回归测试总结')
print('='*60)

total = results['passed'] + results['failed']
print(f'总测试数: {total}')
print(f'✅ 通过: {results["passed"]}')
print(f'❌ 失败: {results["failed"]}')
print(f'通过率: {results["passed"]/total*100:.1f}%')

# 列出失败的测试
if results['failed'] > 0:
    print('\n失败的测试:')
    for name, passed, error in results['tests']:
        if not passed:
            print(f'  ❌ {name}')
            if error:
                print(f'     错误: {error}')

print('\n' + '='*60)
if results['failed'] == 0:
    print('🎉 所有回归测试通过！')
    print('='*60)
    sys.exit(0)
else:
    print(f'⚠️  有 {results["failed"]} 个测试失败')
    print('='*60)
    sys.exit(1)
