#!/usr/bin/env python3
"""
测试 Web API 调用链路

模拟从 Web 页面 /providers 点击测试按钮的完整调用链路
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider

def main():
    print("=" * 60)
    print("测试 Web API 调用链路")
    print("=" * 60)
    print()
    
    provider_id = 'yahoo'
    
    print(f"步骤 1: 模拟 Web API 调用 /api/v1/providers/{provider_id}/test")
    print("-" * 60)
    
    # 这里调用的是 BaseDataProvider.test_provider（静态方法）
    # 相当于 API 端点 test_provider_connection 的逻辑
    result = BaseDataProvider.test_provider(provider_id, credential='')
    
    print()
    print("测试结果:")
    print("-" * 60)
    print(f"状态: {result.get('status')}")
    print(f"测试结果: {result.get('test_result')}")
    print(f"可用性: {result.get('available')}")
    print(f"消息: {result.get('message')}")
    
    if 'details' in result:
        print(f"\n详细信息:")
        for key, value in result['details'].items():
            print(f"  {key}: {value}")
    
    print()
    print("=" * 60)
    if result.get('status') == 'success':
        print("✅ 测试通过")
    else:
        print("❌ 测试失败")
    print("=" * 60)

if __name__ == '__main__':
    main()
