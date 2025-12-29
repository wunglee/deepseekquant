"""
分时数据API手动测试脚本

用途：启动API服务器并测试分时数据端点

运行方式：
    python -m core_bak_refactored.tests.manual.test_intraday_api_manual
    
然后访问：
    http://localhost:5000/api/v1/intraday/data?symbol=000300.SH
"""

import logging

import pandas as pd
import requests

from core_bak_refactored.app.quality_monitoring.api_service import DataQualityAPIService
from core_bak_refactored.app.quality_monitoring.monitoring_service import QualityMonitoringService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_api_call():
    """测试API调用"""
    url = "http://localhost:5000/api/v1/intraday/data"
    
    # 测试用例1: 沪深300
    params = {
        'symbol': '000300.SH',
        'trade_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }
    
    logger.info(f"发送请求: {url}?{params}")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"响应状态: {data['status']}")
        
        if data['status'] == 'success':
            intraday = data['data']
            logger.info(f"✅ 成功获取分时数据:")
            logger.info(f"  - 代码: {intraday['symbol']}")
            logger.info(f"  - 名称: {intraday['name']}")
            logger.info(f"  - 当前价: {intraday['current_price']}")
            logger.info(f"  - 昨收价: {intraday['yesterday_close']}")
            logger.info(f"  - 涨跌额: {intraday['change']}")
            logger.info(f"  - 涨跌幅: {intraday['change_percent']}%")
            logger.info(f"  - 分时点数: {len(intraday['times'])}")
            logger.info(f"  - 买盘档位: {len(intraday['order_book']['bids'])}")
            logger.info(f"  - 卖盘档位: {len(intraday['order_book']['asks'])}")
            logger.info(f"  - 成交明细: {len(intraday['trade_records'])}")
            
            # 打印部分数据示例
            if intraday['times']:
                logger.info(f"\n前5个分时点:")
                for i in range(min(5, len(intraday['times']))):
                    logger.info(f"  {intraday['times'][i]}: 价格={intraday['prices'][i]}, 成交量={intraday['volumes'][i]}")
            
            # 打印盘口数据
            logger.info(f"\n买一: 价格={intraday['order_book']['bids'][0]['price']}, 量={intraday['order_book']['bids'][0]['volume']}")
            logger.info(f"卖一: 价格={intraday['order_book']['asks'][0]['price']}, 量={intraday['order_book']['asks'][0]['volume']}")
            
            return True
        else:
            logger.error(f"❌ 请求失败: {data.get('message')}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ 网络请求失败: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return False


def start_server():
    """启动API服务器"""
    logger.info("正在启动API服务器...")
    
    # 创建监控服务
    monitoring_service = QualityMonitoringService()
    
    # 创建API服务
    api_service = DataQualityAPIService(monitoring_service)
    
    # 启动服务器
    logger.info("API服务器已启动，访问地址: http://localhost:5000")
    logger.info("分时数据端点: http://localhost:5000/api/v1/intraday/data?symbol=000300.SH")
    logger.info("\n按 Ctrl+C 停止服务器\n")
    
    # 使用socketio启动（支持实时推送）
    api_service.socketio.run(
        api_service.app,
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False
    )


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # 仅测试API调用
        logger.info("测试模式：调用API端点")
        success = test_api_call()
        sys.exit(0 if success else 1)
    else:
        # 启动服务器
        start_server()
