"""
市场配置管理器

职责：管理不同市场的配置参数（纯配置，无业务逻辑）
定位：业务层配置管理
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class MarketConfigManager:
    """市场配置管理器（业务层）"""
    
    def __init__(self):
        self.market_registry = self._initialize_market_registry()
    
    def _initialize_market_registry(self) -> Dict[str, Dict]:
        """初始化市场基础参数"""
        return {
            'CN': {
                'name': '中国A股',
                'currency': 'CNY',
                'timezone': 'Asia/Shanghai',
                'trading_hours': '09:30-11:30,13:00-15:00',
                'settlement_days': 1,
                'regulatory_body': 'CSRC',
                'market_cap_category': 'emerging',
                'default_trading_days': 245
            },
            'US': {
                'name': '美国股市',
                'currency': 'USD', 
                'timezone': 'America/New_York',
                'trading_hours': '09:30-16:00',
                'after_hours': '16:00-20:00',
                'pre_market': '04:00-09:30',
                'settlement_days': 2,
                'regulatory_body': 'SEC',
                'market_cap_category': 'developed',
                'default_trading_days': 252
            },
            'HK': {
                'name': '香港股市',
                'currency': 'HKD',
                'timezone': 'Asia/Hong_Kong', 
                'trading_hours': '09:30-12:00,13:00-16:00',
                'pre_market': '09:00-09:30',
                'settlement_days': 2,
                'regulatory_body': 'SFC',
                'market_cap_category': 'developed',
                'default_trading_days': 247
            },
            'JP': {
                'name': '日本股市',
                'currency': 'JPY',
                'timezone': 'Asia/Tokyo',
                'trading_hours': '09:00-11:30,12:30-15:00',
                'settlement_days': 2,
                'regulatory_body': 'FSA',
                'market_cap_category': 'developed',
                'default_trading_days': 245
            },
            'EU': {
                'name': '欧洲股市',
                'currency': 'EUR',
                'timezone': 'Europe/Paris',
                'trading_hours': '09:00-17:30',
                'settlement_days': 2,
                'regulatory_body': 'ESMA',
                'market_cap_category': 'developed',
                'default_trading_days': 255
            }
        }
    
    def get_market_info(self, market_code: str) -> Dict[str, Any]:
        """获取市场基本信息"""
        return self.market_registry.get(market_code, {})
    
    def validate_market_config(self, config: Dict) -> List[str]:
        """验证市场配置有效性"""
        errors = []
        market_type = config.get('market_type', 'CN')
        
        if market_type not in self.market_registry:
            errors.append(f"不支持的市场类型: {market_type}")
        
        market_configs = config.get('market_configs', {})
        if market_type not in market_configs:
            errors.append(f"缺少{market_type}市场的具体配置")
        
        return errors

    def generate_config_template(self, market_type: str) -> Dict[str, Any]:
        """生成配置模板"""
        if market_type not in self.market_registry:
            logger.warning(f"不支持的市场类型{market_type}，回退到CN")
            market_type = 'CN'
        
        market_info = self.market_registry[market_type]
        
        # 业务参数配置
        base_template = {
            'market_type': market_type,
            'trading_days_per_year': market_info.get('default_trading_days', 252),
            'market_configs': {
                market_type: self._build_market_specific_config(market_type, market_info)
            },
            'confidence_levels': {
                'daily_monitoring': 0.95,
                'risk_limit': 0.99,
                'regulatory_reporting': 0.99
            },
            'dynamic_risk_free_rate': None,
            'log_level': 'INFO',
            'performance_monitoring': {
                'enable_calculation_timing': True,
                'enable_memory_monitoring': False,
                'sample_size_warning_threshold': 50
            }
        }
        
        return base_template
    
    def _build_market_specific_config(self, market_type: str, market_info: Dict) -> Dict[str, Any]:
        """构建市场特定配置（业务层）"""
        config = {
            'trading_days': market_info.get('default_trading_days', 252),
            'risk_free_rate': self._get_default_risk_free_rate(market_type),
            'trading_hours': self._get_default_trading_hours(market_type),
            'risk_premium_base': self._get_default_risk_premium(market_type),
            'anomaly_detection_enabled': True,
            'conservative_adjustment': True,
            'volatility_scaling': True
        }
        
        # 市场特定机制配置
        if market_type == 'CN':
            config.update({
                'has_limit_up_down': True,
                'limit_thresholds': {
                    'main_board': 0.10,
                    'gem': 0.20,
                    'st': 0.05,
                    'kcb': 0.20
                }
            })
        elif market_type == 'US':
            config.update({
                'has_limit_up_down': False,
                'circuit_breaker_levels': [0.07, 0.13, 0.20],
                'luld_threshold': 0.05,
                'luld_window': 5
            })
        else:
            config.update({
                'has_limit_up_down': False
            })
        
        return config

    def _get_default_trading_hours(self, market_type: str) -> Dict[str, str]:
        """获取默认交易时间"""
        trading_hours_map = {
            'CN': {'regular': '09:30-11:30,13:00-15:00', 'pre_market': '', 'after_hours': ''},
            'US': {'regular': '09:30-16:00', 'pre_market': '04:00-09:30', 'after_hours': '16:00-20:00'},
            'HK': {'regular': '09:30-12:00,13:00-16:00', 'pre_market': '09:00-09:30', 'after_hours': ''},
            'JP': {'regular': '09:00-11:30,12:30-15:00', 'pre_market': '', 'after_hours': ''},
            'EU': {'regular': '09:00-17:30', 'pre_market': '08:00-09:00', 'after_hours': ''}
        }
        return trading_hours_map.get(market_type, {'regular': '09:30-16:00'})

    def _get_default_risk_premium(self, market_type: str) -> float:
        """获取默认风险溢价（业务参数）"""
        premium_map = {
            'CN': 0.015,
            'US': 0.010,
            'HK': 0.020,
            'JP': 0.008,
            'EU': 0.009
        }
        return premium_map.get(market_type, 0.01)
    
    def _get_default_risk_free_rate(self, market_type: str) -> float:
        """获取默认无风险利率（业务参数）"""
        rate_map = {
            'CN': 0.03,
            'US': 0.045,
            'HK': 0.035,
            'JP': 0.005,
            'EU': 0.025
        }
        return rate_map.get(market_type, 0.03)
