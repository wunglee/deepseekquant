"""
市场配置管理器（共享业务模块）

职责：管理不同市场的配置参数（纯配置，无业务逻辑）
定位：业务基础共享
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from core_bak_refactored.core.share.market.market_enums import MarketCode

logger = logging.getLogger(__name__)


class MarketConfigManager:
    """市场配置管理器（共享业务基础）"""
    
    def __init__(self):
        self.market_registry = self._initialize_market_registry()
    
    def _initialize_market_registry(self) -> Dict[str, Dict]:
        """初始化市场基础参数"""
        return {
            MarketCode.CN.value: {
                'name': '中国A股',
                'currency': 'CNY',
                'timezone': 'Asia/Shanghai',
                'trading_hours': '09:30-11:30,13:00-15:00',
                'settlement_days': 1,
                'regulatory_body': 'CSRC',
                'market_cap_category': 'emerging',
                'default_trading_days': 245
            },
            MarketCode.US.value: {
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
            MarketCode.HK.value: {
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
            MarketCode.JP.value: {
                'name': '日本股市',
                'currency': 'JPY',
                'timezone': 'Asia/Tokyo',
                'trading_hours': '09:00-11:30,12:30-15:00',
                'settlement_days': 2,
                'regulatory_body': 'FSA',
                'market_cap_category': 'developed',
                'default_trading_days': 245
            },
            MarketCode.EU.value: {
                'name': '欧洲股市',
                'currency': 'EUR',
                'timezone': 'Europe/Paris',
                'trading_hours': '09:00-17:30',
                'settlement_days': 2,
                'regulatory_body': 'ESMA',
                'market_cap_category': 'developed',
                'default_trading_days': 255
            },
            MarketCode.SG.value: {
                'name': '新加坡股市',
                'currency': 'SGD',
                'timezone': 'Asia/Singapore',
                'trading_hours': '09:00-12:00,13:00-17:00',
                'settlement_days': 2,
                'regulatory_body': 'MAS',
                'market_cap_category': 'developed',
                'default_trading_days': 250
            }
        }
    
    def get_market_info(self, market_code: str) -> Dict[str, Any]:
        """获取市场基本信息"""
        return self.market_registry.get(market_code, {})
    
    def validate_market_config(self, config: Dict) -> List[str]:
        """验证市场配置有效性（使用枚举）"""
        errors = []
        market_type = config.get('market_type', MarketCode.CN.value)
        
        # 使用枚举验证市场代码
        if not MarketCode.is_valid(market_type):
            errors.append(f"不支持的市场类型: {market_type}, 支持: {MarketCode.get_all_codes()}")
        
        market_configs = config.get('market_configs', {})
        if market_type not in market_configs:
            errors.append(f"缺少{market_type}市场的具体配置")
        
        return errors

    def generate_config_template(self, market_type: str) -> Dict[str, Any]:
        """生成配置模板（使用枚举验证）"""
        # 验证市场类型
        if not MarketCode.is_valid(market_type):
            logger.warning(f"不支持的市场类型{market_type}，回退到CN")
            market_type = MarketCode.CN.value
        
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
        """构建市场特定配置（业务层 + 专家建议优化）"""
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
        if market_type == MarketCode.CN.value:
            config.update({
                'has_limit_up_down': True,
                'limit_thresholds': {
                    'main_board': 0.10,
                    'gem': 0.20,
                    'st': 0.05,
                    'kcb': 0.20
                },
                # 专家建议：A股风险参数优化 (第14轮微调)
                'var_method_priority': 'historical_simulation',  # A股更适合历史模拟
                'covariance_lookback': 126,  # 半年度滚动（约6个月交易日）
                'jump_adjustment_coef': 0.035,  # 专家微调: 0.03→0.035 (A股跳跃更显著)
                'max_jump_adjustment': 0.18,  # 专家微调: 0.15→0.18
                'evt_threshold': 0.85,  # 较低阈值适应频繁跳跃
                'limit_adjustment_enabled': True,  # 启用涨跌停调整
                'min_required_returns': 30,  # 最小样本量要求
                'volatility_persistence': 0.94,  # 波动率持续性中等
                'liquidity_risk_weight': 1.2,  # 流动性风险权重较高
                'political_risk_premium': 0.008,  # 政治风险溢价
                # 专家建议：持仓风险参数（第1轮评审 - Almgren et al. 2005）
                'price_impact_alpha': 0.55,  # 散户主导，冲击系数高（范围0.5-0.7）
                'price_impact_beta': 0.52,   # 非线性较弱（范围0.5-0.55）
                'default_spread': 0.0025,    # A股典型spread（0.25%）
                # 专家建议：参与率限制（第2轮评审调整 - 考虑T+1限制）
                'participation_limits': {
                    'NORMAL': 0.08,    # 从10%降至8%（T+1限制）
                    'VOLATILE': 0.05,  # 市场波动加剧
                    'EXTREME': 0.02    # 极端行情
                },
                # 专家建议：市场状态阈值（第2轮评审优化 - 基于滚动分位数校准）
                'state_thresholds': {
                    'normal_vol_max': 1.15,      # 从1.2调低（A股波动容忍度低）
                    'normal_volume_min': 0.75,   # 从0.8调低（成交量波动大）
                    'volatile_vol_max': 1.4,     # 从1.5调低
                    'volatile_volume_min': 0.55  # 从0.6调低
                },
                # 专家建议（第2轮评审 P0）：市值分层参数
                'market_cap_tiers': {
                    'large_cap': {'impact_discount': 0.8, 'liquidity_buffer': 1.2},
                    'mid_cap': {'impact_discount': 1.0, 'liquidity_buffer': 1.0},
                    'small_cap': {'impact_discount': 1.3, 'liquidity_buffer': 0.7}
                },
                # 专家建议（第2轮评审 P0）：日内时段调整
                'intraday_adjustments': {
                    'open_30min': 1.5,    # 开盘30分钟冲击扩大50%
                    'close_30min': 1.3,   # 收盘30分钟冲击扩大30%
                    'lunch_break': 0.8    # 午间休市冲击减少20%
                },
                # 专家建议（第2轮评审 P0重构）：流动性成本折扣配置
                'liquidity_cost_discount': {
                    'cn_t1_single_day': 0.95,        # A股T+1限制，1天折扣
                    'cn_penalty_factor': 0.85,       # A股多日额外惩罚系数
                    'dynamic_bound_increment': 0.05, # 天数增长系数
                    'dynamic_bound_max_increment': 0.3,  # 最大增量
                    'dynamic_bound_cap': 0.8,        # 动态下限上限
                    'base_lower_bounds': {           # 各市场基础下限
                        MarketCode.CN.value: 0.6,
                        MarketCode.US.value: 0.4,
                        MarketCode.HK.value: 0.55,
                        MarketCode.JP.value: 0.5,
                        MarketCode.SG.value: 0.65,
                        MarketCode.EU.value: 0.45
                    },
                    'market_adjustments': {          # 市场类型调整系数
                        MarketCode.US.value: 0.95,
                        MarketCode.HK.value: 0.88,
                        MarketCode.JP.value: 0.92,
                        MarketCode.SG.value: 0.85,
                        MarketCode.EU.value: 0.94
                    },
                    'liquidity_adjustments': {       # 流动性调整系数
                        'top_20%': 0.96,
                        'mid_60%': 0.90,
                        'bottom_20%': 0.82
                    },
                    'simple_thresholds': {           # 简单分类阈值（回退用）
                        'high_liquidity': 10_000_000,
                        'mid_liquidity': 1_000_000
                    },
                    'quantile_min_samples': 100      # 动态分位数最小样本数
                },
                # 涨跌停与停牌（延后至5B后续迭代实施）
                'price_limit': 0.10,
                'halt_risk_factor': 0.15,
                'consecutive_limit_days_p95': 3
            })
            config['volatility_spike_threshold'] = 0.05
            config['limit_hit_ratio_threshold'] = 0.25
            config['major_event_sensitivity'] = 'MEDIUM'
            config['volatility_tiers'] = {
                'NORMAL': {'max': 0.05, 'cache_ttl': 3600},
                'MEDIUM': {'max': 0.075, 'cache_ttl': 1800},
                'HIGH': {'max': 0.10, 'cache_ttl': 600},
                'EXTREME': {'max': float('inf'), 'cache_ttl': 60}
            }
            config['event_weights'] = {
                'circuit_breaker': 0.6,
                'extreme_correlation': 0.2,
                'limit_hits': 0.4,
                'major_event': 0.3
            }
        elif market_type == MarketCode.US.value:
            config.update({
                'has_limit_up_down': False,
                'circuit_breaker_levels': [0.07, 0.13, 0.20],
                'luld_threshold': 0.05,
                'luld_window': 5,
                # 专家建议：美股风险参数优化 (第14轮微调)
                'var_method_priority': 't_distribution',  # 美股适合参数法
                'covariance_lookback': 756,  # 专家微调: 504→756 (3年，美股长记忆性)
                'jump_adjustment_coef': 0.02,  # 跳跃相对较少
                'max_jump_adjustment': 0.12,  # 美股相对稳定
                'evt_threshold': 0.90,  # 标准阈值
                'limit_adjustment_enabled': False,
                'min_required_returns': 50,
                'volatility_persistence': 0.97,  # 高度持续（机构主导）
                'liquidity_risk_weight': 0.85,  # 专家优化: 0.8→0.85 (反映近期流动性变化)
                'political_risk_premium': 0.003,  # 政治风险低
                # 专家建议：持仓风险参数（第1轮评审 - Kissell 2013）
                'price_impact_alpha': 0.25,  # 机构主导，流动性好，冲击低（范围0.2-0.3）
                'price_impact_beta': 0.65,   # 算法交易优化（范围0.6-0.7）
                'default_spread': 0.0015,    # 美股典型spread（0.15%）
                # 专家建议：参与率限制（按市场状态）
                'participation_limits': {
                    'NORMAL': 0.15,    # 美股深度好，允许更高参与率
                    'VOLATILE': 0.08,  # 波动加剧
                    'EXTREME': 0.03    # 极端行情
                },
                # 专家建议：市场状态阈值
                'state_thresholds': {
                    'normal_vol_max': 1.3,      # 美股波动容忍度较高
                    'normal_volume_min': 0.7,
                    'volatile_vol_max': 1.8,
                    'volatile_volume_min': 0.5
                },
                # 无涨跌停
                'price_limit': None,
                'halt_risk_factor': 0.05,
                # 专家建议（第2轮评审 P0）：市值分层参数
                'market_cap_tiers': {
                    'large_cap': {'impact_discount': 0.7, 'liquidity_buffer': 1.3},
                    'mid_cap': {'impact_discount': 1.0, 'liquidity_buffer': 1.0},
                    'small_cap': {'impact_discount': 1.4, 'liquidity_buffer': 0.6}
                },
                # 专家建议（第2轮评审 P0）：日内时段调整
                'intraday_adjustments': {
                    'open_30min': 1.4,    # 开盘30分钟冲击扩大40%
                    'close_30min': 1.2,   # 收盘30分钟冲击扩大20%
                    'after_hours': 0.9    # 盘后交易冲击较低
                },
                # 专家建议（第2轮评审 P0重构）：流动性成本折扣配置
                'liquidity_cost_discount': {
                    # 继承通用配置即可，无T+1限制
                }
            })
            config['volatility_spike_threshold'] = 0.03
            config['limit_hit_ratio_threshold'] = 0.20
            config['major_event_sensitivity'] = 'HIGH'
            config['volatility_tiers'] = {
                'NORMAL': {'max': 0.05, 'cache_ttl': 3600},
                'MEDIUM': {'max': 0.075, 'cache_ttl': 1800},
                'HIGH': {'max': 0.10, 'cache_ttl': 600},
                'EXTREME': {'max': float('inf'), 'cache_ttl': 60}
            }
            config['event_weights'] = {
                'circuit_breaker': 0.4,
                'extreme_correlation': 0.3,
                'limit_hits': 0.2,
                'major_event': 0.5
            }
        elif market_type == MarketCode.HK.value:
            config.update({
                'has_limit_up_down': False,
                # 专家建议：港股风险参数优化 (第14轮微调)
                'var_method_priority': 'evt',  # 港股极端风险更多
                'covariance_lookback': 378,  # 专家微调: 252→378 (1.5年，受多重因素影响)
                'jump_adjustment_coef': 0.030,  # 专家微调: 0.025→0.030 (介于A股和美股之间)
                'max_jump_adjustment': 0.15,  # 港股特殊调整
                'evt_threshold': 0.86,  # 专家微调: 0.88→0.86 (尾部风险显著)
                'limit_adjustment_enabled': True,  # 港股部分板块有限制
                'min_required_returns': 50,
                'volatility_persistence': 0.92,  # 资金流动影响波动率持续性
                'liquidity_risk_weight': 1.1,  # 受资金流动影响
                'political_risk_premium': 0.015,  # 地缘政治风险高
                # 专家建议：持仓风险参数（第2轮评审微调 - Chan & Lakonishok 1995）
                'price_impact_alpha': 0.42,  # 国际资金影响，从0.45调至0.42
                'price_impact_beta': 0.58,   # 国际资金流动影响（范围0.55-0.6）
                'default_spread': 0.0020,    # 港股典型spread（0.20%）
                # 专家建议：参与率限制（按市场状态）
                'participation_limits': {
                    'NORMAL': 0.12,    # 混合市场
                    'VOLATILE': 0.06,  # 波动加剧
                    'EXTREME': 0.025   # 极端行情
                },
                # 专家建议：市场状态阈值
                'state_thresholds': {
                    'normal_vol_max': 1.25,     # 港股介于中美之间
                    'normal_volume_min': 0.75,
                    'volatile_vol_max': 1.6,
                    'volatile_volume_min': 0.55
                },
                # 无涨跌停但有限制
                'price_limit': None,
                'halt_risk_factor': 0.10,
                # 专家建议（第2轮评审 P0）：市值分层参数
                'market_cap_tiers': {
                    'large_cap': {'impact_discount': 0.75, 'liquidity_buffer': 1.25},
                    'mid_cap': {'impact_discount': 1.0, 'liquidity_buffer': 1.0},
                    'small_cap': {'impact_discount': 1.35, 'liquidity_buffer': 0.65}
                },
                # 专家建议（第2轮评审 P0）：日内时段调整
                'intraday_adjustments': {
                    'open_30min': 1.45,   # 开盘30分钟冲击扩大45%
                    'close_30min': 1.25,  # 收盘30分钟冲击扩大25%
                    'lunch_break': 0.85   # 午间休市冲击较低
                }
            })
            config['volatility_spike_threshold'] = 0.04
            config['limit_hit_ratio_threshold'] = 0.25
            config['major_event_sensitivity'] = 'HIGH'
            config['volatility_tiers'] = {
                'NORMAL': {'max': 0.05, 'cache_ttl': 3600},
                'MEDIUM': {'max': 0.075, 'cache_ttl': 1800},
                'HIGH': {'max': 0.10, 'cache_ttl': 600},
                'EXTREME': {'max': float('inf'), 'cache_ttl': 60}
            }
        elif market_type == MarketCode.JP.value:
            config.update({
                'has_limit_up_down': False,
                # 专家建议：日本市场参数配置 (第14轮补充)
                'var_method_priority': 't_distribution',  # 货币政策主导，收益率分布相对稳定
                'covariance_lookback': 504,  # 2年（日本央行政策持续性强）
                'jump_adjustment_coef': 0.022,  # 通缩环境跳跃较小
                'max_jump_adjustment': 0.15,  # 黑田经济学期间有政策跳跃
                'evt_threshold': 0.88,  # 尾部风险中等
                'limit_adjustment_enabled': False,
                'min_required_returns': 60,  # 通缩环境需要更多样本
                'volatility_persistence': 0.95,  # 货币政策主导，波动率高度持续
                'liquidity_risk_weight': 0.9,  # 流动性充足
                'political_risk_premium': 0.005,  # 政治风险中等
                'deflation_risk_adjustment': 0.01,  # 通缩风险调整系数
                # 专家建议：持仓风险参数（第1轮评审 - Iihara et al. 2001）
                'price_impact_alpha': 0.35,  # 机构主导，流动性集中（范围0.3-0.4）
                'price_impact_beta': 0.62,   # 交叉持股影响（范围0.6-0.65）
                'default_spread': 0.0018,    # 日股典型spread（0.18%）
                # 专家建议：参与率限制（按市场状态）
                'participation_limits': {
                    'NORMAL': 0.13,    # 机构主导
                    'VOLATILE': 0.07,  # 波动加剧
                    'EXTREME': 0.03    # 极端行情
                },
                # 专家建议：市场状态阈值
                'state_thresholds': {
                    'normal_vol_max': 1.25,
                    'normal_volume_min': 0.75,
                    'volatile_vol_max': 1.6,
                    'volatile_volume_min': 0.55
                },
                # 无涨跌停
                'price_limit': None,
                'halt_risk_factor': 0.06,
                # 专家建议（第2轮评审 P0）：市值分层参数
                'market_cap_tiers': {
                    'large_cap': {'impact_discount': 0.75, 'liquidity_buffer': 1.2},
                    'mid_cap': {'impact_discount': 1.0, 'liquidity_buffer': 1.0},
                    'small_cap': {'impact_discount': 1.3, 'liquidity_buffer': 0.7}
                },
                # 专家建议（第2轮评审 P0）：日内时段调整
                'intraday_adjustments': {
                    'open_30min': 1.4,    # 开盘30分钟冲击扩大40%
                    'close_30min': 1.25,  # 收盘30分钟冲击扩大25%
                    'lunch_break': 0.85   # 午间休市冲击较低
                }
            })
            config['volatility_spike_threshold'] = 0.04
            config['limit_hit_ratio_threshold'] = 0.25
            config['major_event_sensitivity'] = 'MEDIUM'
            config['volatility_tiers'] = {
                'NORMAL': {'max': 0.05, 'cache_ttl': 3600},
                'MEDIUM': {'max': 0.075, 'cache_ttl': 1800},
                'HIGH': {'max': 0.10, 'cache_ttl': 600},
                'EXTREME': {'max': float('inf'), 'cache_ttl': 60}
            }
        elif market_type == MarketCode.EU.value:
            config.update({
                'has_limit_up_down': False,
                # 专家建议：欧洲市场参数配置 (第14轮补充 + 第15轮优化)
                'var_method_priority': 'historical_simulation',  # 政治事件驱动性强
                'covariance_lookback': 252,  # 1年（政治周期短，政治不确定性高）
                'jump_adjustment_coef': 0.025,  # 政治事件引发跳跃
                'max_jump_adjustment': 0.15,  # 政治事件风险
                'evt_threshold': 0.87,  # 政治尾部风险
                'limit_adjustment_enabled': False,
                'min_required_returns': 45,  # 政治事件影响估计
                'volatility_persistence': 0.90,  # 政治事件降低持续性
                'liquidity_risk_weight': 1.0,  # 跨国流动性差异
                'political_risk_premium': 0.010,  # 欧盟政治风险
                'brexit_risk_weight': self._get_brexit_risk_weight(),  # 动态衰减机制
                'banking_sector_risk': 0.008,  # 银行体系风险溢价
                # 专家建议：持仓风险参数（第1轮评审 - Huang & Stoll 1997）
                'price_impact_alpha': 0.30,  # 市场分散但深度好（范围0.25-0.35）
                'price_impact_beta': 0.63,   # MiFID II改善透明度（范围0.6-0.65）
                'default_spread': 0.0016,    # 欧洲典型spread（0.16%）
                # 专家建议：参与率限制（按市场状态）
                'participation_limits': {
                    'NORMAL': 0.14,    # 市场深度好
                    'VOLATILE': 0.07,  # 波动加剧
                    'EXTREME': 0.03    # 极端行情
                },
                # 专家建议：市场状态阈值
                'state_thresholds': {
                    'normal_vol_max': 1.25,
                    'normal_volume_min': 0.75,
                    'volatile_vol_max': 1.65,
                    'volatile_volume_min': 0.55
                },
                # 无涨跌停
                'price_limit': None,
                'halt_risk_factor': 0.07,
                # 专家建议（第2轮评审 P0）：市值分层参数
                'market_cap_tiers': {
                    'large_cap': {'impact_discount': 0.7, 'liquidity_buffer': 1.25},
                    'mid_cap': {'impact_discount': 1.0, 'liquidity_buffer': 1.0},
                    'small_cap': {'impact_discount': 1.35, 'liquidity_buffer': 0.65}
                },
                # 专家建议（第2轮评审 P0）：日内时段调整
                'intraday_adjustments': {
                    'open_30min': 1.35,   # 开盘30分钟冲击扩大35%
                    'close_30min': 1.2,   # 收盘30分钟冲击扩大20%
                    'midday': 0.9         # 中午时段冲击较低
                }
            })
            config['volatility_spike_threshold'] = 0.035
            config['limit_hit_ratio_threshold'] = 0.20
            config['major_event_sensitivity'] = 'HIGH'
            config['volatility_tiers'] = {
                'NORMAL': {'max': 0.05, 'cache_ttl': 3600},
                'MEDIUM': {'max': 0.075, 'cache_ttl': 1800},
                'HIGH': {'max': 0.10, 'cache_ttl': 600},
                'EXTREME': {'max': float('inf'), 'cache_ttl': 60}
            }
        elif market_type == MarketCode.SG.value:
            config.update({
                'has_limit_up_down': False,
                # 专家建议：新加坡市场参数配置 (第14轮补充)
                'var_method_priority': 'evt',  # 小型开放经济体对全球冲击敏感
                'covariance_lookback': 189,  # 9个月（全球资本流动快速变化）
                'jump_adjustment_coef': 0.028,  # 全球资本流动引发跳跃
                'max_jump_adjustment': 0.15,  # 外部冲击风险
                'evt_threshold': 0.84,  # 较低阈值（外部冲击敏感）
                'limit_adjustment_enabled': False,
                'min_required_returns': 40,  # 市场规模小但数据质量高
                'volatility_persistence': 0.88,  # 外部依赖性强，持续性较低
                'liquidity_risk_weight': 1.25,  # 专家优化: 1.3→1.25 (反映良好市场基础设施)
                'political_risk_premium': 0.006,  # 政治稳定但外部依赖
                'trade_openness_risk': 0.012,  # 贸易开放度风险溢价（贸易依存度300%+）
                'currency_risk_weight': 1.25,  # 汇率政策风险
                # 专家建议：持仓风险参数（第1轮评审 - Biais et al. 1995）
                'price_impact_alpha': 0.50,  # 市场规模小，国际资金敏感（范围0.45-0.55）
                'price_impact_beta': 0.54,   # 外部冲击影响（范围0.5-0.55）
                'default_spread': 0.0028,    # 新加坡典型spread（0.28%）
                # 专家建议：参与率限制（按市场状态）
                'participation_limits': {
                    'NORMAL': 0.10,    # 小市场限制较严
                    'VOLATILE': 0.05,  # 波动加剧
                    'EXTREME': 0.02    # 极端行情
                },
                # 专家建议：市场状态阈值
                'state_thresholds': {
                    'normal_vol_max': 1.3,
                    'normal_volume_min': 0.7,
                    'volatile_vol_max': 1.7,
                    'volatile_volume_min': 0.5
                },
                # 无涨跌停
                'price_limit': None,
                'halt_risk_factor': 0.08,
                # 专家建议（第2轮评审 P0）：市值分层参数
                'market_cap_tiers': {
                    'large_cap': {'impact_discount': 0.75, 'liquidity_buffer': 1.15},
                    'mid_cap': {'impact_discount': 1.0, 'liquidity_buffer': 1.0},
                    'small_cap': {'impact_discount': 1.4, 'liquidity_buffer': 0.6}
                },
                # 专家建议（第2轮评审 P0）：日内时段调整
                'intraday_adjustments': {
                    'open_30min': 1.5,    # 开盘30分钟冲击扩大50%
                    'close_30min': 1.3,   # 收盘30分钟冲击扩大30%
                    'lunch_break': 0.8    # 午间休市冲击减少20%
                }
            })
            config['volatility_spike_threshold'] = 0.05
            config['limit_hit_ratio_threshold'] = 0.30
            config['major_event_sensitivity'] = 'MEDIUM'
            config['volatility_tiers'] = {
                'NORMAL': {'max': 0.05, 'cache_ttl': 3600},
                'MEDIUM': {'max': 0.075, 'cache_ttl': 1800},
                'HIGH': {'max': 0.10, 'cache_ttl': 600},
                'EXTREME': {'max': float('inf'), 'cache_ttl': 60}
            }
        else:
            # 其他未配置市场使用默认参数
            config.update({
                'has_limit_up_down': False,
                'var_method_priority': 'normal',
                'covariance_lookback': 252,
                'jump_adjustment_coef': 0.02,
                'max_jump_adjustment': 0.15,
                'evt_threshold': 0.90,
                'limit_adjustment_enabled': False,
                'min_required_returns': 50,
                'volatility_persistence': 0.92,
                'liquidity_risk_weight': 1.0,
                'political_risk_premium': 0.008
            })
        
        return config

    def _get_brexit_risk_weight(self, as_of_date: Optional[datetime] = None) -> float:
        """
        计算动态Brexit风险权重（专家建议第15轮）
        
        Parameters:
        as_of_date: 计算日期（默认为当前日期）
        
        Returns:
        float: 动态调整后的风险权重（1.0-1.15）
        
        衰减机制:
        - Brexit正式生效: 2020-01-31
        - 基准权重: 1.15
        - 衰减率: 每年5% (0.95)
        - 下限: 1.0 (中性水平)
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # Brexit正式生效日期: 2020-01-31
        brexit_date = datetime(2020, 1, 31)
        years_since_brexit = (as_of_date - brexit_date).days / 365.25
        
        # 每年衰减5%
        base_weight = 1.15
        decay_rate = 0.95
        adjusted_weight = base_weight * (decay_rate ** years_since_brexit)
        
        # 设置下限1.0（不会低于中性水平）
        return max(adjusted_weight, 1.0)
    
    def _get_default_trading_hours(self, market_type: str) -> Dict[str, str]:
        """获取默认交易时间"""
        trading_hours_map = {
            MarketCode.CN.value: {'regular': '09:30-11:30,13:00-15:00', 'pre_market': '', 'after_hours': ''},
            MarketCode.US.value: {'regular': '09:30-16:00', 'pre_market': '04:00-09:30', 'after_hours': '16:00-20:00'},
            MarketCode.HK.value: {'regular': '09:30-12:00,13:00-16:00', 'pre_market': '09:00-09:30', 'after_hours': ''},
            MarketCode.JP.value: {'regular': '09:00-11:30,12:30-15:00', 'pre_market': '', 'after_hours': ''},
            MarketCode.EU.value: {'regular': '09:00-17:30', 'pre_market': '08:00-09:00', 'after_hours': ''},
            MarketCode.SG.value: {'regular': '09:00-12:00,13:00-17:00', 'pre_market': '', 'after_hours': ''}
        }
        return trading_hours_map.get(market_type, {'regular': '09:30-16:00'})

    def _get_default_risk_premium(self, market_type: str) -> float:
        """获取默认风险溢价（业务参数 + 专家建议第14轮补充）"""
        premium_map = {
            MarketCode.CN.value: 0.015,  # 新兴市场溢价
            MarketCode.US.value: 0.010,  # 成熟市场基准
            MarketCode.HK.value: 0.020,  # 地缘政治溢价
            MarketCode.JP.value: 0.008,  # 通缩环境溢价较低
            MarketCode.EU.value: 0.012,  # 政治风险溢价（专家微调: 0.009→0.012）
            MarketCode.SG.value: 0.014   # 小型开放经济体溢价（专家微调: 0.012→0.014）
        }
        return premium_map.get(market_type, 0.01)
    
    def _get_default_risk_free_rate(self, market_type: str) -> float:
        """获取默认无风险利率（业务参数）"""
        rate_map = {
            MarketCode.CN.value: 0.03,
            MarketCode.US.value: 0.045,
            MarketCode.HK.value: 0.035,
            MarketCode.JP.value: 0.005,
            MarketCode.EU.value: 0.025,
            MarketCode.SG.value: 0.030
        }
        return rate_map.get(market_type, 0.03)
