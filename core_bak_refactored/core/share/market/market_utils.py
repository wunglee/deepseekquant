"""
市场工具类（领域层共享）

职责：
- 提供市场识别、推断等基础功能
- 支持从 symbol/index_id 推断市场类型
- 可被所有层（应用层、领域层）复用
"""

from typing import Optional
from core_bak_refactored.core.share.market.market_enums import MarketCode


class MarketUtils:
    """市场工具类
    
    提供市场相关的通用工具方法
    """
    
    @staticmethod
    def infer_market_from_symbol(symbol: str) -> MarketCode:
        """从股票/指数代码推断市场类型
        
        Args:
            symbol: 股票/指数代码（如 '000300.SH', '^GSPC', 'HSI'）
        
        Returns:
            MarketCode: 推断出的市场代码枚举
        
        Examples:
            >>> MarketUtils.infer_market_from_symbol('000300.SH')
            <MarketCode.CN: 'CN'>
            >>> MarketUtils.infer_market_from_symbol('^GSPC')
            <MarketCode.US: 'US'>
            >>> MarketUtils.infer_market_from_symbol('HSI')
            <MarketCode.HK: 'HK'>
            >>> MarketUtils.infer_market_from_symbol('0700.HK')
            <MarketCode.HK: 'HK'>
            >>> MarketUtils.infer_market_from_symbol('N225')
            <MarketCode.JP: 'JP'>
        
        规则：
            - A股市场：.SH（上海）、.SZ（深圳）、.CN
            - 港股市场：.HK、.HKG、HSI（恒生指数）
            - 美股市场：^开头（如 ^GSPC、^DJI、^IXIC）
            - 日本市场：N225（日经指数）
            - 欧洲市场：.EU
            - 新加坡：.SG
            - 默认：MarketCode.CN
        """
        if not symbol:
            return MarketCode.CN
        
        symbol_upper = symbol.upper()
        
        # A股市场（上海/深圳）
        if any(symbol_upper.endswith(suffix) for suffix in ['.SH', '.SZ', '.CN']):
            return MarketCode.CN
        
        # 港股市场
        if any(symbol_upper.endswith(suffix) for suffix in ['.HK', '.HKG']) or symbol_upper == 'HSI':
            return MarketCode.HK
        
        # 美股市场（^GSPC, ^DJI, ^IXIC 等）
        if symbol_upper.startswith('^'):
            return MarketCode.US
        
        # 日本市场
        if symbol_upper == 'N225':
            return MarketCode.JP
        
        # 欧洲市场
        if symbol_upper.endswith('.EU'):
            return MarketCode.EU
        
        # 新加坡市场
        if symbol_upper.endswith('.SG'):
            return MarketCode.SG
        
        # 美股市场（.US 后缀）
        if symbol_upper.endswith('.US'):
            return MarketCode.US
        
        # 默认为 A股市场
        return MarketCode.CN
    
    @staticmethod
    def infer_market_from_metadata(metadata: dict) -> Optional[MarketCode]:
        """从元数据中提取市场类型
        
        Args:
            metadata: 元数据字典（可能包含 'market_type' 或 'market' 字段）
        
        Returns:
            MarketCode: 提取出的市场代码枚举，如果无法提取则返回 None
        
        Examples:
            >>> MarketUtils.infer_market_from_metadata({'market_type': 'CN'})
            <MarketCode.CN: 'CN'>
            >>> MarketUtils.infer_market_from_metadata({'market': MarketCode.US})
            <MarketCode.US: 'US'>
            >>> MarketUtils.infer_market_from_metadata({'other': 'data'})
            None
        """
        if not metadata:
            return None
        
        # 尝试从 market_type 字段提取
        market_type = metadata.get('market_type')
        if market_type:
            if isinstance(market_type, MarketCode):
                return market_type
            if isinstance(market_type, str) and MarketCode.is_valid(market_type.upper()):
                return MarketCode(market_type.upper())
        
        # 尝试从 market 字段提取
        market = metadata.get('market')
        if market:
            if isinstance(market, MarketCode):
                return market
            if isinstance(market, str) and MarketCode.is_valid(market.upper()):
                return MarketCode(market.upper())
        
        return None
    
    @staticmethod
    def detect_market_with_fallback(symbol: str = None, metadata: dict = None) -> MarketCode:
        """综合检测市场类型（优先元数据，其次 symbol 启发式）
        
        Args:
            symbol: 股票/指数代码
            metadata: 元数据字典
        
        Returns:
            MarketCode: 检测出的市场代码枚举
        
        Examples:
            >>> # 优先使用元数据
            >>> MarketUtils.detect_market_with_fallback(
            ...     symbol='000300.SH',
            ...     metadata={'market_type': 'US'}
            ... )
            <MarketCode.US: 'US'>
            
            >>> # 元数据缺失时使用 symbol
            >>> MarketUtils.detect_market_with_fallback(symbol='000300.SH')
            <MarketCode.CN: 'CN'>
            
            >>> # 都缺失时返回默认值
            >>> MarketUtils.detect_market_with_fallback()
            <MarketCode.CN: 'CN'>
        """
        # 1. 优先使用元数据
        if metadata:
            market = MarketUtils.infer_market_from_metadata(metadata)
            if market:
                return market
        
        # 2. 其次使用 symbol 启发式推断
        if symbol:
            return MarketUtils.infer_market_from_symbol(symbol)
        
        # 3. 默认为 A股市场
        return MarketCode.CN
    
    @staticmethod
    def is_trading_day_aware(market: MarketCode) -> bool:
        """判断市场是否需要交易日判断
        
        Args:
            market: 市场代码
        
        Returns:
            bool: 是否需要交易日判断
        
        Note:
            某些市场（如虚拟货币）是 7x24 交易，不需要交易日判断
        """
        # 所有实体股票市场都需要交易日判断
        return market in [
            MarketCode.CN,
            MarketCode.US,
            MarketCode.HK,
            MarketCode.JP,
            MarketCode.EU,
            MarketCode.SG
        ]
