"""基本面数据服务 - 获取公司财务、估值、比率等基本面数据"""
from typing import Any, Dict
import logging
from datetime import datetime


class FundamentalDataService:
    """基本面数据服务（职责单一：公司基本面分析）"""

    def __init__(self) -> None:
        self.logger = logging.getLogger('DeepSeekQuant.FundamentalDataService')

    async def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """获取基本面数据（多数据源尝试）"""
        try:
            fundamentals: Dict[str, Any] = {}
            # 尝试Yahoo Finance
            try:
                yf_fundamentals = await self._get_yahoo_fundamentals(symbol)
                fundamentals.update(yf_fundamentals)
            except Exception as e:
                self.logger.debug(f"Yahoo Finance基本面数据获取失败: {e}")
            # 尝试Alpha Vantage
            if not fundamentals:
                try:
                    av_fundamentals = await self._get_alpha_vantage_fundamentals(symbol)
                    fundamentals.update(av_fundamentals)
                except Exception as e:
                    self.logger.debug(f"Alpha Vantage基本面数据获取失败: {e}")
            # 尝试其他数据源
            if not fundamentals:
                try:
                    other_fundamentals = await self._get_other_fundamentals(symbol)
                    fundamentals.update(other_fundamentals)
                except Exception as e:
                    self.logger.debug(f"其他基本面数据源获取失败: {e}")
            if not fundamentals:
                raise ValueError("无法获取基本面数据")
            # 计算衍生指标
            fundamentals.update(self._calculate_fundamental_ratios(fundamentals))
            return fundamentals
        except Exception as e:
            self.logger.error(f"获取 {symbol} 基本面数据失败: {e}")
            return {'error': str(e)}

    async def _get_yahoo_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """从Yahoo Finance获取基本面数据"""
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow
        return {
            'company_name': info.get('longName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'trailing_pe': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSales'),
            'dividend_yield': info.get('dividendYield'),
            'profit_margins': info.get('profitMargins'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'return_on_equity': info.get('returnOnEquity'),
            'return_on_assets': info.get('returnOnAssets'),
            'beta': info.get('beta'),
            '52_week_high': info.get('fiftyTwoWeekHigh'),
            '52_week_low': info.get('fiftyTwoWeekLow'),
            'analyst_recommendation': info.get('recommendationKey'),
            'number_of_analysts': info.get('numberOfAnalystOpinions'),
            'target_price': info.get('targetMeanPrice'),
            'total_revenue': financials.loc['Total Revenue'].iloc[0] if not financials.empty else None,
            'net_income': financials.loc['Net Income'].iloc[0] if not financials.empty else None,
            'total_assets': balance_sheet.loc['Total Assets'].iloc[0] if not balance_sheet.empty else None,
            'total_liabilities': balance_sheet.loc['Total Liabilities'].iloc[0] if not balance_sheet.empty else None,
            'operating_cash_flow': cash_flow.loc['Operating Cash Flow'].iloc[0] if not cash_flow.empty else None,
            'free_cash_flow': cash_flow.loc['Free Cash Flow'].iloc[0] if not cash_flow.empty else None,
            'data_source': 'yahoo',
            'last_updated': datetime.now().isoformat()
        }

    async def _get_alpha_vantage_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """从Alpha Vantage获取基本面数据（占位）"""
        return {}

    async def _get_other_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """从其他数据源获取基本面数据（占位）"""
        return {}

    def _calculate_fundamental_ratios(self, fundamentals: Dict) -> Dict[str, Any]:
        """计算基本面比率指标"""
        ratios: Dict[str, Any] = {}
        # 估值比率
        if fundamentals.get('market_cap') and fundamentals.get('total_revenue'):
            ratios['ev_to_sales'] = fundamentals.get('enterprise_value', 0) / fundamentals['total_revenue']
        if fundamentals.get('enterprise_value') and fundamentals.get('ebitda'):
            ratios['ev_to_ebitda'] = fundamentals['enterprise_value'] / fundamentals['ebitda']
        # 盈利能力比率
        if fundamentals.get('net_income') and fundamentals.get('total_assets'):
            ratios['return_on_assets'] = fundamentals['net_income'] / fundamentals['total_assets']
        if fundamentals.get('net_income') and fundamentals.get('shareholder_equity'):
            ratios['return_on_equity'] = fundamentals['net_income'] / fundamentals['shareholder_equity']
        # 财务健康比率
        if fundamentals.get('total_debt') and fundamentals.get('shareholder_equity'):
            ratios['debt_to_equity'] = fundamentals['total_debt'] / fundamentals['shareholder_equity']
        if fundamentals.get('operating_cash_flow') and fundamentals.get('total_debt'):
            ratios['cash_flow_to_debt'] = fundamentals['operating_cash_flow'] / fundamentals['total_debt']
        # 增长比率
        if fundamentals.get('revenue_growth'):
            ratios['revenue_growth_3y'] = fundamentals['revenue_growth']
        if fundamentals.get('eps_growth'):
            ratios['eps_growth_3y'] = fundamentals['eps_growth']
        # 效率比率
        if fundamentals.get('total_revenue') and fundamentals.get('total_assets'):
            ratios['asset_turnover'] = fundamentals['total_revenue'] / fundamentals['total_assets']
        return ratios
