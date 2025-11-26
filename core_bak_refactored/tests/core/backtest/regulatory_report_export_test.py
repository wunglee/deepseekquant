import unittest
import tempfile
import json
from pathlib import Path

from core_bak_refactored.core.backtest._fragments.event_window_backtester import EventWindowBacktester
from core_bak_refactored.core.data._fragments.historical_data_provider import MockHistoricalDataProvider
from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio_builder import SyntheticPortfolioBuilder
from core_bak_refactored.core.backtest._fragments.stress_test_result import from_backtest_result
from core_bak_refactored.core.backtest._fragments.regulatory_report_exporter import RegulatoryReportExporter


class RegulatoryReportExportTest(unittest.TestCase):
    """
    监管报告格式导出测试（业务目标5）
    
    目标：验证Excel、JSON、PDF三种格式输出功能
    """

    def setUp(self):
        provider = MockHistoricalDataProvider()
        self.backtester = EventWindowBacktester(provider)
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        
        # 运行回测获取结果
        results = self.backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=None,
            benchmark_index='000300.SH',
        )
        self.std_results = [from_backtest_result(r) for r in results[:3]]  # 取3个样本

    def test_export_to_excel(self):
        """测试Excel格式导出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "stress_test_report.xlsx")
            
            # 导出Excel
            RegulatoryReportExporter.to_excel(self.std_results, output_path)
            
            # 验证文件存在
            self.assertTrue(Path(output_path).exists())
            
            # 验证文件大小合理（>1KB）
            file_size = Path(output_path).stat().st_size
            self.assertGreater(file_size, 1024)

    def test_export_to_json(self):
        """测试JSON格式导出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "stress_test_report.json")
            
            # 导出JSON
            RegulatoryReportExporter.to_json(self.std_results, output_path)
            
            # 验证文件存在
            self.assertTrue(Path(output_path).exists())
            
            # 验证JSON结构
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.assertEqual(data['report_type'], 'stress_test')
            self.assertEqual(data['version'], '1.0')
            self.assertEqual(data['total_scenarios'], 3)
            self.assertEqual(len(data['results']), 3)
            
            # 验证必需字段存在
            for result in data['results']:
                self.assertIn('report_id', result)
                self.assertIn('portfolio_id', result)
                self.assertIn('scenario_id', result)
                self.assertIn('metadata', result)

    def test_export_to_pdf_optional(self):
        """测试PDF格式导出（可选功能）"""
        try:
            import reportlab
        except ImportError:
            self.skipTest("reportlab未安装，跳过PDF测试")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "stress_test_report.pdf")
            
            # 导出PDF
            RegulatoryReportExporter.to_pdf(self.std_results, output_path)
            
            # 验证文件存在
            self.assertTrue(Path(output_path).exists())
            
            # 验证PDF文件头
            with open(output_path, 'rb') as f:
                header = f.read(4)
                self.assertEqual(header, b'%PDF')


if __name__ == '__main__':
    unittest.main()
