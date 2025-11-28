"""统一监管报告导出测试文件
合并了：
- regulatory_report_exporter_test.py（基础导出测试）
- regulatory_report_export_test.py（Excel/JSON/PDF导出测试）
- regulatory_report_completeness_test.py（字段完整性测试）
"""

import unittest
import json
import tempfile
from pathlib import Path

from core_bak_refactored.core.backtest._fragments.event_window_backtester import EventWindowBacktester
from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider
from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio_builder import SyntheticPortfolioBuilder
from core_bak_refactored.core.backtest._fragments.regulatory_report_exporter import RegulatoryReportExporter
from core_bak_refactored.core.backtest._fragments.stress_test_result import StressTestResult, from_backtest_result


class RegulatoryReportExporterBasicTest(unittest.TestCase):
    """基础导出功能测试"""
    
    def setUp(self):
        self.results = [
            StressTestResult(
                report_id='r1',
                portfolio_id='P1',
                scenario_id='S1',
                stress_loss_percentage=0.12,
                metadata={
                    'event_name': 'Event1',
                    'period': ('2020-01-01', '2020-02-01'),
                    'predicted_loss': 0.10,
                    'actual_loss': 0.12,
                    'prediction_error': 0.02,
                    'benchmark_index': '000300.SH'
                }
            )
        ]

    def test_export_to_json(self):
        out = Path('reg_export_test.json')
        try:
            RegulatoryReportExporter.to_json(self.results, str(out))
            self.assertTrue(out.exists())
            data = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(data.get('total_scenarios'), 1)
            self.assertIsInstance(data.get('results'), list)
        finally:
            if out.exists():
                out.unlink()

    def test_export_to_excel(self):
        out = Path('reg_export_test.xlsx')
        try:
            RegulatoryReportExporter.to_excel(self.results, str(out))
            self.assertTrue(out.exists())
            import openpyxl
            wb = openpyxl.load_workbook(str(out))
            self.assertIn('压力测试报告', wb.sheetnames)
        finally:
            if out.exists():
                out.unlink()


class RegulatoryReportExportFormatsTest(unittest.TestCase):
    """多格式导出测试（业务目标5）"""

    def setUp(self):
        provider = MockHistoricalDataProvider()
        self.backtester = EventWindowBacktester(provider)
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        
        results = self.backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=None,
            benchmark_index='000300.SH',
        )
        self.std_results = [from_backtest_result(r) for r in results[:3]]

    def test_export_to_excel_with_backtest_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "stress_test_report.xlsx")
            RegulatoryReportExporter.to_excel(self.std_results, output_path)
            
            self.assertTrue(Path(output_path).exists())
            file_size = Path(output_path).stat().st_size
            self.assertGreater(file_size, 1024)

    def test_export_to_json_with_backtest_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "stress_test_report.json")
            RegulatoryReportExporter.to_json(self.std_results, output_path)
            
            self.assertTrue(Path(output_path).exists())
            
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.assertEqual(data['report_type'], 'stress_test')
            self.assertEqual(data['version'], '1.0')
            self.assertEqual(data['total_scenarios'], 3)
            self.assertEqual(len(data['results']), 3)
            
            for result in data['results']:
                self.assertIn('report_id', result)
                self.assertIn('portfolio_id', result)
                self.assertIn('scenario_id', result)
                self.assertIn('metadata', result)

    def test_export_to_pdf_optional(self):
        try:
            import reportlab
        except ImportError:
            self.skipTest("reportlab未安装，跳过PDF测试")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "stress_test_report.pdf")
            RegulatoryReportExporter.to_pdf(self.std_results, output_path)
            
            self.assertTrue(Path(output_path).exists())
            
            with open(output_path, 'rb') as f:
                header = f.read(4)
                self.assertEqual(header, b'%PDF')


class RegulatoryReportCompletenessTest(unittest.TestCase):
    """字段完整性测试（业务目标5）"""

    def setUp(self):
        provider = MockHistoricalDataProvider()
        self.backtester = EventWindowBacktester(provider)
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()

    def test_regulatory_report_completeness(self):
        results = self.backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=None,
            benchmark_index='000300.SH',
        )
        self.assertGreaterEqual(len(results), 5)

        std_results = [from_backtest_result(r) for r in results]
        
        required_top = [
            'report_id',
            'portfolio_id',
            'scenario_id',
            'var_normal',
            'var_stressed',
            'stress_loss_amount',
            'stress_loss_percentage',
            'recovery_period',
            'risk_decomposition',
            'triggered_actions',
            'recommended_actions',
            'compliance_status',
            'metadata',
        ]
        required_meta = [
            'event_name',
            'period',
            'predicted_loss',
            'actual_loss',
            'prediction_error',
            'benchmark_index',
        ]
        TOTAL_REQUIRED = len(required_top) + len(required_meta)

        for s in std_results:
            d = s.to_dict()
            present = 0
            
            for k in required_top:
                if k in d:
                    present += 1
            
            meta = d.get('metadata', {}) or {}
            for mk in required_meta:
                if mk in meta and meta[mk] is not None:
                    present += 1
            
            completeness_ratio = present / TOTAL_REQUIRED
            self.assertGreaterEqual(
                completeness_ratio,
                0.95,
                msg=f"字段完整性不足：{present}/{TOTAL_REQUIRED} ({completeness_ratio:.1%})",
            )


if __name__ == '__main__':
    unittest.main()
