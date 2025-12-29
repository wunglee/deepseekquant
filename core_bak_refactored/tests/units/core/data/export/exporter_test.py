import json
import tempfile
from pathlib import Path

import pytest

from core_bak_refactored.core.data.export.exporter import DataExporter


class TestDataExporter:
    """测试数据导出器。"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_data(self):
        """示例数据。"""
        return [
            {'symbol': 'AAPL', 'price': 150.0, 'volume': 1000},
            {'symbol': 'GOOGL', 'price': 140.0, 'volume': 2000}
        ]

    def test_init(self, temp_dir):
        """测试初始化。"""
        exporter = DataExporter(temp_dir)
        assert exporter.output_dir == Path(temp_dir)

    def test_export_to_csv(self, temp_dir, sample_data):
        """测试导出CSV。"""
        exporter = DataExporter(temp_dir)
        result = exporter.export_to_csv(sample_data, 'test.csv')
        
        assert result is True
        assert (Path(temp_dir) / 'test.csv').exists()

    def test_export_to_csv_empty(self, temp_dir):
        """测试导出空数据。"""
        exporter = DataExporter(temp_dir)
        result = exporter.export_to_csv([], 'test.csv')
        
        assert result is False

    def test_export_to_json(self, temp_dir, sample_data):
        """测试导出JSON。"""
        exporter = DataExporter(temp_dir)
        result = exporter.export_to_json(sample_data, 'test.json')
        
        assert result is True
        
        # 验证JSON内容
        with open(Path(temp_dir) / 'test.json', 'r') as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[0]['symbol'] == 'AAPL'

    def test_export_batch_to_csv(self, temp_dir, sample_data):
        """测试批量导出CSV。"""
        exporter = DataExporter(temp_dir)
        
        data_dict = {
            'aapl': sample_data[:1],
            'googl': sample_data[1:]
        }
        
        count = exporter.export_batch_to_csv(data_dict, 'batch')
        
        assert count == 2
        assert (Path(temp_dir) / 'batch_aapl.csv').exists()
        assert (Path(temp_dir) / 'batch_googl.csv').exists()

    def test_export_with_metadata(self, temp_dir, sample_data):
        """测试导出带元数据。"""
        exporter = DataExporter(temp_dir)
        
        metadata = {
            'source': 'test',
            'timestamp': '2024-01-01',
            'count': len(sample_data)
        }
        
        result = exporter.export_with_metadata(sample_data, 'test_meta.json', metadata)
        
        assert result is True
        
        # 验证元数据
        with open(Path(temp_dir) / 'test_meta.json', 'r') as f:
            data = json.load(f)
            assert 'metadata' in data
            assert 'data' in data
            assert data['metadata']['source'] == 'test'

    def test_get_export_summary(self, temp_dir, sample_data):
        """测试获取导出摘要。"""
        exporter = DataExporter(temp_dir)
        
        exporter.export_to_csv(sample_data, 'test1.csv')
        exporter.export_to_json(sample_data, 'test2.json')
        
        summary = exporter.get_export_summary()
        
        assert summary['total_files'] >= 2
        assert summary['file_types']['csv'] >= 1
        assert summary['file_types']['json'] >= 1
