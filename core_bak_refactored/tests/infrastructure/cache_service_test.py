import unittest
from datetime import datetime

from core_bak_refactored.infrastructure.cache_service import (
    CacheService, CacheConfig, CacheKeyGenerator
)


class CacheServiceTest(unittest.TestCase):
    def setUp(self):
        self.service = CacheService(CacheConfig(l1_maxsize=10, l1_ttl_seconds=60))

    def test_set_get_invalidate(self):
        key = 'test:key:1'
        self.assertIsNone(self.service.get(key))
        self.service.set(key, {'v': 1})
        self.assertEqual(self.service.get(key), {'v': 1})
        # invalidate and ensure gone
        self.service.invalidate(key)
        self.assertIsNone(self.service.get(key))
        # metrics should reflect hits/misses
        m = self.service.metrics.to_dict()
        self.assertIn('l1_hit_rate', m)
        self.assertIn('overall_hit_rate', m)

    def test_generate_key_basic(self):
        components = {
            'market': 'CN',
            'symbols': ['000300.SH', '600036.SS'],
            'model_type': 'risk',
            'time_window': datetime(2025, 1, 1, 10, 15),
            'params': {'alpha': 0.1, 'beta': 0.2}
        }
        key = CacheKeyGenerator.generate_key(components, data_version='v1.1')
        self.assertTrue(key.startswith('v1.1:'))
        parts = key.split(':')
        self.assertEqual(len(parts), 4)


if __name__ == '__main__':
    unittest.main()
