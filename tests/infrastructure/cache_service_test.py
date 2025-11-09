import unittest
from infrastructure.interfaces import InfrastructureProvider

class TestCacheService(unittest.TestCase):
    def test_get_set(self):
        svc = InfrastructureProvider.get('cache')
        svc.initialize()
        r_set = svc.process(op='set', key='k', value='v')
        self.assertEqual(r_set['status'], 'success')
        r_get = svc.process(op='get', key='k')
        self.assertEqual(r_get['value'], 'v')
        svc.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
