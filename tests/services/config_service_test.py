import unittest
from core.services.config_service import ConfigService

class TestConfigService(unittest.TestCase):
    def test_get_and_set(self):
        svc = ConfigService()
        svc.initialize()
        r1 = svc.process(action='get', key='system.name')
        self.assertEqual(r1['status'], 'success')
        r2 = svc.process(action='set', key='system.name', value='DeepSeekQuant')
        self.assertEqual(r2['status'], 'success')
        svc.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
