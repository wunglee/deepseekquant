import unittest
from infrastructure.interfaces import InfrastructureProvider

class TestServicesHealth(unittest.TestCase):
    def test_basic_health(self):
        services = [
            InfrastructureProvider.get('config'),
            InfrastructureProvider.get('event_bus'),
            InfrastructureProvider.get('cache'),
        ]
        for svc in services:
            self.assertTrue(svc.initialize())
            health = svc.get_health_status()
            self.assertIn('is_healthy', health)
            svc.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
