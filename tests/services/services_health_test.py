import unittest
from core.services.config_service import ConfigService
from core.services.event_bus_service import EventBusService
from core.services.cache_service import CacheService

class TestServicesHealth(unittest.TestCase):
    def test_basic_health(self):
        services = [ConfigService(), EventBusService(), CacheService()]
        for svc in services:
            self.assertTrue(svc.initialize())
            health = svc.get_health_status()
            self.assertIn('is_healthy', health)
            svc.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
