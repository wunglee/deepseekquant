import unittest
from infrastructure.interfaces import InfrastructureProvider

class TestEventBusService(unittest.TestCase):
    def test_publish_subscribe(self):
        svc = InfrastructureProvider.get('event_bus')
        svc.initialize()
        received = []
        svc.subscribe('topic', lambda e: received.append(e))
        svc.publish('topic', {'msg': 'hello'})
        self.assertEqual(received[0]['msg'], 'hello')
        svc.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
