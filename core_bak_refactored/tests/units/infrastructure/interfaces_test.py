import unittest

from core_bak_refactored.infrastructure.interfaces import (
    InfrastructureProvider
)


class InterfacesTest(unittest.TestCase):
    def test_infrastructure_provider_register_and_get(self):
        # Test registering a custom factory
        def test_factory():
            return {'test': True}
        InfrastructureProvider.register('test_service', test_factory)
        instance = InfrastructureProvider.get('test_service')
        self.assertEqual(instance, {'test': True})

    def test_infrastructure_provider_get_nonexistent(self):
        with self.assertRaises(KeyError):
            InfrastructureProvider.get('nonexistent_service')


if __name__ == '__main__':
    unittest.main()
