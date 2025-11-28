import unittest

from core_bak_refactored.core.risk.international_enhancements import InternationalEnhancements


class InternationalEnhancementsTest(unittest.TestCase):
    def test_enhancements_instantiation(self):
        enhancements = InternationalEnhancements()
        self.assertIsNotNone(enhancements)


if __name__ == '__main__':
    unittest.main()
