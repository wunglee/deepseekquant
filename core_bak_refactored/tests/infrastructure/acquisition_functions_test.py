import unittest
import numpy as np
from unittest.mock import Mock

from core_bak_refactored.infrastructure.acquisition_functions import AcquisitionFunction


class AcquisitionFunctionsTest(unittest.TestCase):
    def setUp(self):
        self.gp_model = Mock()
        self.scaler = Mock()
        self.scaler.transform = Mock(return_value=np.array([[0.5, 0.5]]))
        
    def test_expected_improvement_initialization(self):
        af = AcquisitionFunction(
            function_type='expected_improvement',
            gp_model=self.gp_model,
            scaler=self.scaler,
            best_value=-0.5,
            objective='minimize'
        )
        self.assertEqual(af.function_type, 'expected_improvement')
        self.assertEqual(af.best_value, -0.5)
        
    def test_upper_confidence_bound_computation(self):
        self.gp_model.predict = Mock(return_value=(np.array([0.3]), np.array([0.1])))
        
        af = AcquisitionFunction(
            function_type='upper_confidence_bound',
            gp_model=self.gp_model,
            scaler=self.scaler,
            best_value=-0.5,
            objective='minimize',
            kappa=2.0
        )
        
        x = np.array([0.5, 0.5])
        value = af.compute(x)
        self.assertIsInstance(value, (float, np.floating))
        

if __name__ == '__main__':
    unittest.main()
