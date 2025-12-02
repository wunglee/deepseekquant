import unittest
import numpy as np

from core_bak_refactored.core.optimization.bayesian import GaussianProcessModel


class GaussianProcessTest(unittest.TestCase):
    def test_model_initialization(self):
        gp = GaussianProcessModel(noise_level=0.1, random_seed=42)
        self.assertIsNotNone(gp.gp)
        self.assertFalse(gp.is_fitted)
        
    def test_model_training(self):
        gp = GaussianProcessModel(random_seed=42)
        X = np.random.rand(20, 2)
        y = np.random.rand(20)
        
        quality = gp.train(X, y)
        self.assertTrue(gp.is_fitted)
        self.assertIsInstance(quality, (float, np.floating))
        
    def test_model_prediction(self):
        gp = GaussianProcessModel(random_seed=42)
        X_train = np.random.rand(20, 2)
        y_train = np.random.rand(20)
        gp.train(X_train, y_train)
        
        X_test = np.random.rand(5, 2)
        y_pred, std = gp.predict(X_test, return_std=True)
        
        self.assertEqual(len(y_pred), 5)
        self.assertEqual(len(std), 5)


if __name__ == '__main__':
    unittest.main()
