import unittest
from core.optimization.bayesian_optimizer import BayesianOptimizer, OptimizationResult
from common import AcquisitionFunctionType, OptimizationObjective

class TestBayesianOptimizer(unittest.TestCase):
    def test_maximize(self):
        opt = BayesianOptimizer()
        def obj_fn(params):
            return params.get('x', 0) ** 2
        
        search_space = {
            'candidates': [{'x': 1}, {'x': 2}, {'x': 3}]
        }
        result = opt.optimize(obj_fn, search_space, objective=OptimizationObjective.MAXIMIZE, max_iter=3)
        self.assertEqual(result.best_params['x'], 3)
        self.assertEqual(result.score, 9)

    def test_minimize(self):
        opt = BayesianOptimizer()
        def obj_fn(params):
            return abs(params.get('x', 0) - 2)
        
        search_space = {
            'candidates': [{'x': 1}, {'x': 2}, {'x': 3}]
        }
        result = opt.optimize(obj_fn, search_space, objective=OptimizationObjective.MINIMIZE, max_iter=3)
        self.assertEqual(result.best_params['x'], 2)
        self.assertEqual(result.score, 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
