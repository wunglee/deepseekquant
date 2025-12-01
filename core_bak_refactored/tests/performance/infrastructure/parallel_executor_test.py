import unittest

from core_bak_refactored.infrastructure.parallel_executor import ParallelExecutor, ParallelConfig


def square(x):
    return x * x


def echo(x):
    return str(x)


class ParallelExecutorTest(unittest.TestCase):
    def test_should_parallelize_thresholds(self):
        ex = ParallelExecutor(ParallelConfig(min_items_for_parallel_cpu=5, min_items_for_parallel_io=3))
        should_cpu, reason_cpu = ex.should_parallelize(4, 'cpu')
        self.assertFalse(should_cpu)
        should_io, reason_io = ex.should_parallelize(2, 'io')
        self.assertFalse(should_io)
        should_cpu2, _ = ex.should_parallelize(6, 'cpu')
        self.assertTrue(should_cpu2)

    def test_map_cpu_intensive_serial_path(self):
        # For small list, executor chooses serial path
        ex = ParallelExecutor(ParallelConfig(min_items_for_parallel_cpu=1000))
        res = ex.map_cpu_intensive(square, [1, 2, 3, 4])
        self.assertEqual(res, [1, 4, 9, 16])

    def test_map_io_intensive_serial_path(self):
        ex = ParallelExecutor(ParallelConfig(min_items_for_parallel_io=1000))
        res = ex.map_io_intensive(echo, [1, 2, 3])
        self.assertEqual(res, ['1', '2', '3'])


if __name__ == '__main__':
    unittest.main()
