from typing import Dict, List
import numpy as np


def calculate_daily_volatility(data: List) -> float:
    if len(data) < 2:
        return 0.0
    returns = []
    for i in range(1, len(data)):
        daily_return = (data[i].close - data[i - 1].close) / data[i - 1].close
        returns.append(daily_return)
    return float(np.std(returns) * np.sqrt(252))
