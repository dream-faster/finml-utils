def round_limit_metric(x: float) -> float:
    x = round(x, 3)
    return min(100.0, max(-100.0, x))
