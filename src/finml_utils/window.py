def calculate_window_size(window_size: int | float, length: int) -> int:
    return window_size if window_size > 1 else int(length * window_size)  # type: ignore
