def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of an empty list")
    return sum(values) / len(values)


# Assert tests
assert mean([1.0, 2.0, 3.0]) == 2.0
assert mean([5.0]) == 5.0
assert mean([-1.0, 1.0]) == 0.0
assert mean([1.5, 2.5, 3.5]) == 2.5
try:
    mean([])
    assert False, "Empty list should raise ValueError"
except ValueError:
    pass
