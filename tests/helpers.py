"""Small builders shared by the statistics tests."""


def outcomes(spec: dict[str, list[int]]) -> dict[str, list[bool]]:
    """Build per-task outcome lists from 1/0 shorthand."""
    return {task: [bool(value) for value in results] for task, results in spec.items()}
