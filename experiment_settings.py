"""Compare experiment settings, including conventions used by older results."""


def differing_settings(first, second, keys):
    """Old results used legacy-raw noise and greedy digit values."""
    differences = []
    for key in keys:
        default = {"noise_mode": "legacy-raw", "digit_mode": "greedy"}.get(key)
        if first.get(key, default) != second.get(key, default):
            differences.append(key)
    return differences
