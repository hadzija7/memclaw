def mask_user_id(user_id: str | int) -> str:
    """Mask the middle of a user identifier for log safety.

    One `*` per hidden character so length is preserved.
    Example: '100951236' -> '100****36'.
    """
    s = str(user_id)
    if len(s) <= 5:
        return "*" * len(s)
    return f"{s[:3]}{'*' * (len(s) - 5)}{s[-2:]}"
