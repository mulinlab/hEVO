"""Locate changed token blocks without assuming base and token indices agree."""


def changed_block(left, right):
    """Return differing positions after removing a common prefix and suffix."""
    start = 0
    while start < min(len(left), len(right)) and int(left[start]) == int(right[start]):
        start += 1
    end_left, end_right = len(left), len(right)
    while (
        end_left > start
        and end_right > start
        and int(left[end_left - 1]) == int(right[end_right - 1])
    ):
        end_left -= 1
        end_right -= 1
    return list(range(start, end_left)), list(range(start, end_right))
