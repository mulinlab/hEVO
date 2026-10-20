"""Clipped interval unions used by the HAR/HAQER renderers."""


def merge_spans(
    spans: list[tuple[int, int]], region_start: int, region_end: int
) -> list[tuple[int, int]]:
    clipped = [
        (max(region_start, start), min(region_end, end))
        for start, end in spans
        if start < region_end and end > region_start
    ]
    clipped = [(start, end) for start, end in clipped if start < end]
    if not clipped:
        return []
    clipped.sort()
    merged = [clipped[0]]
    for start, end in clipped[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
