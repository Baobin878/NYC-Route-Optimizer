from math import radians, sin, cos, sqrt, atan2
from typing import Tuple


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two points
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers

    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c