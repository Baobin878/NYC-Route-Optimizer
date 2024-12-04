from datetime import datetime
import pandas as pd
import numpy as np


class TrafficAnalyzer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.traffic_patterns = self._analyze_traffic_patterns()

    def _analyze_traffic_patterns(self):
        """Analyze historical traffic patterns by hour and weekday"""
        patterns = {}

        # Group by hour and weekday to get average density
        hourly_patterns = self.df.groupby(['Hour', 'Weekday']).size()
        max_density = hourly_patterns.max()

        # Normalize patterns to get multipliers
        for (hour, weekday), count in hourly_patterns.items():
            patterns[(hour, weekday)] = 1 + (count / max_density)

        return patterns

    def get_traffic_multiplier(self, hour: int, weekday: int) -> float:
        """Get traffic multiplier for given time"""
        return self.traffic_patterns.get((hour, weekday), 1.0)