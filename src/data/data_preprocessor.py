from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.nyc_bounds = {
            'lat_min': 40.5,
            'lat_max': 41.0,
            'lon_min': -74.1,
            'lon_max': -73.7
        }
        logger.info("DataPreprocessor initialized")

    def preprocess(self, input_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info(f"Starting preprocessing of {len(input_df)} records")

            # Create a copy to avoid modifying the input
            df = input_df.copy()

            # Apply preprocessing steps
            df = self._clean_data(df)
            df = self._filter_coordinates(df)
            df = self._convert_datetime(df)
            df = self._add_time_features(df)

            logger.info(f"Preprocessing complete. Output shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning data...")
        df = df.drop_duplicates()
        df = df.dropna(subset=['Date/Time', 'Lat', 'Lon', 'Base'])
        return df

    def _filter_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filtering coordinates...")
        valid_coords = (
                (df['Lat'].between(self.nyc_bounds['lat_min'], self.nyc_bounds['lat_max'])) &
                (df['Lon'].between(self.nyc_bounds['lon_min'], self.nyc_bounds['lon_max']))
        )
        return df[valid_coords].copy()

    def _convert_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Converting datetime...")
        df['Date/Time'] = pd.to_datetime(df['Date/Time'])
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding time features...")
        df['Hour'] = df['Date/Time'].dt.hour
        df['Weekday'] = df['Date/Time'].dt.weekday
        df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
        return df

    def get_statistics(self, df: pd.DataFrame) -> Dict:
        return {
            'total_records': len(df),
            'valid_coordinates': len(self._filter_coordinates(df)),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': len(df[df.duplicated()]),
            'coordinates_range': {
                'lat': {'min': df['Lat'].min(), 'max': df['Lat'].max()},
                'lon': {'min': df['Lon'].min(), 'max': df['Lon'].max()}
            }
        }