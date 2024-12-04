import unittest
from pathlib import Path
import sys
import pandas as pd
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_preprocessor import DataPreprocessor
from src.data.data_loader import UberDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataProcess(unittest.TestCase):
    """Test cases for data loading and preprocessing"""


    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all tests"""
        # Set up paths
        cls.project_root = Path(__file__).parent.parent
        cls.data_dir = cls.project_root / 'data'

        # Initialize components
        cls.data_loader = UberDataLoader(data_dir=cls.data_dir)
        cls.preprocessor = DataPreprocessor()

        try:
            # Try to load data
            logger.info(f"Looking for data files in: {cls.data_dir}")
            available_files = cls.data_loader.list_available_files()
            logger.info(f"Available data files: {available_files}")

            # Load raw data
            cls.raw_data = cls.data_loader.load_raw_data()
            logger.info(f"Loaded {len(cls.raw_data)} records for testing")

            # Print data info
            logger.info("Data columns:")
            for col in cls.raw_data.columns:
                logger.info(f"  - {col}: {cls.raw_data[col].dtype}")

        except Exception as e:
            logger.error(f"Error in test setup: {str(e)}")
            raise

    def test_1_data_loader(self):
        """Test data loading functionality"""
        logger.info("Testing data loader...")

        # Test loading raw data
        data = self.data_loader.load_raw_data()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)

        # Check required columns
        required_columns = ['Date/Time', 'Lat', 'Lon', 'Base']
        for col in required_columns:
            self.assertIn(col, data.columns)

        # Check data is not empty
        self.assertGreater(len(data), 0)

        # Print data info for debugging
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Columns: {data.columns.tolist()}")

    def test_2_data_cleaning(self):
        """Test data cleaning functionality"""
        logger.info("Testing data cleaning...")

        # Create test data with some issues
        test_df = self.raw_data.copy()
        test_df.loc[0, 'Lat'] = None  # Add null value
        test_df = pd.concat([test_df, test_df.head(1)])  # Add duplicate

        # Clean data
        cleaned_df = self.preprocessor._clean_data(test_df)

        # Check results
        self.assertLess(len(cleaned_df), len(test_df))  # Should remove problematic rows
        self.assertFalse(cleaned_df.duplicated().any())  # No duplicates
        self.assertFalse(cleaned_df[['Lat', 'Lon', 'Base']].isnull().any().any())  # No nulls

    def test_3_coordinate_filtering(self):
        """Test coordinate filtering"""
        logger.info("Testing coordinate filtering...")

        # Create test data with out-of-bounds coordinates
        test_df = self.raw_data.copy()
        test_df.loc[0, 'Lat'] = 45.0  # Outside NYC
        test_df.loc[1, 'Lon'] = -75.0  # Outside NYC

        # Filter coordinates
        filtered_df = self.preprocessor._filter_coordinates(test_df)

        # Check results
        self.assertLess(len(filtered_df), len(test_df))

        # Verify bounds
        bounds = self.preprocessor.nyc_bounds
        self.assertTrue(
            filtered_df['Lat'].between(bounds['lat_min'], bounds['lat_max']).all()
        )
        self.assertTrue(
            filtered_df['Lon'].between(bounds['lon_min'], bounds['lon_max']).all()
        )

    def test_4_datetime_conversion(self):
        """Test datetime conversion"""
        logger.info("Testing datetime conversion...")

        # Create test data
        test_df = self.raw_data.copy()

        # Convert datetime
        converted_df = self.preprocessor._convert_datetime(test_df)

        # Check results
        self.assertIsInstance(converted_df['Date/Time'].iloc[0], pd.Timestamp)
        self.assertTrue(converted_df['Date/Time'].notna().all())

    def test_5_time_features(self):
        """Test time feature generation"""
        logger.info("Testing time feature generation...")

        # Prepare test data
        test_df = self.raw_data.copy()
        test_df = self.preprocessor._convert_datetime(test_df)

        # Generate time features
        featured_df = self.preprocessor._add_time_features(test_df)

        # Check results
        self.assertIn('Hour', featured_df.columns)
        self.assertIn('Weekday', featured_df.columns)
        self.assertIn('IsWeekend', featured_df.columns)

        # Verify ranges
        self.assertTrue(featured_df['Hour'].between(0, 23).all())
        self.assertTrue(featured_df['Weekday'].between(0, 6).all())
        self.assertTrue(featured_df['IsWeekend'].isin([0, 1]).all())

    def test_6_full_pipeline(self):
        """Test complete preprocessing pipeline"""
        logger.info("Testing full preprocessing pipeline...")

        # Run full preprocessing
        processed_df = self.preprocessor.preprocess(self.raw_data)

        # Verify results
        self.assertIsNotNone(processed_df)
        self.assertGreater(len(processed_df), 0)

        # Check all expected columns
        expected_columns = [
            'Date/Time', 'Lat', 'Lon', 'Base',
            'Hour', 'Weekday', 'IsWeekend'
        ]
        for col in expected_columns:
            self.assertIn(col, processed_df.columns)

        # Verify data quality
        self.assertFalse(processed_df.isnull().any().any())

        # Print summary
        logger.info(f"Input shape: {self.raw_data.shape}")
        logger.info(f"Output shape: {processed_df.shape}")
        logger.info(f"Columns: {processed_df.columns.tolist()}")

    def tearDown(self):
        """Clean up after each test"""
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)