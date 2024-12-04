import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UberDataLoader:
    """Handler for loading Uber ride data"""

    def __init__(self, data_dir: str = None):
        """
        Initialize data loader
        Args:
            data_dir: Directory containing data files (optional)
        """
        if data_dir is None:
            # If no directory specified, use the default project structure
            self.data_dir = Path(__file__).parent.parent.parent / 'data'
        else:
            self.data_dir = Path(data_dir)

        logger.info(f"Data directory set to: {self.data_dir}")

        # Define file names
        self.file_names = {
            'raw': 'uber-raw-data-apr24.csv',
            'sampled': 'sampled_cleaned_data.csv',
            'final': 'final_cleaned_data.csv'
        }

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw Uber ride data"""
        try:
            # Try loading files in order of preference
            for file_type, file_name in self.file_names.items():
                file_path = self.data_dir / file_name
                if file_path.exists():
                    logger.info(f"Loading data from {file_path}")
                    df = pd.read_csv(file_path)
                    logger.info(f"Loaded {len(df)} records from {file_name}")
                    return df

            # If no files found, list available files and raise error
            available_files = list(self.data_dir.glob('*.csv'))
            logger.error(f"Available CSV files: {[f.name for f in available_files]}")
            raise FileNotFoundError(
                f"No data files found in {self.data_dir}. "
                f"Need either {', '.join(self.file_names.values())}"
            )

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def load_sampled_data(self) -> pd.DataFrame:
        """Load sampled and cleaned data"""
        try:
            file_path = self.data_dir / self.file_names['sampled']
            if file_path.exists():
                logger.info(f"Loading sampled data from {file_path}")
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} records from sampled data")
                return df
            else:
                logger.warning(f"Sampled data file not found: {file_path}")
                return self.load_raw_data()

        except Exception as e:
            logger.error(f"Error loading sampled data: {str(e)}")
            raise

    def save_processed_data(self, df: pd.DataFrame, file_type: str = 'sampled'):
        """
        Save processed data
        Args:
            df: DataFrame to save
            file_type: Type of file to save ('sampled' or 'final')
        """
        try:
            if file_type not in self.file_names:
                raise ValueError(f"Invalid file type. Must be one of: {list(self.file_names.keys())}")

            output_path = self.data_dir / self.file_names[file_type]
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} records to {output_path}")

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def list_available_files(self) -> list:
        """List all available CSV files in the data directory"""
        try:
            csv_files = list(self.data_dir.glob('*.csv'))
            return [f.name for f in csv_files]
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            raise