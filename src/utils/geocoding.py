# src/utils/geocoding.py

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GeocodingHelper:
    """Helper class for geocoding operations"""

    def __init__(self):
        """Initialize geocoder"""
        self.geocoder = Nominatim(user_agent="uber_route_optimizer")

    def get_address(self, lat: float, lon: float) -> Optional[str]:
        """Get address from coordinates"""
        try:
            location = self.geocoder.reverse((lat, lon))
            return location.address if location else None
        except GeocoderTimedOut:
            logger.warning(f"Geocoding timed out for coordinates: {lat}, {lon}")
            return None
        except Exception as e:
            logger.error(f"Error in geocoding: {str(e)}")
            return None

    def get_coordinates(self, address: str) -> Optional[Tuple[float, float]]:
        """Get coordinates from address"""
        try:
            location = self.geocoder.geocode(address)
            if location:
                return (location.latitude, location.longitude)
            return None
        except GeocoderTimedOut:
            logger.warning(f"Geocoding timed out for address: {address}")
            return None
        except Exception as e:
            logger.error(f"Error in geocoding: {str(e)}")
            return None