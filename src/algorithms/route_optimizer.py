import requests
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
from dataclasses import dataclass
import json
from math import radians, sin, cos, sqrt, atan2, degrees

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RouteSegment:
    distance: float  # meters
    duration: float  # seconds
    instruction: str
    name: str
    coordinates: List[float]


class OSRMRouteOptimizer:
    def __init__(self):
        """Initialize OSRM route optimizer"""
        self.base_url = "http://router.project-osrm.org/route/v1"
        self.geocode_url = "https://nominatim.openstreetmap.org/search"
        self.reverse_geocode_url = "https://nominatim.openstreetmap.org/reverse"

    def find_route(self,
                   start: Tuple[float, float],
                   end: Tuple[float, float],
                   optimize_for: str = 'distance') -> Optional[Dict]:
        """Find route between two points"""
        try:
            # Format coordinates (OSRM expects lng,lat order)
            coords = f"{start[1]},{start[0]};{end[1]},{end[0]}"

            # Configure routing parameters
            params = {
                'overview': 'full',
                'steps': 'true',
                'annotations': 'true',
                'geometries': 'geojson',
                'alternatives': 'true'
            }

            # Request route
            url = f"{self.base_url}/driving/{coords}"
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            if 'routes' not in data or not data['routes']:
                logger.error("No routes found")
                return None

            # Process the best route
            best_route = self._select_optimal_route(data['routes'], optimize_for)
            if not best_route:
                return None

            # Format response
            return {
                'path': self._extract_path(best_route['geometry']),
                'distance': best_route['distance'] / 1000,  # Convert to km
                'estimated_time': best_route['duration'] / 60,  # Convert to minutes
                'directions': self._generate_directions(best_route['legs'][0]['steps']),
                'traffic_level': self._estimate_traffic_level(best_route)
            }

        except Exception as e:
            logger.error(f"Error finding route: {str(e)}")
            return None

    def _select_optimal_route(self, routes: List[Dict], optimize_for: str) -> Optional[Dict]:
        """Select the best route based on optimization criteria"""
        if not routes:
            return None

        if optimize_for == 'distance':
            return min(routes, key=lambda r: r['distance'])
        else:  # optimize for time
            return min(routes, key=lambda r: r['duration'])

    def _extract_path(self, geometry: Dict) -> List[List[float]]:
        """Extract path coordinates from GeoJSON geometry"""
        try:
            if geometry['type'] != 'LineString':
                return []

            # Convert from [lon, lat] to [lat, lon] format
            return [[coord[1], coord[0]] for coord in geometry['coordinates']]
        except Exception as e:
            logger.error(f"Error extracting path: {str(e)}")
            return []

    def _generate_directions(self, steps: List[Dict]) -> List[Dict]:
        """Generate human-readable directions from OSRM steps"""
        directions = []
        total_distance = 0

        try:
            for step in steps:
                # Calculate distance for this step
                distance = step['distance']
                total_distance += distance

                # Format instruction
                instruction = self._format_instruction(step)

                # Extract coordinates (convert from lon,lat to lat,lon)
                coords = self._extract_path(step['geometry'])[0]

                directions.append({
                    'instruction': instruction,
                    'distance': f"{distance / 1000:.2f} km",
                    'street': step.get('name', ''),
                    'coordinates': coords,
                    'total_distance': f"{total_distance / 1000:.2f} km"
                })

            # Add final destination
            if directions:
                directions.append({
                    'instruction': "Arrive at destination",
                    'distance': "0 km",
                    'street': directions[-1]['street'],
                    'coordinates': self._extract_path(steps[-1]['geometry'])[-1],
                    'total_distance': f"{total_distance / 1000:.2f} km"
                })

        except Exception as e:
            logger.error(f"Error generating directions: {str(e)}")

        return directions

    def _format_instruction(self, step: Dict) -> str:
        """Format human-readable instruction from OSRM step"""
        try:
            maneuver = step['maneuver']
            instruction = ""

            # Handle different maneuver types
            if maneuver['type'] == 'turn':
                modifier = maneuver.get('modifier', '').replace('_', ' ')
                instruction = f"Turn {modifier}"
            elif maneuver['type'] == 'new name':
                instruction = "Continue onto"
            elif maneuver['type'] == 'depart':
                instruction = "Head"
            elif maneuver['type'] == 'arrive':
                instruction = "Arrive at destination"
            else:
                instruction = maneuver['type'].capitalize().replace('_', ' ')

            # Add street name if available
            if step.get('name'):
                instruction = f"{instruction} {step['name']}"

            return instruction.strip()

        except Exception as e:
            logger.error(f"Error formatting instruction: {str(e)}")
            return "Continue"

    def _estimate_traffic_level(self, route: Dict) -> str:
        """Estimate traffic level based on duration/distance ratio"""
        try:
            if 'duration' not in route or 'distance' not in route:
                return 'Unknown'

            # Calculate average speed (m/s)
            avg_speed = route['distance'] / route['duration']

            # Classify traffic based on average speed
            if avg_speed < 5:  # Less than 18 km/h
                return 'Heavy'
            elif avg_speed < 8:  # Less than 28.8 km/h
                return 'Moderate'
            else:
                return 'Light'

        except Exception as e:
            logger.error(f"Error estimating traffic: {str(e)}")
            return 'Unknown'

    def get_address_details(self, coords: Tuple[float, float]) -> Optional[Dict]:
        """Get address details using OpenStreetMap Nominatim"""
        try:
            params = {
                'format': 'json',
                'lat': coords[0],
                'lon': coords[1],
                'addressdetails': 1,
                'accept-language': 'en'
            }

            headers = {
                'User-Agent': 'UberRouteOptimizer/1.0'  # Required by Nominatim
            }

            response = requests.get(
                self.reverse_geocode_url,
                params=params,
                headers=headers
            )
            response.raise_for_status()

            data = response.json()
            if not data:
                return None

            return {
                'display_name': data.get('display_name', ''),
                'address': data.get('address', {}),
                'coordinates': [float(data.get('lat', 0)), float(data.get('lon', 0))]
            }

        except Exception as e:
            logger.error(f"Error getting address details: {str(e)}")
            return None

    def get_live_traffic(self, route: Dict) -> Dict:
        """Get live traffic information for route"""
        try:
            segments = []
            overall_level = 'Unknown'

            if 'path' in route and len(route['path']) > 1:
                # Process each segment of the route
                for i in range(len(route['path']) - 1):
                    segment = {
                        'start': route['path'][i],
                        'end': route['path'][i + 1],
                        'traffic_level': self._estimate_segment_traffic(
                            route['path'][i],
                            route['path'][i + 1]
                        )
                    }
                    segments.append(segment)

                # Calculate overall traffic level
                traffic_levels = [s['traffic_level'] for s in segments]
                if traffic_levels:
                    if 'Heavy' in traffic_levels:
                        overall_level = 'Heavy'
                    elif 'Moderate' in traffic_levels:
                        overall_level = 'Moderate'
                    else:
                        overall_level = 'Light'

            return {
                'overall_level': overall_level,
                'segments': segments
            }

        except Exception as e:
            logger.error(f"Error getting live traffic: {str(e)}")
            return {
                'overall_level': 'Unknown',
                'segments': []
            }

    def _estimate_segment_traffic(self, start: List[float], end: List[float]) -> str:
        """Estimate traffic level for a route segment"""
        try:
            # Format coordinates
            coords = f"{start[1]},{start[0]};{end[1]},{end[0]}"

            # Get route for segment
            url = f"{self.base_url}/driving/{coords}"
            response = requests.get(url, params={'overview': 'false'})
            response.raise_for_status()

            data = response.json()
            if 'routes' in data and data['routes']:
                route = data['routes'][0]
                return self._estimate_traffic_level(route)

            return 'Unknown'

        except Exception:
            return 'Unknown'