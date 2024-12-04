
import folium
from folium import plugins
from typing import List, Tuple, Dict
import logging
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RouteVisualizer:
    """Visualizes routes on an interactive map"""

    def __init__(self):
        """Initialize visualizer with styling options"""
        self.colors = {
            'route': '#4285F4',  # Google Maps blue
            'start': '#34A853',  # Google Maps green
            'end': '#EA4335',  # Google Maps red
            'turn': '#FBBC05'  # Google Maps yellow
        }
        self.map = None

    def create_route_map(self,
                         route_data: Dict,
                         center: Tuple[float, float] = None) -> folium.Map:
        """
        Create an interactive map with the route

        Args:
            route_data: Dictionary containing route information
                - path: List of (lat, lon) coordinates
                - turns: List of turn instructions
                - distance: Total distance
                - estimated_time: Estimated time
            center: Optional center coordinates for map
        """
        try:
            # Get center point if not provided
            if not center:
                center = self._get_center_point(route_data['path'])

            # Create base map
            self.map = folium.Map(
                location=center,
                zoom_start=14,
                tiles='cartodbdark_matter'
            )

            # Add route line
            self._add_route_line(route_data['path'])

            # Add markers
            self._add_markers(route_data)

            # Add route information
            self._add_info_box(route_data)

            return self.map

        except Exception as e:
            logger.error(f"Error creating route map: {str(e)}")
            raise

    def _add_route_line(self, coordinates: List[Tuple[float, float]]):
        """Add the main route line to map"""
        folium.PolyLine(
            locations=coordinates,
            weight=4,
            color=self.colors['route'],
            opacity=0.8
        ).add_to(self.map)

    def _add_markers(self, route_data: Dict):
        """Add all markers (start, end, turns) to map"""
        # Start marker
        folium.CircleMarker(
            location=route_data['path'][0],
            radius=8,
            color=self.colors['start'],
            fill=True,
            popup='Start',
            fill_opacity=1.0
        ).add_to(self.map)

        # End marker
        folium.CircleMarker(
            location=route_data['path'][-1],
            radius=8,
            color=self.colors['end'],
            fill=True,
            popup='Destination',
            fill_opacity=1.0
        ).add_to(self.map)

        # Turn points
        for i, turn in enumerate(route_data.get('turns', [])):
            folium.CircleMarker(
                location=turn['location'],
                radius=6,
                color=self.colors['turn'],
                fill=True,
                popup=f"Turn {i + 1}: {turn['instruction']}",
                fill_opacity=0.7
            ).add_to(self.map)

    def _add_info_box(self, route_data: Dict):
        """Add route information box to map"""
        info_html = f'''
            <div style="position: fixed; 
                        top: 10px; 
                        right: 10px; 
                        width: 250px;
                        height: auto;
                        z-index:9999;
                        background-color: white;
                        border-radius: 10px;
                        padding: 10px;
                        font-family: Arial;
                        box-shadow: 0 0 10px rgba(0,0,0,0.5);">
                <h4 style="margin:0;">Route Information</h4>
                <hr style="margin: 5px 0;">
                <p><b>Distance:</b> {route_data['distance']:.1f} km</p>
                <p><b>Est. Time:</b> {route_data['estimated_time']:.0f} min</p>
                <div style="margin-top: 10px;">
                    <b>Turn-by-turn directions:</b>
                    <ol style="margin: 5px 0; padding-left: 20px;">
                        {self._generate_directions_html(route_data.get('turns', []))}
                    </ol>
                </div>
            </div>
        '''
        self.map.get_root().html.add_child(folium.Element(info_html))

    def _generate_directions_html(self, turns: List[Dict]) -> str:
        """Generate HTML for turn-by-turn directions"""
        return ''.join([
            f"<li style='margin:3px 0;'>{turn['instruction']}</li>"
            for turn in turns
        ])

    def _get_center_point(self, coordinates: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate center point of route"""
        lats, lons = zip(*coordinates)
        return (sum(lats) / len(lats), sum(lons) / len(lons))

    def save_map(self, filename: str = 'route_map.html'):
        """Save map to HTML file"""
        if self.map:
            self.map.save(filename)
            logger.info(f"Map saved to {filename}")
        else:
            raise ValueError("No map to save. Create route map first.")





