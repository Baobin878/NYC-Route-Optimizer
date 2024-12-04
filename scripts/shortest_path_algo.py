import folium
from folium import plugins
import requests
import polyline
from typing import List, Tuple, Dict
from math import radians, sin, cos, sqrt, atan2


class UberRouteVisualizer:
    def __init__(self):
        """Initialize the route visualizer"""
        self.waypoints = []
        self.OSRM_URL = "http://router.project-osrm.org/route/v1/driving/"

    def get_route(self, start_coords: Tuple[float, float], end_coords: Tuple[float, float]) -> dict:
        """Get route from OSRM API"""
        url = f"{self.OSRM_URL}{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "true",
            "annotations": "true"
        }
        response = requests.get(url, params=params)
        return response.json()

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def visualize_route(self, start_coords: Tuple[float, float], end_coords: Tuple[float, float],
                        output_file: str = 'uber_route_map.html'):
        """Create a Google Maps style visualization using OSRM routing"""
        # Get route from OSRM
        route_data = self.get_route(start_coords, end_coords)

        if 'routes' not in route_data:
            print("Error getting route data")
            return

        # Extract route coordinates and instructions
        route = route_data['routes'][0]
        coordinates = route['geometry']['coordinates']
        steps = route['legs'][0]['steps']

        # Calculate center point
        center_lat = (start_coords[0] + end_coords[0]) / 2
        center_lon = (start_coords[1] + end_coords[1]) / 2

        # Create the map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=16,
            tiles='cartodbdark_matter'
        )

        # Create route line with actual street geometry
        route_coords = [[coord[1], coord[0]] for coord in coordinates]
        folium.PolyLine(
            route_coords,
            weight=4,
            color='#4285F4',  # Google Maps blue
            opacity=0.8
        ).add_to(m)

        # Add start marker (green)
        folium.CircleMarker(
            location=[start_coords[0], start_coords[1]],
            radius=8,
            color='#34A853',  # Google Maps green
            fill=True,
            popup='Pickup Location<br>Vernon Blvd',
            fill_opacity=1.0
        ).add_to(m)

        # Add turn points from steps
        for step in steps:
            if step['maneuver']['type'] != 'depart' and step['maneuver']['type'] != 'arrive':
                location = step['maneuver']['location']
                folium.CircleMarker(
                    location=[location[1], location[0]],
                    radius=4,
                    color='#FBBC05',  # Google Maps yellow
                    fill=True,
                    popup=step['maneuver']['type'],
                    fill_opacity=0.7
                ).add_to(m)

        # Add end marker (red)
        folium.CircleMarker(
            location=[end_coords[0], end_coords[1]],
            radius=8,
            color='#EA4335',  # Google Maps red
            fill=True,
            popup='Dropoff Location<br>Jackson Ave',
            fill_opacity=1.0
        ).add_to(m)

        # Calculate total distance and time
        distance_km = route['distance'] / 1000  # Convert meters to kilometers
        time_minutes = max(1, int(route['duration'] / 60))  # Convert seconds to minutes

        # Generate turn-by-turn directions
        directions_html = "".join([
            f"<p style='margin:5px 0;font-size:12px;'>{i + 1}. {step['maneuver']['type'].title()}: {step.get('name', '')}</p>"
            for i, step in enumerate(steps)
        ])

        # Add route information box
        title_html = f'''
            <div style="position: fixed; 
                        top: 10px; 
                        left: 50px; 
                        width: 300px;
                        height: auto;
                        z-index:9999;
                        background-color: white;
                        border-radius: 10px;
                        padding: 10px;
                        font-family: Arial;
                        box-shadow: 0 0 10px rgba(0,0,0,0.5);">
                <h4 style="margin:0;">Uber Route</h4>
                <p style="margin:5px 0;">Distance: {distance_km:.1f} km</p>
                <p style="margin:5px 0;">Est. Time: {time_minutes} min</p>
                <p style="margin:5px 0;font-size:12px;">Pickup: Vernon Blvd</p>
                <p style="margin:5px 0;font-size:12px;">Dropoff: Jackson Ave</p>
                <hr style="margin:10px 0;">
                <h5 style="margin:5px 0;">Directions:</h5>
                {directions_html}
            </div>
            '''
        m.get_root().html.add_child(folium.Element(title_html))

        m.save(output_file)
        print(f"Map saved to {output_file}")
        return m


def main():
    # Create visualizer
    viz = UberRouteVisualizer()

    # Define start and end coordinates
    start_coords = (40.744876, -73.953915)  # Vernon Blvd
    end_coords = (40.747726, -73.949463)  # Jackson Ave

    # Create visualization
    viz.visualize_route(start_coords, end_coords)


if __name__ == "__main__":
    main()