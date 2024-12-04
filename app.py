from flask import Flask, render_template, request, jsonify
from typing import Dict, Tuple
import logging
import os
from datetime import datetime

# Import the OSRM optimizer
from src.algorithms.route_optimizer import OSRMRouteOptimizer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

NYC_BOUNDS = {
    'min_lat': 40.4774,
    'max_lat': 40.9176,
    'min_lon': -74.2591,
    'max_lon': -73.7004
}


def init_services():
    """Initialize the route optimization service"""
    try:
        logger.info("Initializing route optimizer...")
        return OSRMRouteOptimizer()  # No parameters needed for OSRM initialization
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        return None


def validate_coordinates(coords: Tuple[float, float], bounds: Dict) -> bool:
    """Validate if coordinates are within NYC bounds"""
    lat, lon = coords
    return (bounds['min_lat'] <= lat <= bounds['max_lat'] and
            bounds['min_lon'] <= lon <= bounds['max_lon'])


# Initialize optimizer once
optimizer = init_services()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/find_route', methods=['POST'])
def find_route():
    """Handle route finding requests"""
    if optimizer is None:
        return jsonify({
            'error': 'Route optimization service not initialized',
            'success': False
        }), 503

    try:
        data = request.json
        start_coords = (float(data['start_lat']), float(data['start_lon']))
        end_coords = (float(data['end_lat']), float(data['end_lon']))

        # Validate coordinates
        if not all(validate_coordinates(coords, NYC_BOUNDS)
                   for coords in [start_coords, end_coords]):
            return jsonify({
                'error': 'Coordinates must be within NYC bounds',
                'bounds': NYC_BOUNDS,
                'success': False
            }), 400

        # Find route using OSRM
        route = optimizer.find_route(
            start=start_coords,
            end=end_coords,
            optimize_for=data.get('optimize_for', 'distance')
        )

        if not route:
            return jsonify({
                'error': 'No route found',
                'success': False
            }), 404

        # Format response
        response = {
            'success': True,
            'path': route['path'],
            'directions': route['directions'],
            'distance': f"{float(route['distance']):.2f} km",
            'estimated_time': f"{int(route['estimated_time'])} minutes",
            'traffic_level': route['traffic_level']
        }

        logger.info(f"Route found - Distance: {route['distance']:.2f}km, "
                    f"Time: {route['estimated_time']}min")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing route request: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/address_lookup', methods=['POST'])
def address_lookup():
    """Look up address details"""
    try:
        data = request.json
        coords = (float(data['lat']), float(data['lon']))

        if not validate_coordinates(coords, NYC_BOUNDS):
            return jsonify({
                'error': 'Coordinates must be within NYC bounds',
                'success': False
            }), 400

        details = optimizer.get_address_details(coords)
        if details:
            return jsonify({
                'success': True,
                'address': details
            })
        return jsonify({
            'error': 'Address not found',
            'success': False
        }), 404

    except Exception as e:
        logger.error(f"Error looking up address: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/traffic', methods=['POST'])
def get_traffic():
    """Get traffic information for a route"""
    try:
        data = request.json
        coords = [(float(p['lat']), float(p['lon'])) for p in data['path']]

        # Validate all coordinates
        if not all(validate_coordinates(coord, NYC_BOUNDS) for coord in coords):
            return jsonify({
                'error': 'All coordinates must be within NYC bounds',
                'success': False
            }), 400

        traffic_info = optimizer.get_live_traffic({'path': coords})
        return jsonify({
            'success': True,
            'traffic': traffic_info
        })

    except Exception as e:
        logger.error(f"Error getting traffic data: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found', 'success': False}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error', 'success': False}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)