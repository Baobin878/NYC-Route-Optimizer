
from typing import Dict, List, Tuple, Union
import pandas as pd
import networkx as nx
from math import radians, sin, cos, sqrt, atan2, degrees
import logging
from pathlib import Path
from collections import defaultdict
import random
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreetGraphBuilder:
    """Builds optimized street network graph for NYC"""

    NYC_BOUNDS = {
        'min_lat': 40.7000,
        'max_lat': 40.8000,
        'min_lon': -74.0200,
        'max_lon': -73.9500
    }

    def __init__(self, data: Union[str, Path, pd.DataFrame]):
        self.df = None
        self.graph = nx.DiGraph()
        self.node_grid = defaultdict(list)
        self.grid_size = 0.001  # About 100m in NYC

        try:
            if isinstance(data, (str, Path)):
                self.df = pd.read_csv(data)
            else:
                self.df = data.copy()
            self._validate_data()
        except Exception as e:
            logger.error(f"Error initializing graph builder: {str(e)}")
            raise

    def _validate_data(self):
        """Validate and clean input data"""
        required_columns = ['Date/Time', 'Lat', 'Lon', 'Base']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Filter coordinates within NYC bounds
        valid_coords = (
                self.df['Lat'].between(self.NYC_BOUNDS['min_lat'], self.NYC_BOUNDS['max_lat']) &
                self.df['Lon'].between(self.NYC_BOUNDS['min_lon'], self.NYC_BOUNDS['max_lon'])
        )
        self.df = self.df[valid_coords].copy()
        logger.info(f"Retained {len(self.df)} valid coordinates")

    def build_graph(self) -> nx.DiGraph:
        try:
            logger.info("Building optimized street network...")

            # Step 1: Create initial nodes and grid structure
            unique_points = self.df[['Lat', 'Lon']].drop_duplicates()
            nodes_created = self._create_nodes(unique_points)
            logger.info(f"Created {nodes_created} nodes")

            # Step 2: Create primary street connections
            edges = self._create_street_segments()
            logger.info(f"Created {edges} edges")

            # Step 3: Add strategic cross connections
            cross_streets = self._add_cross_connections()
            logger.info(f"Added {cross_streets} cross-street connections")

            # Step 4: Ensure strong connectivity
            self._ensure_strong_connectivity()

            # Step 5: Optimize edges
            removed = self._optimize_edges()
            logger.info(f"Removed {removed} redundant edges")

            return self.graph

        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            raise

    def _ensure_strong_connectivity(self):
        """Ensure strong connectivity between all parts of the graph"""
        components = list(nx.weakly_connected_components(self.graph))
        if len(components) > 1:
            main_component = max(components, key=len)
            logger.info(f"Found {len(components)} components, connecting to main component")

            # Process components by size
            other_components = sorted(
                [c for c in components if c != main_component],
                key=len, reverse=True
            )

            for component in other_components:
                # Find multiple connection points
                connections_added = 0
                for node1 in component:
                    pos1 = (self.graph.nodes[node1]['lat'],
                            self.graph.nodes[node1]['lon'])

                    # Find closest nodes in main component
                    candidates = []
                    for node2 in main_component:
                        pos2 = (self.graph.nodes[node2]['lat'],
                                self.graph.nodes[node2]['lon'])
                        dist = self._haversine_distance(pos1[0], pos1[1],
                                                        pos2[0], pos2[1])
                        if dist <= 0.3:  # Increased connection radius
                            candidates.append((node2, dist))

                    # Add multiple connections if possible
                    candidates.sort(key=lambda x: x[1])
                    for node2, dist in candidates[:3]:  # Add up to 3 connections
                        self.graph.add_edge(node1, node2,
                                            distance=dist,
                                            time=dist / 30 * 60)
                        self.graph.add_edge(node2, node1,
                                            distance=dist,
                                            time=dist / 30 * 60)
                        connections_added += 1

                    if connections_added >= 3:
                        break

                logger.info(f"Connected component of size {len(component)} "
                            f"with {connections_added} connections")

    def _optimize_edges(self):
        """Optimize graph by removing redundant edges while maintaining connectivity"""
        initial_edges = self.graph.number_of_edges()
        edges_to_remove = []

        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) > 6:  # Increased from 4
                # Keep connections that follow street grid
                street_aligned = []
                cross_streets = []

                for neighbor in neighbors:
                    pos1 = (self.graph.nodes[node]['lat'],
                            self.graph.nodes[node]['lon'])
                    pos2 = (self.graph.nodes[neighbor]['lat'],
                            self.graph.nodes[neighbor]['lon'])

                    if self._is_street_aligned(pos1[0], pos1[1], pos2[0], pos2[1]):
                        street_aligned.append((neighbor,
                                               self.graph[node][neighbor]['distance']))
                    else:
                        cross_streets.append((neighbor,
                                              self.graph[node][neighbor]['distance']))

                # Keep top street-aligned connections
                street_aligned.sort(key=lambda x: x[1])
                keep_street = street_aligned[:4] if street_aligned else []

                # Keep top cross-street connections
                cross_streets.sort(key=lambda x: x[1])
                keep_cross = cross_streets[:2] if cross_streets else []

                # Mark others for removal
                keep_nodes = {n for n, _ in keep_street + keep_cross}
                edges_to_remove.extend([
                    (node, n) for n in neighbors if n not in keep_nodes
                ])

        self.graph.remove_edges_from(edges_to_remove)
        return initial_edges - self.graph.number_of_edges()

    def _create_nodes(self, points: pd.DataFrame) -> int:
        """Create nodes and add to spatial grid"""
        nodes_created = 0
        for idx, point in points.iterrows():
            lat, lon = float(point['Lat']), float(point['Lon'])
            node_id = f"node_{idx}"
            self.graph.add_node(node_id, lat=lat, lon=lon)

            # Add to spatial grid
            cell = self._get_grid_cell(lat, lon)
            self.node_grid[cell].append(node_id)
            nodes_created += 1

        return nodes_created

    def _is_valid_street_connection(self, pos1: Tuple[float, float],
                                    pos2: Tuple[float, float],
                                    street: str) -> bool:
        """Check if connection follows street pattern"""
        lat1, lon1 = pos1
        lat2, lon2 = pos2

        # For avenues (primarily north-south)
        if 'avenue' in street:
            lon_diff = abs(lon1 - lon2)
            if lon_diff > 0.0005:  # Ensure staying on same avenue
                return False
            return True

        # For streets (primarily east-west)
        if 'street' in street:
            lat_diff = abs(lat1 - lat2)
            if lat_diff > 0.0005:  # Ensure staying on same street
                return False
            return True

        return False

    def _create_street_segments(self) -> int:
        """Create edges between aligned nodes"""
        edges_added = 0
        processed = set()
        max_distance = 0.3  # Increased from 0.2 for better connectivity

        # Get list of cells once
        cells = list(self.node_grid.keys())

        for cell in cells:
            # Get nodes in current and adjacent cells
            nearby_nodes = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    adj_cell = (cell[0] + i, cell[1] + j)
                    if adj_cell in self.node_grid:
                        nearby_nodes.extend(self.node_grid[adj_cell])

            # Process nodes in current neighborhood
            for i, node1 in enumerate(nearby_nodes):
                if node1 not in self.graph.nodes:
                    continue

                pos1 = (self.graph.nodes[node1]['lat'],
                        self.graph.nodes[node1]['lon'])

                # Only look at remaining nodes to avoid duplicate checks
                for node2 in nearby_nodes[i + 1:]:
                    if node2 not in self.graph.nodes or (node1, node2) in processed:
                        continue

                    pos2 = (self.graph.nodes[node2]['lat'],
                            self.graph.nodes[node2]['lon'])

                    # Check if nodes are aligned with street grid
                    if self._is_street_aligned(pos1[0], pos1[1], pos2[0], pos2[1]):
                        dist = self._haversine_distance(pos1[0], pos1[1],
                                                        pos2[0], pos2[1])
                        if dist <= max_distance:
                            self.graph.add_edge(node1, node2,
                                                distance=dist,
                                                time=dist / 30 * 60)
                            self.graph.add_edge(node2, node1,
                                                distance=dist,
                                                time=dist / 30 * 60)
                            edges_added += 2
                            processed.add((node1, node2))
                            processed.add((node2, node1))

            if edges_added % 1000 == 0 and edges_added > 0:
                logger.info(f"Created {edges_added} edges")

        return edges_added

    def _add_cross_connections(self) -> int:
        """Add connections between intersecting streets"""
        added = 0
        processed = set()
        max_distance = 0.15  # Maximum distance for cross connections

        cells = list(self.node_grid.keys())

        for cell in cells:
            # Get nodes from current and adjacent cells
            nearby_nodes = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    adj_cell = (cell[0] + i, cell[1] + j)
                    if adj_cell in self.node_grid:
                        nearby_nodes.extend(self.node_grid[adj_cell])

            # Process nodes in current neighborhood
            for i, node1 in enumerate(nearby_nodes):
                if node1 not in self.graph.nodes:
                    continue

                pos1 = (self.graph.nodes[node1]['lat'],
                        self.graph.nodes[node1]['lon'])

                for node2 in nearby_nodes[i + 1:]:
                    if node2 not in self.graph.nodes or (node1, node2) in processed:
                        continue

                    pos2 = (self.graph.nodes[node2]['lat'],
                            self.graph.nodes[node2]['lon'])

                    # Check if forms valid intersection
                    if self._is_intersection(pos1[0], pos1[1], pos2[0], pos2[1]):
                        dist = self._haversine_distance(pos1[0], pos1[1],
                                                        pos2[0], pos2[1])
                        if dist <= max_distance:
                            self.graph.add_edge(node1, node2,
                                                distance=dist,
                                                time=dist / 30 * 60)
                            self.graph.add_edge(node2, node1,
                                                distance=dist,
                                                time=dist / 30 * 60)
                            added += 2
                            processed.add((node1, node2))
                            processed.add((node2, node1))

                if added % 1000 == 0 and added > 0:
                    logger.info(f"Added {added} cross connections")

        return added
    def _sort_nodes_by_street(self, nodes: List[str], major_streets: Dict) -> Dict[str, List[str]]:
        """Group nodes by streets and avenues"""
        street_nodes = defaultdict(list)

        for node in nodes:
            lat = self.graph.nodes[node]['lat']
            lon = self.graph.nodes[node]['lon']

            # Check if node is on a major street/avenue
            for avenue in major_streets['avenues']:
                if abs(lon - avenue) < 0.0005:  # ~50m tolerance
                    street_nodes[f'avenue_{avenue}'].append(node)

            for street in major_streets['streets']:
                if abs(lat - street) < 0.0005:  # ~50m tolerance
                    street_nodes[f'street_{street}'].append(node)

        # Sort nodes along each street/avenue
        for street in street_nodes:
            if 'avenue' in street:
                street_nodes[street].sort(key=lambda n: self.graph.nodes[n]['lat'])
            else:
                street_nodes[street].sort(key=lambda n: self.graph.nodes[n]['lon'])

        return street_nodes

    def _is_intersection(self, pos1: Tuple[float, float],
                         pos2: Tuple[float, float]) -> bool:
        """Check if points form a valid street intersection"""
        lat1, lon1 = pos1
        lat2, lon2 = pos2

        # Check if one point is on avenue and other on street
        lat_diff = abs(lat1 - lat2)
        lon_diff = abs(lon1 - lon2)

        return (lat_diff < 0.0005 and lon_diff < 0.0005)  # ~50m tolerance

    def _add_cross_connections(self) -> int:
        """Add connections between intersecting streets"""
        added = 0
        manhattan_angle = 29  # Manhattan grid angle
        processed = set()

        # Get list of cells once
        cells = list(self.node_grid.keys())

        for cell in cells:
            # Get nodes only from current and adjacent cells
            nearby_nodes = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    adj_cell = (cell[0] + i, cell[1] + j)
                    if adj_cell in self.node_grid:
                        nearby_nodes.extend(self.node_grid[adj_cell])

            # Process nodes in current neighborhood
            for i, node1 in enumerate(nearby_nodes):
                if node1 not in self.graph.nodes:
                    continue

                pos1 = (self.graph.nodes[node1]['lat'],
                        self.graph.nodes[node1]['lon'])

                # Only look at remaining nodes to avoid duplicate checks
                for node2 in nearby_nodes[i + 1:]:
                    if node2 not in self.graph.nodes or (node1, node2) in processed:
                        continue

                    pos2 = (self.graph.nodes[node2]['lat'],
                            self.graph.nodes[node2]['lon'])

                    # Check if forms valid street intersection
                    lat_diff = abs(pos1[0] - pos2[0])
                    lon_diff = abs(pos1[1] - pos2[1])

                    if lat_diff < 0.0005 and lon_diff < 0.0005:  # About 50m
                        dist = self._haversine_distance(pos1[0], pos1[1],
                                                        pos2[0], pos2[1])
                        if dist <= 0.05:  # 50m maximum for intersections
                            self.graph.add_edge(node1, node2,
                                                distance=dist,
                                                time=dist / 30 * 60)
                            self.graph.add_edge(node2, node1,
                                                distance=dist,
                                                time=dist / 30 * 60)
                            added += 2
                            processed.add((node1, node2))
                            processed.add((node2, node1))

                if added % 1000 == 0 and added > 0:
                    logger.info(f"Added {added} cross connections")

        return added

    def _is_valid_cross_connection(self, pos1, pos2):
        """Check if points form a valid cross street connection"""
        lat1, lon1 = pos1
        lat2, lon2 = pos2

        # Must be close enough
        lat_diff = abs(lat1 - lat2)
        lon_diff = abs(lon1 - lon2)

        if lat_diff > 0.0005 or lon_diff > 0.0005:
            return False

        # Must be at an intersection
        angle = abs(degrees(atan2(lat_diff, lon_diff)) % 90)
        return 85 <= angle <= 95  # Approximately perpendicular
    def _ensure_connectivity(self):
        """Ensure graph is connected efficiently"""
        if not nx.is_weakly_connected(self.graph):
            components = list(nx.weakly_connected_components(self.graph))
            main_component = max(components, key=len)
            logger.info(f"Found {len(components)} components, connecting to main component")

            # Process smaller components first
            other_components = sorted(
                [c for c in components if c != main_component],
                key=len, reverse=True
            )

            # Connect each component using nearest neighbors
            for component in other_components:
                # Sample points from each component for efficiency
                sample_size = min(10, len(component))
                component_sample = set(random.sample(list(component), sample_size))
                main_sample = set(random.sample(list(main_component),
                                                min(50, len(main_component))))

                # Find closest pair between samples
                min_dist = float('inf')
                best_pair = None

                for node1 in component_sample:
                    pos1 = (self.graph.nodes[node1]['lat'],
                            self.graph.nodes[node1]['lon'])

                    # Use grid to find nearby nodes in main component
                    cell = self._get_grid_cell(pos1[0], pos1[1])
                    nearby_nodes = set()

                    # Check neighboring cells
                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            nearby_cell = (cell[0] + i, cell[1] + j)
                            if nearby_cell in self.node_grid:
                                nearby_nodes.update(
                                    n for n in self.node_grid[nearby_cell]
                                    if n in main_sample
                                )

                    # Check distances to nearby nodes
                    for node2 in nearby_nodes:
                        pos2 = (self.graph.nodes[node2]['lat'],
                                self.graph.nodes[node2]['lon'])
                        dist = self._haversine_distance(pos1[0], pos1[1],
                                                        pos2[0], pos2[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (node1, node2)

                if best_pair:
                    # Add connection
                    node1, node2 = best_pair
                    self.graph.add_edge(node1, node2,
                                        distance=min_dist,
                                        time=min_dist / 30 * 60)
                    self.graph.add_edge(node2, node1,
                                        distance=min_dist,
                                        time=min_dist / 30 * 60)
                    logger.info(f"Connected component of size {len(component)} "
                                f"with distance {min_dist:.3f}km")

    def _prune_redundant_edges(self) -> int:
        """Remove unnecessary edges while maintaining connectivity"""
        edges_before = self.graph.size()
        edges_to_remove = []

        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) > 4:  # If too many connections
                # Keep only closest 4 neighbors
                distances = [(n, self.graph[node][n]['distance']) for n in neighbors]
                distances.sort(key=lambda x: x[1])

                # Mark excess edges for removal
                edges_to_remove.extend([
                    (node, n) for n, _ in distances[4:]
                ])

        self.graph.remove_edges_from(edges_to_remove)
        return edges_before - self.graph.size()

    # Helper methods
    def _get_grid_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """Get grid cell indices for coordinates"""
        return (int(lat / self.grid_size), int(lon / self.grid_size))

    def _get_nearby_nodes(self, cell: Tuple[int, int]) -> List[str]:
        """Get nodes from cell and adjacent cells"""
        nodes = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                nearby_cell = (cell[0] + i, cell[1] + j)
                nodes.extend(self.node_grid[nearby_cell])
        return nodes

    def _is_street_aligned(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> bool:
        """Check if points align with Manhattan street grid"""
        lat_diff = abs(lat1 - lat2)
        lon_diff = abs(lon1 - lon2)

        # Direct alignment
        if lat_diff < 0.0001 or lon_diff < 0.0001:
            return True

        # Manhattan grid alignment
        angle = degrees(atan2(lat_diff, lon_diff)) % 90
        return abs(angle - 29) < 5 or abs(angle - 61) < 5

    def _get_angle(self, pos1: Tuple[float, float],
                   pos2: Tuple[float, float]) -> float:
        """Calculate angle between two points"""
        return degrees(atan2(pos2[0] - pos1[0], pos2[1] - pos1[1]))

    def _haversine_distance(self, lat1: float, lon1: float,
                            lat2: float, lon2: float) -> float:
        """Calculate great circle distance between points"""
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c


