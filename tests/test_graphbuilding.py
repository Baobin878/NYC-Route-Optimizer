import unittest
from pathlib import Path
import sys
import pandas as pd
import networkx as nx
import logging

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.graph_builder import StreetGraphBuilder
from src.data.data_preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestGraphBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.data_path = project_root / 'data' / 'sampled_cleaned_data.csv'
        cls.preprocessor = DataPreprocessor()

        try:
            # Load and preprocess test data
            raw_data = pd.read_csv(str(cls.data_path))
            cls.test_data = cls.preprocessor.preprocess(raw_data)
            logger.info(f"Prepared {len(cls.test_data)} records for testing")
        except Exception as e:
            logger.error(f"Error preparing test data: {str(e)}")
            raise

    def setUp(self):
        """Set up for each test"""
        self.graph_builder = StreetGraphBuilder(self.test_data)

    def test_1_graph_creation(self):
        """Test basic graph creation"""
        logger.info("Testing graph creation...")

        graph = self.graph_builder.build_graph()

        # Basic checks
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)

        logger.info(f"Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    def test_2_node_attributes(self):
        """Test node attributes"""
        logger.info("Testing node attributes...")

        graph = self.graph_builder.build_graph()
        first_node = list(graph.nodes)[0]

        # Check node attributes
        self.assertIn('lat', graph.nodes[first_node])
        self.assertIn('lon', graph.nodes[first_node])
        self.assertIsInstance(graph.nodes[first_node]['lat'], float)
        self.assertIsInstance(graph.nodes[first_node]['lon'], float)

    def test_3_edge_attributes(self):
        """Test edge attributes"""
        logger.info("Testing edge attributes...")

        graph = self.graph_builder.build_graph()

        # Get first edge that exists
        if len(graph.edges) > 0:
            first_edge = list(graph.edges)[0]
            edge_data = graph.edges[first_edge]

            # Check edge attributes
            self.assertIn('distance', edge_data)
            self.assertIn('weight', edge_data)
            self.assertGreater(edge_data['distance'], 0)

            logger.info(f"Edge attributes: {edge_data.keys()}")

    def test_4_distance_calculation(self):
        """Test distance calculations"""
        logger.info("Testing distance calculations...")

        # Test with known coordinates
        lat1, lon1 = 40.7128, -74.0060  # New York City
        lat2, lon2 = 40.7614, -73.9776  # Central Park

        distance = self.graph_builder._haversine_distance(lat1, lon1, lat2, lon2)

        # Distance should be reasonable
        self.assertGreater(distance, 0)
        self.assertLess(distance, 10)  # Should be less than 10km

        logger.info(f"Calculated distance: {distance:.2f} km")

    def test_5_graph_connectivity(self):
        """Test graph connectivity"""
        logger.info("Testing graph connectivity...")

        graph = self.graph_builder.build_graph()

        # For directed graph, use weakly_connected_components
        components = list(nx.weakly_connected_components(graph))
        largest_component = max(components, key=len)

        logger.info(f"Number of components: {len(components)}")
        logger.info(f"Largest component size: {len(largest_component)}")

        # Graph should have some connected components
        self.assertGreater(len(components), 0)
        self.assertGreater(len(largest_component), 1)

    def test_6_nearest_node(self):
        """Test nearest node finding"""
        logger.info("Testing nearest node finding...")

        # Build graph first
        graph = self.graph_builder.build_graph()

        # Test coordinates (NYC area)
        test_lat, test_lon = 40.7128, -74.0060

        # Find nearest node
        nearest = self.graph_builder.get_nearest_node(test_lat, test_lon)

        # Verify result
        self.assertIsNotNone(nearest)
        self.assertIn(nearest, graph.nodes)

        # Get node coordinates
        node_lat = graph.nodes[nearest]['lat']
        node_lon = graph.nodes[nearest]['lon']

        logger.info(f"Test point: ({test_lat}, {test_lon})")
        logger.info(f"Nearest node: ({node_lat}, {node_lon})")

    def tearDown(self):
        """Clean up after each test"""
        self.graph_builder = None


if __name__ == '__main__':
    unittest.main(verbosity=2)