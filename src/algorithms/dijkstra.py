
# src/algorithms/dijkstra.py
import networkx as nx
from typing import Dict, List, Tuple, Optional
from heapq import heappush, heappop


def dijkstra_path(graph: nx.DiGraph,
                  start: str,
                  end: str,
                  weight: str = 'distance') -> List[str]:
    """
    Find the shortest path between two nodes using Dijkstra's algorithm

    Args:
        graph: NetworkX graph
        start: Starting node
        end: End node
        weight: Edge weight attribute

    Returns:
        List of nodes representing the path
    """
    # Initialize distances and predecessors
    distances = {node: float('infinity') for node in graph.nodes()}
    distances[start] = 0
    predecessors = {node: None for node in graph.nodes()}

    # Priority queue for nodes to visit
    # Format: (distance, node)
    pq = [(0, start)]

    # Keep track of visited nodes
    visited = set()

    while pq:
        # Get node with smallest distance
        current_distance, current_node = heappop(pq)

        # Skip if we've already processed this node
        if current_node in visited:
            continue

        # Mark as visited
        visited.add(current_node)

        # Exit if we reach the end
        if current_node == end:
            break

        # Check all neighbors
        for neighbor in graph.neighbors(current_node):
            if neighbor in visited:
                continue

            # Calculate distance through current node
            edge_weight = graph[current_node][neighbor].get(weight, 1.0)
            distance = current_distance + edge_weight

            # If we found a shorter path, update it
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heappush(pq, (distance, neighbor))

    # No path found
    if distances[end] == float('infinity'):
        raise nx.NetworkXNoPath(f"No path between {start} and {end}")

    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors[current]

    return path[::-1]  # Reverse path to go from start to end


def dijkstra_path_length(graph: nx.DiGraph,
                         start: str,
                         end: str,
                         weight: str = 'distance') -> float:
    """Calculate the length of the shortest path using Dijkstra's algorithm"""
    path = dijkstra_path(graph, start, end, weight)
    return sum(graph[path[i]][path[i + 1]].get(weight, 1.0)
               for i in range(len(path) - 1))


def all_pairs_dijkstra_path_length(graph: nx.DiGraph,
                                   weight: str = 'distance') -> Dict[Tuple[str, str], float]:
    """Calculate shortest path lengths between all pairs of nodes"""
    path_lengths = {}

    for start in graph.nodes():
        # Run Dijkstra's from each start node
        distances = {node: float('infinity') for node in graph.nodes()}
        distances[start] = 0
        pq = [(0, start)]
        visited = set()

        while pq:
            current_distance, current_node = heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)
            path_lengths[(start, current_node)] = current_distance

            for neighbor in graph.neighbors(current_node):
                if neighbor in visited:
                    continue

                edge_weight = graph[current_node][neighbor].get(weight, 1.0)
                distance = current_distance + edge_weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heappush(pq, (distance, neighbor))

    return path_lengths


