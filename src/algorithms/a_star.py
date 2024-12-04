# src/algorithms/a_star.py
import networkx as nx
from typing import Dict, List, Tuple, Optional, Callable
from heapq import heappush, heappop
from math import radians, sin, cos, sqrt, atan2


def astar_path(graph: nx.DiGraph,
               start: str,
               end: str,
               weight: str = 'distance',
               heuristic: Optional[Callable] = None) -> List[str]:
    """
    Find the shortest path between two nodes using A* algorithm

    Args:
        graph: NetworkX graph
        start: Starting node
        end: End node
        weight: Edge weight attribute
        heuristic: Optional heuristic function

    Returns:
        List of nodes representing the path
    """
    if not heuristic:
        # Default heuristic using straight-line distance
        def heuristic(n1: str, n2: str) -> float:
            # Get coordinates
            lat1, lon1 = graph.nodes[n1]['lat'], graph.nodes[n1]['lon']
            lat2, lon2 = graph.nodes[n2]['lat'], graph.nodes[n2]['lon']

            # Calculate Haversine distance
            R = 6371  # Earth's radius in km
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c

    # Initialize data structures
    frontier = []  # Priority queue
    heappush(frontier, (0, start))

    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current = heappop(frontier)[1]

        # Exit if we reach the goal
        if current == end:
            break

        # Check all neighbors
        for next_node in graph.neighbors(current):
            # Calculate new cost
            new_cost = cost_so_far[current] + graph[current][next_node].get(weight, 1.0)

            # If we found a better path, update it
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                # Priority = cost + heuristic
                priority = new_cost + heuristic(next_node, end)
                heappush(frontier, (priority, next_node))
                came_from[next_node] = current

    # Reconstruct path
    if end not in came_from:
        raise nx.NetworkXNoPath(f"No path between {start} and {end}")

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = came_from[current]

    return path[::-1]  # Reverse path to go from start to end


def astar_path_length(graph: nx.DiGraph,
                      start: str,
                      end: str,
                      weight: str = 'distance') -> float:
    """Calculate the length of the shortest path using A*"""
    path = astar_path(graph, start, end, weight)
    return sum(graph[path[i]][path[i + 1]].get(weight, 1.0)
               for i in range(len(path) - 1))


