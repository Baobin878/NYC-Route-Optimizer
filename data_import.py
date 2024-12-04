def dijkstra(self, start: str, end: str) -> Tuple[float, List[str]]:

    # Initialize distances
    self.distances = {node: float('infinity') for node in self.graph}
    self.distances[start] = 0

    # Initialize priority queue
    pq = [(0, start)]
    paths = {start: [start]}

    while pq:
        current_distance, current = heapq.heappop(pq)

        # Found destination
        if current == end:
            return current_distance, paths[current]

        # Already found a better path
        if current_distance > self.distances[current]:
            continue

        # Check neighbors
        for neighbor, weight in self.graph[current].items():
            distance = current_distance + weight

            # Found better path
            if distance < self.distances[neighbor]:
                self.distances[neighbor] = distance
                paths[neighbor] = paths[current] + [neighbor]
                heapq.heappush(pq, (distance, neighbor))

    return float('infinity'), []