import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Graph:
    """
    Graph class for initializing and managing a directed graph.
    
    Attributes:
        graph: Dictionary where keys represent nodes, and values are lists of nodes connected from the key node.
        weight: Dictionary where keys represent nodes, and values are lists of weights corresponding to edges connected from the key node.
        heuristic: Dictionary where keys represent nodes, and values are heuristic values from the source to the goal.
    """

    def __init__(self):
        """
        Initializes the graph, weight, and heuristic dictionaries.
        """
        self.graph = {}
        self.weight = {}
        self.heuristic = {}

    def addEdge(self, o, d, w=1):
        """
        Adds an edge from node o to node d in the directed graph.

        Parameters:
            o: Origin/start/current node.
            d: Destination node.
            w: Weight of the edge (default = 1).
        """
        if o not in self.graph:
            self.graph[o] = []
            self.weight[o] = []
            self.heuristic[o] = 100
        if d not in self.graph:
            self.graph[d] = []
            self.weight[d] = []
            self.heuristic[d] = 100
        self.graph[o].append(d)
        self.weight[o].append(w)

    def addHeuristics(self, o, h):
        """
        Adds heuristic value to the specified node.

        Parameters:
            o: Node for which to add a heuristic value.
            h: Heuristic value (default value = 100).
        """
        self.heuristic[o] = h

class Algorithm:
    """
    This class contains searching techniques that can be used on a directed Graph.
    Parameters:
        g: Directed graph.
        o: Origin/start/current node.
        d: Destination node.
    """

    def DFS(self, g, o, d):
        """
        Depth First Search on a directed graph.

        Parameters:
            g: Directed graph.
            o: Origin/start/current node.
            d: Destination node.
        """
        visited = set()
        stack = [(o, [o])]  # Use a stack to store both the node and its path
        total_path = []
        while stack:
            node, path = stack.pop()
            total_path.append(path)
            if node == d:
                print(path)
                return total_path
            if node not in visited:
                visited.add(node)
                for neighbor in g.graph[node]:
                    stack.append((neighbor, path + [neighbor]))
        return None

    def BFS(self, g, o, d):
        """
        This implements Breadth First Search on a given graph.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        visited = set()
        queue = [(o, [o])]  # Use a queue to store both the node and its path
        total_path = []
        while queue:
            node, path = queue.pop(0)
            total_path.append(path)
            if node == d:
                print(path)
                return total_path
            if node not in visited:
                visited.add(node)
                for neighbor in g.graph[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        return None

    def BMS(self, g, o, d):
        """
        This implements British Museum Search on a given graph.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        paths = []
        stack = [(o, [o])]
        while stack:
            node, path = stack.pop()
            paths.append(path)
            for neighbor in g.graph[node]:
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
        return paths
    
    def HC(self, g, o, d):
        """
        This implements Hill Climbing on a given graph.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        path = []
        total_path = []
        visited = set()
        node = o
        while node != d:
            path.append(node)
            visited.add(node)
            neighbors = g.graph[node]
            neighbor_heuristics = [g.heuristic[neighbor] for neighbor in neighbors]
            best_neighbor = neighbors[neighbor_heuristics.index(min(neighbor_heuristics))]
            if best_neighbor in visited:
                return total_path
            node = best_neighbor
            total_path.append(list(path[:]))
        path.append(d)
        total_path.append(list(path[:]))
        print(path)
        return total_path
    
    def BS(self, g, o, d, bw=1):
        """
        This implements Beam Search on a given graph.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
            bw : determines the beam width required (default value = 1)
        """
        beam = [(g.heuristic[o], (o, [o]))]
        total_path = []
        while beam:
            beam.sort(key=lambda x: x[0])
            best_paths = beam[:bw]
            beam = []
            for misc, (node, path) in best_paths:
                total_path.append(path)
                if node == d:
                    print(path)
                    return total_path
                for neighbor in g.graph[node]:
                    if neighbor not in path:
                        heuristic_score = g.heuristic[neighbor]
                        new_path = path + [neighbor]
                        beam.append((heuristic_score, (neighbor, new_path)))
        return None

    def Oracle(self, g, o, d):
        """
        Oracle search performing an exhaustive search to find all possible paths.
        Returns a list of tuples, each containing a path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        all_paths = []
        total_path = []
        stack = [(o, [], 0)]  # (node, path, cost)
        while stack:
            current, path, cost = stack.pop()
            total_path.append(path+[current])
            if current == d:
                all_paths.append((path + [current], cost))
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        stack.append((neighbor, path + [current], cost + weight))
        print(all_paths)
        return total_path

    def BB(self, g, o, d):
        """
        Branch and Bound algorithm to find the optimal path.
        Returns the optimal path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        best_path = None
        best_cost = float('inf')  # Initialize with positive infinity

        # Priority queue implemented as a list of tuples (cost, node, path)
        priority_queue = [(0, o, [])]
        total_path = []

        while priority_queue:
            # Find the path with the lowest cost in the priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] < priority_queue[min_index][0]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)
            total_path.append(path+[current])
            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        if cost+weight<=best_cost:
                        # Add the neighbor to the priority queue with updated cost
                            priority_queue.append((cost + weight, neighbor, path + [current]))
                
        print(best_path, best_cost)
        return total_path

    def EL(self, g, o, d):
        """
        Branch and Bound algorithm with an extended list.
        Returns the optimal path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        best_path = None
        best_cost = float('inf')  # Initialize with positive infinity

        # Priority queue implemented as a list of tuples (cost, node, path)
        priority_queue = [(0, o, [])]
        total_path = []

        # Extended list to keep track of visited nodes
        extended_list = {node: False for node in g.graph}

        while priority_queue:
            # Find the path with the lowest cost in the priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] < priority_queue[min_index][0]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)
            
            total_path.append(path+[current])
            
            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if not extended_list[current] and not extended_list[neighbor]:
                        if cost+weight<=best_cost:
                        # Add the neighbor to the priority queue with updated cost
                            priority_queue.append((cost + weight, neighbor, path + [current]))
            extended_list[current] = True
        print(best_path, best_cost)
        return total_path
    
    def EH(self, g, o, d):
        """
        Branch and Bound algorithm with estimated heuristics.
        Returns the optimal path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        best_path = None
        best_cost = float('inf')  # Initialize with positive infinity

        # Priority queue implemented as a list of tuples (cost, node, path)
        priority_queue = [(0, o, [])]
        total_path = []

        while priority_queue:
            # Find the path with the lowest cost in the priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] + g.heuristic[priority_queue[i][1]] < priority_queue[min_index][0] + g.heuristic[priority_queue[min_index][1]]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)

            total_path.append(path+[current])
            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        if cost+weight+g.heuristic[current]<=best_cost:
                            # Add the neighbor to the priority queue with updated cost
                            priority_queue.append((cost + weight, neighbor, path + [current]))

        print(best_path, best_cost)
        return total_path
    
    def Astar(self, g, o, d):
        """
        Branch and Bound algorithm with extended list and estimated heuristics.
        Returns the optimal path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        best_path = None
        best_cost = float('inf')  # Initialize with positive infinity

        # Priority queue implemented as a list of tuples (cost, node, path)
        priority_queue = [(0, o, [])]
        total_path = []

        # Extended list to keep track of visited nodes
        extended_list = {node: False for node in g.graph}

        while priority_queue:
            # Find the path with the lowest cost in the priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] + g.heuristic[priority_queue[i][1]] < priority_queue[min_index][0] + g.heuristic[priority_queue[min_index][1]]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)
        
            # Use the extended list to track visited nodes for the current path
            visited = set(path)
            total_path.append(path+[current])

            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if not extended_list[current] and not extended_list[neighbor] and neighbor not in visited:
                        if cost+weight+g.heuristic[current]<=best_cost:
                        # Add the neighbor to the priority queue with updated cost
                            priority_queue.append((cost + weight, neighbor, path + [current]))

            # Mark the current node as visited in the global set
            extended_list[current] = True

        print(best_path, best_cost)
        return total_path
    
    def BestFirstSearch(self, g, o, d):
        """
        Best-First Search algorithm.
        Returns the optimal path.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        best_path = None

        # Priority queue implemented as a list of tuples (heuristic, node, path)
        priority_queue = [(g.heuristic[o], o, [])]
        total_path = []

        while priority_queue:
            # Find the path with the lowest heuristic value in the priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] < priority_queue[min_index][0]:
                    min_index = i
            heuristic, current, path = priority_queue.pop(min_index)

            total_path.append(path+[current])

            if current == d:
                # Destination reached, update best_path
                best_path = path + [current]
                print(best_path)
                return total_path
            else:
                for neighbor in g.graph[current]:
                    if neighbor not in path:
                        # Add the neighbor to the priority queue with updated heuristic
                        priority_queue.append((g.heuristic[neighbor], neighbor, path + [current]))

        print(best_path)
        return total_path
    def Astar_with_heuristics(self, g, o, d):
        """
        A* search algorithm with heuristics.
        Returns the optimal path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        best_path = None
        best_cost = float('inf')  # Initialize with positive infinity

        # Priority queue implemented as a list of tuples (cost, node, path)
        priority_queue = [(g.heuristic[o], 0, o, [])]
        total_path = []

        while priority_queue:
            # Find the path with the lowest f-value (cost + heuristic) in the priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][1] + g.heuristic[priority_queue[i][2]] < priority_queue[min_index][1] + g.heuristic[priority_queue[min_index][2]]:
                    min_index = i
            heuristic, cost, current, path = priority_queue.pop(min_index)

            total_path.append(path + [current])

            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        new_cost = cost + weight
                        if new_cost + g.heuristic[neighbor] <= best_cost:
                            # Add the neighbor to the priority queue with updated cost and f-value
                            priority_queue.append((g.heuristic[neighbor], new_cost, neighbor, path + [current]))

        print(best_path, best_cost)
        return total_path
    def Oracle_with_cost_and_heuristics(self, g, o, d):
        """
        Oracle search with both edge costs and heuristics.
        Returns a list of tuples, each containing a path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        all_paths = []
        total_path = []
        stack = [(o, [], 0)]  # (node, path, cost)
        while stack:
            current, path, cost = stack.pop()
            total_path.append(path + [current])
            if current == d:
                all_paths.append((path + [current], cost))
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        stack.append((neighbor, path + [current], cost + weight))
        print(all_paths)
        return total_path


class GraphVisualization:
    def visualize_traversal(self, g, o, d, traversal_algorithm, bw=1):
        G = nx.DiGraph()
        for node, neighbors in g.graph.items():
            for neighbor, weight in zip(neighbors, g.weight[node]):
                G.add_edge(node, neighbor, weight=weight)

        if traversal_algorithm.__name__ == "BS":
            paths = traversal_algorithm(g, o, d, bw)
        else:
            paths = traversal_algorithm(g, o, d)
        pos = nx.planar_layout(G)  # You can choose a different layout if you prefer.

        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            node_labels = {node: f"{node}\nH:{g.heuristic[node]}" for node in G.nodes()}  # Include heuristic values in node labels
            # Draw the graph
            nx.draw(G, pos, with_labels=True, node_size=900, font_size=10, node_color='orange', font_color='black', font_weight='bold', labels=node_labels, ax=ax)
            edge_labels = {(node, neighbor): G[node][neighbor]['weight'] for node, neighbor in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8, ax=ax)

            # Highlight the path up to the current step
            if frame < len(paths):
                path = paths[frame]
                path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, ax=ax)

        ani = FuncAnimation(fig, update, frames=len(paths) + 1, repeat=False, interval=1000)  # Adjust the interval to control animation speed
        plt.show()

choice = input("Press Enter to continue with default values, otherwise enter 1")
g = Graph()
algo = Algorithm()

if choice == '':
    # Add directed edges and heuristics for the provided graph
    g.addEdge('p', 'a', 4)
    g.addEdge('p', 'r', 4)
    g.addEdge('r', 'e', 1)
    g.addEdge('e', 's', 1)
    g.addEdge('n', 's', 6)
    g.addEdge('l', 'n', 5)
    g.addEdge('m', 'l', 2)
    g.addEdge('a', 'm', 3)
    g.addEdge('p', 'c', 4)
    g.addEdge('c', 'm', 6)
    g.addEdge('m', 'v', 5)
    g.addEdge('c', 'v', 3)
    g.addEdge('v', 'n', 5)
    g.addEdge('v', 's', 4)
    g.addEdge('e', 'v', 5)
    g.addEdge('c', 'r', 2)

    g.addHeuristics('p', 0)  # Heuristics for source 'p'
    g.addHeuristics('a', 4)
    g.addHeuristics('r', 3)
    g.addHeuristics('e', 2)
    g.addHeuristics('s', 0)  # Heuristics for goal 's'
    g.addHeuristics('n', 2)
    g.addHeuristics('l', 5)
    g.addHeuristics('m', 6)
    g.addHeuristics('c', 4)
    g.addHeuristics('v', 1)

else:
    pass


#GraphVisualization().visualize_traversal(g, 'p', 's', algo.BFS)#bredth first search
#GraphVisualization().visualize_traversal(g, 'p', 's', algo.DFS)#depth first search
#GraphVisualization().visualize_traversal(g, 'p', 's', algo.HC)#hill climbing
#GraphVisualization().visualize_traversal(g, 'p', 's', algo.BS)#beam search
#GraphVisualization().visualize_traversal(g, 'p', 's', algo.Oracle)
#GraphVisualization().visualize_traversal(g, 'p', 's',algo.BB)#branch and bound
#GraphVisualization().visualize_traversal(g, 'p', 's', algo.EL)# branch and bound with extended list
#GraphVisualization().visualize_traversal(g, 'p', 's', algo.EH)#branch and bound with heurestics
#GraphVisualization().visualize_traversal(g, 'p', 's', algo.Astar)#A*
#GraphVisualization().visualize_traversal(g, 'p', 's', algo.BestFirstSearch)
#GraphVisualization().visualize_traversal(g, 'p', 's', algo.Oracle_with_cost_and_heuristics )
#GraphVisualization().visualize_traversal(g, 'p', 's', algo.Astar_with_heuristics)
