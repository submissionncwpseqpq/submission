import networkx as nx
import math
import pandas as pd
import heapq
from tqdm import tqdm

class TransitiveLabelsGraph:
    def __init__(self, gt_tagging):
        self.gt_tagging = gt_tagging
        
        annotations_df = pd.DataFrame(gt_tagging, columns=['key', 'value'])
        annotations_df['query'] = annotations_df['key'].apply(lambda x: x[0])
        annotations_df['gallery'] = annotations_df['key'].apply(lambda x: x[1])
        annotations_df = annotations_df.drop(columns=['key'])
        
        self.annotations_df = annotations_df

    def create_gt_graphs(self, add_transitive_edges=True, max_positive_length=7):
        G = nx.Graph()
        
        for _, row in self.annotations_df.iterrows():
            query = row['query']
            gallery = row['gallery']
            value = row['value']
            
            weight = 1 if value == 1 else math.inf
        
            G.add_node(query)
            G.add_node(gallery)
            G.add_edge(query, gallery, weight=weight)
        
        if add_transitive_edges:
            G, self.all_paths_within = self.add_transitive_edges(G, max_positive_length)
        

    def add_transitive_edges(self, G, max_positive_length=7, all_paths_within_limit=None):
        if all_paths_within_limit is None:
            all_paths_within_limit = self.get_added_edges(G, max_positive_length)
            
        remove = []
        for (start, end), (path, weight) in tqdm(all_paths_within_limit.items(), desc="Adding edges"):
            if start != end and not G.has_edge(start, end):
                G.add_edge(start, end, weight=weight)
            else:
                remove.append((start, end))

        # Remove edges that already existed in the graph, only used for reporting                
        for (start, end) in tqdm(remove, desc="Removing edges"):
            del all_paths_within_limit[(start, end)]

        print("Found edges: ", len(all_paths_within_limit))
        
        return G, all_paths_within_limit
    
    def get_added_edges(G, max_positive_length=10):
        all_paths_within_limit = {}
    
        for node in tqdm(G.nodes, desc="Computing paths"):
            paths_from_node = dijkstra(node, G, max_positive_length)
            all_paths_within_limit.update(paths_from_node)
        
        return all_paths_within_limit
            
            
def dijkstra(start, graph, max_weight):
    pq = [(0, start, [start])]
    visited = set()
    paths = {}
    shortest_paths = {start: [start]}

    while pq:
        weight, node, _ = heapq.heappop(pq)

        if node not in visited:
            visited.add(node)
            path = shortest_paths[node]
            if node != start and weight <= max_weight:
                paths[(start, node)] = (path, weight)

            for neighbor, attrs in graph[node].items():
                edge_weight = attrs['weight']
                if weight + edge_weight <= max_weight and neighbor not in visited:
                    heapq.heappush(pq, (weight + edge_weight, neighbor, None))
                    shortest_paths[neighbor] = path + [neighbor]

    return paths