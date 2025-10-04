"""
Advanced Graph Generator for Testing and Demonstration
Creates various types of graphs to test visualization capabilities
"""

import networkx as nx
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class GraphConfig:
    """Configuration for graph generation"""
    graph_type: str
    nodes: int
    edges: int = None
    density: float = None
    communities: int = None
    seed: int = None
    additional_params: Dict[str, Any] = None

class GraphGenerator:
    """Advanced graph generator for testing visualization capabilities"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_graph(self, config: GraphConfig) -> nx.Graph:
        """Generate graph based on configuration"""
        
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        if config.graph_type == "erdos_renyi":
            return self._generate_erdos_renyi(config)
        elif config.graph_type == "barabasi_albert":
            return self._generate_barabasi_albert(config)
        elif config.graph_type == "watts_strogatz":
            return self._generate_watts_strogatz(config)
        elif config.graph_type == "stochastic_block":
            return self._generate_stochastic_block(config)
        elif config.graph_type == "scale_free":
            return self._generate_scale_free(config)
        elif config.graph_type == "small_world":
            return self._generate_small_world(config)
        elif config.graph_type == "random_tree":
            return self._generate_random_tree(config)
        elif config.graph_type == "complete":
            return self._generate_complete(config)
        elif config.graph_type == "bipartite":
            return self._generate_bipartite(config)
        elif config.graph_type == "hierarchical":
            return self._generate_hierarchical(config)
        elif config.graph_type == "temporal":
            return self._generate_temporal(config)
        elif config.graph_type == "financial":
            return self._generate_financial_network(config)
        elif config.graph_type == "social":
            return self._generate_social_network(config)
        elif config.graph_type == "biological":
            return self._generate_biological_network(config)
        elif config.graph_type == "transportation":
            return self._generate_transportation_network(config)
        else:
            raise ValueError(f"Unknown graph type: {config.graph_type}")
    
    def _generate_erdos_renyi(self, config: GraphConfig) -> nx.Graph:
        """Generate Erdos-Renyi random graph"""
        
        if config.density is not None:
            p = config.density
        elif config.edges is not None:
            p = (2 * config.edges) / (config.nodes * (config.nodes - 1))
        else:
            p = 0.1  # Default density
        
        return nx.erdos_renyi_graph(config.nodes, p, seed=config.seed)
    
    def _generate_barabasi_albert(self, config: GraphConfig) -> nx.Graph:
        """Generate Barabasi-Albert scale-free graph"""
        
        m = config.additional_params.get('m', 2) if config.additional_params else 2
        return nx.barabasi_albert_graph(config.nodes, m, seed=config.seed)
    
    def _generate_watts_strogatz(self, config: GraphConfig) -> nx.Graph:
        """Generate Watts-Strogatz small-world graph"""
        
        k = config.additional_params.get('k', 4) if config.additional_params else 4
        p = config.additional_params.get('p', 0.1) if config.additional_params else 0.1
        
        return nx.watts_strogatz_graph(config.nodes, k, p, seed=config.seed)
    
    def _generate_stochastic_block(self, config: GraphConfig) -> nx.Graph:
        """Generate stochastic block model graph"""
        
        if config.communities is None:
            communities = 3
        else:
            communities = config.communities
        
        # Create community sizes
        community_sizes = [config.nodes // communities] * communities
        remainder = config.nodes % communities
        for i in range(remainder):
            community_sizes[i] += 1
        
        # Create probability matrix
        p_in = config.additional_params.get('p_in', 0.3) if config.additional_params else 0.3
        p_out = config.additional_params.get('p_out', 0.05) if config.additional_params else 0.05
        
        probs = np.full((communities, communities), p_out)
        np.fill_diagonal(probs, p_in)
        
        return nx.stochastic_block_model(community_sizes, probs, seed=config.seed)
    
    def _generate_scale_free(self, config: GraphConfig) -> nx.Graph:
        """Generate scale-free graph with power-law degree distribution"""
        
        # Use Barabasi-Albert as base
        m = config.additional_params.get('m', 2) if config.additional_params else 2
        return nx.barabasi_albert_graph(config.nodes, m, seed=config.seed)
    
    def _generate_small_world(self, config: GraphConfig) -> nx.Graph:
        """Generate small-world graph"""
        
        k = config.additional_params.get('k', 4) if config.additional_params else 4
        p = config.additional_params.get('p', 0.1) if config.additional_params else 0.1
        
        return nx.watts_strogatz_graph(config.nodes, k, p, seed=config.seed)
    
    def _generate_random_tree(self, config: GraphConfig) -> nx.Graph:
        """Generate random tree"""
        
        return nx.random_tree(config.nodes, seed=config.seed)
    
    def _generate_complete(self, config: GraphConfig) -> nx.Graph:
        """Generate complete graph"""
        
        return nx.complete_graph(config.nodes)
    
    def _generate_bipartite(self, config: GraphConfig) -> nx.Graph:
        """Generate bipartite graph"""
        
        n1 = config.additional_params.get('n1', config.nodes // 2) if config.additional_params else config.nodes // 2
        n2 = config.nodes - n1
        p = config.additional_params.get('p', 0.1) if config.additional_params else 0.1
        
        return nx.bipartite.random_graph(n1, n2, p, seed=config.seed)
    
    def _generate_hierarchical(self, config: GraphConfig) -> nx.Graph:
        """Generate hierarchical graph"""
        
        # Create a tree-like structure with additional edges
        graph = nx.random_tree(config.nodes, seed=config.seed)
        
        # Add some additional edges to create hierarchy
        additional_edges = config.additional_params.get('additional_edges', 10) if config.additional_params else 10
        
        for _ in range(additional_edges):
            u = random.randint(0, config.nodes - 1)
            v = random.randint(0, config.nodes - 1)
            if u != v and not graph.has_edge(u, v):
                graph.add_edge(u, v)
        
        return graph
    
    def _generate_temporal(self, config: GraphConfig) -> nx.Graph:
        """Generate temporal graph (simplified version)"""
        
        # Create base graph
        graph = nx.erdos_renyi_graph(config.nodes, 0.1, seed=config.seed)
        
        # Add temporal attributes (simplified)
        for node in graph.nodes():
            graph.nodes[node]['timestamp'] = random.randint(0, 100)
        
        for edge in graph.edges():
            graph.edges[edge]['timestamp'] = random.randint(0, 100)
        
        return graph
    
    def _generate_financial_network(self, config: GraphConfig) -> nx.Graph:
        """Generate financial network graph"""
        
        # Create a scale-free network with financial characteristics
        graph = nx.barabasi_albert_graph(config.nodes, 2, seed=config.seed)
        
        # Add financial attributes
        for node in graph.nodes():
            graph.nodes[node]['type'] = random.choice(['bank', 'hedge_fund', 'insurance', 'pension_fund'])
            graph.nodes[node]['size'] = random.uniform(0.1, 1.0)
            graph.nodes[node]['risk'] = random.uniform(0.0, 1.0)
        
        for edge in graph.edges():
            graph.edges[edge]['weight'] = random.uniform(0.1, 1.0)
            graph.edges[edge]['type'] = random.choice(['loan', 'derivative', 'equity', 'bond'])
        
        return graph
    
    def _generate_social_network(self, config: GraphConfig) -> nx.Graph:
        """Generate social network graph"""
        
        # Create a small-world network with social characteristics
        graph = nx.watts_strogatz_graph(config.nodes, 4, 0.1, seed=config.seed)
        
        # Add social attributes
        for node in graph.nodes():
            graph.nodes[node]['age'] = random.randint(18, 80)
            graph.nodes[node]['gender'] = random.choice(['M', 'F', 'Other'])
            graph.nodes[node]['location'] = random.choice(['US', 'EU', 'Asia', 'Other'])
            graph.nodes[node]['influence'] = random.uniform(0.0, 1.0)
        
        for edge in graph.edges():
            graph.edges[edge]['strength'] = random.uniform(0.1, 1.0)
            graph.edges[edge]['type'] = random.choice(['friend', 'colleague', 'family', 'acquaintance'])
        
        return graph
    
    def _generate_biological_network(self, config: GraphConfig) -> nx.Graph:
        """Generate biological network graph"""
        
        # Create a scale-free network with biological characteristics
        graph = nx.barabasi_albert_graph(config.nodes, 2, seed=config.seed)
        
        # Add biological attributes
        for node in graph.nodes():
            graph.nodes[node]['type'] = random.choice(['protein', 'gene', 'metabolite', 'pathway'])
            graph.nodes[node]['expression'] = random.uniform(0.0, 1.0)
            graph.nodes[node]['function'] = random.choice(['metabolism', 'signaling', 'regulation', 'structure'])
        
        for edge in graph.edges():
            graph.edges[edge]['weight'] = random.uniform(0.1, 1.0)
            graph.edges[edge]['type'] = random.choice(['interaction', 'regulation', 'binding', 'catalysis'])
        
        return graph
    
    def _generate_transportation_network(self, config: GraphConfig) -> nx.Graph:
        """Generate transportation network graph"""
        
        # Create a network with transportation characteristics
        graph = nx.erdos_renyi_graph(config.nodes, 0.05, seed=config.seed)
        
        # Add transportation attributes
        for node in graph.nodes():
            graph.nodes[node]['type'] = random.choice(['airport', 'station', 'port', 'hub'])
            graph.nodes[node]['capacity'] = random.randint(100, 10000)
            graph.nodes[node]['location'] = (random.uniform(-180, 180), random.uniform(-90, 90))
        
        for edge in graph.edges():
            graph.edges[edge]['distance'] = random.uniform(10, 1000)
            graph.edges[edge]['type'] = random.choice(['flight', 'train', 'ship', 'truck'])
            graph.edges[edge]['capacity'] = random.randint(50, 500)
        
        return graph
    
    def generate_test_suite(self) -> List[Tuple[str, nx.Graph]]:
        """Generate a comprehensive test suite of graphs"""
        
        test_graphs = []
        
        # Small graphs
        test_graphs.append(("Small ER", self.generate_graph(GraphConfig("erdos_renyi", 50, density=0.1))))
        test_graphs.append(("Small BA", self.generate_graph(GraphConfig("barabasi_albert", 50))))
        test_graphs.append(("Small WS", self.generate_graph(GraphConfig("watts_strogatz", 50))))
        
        # Medium graphs
        test_graphs.append(("Medium ER", self.generate_graph(GraphConfig("erdos_renyi", 500, density=0.05))))
        test_graphs.append(("Medium BA", self.generate_graph(GraphConfig("barabasi_albert", 500))))
        test_graphs.append(("Medium SBM", self.generate_graph(GraphConfig("stochastic_block", 500, communities=5))))
        
        # Large graphs
        test_graphs.append(("Large ER", self.generate_graph(GraphConfig("erdos_renyi", 2000, density=0.01))))
        test_graphs.append(("Large BA", self.generate_graph(GraphConfig("barabasi_albert", 2000))))
        test_graphs.append(("Large SBM", self.generate_graph(GraphConfig("stochastic_block", 2000, communities=10))))
        
        # Very large graphs
        test_graphs.append(("Very Large ER", self.generate_graph(GraphConfig("erdos_renyi", 5000, density=0.005))))
        test_graphs.append(("Very Large BA", self.generate_graph(GraphConfig("barabasi_albert", 5000))))
        
        # Specialized graphs
        test_graphs.append(("Financial Network", self.generate_graph(GraphConfig("financial", 1000))))
        test_graphs.append(("Social Network", self.generate_graph(GraphConfig("social", 1000))))
        test_graphs.append(("Biological Network", self.generate_graph(GraphConfig("biological", 1000))))
        test_graphs.append(("Transportation Network", self.generate_graph(GraphConfig("transportation", 1000))))
        
        # Tree and hierarchical
        test_graphs.append(("Random Tree", self.generate_graph(GraphConfig("random_tree", 200))))
        test_graphs.append(("Hierarchical", self.generate_graph(GraphConfig("hierarchical", 500))))
        
        # Complete and bipartite
        test_graphs.append(("Complete Small", self.generate_graph(GraphConfig("complete", 20))))
        test_graphs.append(("Bipartite", self.generate_graph(GraphConfig("bipartite", 100))))
        
        return test_graphs
    
    def generate_challenge_graphs(self) -> List[Tuple[str, nx.Graph]]:
        """Generate graphs specifically for the visualization challenge"""
        
        challenge_graphs = []
        
        # Challenge 1: Dense graph
        challenge_graphs.append(("Dense Graph", self.generate_graph(GraphConfig("erdos_renyi", 1000, density=0.3))))
        
        # Challenge 2: Sparse graph
        challenge_graphs.append(("Sparse Graph", self.generate_graph(GraphConfig("erdos_renyi", 1000, density=0.001))))
        
        # Challenge 3: High clustering
        challenge_graphs.append(("High Clustering", self.generate_graph(GraphConfig("watts_strogatz", 1000, additional_params={'k': 20, 'p': 0.01}))))
        
        # Challenge 4: Many communities
        challenge_graphs.append(("Many Communities", self.generate_graph(GraphConfig("stochastic_block", 1000, communities=20))))
        
        # Challenge 5: Scale-free
        challenge_graphs.append(("Scale-Free", self.generate_graph(GraphConfig("barabasi_albert", 1000, additional_params={'m': 1}))))
        
        # Challenge 6: Very large
        challenge_graphs.append(("Very Large", self.generate_graph(GraphConfig("erdos_renyi", 10000, density=0.001))))
        
        # Challenge 7: Mixed structure
        challenge_graphs.append(("Mixed Structure", self.generate_graph(GraphConfig("stochastic_block", 2000, communities=5, additional_params={'p_in': 0.2, 'p_out': 0.01}))))
        
        # Challenge 8: Financial network
        challenge_graphs.append(("Financial Network", self.generate_graph(GraphConfig("financial", 2000))))
        
        # Challenge 9: Social network
        challenge_graphs.append(("Social Network", self.generate_graph(GraphConfig("social", 2000))))
        
        # Challenge 10: Biological network
        challenge_graphs.append(("Biological Network", self.generate_graph(GraphConfig("biological", 2000))))
        
        return challenge_graphs
    
    def save_graph(self, graph: nx.Graph, filename: str, format: str = "graphml"):
        """Save graph to file"""
        
        if format == "graphml":
            nx.write_graphml(graph, filename)
        elif format == "gexf":
            nx.write_gexf(graph, filename)
        elif format == "gml":
            nx.write_gml(graph, filename)
        elif format == "edgelist":
            nx.write_edgelist(graph, filename)
        elif format == "adjlist":
            nx.write_adjlist(graph, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"📁 Graph saved to {filename}")
    
    def load_graph(self, filename: str, format: str = "graphml") -> nx.Graph:
        """Load graph from file"""
        
        if format == "graphml":
            return nx.read_graphml(filename)
        elif format == "gexf":
            return nx.read_gexf(filename)
        elif format == "gml":
            return nx.read_gml(filename)
        elif format == "edgelist":
            return nx.read_edgelist(filename)
        elif format == "adjlist":
            return nx.read_adjlist(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_graph_statistics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Get comprehensive statistics about a graph"""
        
        stats = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_connected": nx.is_connected(graph),
            "number_of_components": nx.number_connected_components(graph),
            "average_clustering": nx.average_clustering(graph),
            "transitivity": nx.transitivity(graph),
            "average_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
            "degree_assortativity": nx.degree_assortativity_coefficient(graph),
        }
        
        # Add path length statistics if connected
        if nx.is_connected(graph):
            stats["average_path_length"] = nx.average_shortest_path_length(graph)
            stats["diameter"] = nx.diameter(graph)
            stats["radius"] = nx.radius(graph)
        
        # Add centrality statistics
        try:
            centrality = nx.betweenness_centrality(graph)
            stats["max_betweenness_centrality"] = max(centrality.values())
            stats["average_betweenness_centrality"] = sum(centrality.values()) / len(centrality)
        except:
            stats["max_betweenness_centrality"] = 0
            stats["average_betweenness_centrality"] = 0
        
        return stats
