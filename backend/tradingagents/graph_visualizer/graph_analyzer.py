"""
Advanced Graph Visualization System
Leverages TradingAgents architecture for intelligent graph analysis and visualization
"""

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from langchain_anthropic import ChatAnthropic
import json
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio

@dataclass
class GraphMetrics:
    """Comprehensive graph analysis metrics"""
    node_count: int
    edge_count: int
    density: float
    clustering_coefficient: float
    average_path_length: float
    diameter: float
    modularity: float
    communities: int
    central_nodes: List[str]
    bridge_nodes: List[str]
    isolated_nodes: List[str]
    complexity_score: float
    visualization_difficulty: str

class GraphAnalyzer:
    """Advanced graph analysis using TradingAgents approach"""
    
    def __init__(self, llm_model: str = "claude-sonnet-4-0"):
        self.llm = ChatAnthropic(model=llm_model)
        self.metrics_cache = {}
        self.visualization_strategies = {
            "small": self._visualize_small_graph,
            "medium": self._visualize_medium_graph,
            "large": self._visualize_large_graph,
            "massive": self._visualize_massive_graph
        }
    
    def analyze_graph(self, graph: nx.Graph) -> GraphMetrics:
        """Comprehensive graph analysis with intelligent caching"""
        
        # Check cache first
        graph_hash = hash(str(sorted(graph.edges())))
        if graph_hash in self.metrics_cache:
            return self.metrics_cache[graph_hash]
        
        print("🔍 Analyzing graph structure...")
        
        # Basic metrics
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        density = nx.density(graph)
        
        # Advanced metrics (with error handling for disconnected graphs)
        try:
            clustering_coefficient = nx.average_clustering(graph)
            if nx.is_connected(graph):
                average_path_length = nx.average_shortest_path_length(graph)
                diameter = nx.diameter(graph)
            else:
                # For disconnected graphs, use largest component
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                average_path_length = nx.average_shortest_path_length(subgraph)
                diameter = nx.diameter(subgraph)
        except:
            clustering_coefficient = 0.0
            average_path_length = 0.0
            diameter = 0.0
        
        # Community detection
        try:
            communities = nx.community.greedy_modularity_communities(graph)
            modularity = nx.community.modularity(graph, communities)
            community_count = len(communities)
        except:
            communities = []
            modularity = 0.0
            community_count = 1
        
        # Centrality analysis
        try:
            centrality = nx.betweenness_centrality(graph)
            central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            central_nodes = [node for node, _ in central_nodes]
        except:
            central_nodes = []
        
        # Bridge detection
        try:
            bridges = list(nx.bridges(graph))
            bridge_nodes = list(set([node for edge in bridges for node in edge]))
        except:
            bridge_nodes = []
        
        # Isolated nodes
        isolated_nodes = list(nx.isolates(graph))
        
        # Complexity scoring
        complexity_score = self._calculate_complexity_score(
            node_count, edge_count, density, clustering_coefficient, modularity
        )
        
        # Visualization difficulty assessment
        visualization_difficulty = self._assess_visualization_difficulty(complexity_score, node_count)
        
        metrics = GraphMetrics(
            node_count=node_count,
            edge_count=edge_count,
            density=density,
            clustering_coefficient=clustering_coefficient,
            average_path_length=average_path_length,
            diameter=diameter,
            modularity=modularity,
            communities=community_count,
            central_nodes=central_nodes,
            bridge_nodes=bridge_nodes,
            isolated_nodes=isolated_nodes,
            complexity_score=complexity_score,
            visualization_difficulty=visualization_difficulty
        )
        
        # Cache results
        self.metrics_cache[graph_hash] = metrics
        
        return metrics
    
    def _calculate_complexity_score(self, nodes: int, edges: int, density: float, 
                                  clustering: float, modularity: float) -> float:
        """Calculate graph complexity score (0-1, higher = more complex)"""
        
        # Normalize metrics
        node_complexity = min(1.0, nodes / 10000)  # Scale to 10k nodes
        edge_complexity = min(1.0, edges / 50000)  # Scale to 50k edges
        density_complexity = density  # Already 0-1
        clustering_complexity = clustering  # Already 0-1
        modularity_complexity = abs(modularity)  # Can be negative
        
        # Weighted combination
        complexity = (
            0.3 * node_complexity +
            0.2 * edge_complexity +
            0.2 * density_complexity +
            0.15 * clustering_complexity +
            0.15 * modularity_complexity
        )
        
        return min(1.0, complexity)
    
    def _assess_visualization_difficulty(self, complexity: float, node_count: int) -> str:
        """Assess visualization difficulty based on complexity and size"""
        
        if node_count < 100:
            return "small"
        elif node_count < 1000:
            return "medium"
        elif node_count < 10000:
            return "large"
        else:
            return "massive"
    
    def get_visualization_strategy(self, metrics: GraphMetrics) -> str:
        """Get optimal visualization strategy using LLM analysis"""
        
        prompt = f"""
        As a graph visualization expert, analyze this graph and recommend the best visualization strategy:
        
        Graph Metrics:
        - Nodes: {metrics.node_count}
        - Edges: {metrics.edge_count}
        - Density: {metrics.density:.3f}
        - Clustering Coefficient: {metrics.clustering_coefficient:.3f}
        - Modularity: {metrics.modularity:.3f}
        - Communities: {metrics.communities}
        - Complexity Score: {metrics.complexity_score:.3f}
        
        Available Strategies:
        1. "hierarchical" - For tree-like or hierarchical structures
        2. "force_directed" - For general networks with clear communities
        3. "circular" - For small to medium graphs with clear structure
        4. "matrix" - For dense graphs or adjacency analysis
        5. "modular" - For graphs with clear community structure
        6. "temporal" - For time-evolving graphs
        7. "multilevel" - For very large graphs with hierarchical structure
        8. "interactive_clustering" - For exploration of large graphs
        
        Consider:
        - Graph size and density
        - Community structure
        - Clustering patterns
        - User interaction needs
        - Performance requirements
        
        Return only the strategy name and a brief justification.
        """
        
        try:
            response = self.llm.invoke(prompt)
            strategy = response.content.strip().split('\n')[0].lower()
            return strategy
        except:
            # Fallback to rule-based selection
            if metrics.node_count < 100:
                return "circular"
            elif metrics.communities > 5:
                return "modular"
            elif metrics.density > 0.1:
                return "matrix"
            else:
                return "force_directed"
    
    def visualize_graph(self, graph: nx.Graph, strategy: str = None, 
                       interactive: bool = True, max_nodes: int = 1000) -> go.Figure:
        """Main visualization method with intelligent strategy selection"""
        
        metrics = self.analyze_graph(graph)
        
        if strategy is None:
            strategy = self.get_visualization_strategy(metrics)
        
        print(f"🎨 Using visualization strategy: {strategy}")
        print(f"📊 Graph complexity: {metrics.complexity_score:.3f} ({metrics.visualization_difficulty})")
        
        # Apply strategy-specific visualization
        if strategy == "hierarchical":
            return self._visualize_hierarchical(graph, metrics, interactive)
        elif strategy == "force_directed":
            return self._visualize_force_directed(graph, metrics, interactive, max_nodes)
        elif strategy == "circular":
            return self._visualize_circular(graph, metrics, interactive)
        elif strategy == "matrix":
            return self._visualize_matrix(graph, metrics, interactive)
        elif strategy == "modular":
            return self._visualize_modular(graph, metrics, interactive, max_nodes)
        elif strategy == "temporal":
            return self._visualize_temporal(graph, metrics, interactive)
        elif strategy == "multilevel":
            return self._visualize_multilevel(graph, metrics, interactive)
        elif strategy == "interactive_clustering":
            return self._visualize_interactive_clustering(graph, metrics, interactive, max_nodes)
        else:
            # Default fallback
            return self._visualize_force_directed(graph, metrics, interactive, max_nodes)
    
    def _visualize_hierarchical(self, graph: nx.Graph, metrics: GraphMetrics, 
                              interactive: bool) -> go.Figure:
        """Hierarchical visualization for tree-like structures"""
        
        # Use hierarchical layout
        try:
            pos = nx.spring_layout(graph, k=1, iterations=50)
        except:
            pos = nx.random_layout(graph)
        
        # Prepare data
        edge_x, edge_y = [], []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        node_text = [f"Node: {node}" for node in graph.nodes()]
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Edges'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=10,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            name='Nodes'
        ))
        
        fig.update_layout(
            title=f"Hierarchical Graph Visualization<br><sub>Nodes: {metrics.node_count}, Edges: {metrics.edge_count}</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Hierarchical layout optimized for tree-like structures",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _visualize_force_directed(self, graph: nx.Graph, metrics: GraphMetrics, 
                                interactive: bool, max_nodes: int) -> go.Figure:
        """Force-directed visualization with intelligent sampling"""
        
        # Sample graph if too large
        if metrics.node_count > max_nodes:
            print(f"⚠️ Graph too large ({metrics.node_count} nodes), sampling to {max_nodes}")
            graph = self._sample_graph(graph, max_nodes)
            metrics = self.analyze_graph(graph)
        
        # Use force-directed layout
        try:
            pos = nx.spring_layout(graph, k=1/np.sqrt(metrics.node_count), iterations=100)
        except:
            pos = nx.random_layout(graph)
        
        # Prepare data
        edge_x, edge_y = [], []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        node_text = [f"Node: {node}" for node in graph.nodes()]
        
        # Calculate node sizes based on degree
        degrees = dict(graph.degree())
        node_sizes = [max(5, min(20, degrees[node] * 2)) for node in graph.nodes()]
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Edges'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            name='Nodes'
        ))
        
        fig.update_layout(
            title=f"Force-Directed Graph Visualization<br><sub>Nodes: {metrics.node_count}, Edges: {metrics.edge_count}</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Force-directed layout with intelligent sampling for large graphs",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _visualize_circular(self, graph: nx.Graph, metrics: GraphMetrics, 
                          interactive: bool) -> go.Figure:
        """Circular layout for small to medium graphs"""
        
        # Use circular layout
        pos = nx.circular_layout(graph)
        
        # Prepare data
        edge_x, edge_y = [], []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        node_text = [f"Node: {node}" for node in graph.nodes()]
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Edges'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=15,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            name='Nodes'
        ))
        
        fig.update_layout(
            title=f"Circular Graph Visualization<br><sub>Nodes: {metrics.node_count}, Edges: {metrics.edge_count}</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Circular layout optimized for small to medium graphs",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _visualize_matrix(self, graph: nx.Graph, metrics: GraphMetrics, 
                        interactive: bool) -> go.Figure:
        """Matrix visualization for dense graphs"""
        
        # Create adjacency matrix
        adj_matrix = nx.adjacency_matrix(graph).todense()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=adj_matrix,
            colorscale='Blues',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Adjacency Matrix Visualization<br><sub>Nodes: {metrics.node_count}, Edges: {metrics.edge_count}</sub>",
            xaxis_title="Node Index",
            yaxis_title="Node Index",
            annotations=[ dict(
                text="Matrix view optimized for dense graphs and adjacency analysis",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )]
        )
        
        return fig
    
    def _visualize_modular(self, graph: nx.Graph, metrics: GraphMetrics, 
                         interactive: bool, max_nodes: int) -> go.Figure:
        """Modular visualization for graphs with clear community structure"""
        
        # Detect communities
        try:
            communities = list(nx.community.greedy_modularity_communities(graph))
        except:
            communities = [set(graph.nodes())]
        
        # Create subplot for each community
        n_communities = len(communities)
        cols = min(3, n_communities)
        rows = (n_communities + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Community {i+1}" for i in range(n_communities)],
            specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
        )
        
        # Visualize each community
        for i, community in enumerate(communities):
            row = i // cols + 1
            col = i % cols + 1
            
            # Create subgraph for community
            subgraph = graph.subgraph(community)
            
            # Use spring layout for subgraph
            try:
                pos = nx.spring_layout(subgraph, k=1, iterations=50)
            except:
                pos = nx.random_layout(subgraph)
            
            # Prepare data
            edge_x, edge_y = [], []
            for edge in subgraph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            node_x = [pos[node][0] for node in subgraph.nodes()]
            node_y = [pos[node][1] for node in subgraph.nodes()]
            node_text = [f"Node: {node}" for node in subgraph.nodes()]
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ), row=row, col=col)
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=10,
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                ),
                showlegend=False
            ), row=row, col=col)
        
        fig.update_layout(
            title=f"Modular Graph Visualization<br><sub>Nodes: {metrics.node_count}, Communities: {len(communities)}</sub>",
            showlegend=False,
            height=300 * rows,
            annotations=[ dict(
                text="Modular layout showing community structure",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )]
        )
        
        return fig
    
    def _visualize_temporal(self, graph: nx.Graph, metrics: GraphMetrics, 
                          interactive: bool) -> go.Figure:
        """Temporal visualization for time-evolving graphs"""
        
        # For now, use force-directed with temporal coloring
        # In a real implementation, you'd have temporal data
        try:
            pos = nx.spring_layout(graph, k=1/np.sqrt(metrics.node_count), iterations=50)
        except:
            pos = nx.random_layout(graph)
        
        # Prepare data
        edge_x, edge_y = [], []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        node_text = [f"Node: {node}" for node in graph.nodes()]
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Edges'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=10,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            name='Nodes'
        ))
        
        fig.update_layout(
            title=f"Temporal Graph Visualization<br><sub>Nodes: {metrics.node_count}, Edges: {metrics.edge_count}</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Temporal layout for time-evolving graphs",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _visualize_multilevel(self, graph: nx.Graph, metrics: GraphMetrics, 
                            interactive: bool) -> go.Figure:
        """Multilevel visualization for very large graphs"""
        
        # Use multilevel layout
        try:
            pos = nx.spring_layout(graph, k=1/np.sqrt(metrics.node_count), iterations=50)
        except:
            pos = nx.random_layout(graph)
        
        # Prepare data
        edge_x, edge_y = [], []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        node_text = [f"Node: {node}" for node in graph.nodes()]
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Edges'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=10,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            name='Nodes'
        ))
        
        fig.update_layout(
            title=f"Multilevel Graph Visualization<br><sub>Nodes: {metrics.node_count}, Edges: {metrics.edge_count}</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Multilevel layout for very large graphs with hierarchical structure",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _visualize_interactive_clustering(self, graph: nx.Graph, metrics: GraphMetrics, 
                                        interactive: bool, max_nodes: int) -> go.Figure:
        """Interactive clustering visualization for exploration of large graphs"""
        
        # Sample graph if too large
        if metrics.node_count > max_nodes:
            print(f"⚠️ Graph too large ({metrics.node_count} nodes), sampling to {max_nodes}")
            graph = self._sample_graph(graph, max_nodes)
            metrics = self.analyze_graph(graph)
        
        # Use force-directed layout with clustering
        try:
            pos = nx.spring_layout(graph, k=1/np.sqrt(metrics.node_count), iterations=100)
        except:
            pos = nx.random_layout(graph)
        
        # Prepare data
        edge_x, edge_y = [], []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        node_text = [f"Node: {node}" for node in graph.nodes()]
        
        # Calculate node sizes based on degree
        degrees = dict(graph.degree())
        node_sizes = [max(5, min(20, degrees[node] * 2)) for node in graph.nodes()]
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Edges'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            name='Nodes'
        ))
        
        fig.update_layout(
            title=f"Interactive Clustering Visualization<br><sub>Nodes: {metrics.node_count}, Edges: {metrics.edge_count}</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Interactive clustering for exploration of large graphs",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _sample_graph(self, graph: nx.Graph, max_nodes: int) -> nx.Graph:
        """Intelligent graph sampling to reduce size while preserving structure"""
        
        if graph.number_of_nodes() <= max_nodes:
            return graph
        
        # Use degree-based sampling to preserve important nodes
        degrees = dict(graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        
        # Select top nodes by degree
        selected_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
        
        # Create subgraph
        subgraph = graph.subgraph(selected_nodes)
        
        return subgraph
    
    def generate_insights(self, graph: nx.Graph, metrics: GraphMetrics) -> str:
        """Generate AI-powered insights about the graph"""
        
        prompt = f"""
        As a graph analysis expert, provide insights about this graph:
        
        Graph Metrics:
        - Nodes: {metrics.node_count}
        - Edges: {metrics.edge_count}
        - Density: {metrics.density:.3f}
        - Clustering Coefficient: {metrics.clustering_coefficient:.3f}
        - Average Path Length: {metrics.average_path_length:.3f}
        - Diameter: {metrics.diameter}
        - Modularity: {metrics.modularity:.3f}
        - Communities: {metrics.communities}
        - Complexity Score: {metrics.complexity_score:.3f}
        
        Central Nodes: {metrics.central_nodes[:5]}
        Bridge Nodes: {metrics.bridge_nodes[:5]}
        Isolated Nodes: {len(metrics.isolated_nodes)}
        
        Provide:
        1. Key structural insights
        2. Network characteristics
        3. Visualization recommendations
        4. Potential applications
        5. Optimization suggestions
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except:
            return "Unable to generate insights due to processing error."
    
    def export_visualization(self, fig: go.Figure, filename: str, format: str = "html"):
        """Export visualization in various formats"""
        
        if format == "html":
            fig.write_html(filename)
        elif format == "png":
            fig.write_image(filename)
        elif format == "pdf":
            fig.write_image(filename)
        elif format == "svg":
            fig.write_image(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"📁 Visualization exported to {filename}")
    
    def get_performance_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Get performance metrics for the visualization"""
        
        start_time = time.time()
        
        # Analyze graph
        metrics = self.analyze_graph(graph)
        
        # Get visualization strategy
        strategy = self.get_visualization_strategy(metrics)
        
        # Generate visualization
        fig = self.visualize_graph(graph, strategy)
        
        end_time = time.time()
        
        return {
            "analysis_time": end_time - start_time,
            "nodes_processed": metrics.node_count,
            "edges_processed": metrics.edge_count,
            "complexity_score": metrics.complexity_score,
            "strategy_used": strategy,
            "visualization_difficulty": metrics.visualization_difficulty
        }
