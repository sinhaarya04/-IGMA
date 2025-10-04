"""
Graph Analyst Agent - Extension of TradingAgents for Graph Visualization Challenge
Leverages our existing multi-agent architecture for graph analysis
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time

def create_graph_analyst(llm, toolkit):
    """Create a graph analyst agent that leverages our existing TradingAgents architecture"""
    
    def graph_analyst_node(state):
        # Extract graph data from state (could be from various sources)
        graph_data = state.get("graph_data", {})
        graph_type = state.get("graph_type", "unknown")
        analysis_requirements = state.get("analysis_requirements", "comprehensive")
        
        # Use our existing LLM for graph analysis
        system_message = f"""You are a Graph Analysis Expert, part of the TradingAgents multi-agent system. Your role is to analyze graph structures and provide insights for visualization decisions.

You have access to the same sophisticated reasoning capabilities as our financial analysts, but applied to graph data:

1. **Graph Structure Analysis**: Analyze nodes, edges, density, clustering
2. **Community Detection**: Identify clusters and communities
3. **Centrality Analysis**: Find important nodes and bridges
4. **Visualization Strategy**: Recommend optimal visualization approach
5. **Performance Assessment**: Evaluate graph complexity and rendering challenges

Current Analysis:
- Graph Type: {graph_type}
- Analysis Requirements: {analysis_requirements}
- Graph Data: {graph_data}

Provide a comprehensive analysis following our TradingAgents methodology:
- Detailed structural insights
- Community and centrality analysis  
- Visualization recommendations
- Performance considerations
- Risk assessment for rendering

End with a clear recommendation in this format:
**GRAPH VISUALIZATION STRATEGY: [STRATEGY_NAME]**
**CONFIDENCE SCORE: [0.0-1.0]**
**REASONING: [Brief explanation]**
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        messages = prompt.format_messages(messages=state["messages"])
        result = llm.invoke(messages)
        
        return {
            "messages": [result],
            "graph_analysis_report": result.content,
            "sender": "graph_analyst",
        }
    
    return graph_analyst_node

class GraphAnalysisToolkit:
    """Toolkit for graph analysis using our existing infrastructure"""
    
    def __init__(self, config):
        self.config = config
    
    def analyze_graph_structure(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze graph structure using our existing analysis patterns"""
        
        # Basic metrics
        metrics = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_connected": nx.is_connected(graph),
            "components": nx.number_connected_components(graph),
            "clustering": nx.average_clustering(graph),
            "transitivity": nx.transitivity(graph)
        }
        
        # Advanced metrics
        try:
            if nx.is_connected(graph):
                metrics["average_path_length"] = nx.average_shortest_path_length(graph)
                metrics["diameter"] = nx.diameter(graph)
        except:
            metrics["average_path_length"] = 0
            metrics["diameter"] = 0
        
        # Centrality analysis
        try:
            centrality = nx.betweenness_centrality(graph)
            metrics["max_centrality"] = max(centrality.values())
            metrics["central_nodes"] = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        except:
            metrics["max_centrality"] = 0
            metrics["central_nodes"] = []
        
        # Community detection
        try:
            communities = list(nx.community.greedy_modularity_communities(graph))
            metrics["communities"] = len(communities)
            metrics["modularity"] = nx.community.modularity(graph, communities)
        except:
            metrics["communities"] = 1
            metrics["modularity"] = 0
        
        return metrics
    
    def get_visualization_strategy(self, metrics: Dict[str, Any], llm) -> str:
        """Get visualization strategy using our existing LLM reasoning"""
        
        prompt = f"""
        As a Graph Visualization Expert in the TradingAgents system, analyze these graph metrics and recommend the optimal visualization strategy:

        Graph Metrics:
        - Nodes: {metrics['nodes']}
        - Edges: {metrics['edges']}
        - Density: {metrics['density']:.3f}
        - Clustering: {metrics['clustering']:.3f}
        - Communities: {metrics['communities']}
        - Modularity: {metrics['modularity']:.3f}
        - Connected: {metrics['is_connected']}
        - Components: {metrics['components']}

        Available Strategies:
        1. "hierarchical" - For tree-like structures
        2. "force_directed" - For general networks with communities
        3. "circular" - For small to medium graphs
        4. "matrix" - For dense graphs
        5. "modular" - For graphs with clear communities
        6. "interactive" - For large graphs requiring exploration
        7. "sampled" - For very large graphs

        Consider our TradingAgents principles:
        - Performance optimization
        - User interpretability
        - Scalability
        - Error handling

        Return only the strategy name and confidence score (0.0-1.0).
        """
        
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except:
            # Fallback to rule-based selection
            if metrics['nodes'] < 100:
                return "circular"
            elif metrics['communities'] > 5:
                return "modular"
            elif metrics['density'] > 0.1:
                return "matrix"
            else:
                return "force_directed"
    
    def create_visualization(self, graph: nx.Graph, strategy: str, metrics: Dict[str, Any]) -> go.Figure:
        """Create visualization using our existing Plotly capabilities"""
        
        if strategy == "circular":
            return self._create_circular_layout(graph, metrics)
        elif strategy == "force_directed":
            return self._create_force_directed_layout(graph, metrics)
        elif strategy == "matrix":
            return self._create_matrix_layout(graph, metrics)
        elif strategy == "modular":
            return self._create_modular_layout(graph, metrics)
        elif strategy == "hierarchical":
            return self._create_hierarchical_layout(graph, metrics)
        elif strategy == "interactive":
            return self._create_interactive_layout(graph, metrics)
        elif strategy == "sampled":
            return self._create_sampled_layout(graph, metrics)
        else:
            return self._create_force_directed_layout(graph, metrics)
    
    def _create_circular_layout(self, graph: nx.Graph, metrics: Dict[str, Any]) -> go.Figure:
        """Create circular layout visualization"""
        
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
            title=f"Circular Graph Visualization<br><sub>Nodes: {metrics['nodes']}, Edges: {metrics['edges']}</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _create_force_directed_layout(self, graph: nx.Graph, metrics: Dict[str, Any]) -> go.Figure:
        """Create force-directed layout visualization"""
        
        # Sample if too large
        if metrics['nodes'] > 1000:
            graph = self._sample_graph(graph, 1000)
            metrics = self.analyze_graph_structure(graph)
        
        try:
            pos = nx.spring_layout(graph, k=1/np.sqrt(metrics['nodes']), iterations=100)
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
            title=f"Force-Directed Graph Visualization<br><sub>Nodes: {metrics['nodes']}, Edges: {metrics['edges']}</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _create_matrix_layout(self, graph: nx.Graph, metrics: Dict[str, Any]) -> go.Figure:
        """Create adjacency matrix visualization"""
        
        adj_matrix = nx.adjacency_matrix(graph).todense()
        
        fig = go.Figure(data=go.Heatmap(
            z=adj_matrix,
            colorscale='Blues',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Adjacency Matrix Visualization<br><sub>Nodes: {metrics['nodes']}, Edges: {metrics['edges']}</sub>",
            xaxis_title="Node Index",
            yaxis_title="Node Index"
        )
        
        return fig
    
    def _create_modular_layout(self, graph: nx.Graph, metrics: Dict[str, Any]) -> go.Figure:
        """Create modular layout for community visualization"""
        
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
            
            subgraph = graph.subgraph(community)
            
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
            title=f"Modular Graph Visualization<br><sub>Nodes: {metrics['nodes']}, Communities: {len(communities)}</sub>",
            showlegend=False,
            height=300 * rows
        )
        
        return fig
    
    def _create_hierarchical_layout(self, graph: nx.Graph, metrics: Dict[str, Any]) -> go.Figure:
        """Create hierarchical layout visualization"""
        
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
            title=f"Hierarchical Graph Visualization<br><sub>Nodes: {metrics['nodes']}, Edges: {metrics['edges']}</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _create_interactive_layout(self, graph: nx.Graph, metrics: Dict[str, Any]) -> go.Figure:
        """Create interactive layout for large graphs"""
        
        # Sample if too large
        if metrics['nodes'] > 1000:
            graph = self._sample_graph(graph, 1000)
            metrics = self.analyze_graph_structure(graph)
        
        try:
            pos = nx.spring_layout(graph, k=1/np.sqrt(metrics['nodes']), iterations=100)
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
            title=f"Interactive Graph Visualization<br><sub>Nodes: {metrics['nodes']}, Edges: {metrics['edges']}</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _create_sampled_layout(self, graph: nx.Graph, metrics: Dict[str, Any]) -> go.Figure:
        """Create sampled layout for very large graphs"""
        
        # Sample the graph
        sampled_graph = self._sample_graph(graph, 1000)
        sampled_metrics = self.analyze_graph_structure(sampled_graph)
        
        # Use force-directed layout on sampled graph
        return self._create_force_directed_layout(sampled_graph, sampled_metrics)
    
    def _sample_graph(self, graph: nx.Graph, max_nodes: int) -> nx.Graph:
        """Intelligent graph sampling"""
        
        if graph.number_of_nodes() <= max_nodes:
            return graph
        
        # Use degree-based sampling
        degrees = dict(graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        selected_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
        
        return graph.subgraph(selected_nodes)
