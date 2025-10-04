"""
TradingAgents Graph Visualizer Demo
Demonstrates advanced graph visualization capabilities for the challenge
"""

import os
import sys
import time
import asyncio
from typing import List, Tuple, Dict, Any
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_analyzer import GraphAnalyzer, GraphMetrics
from graph_generator import GraphGenerator, GraphConfig

class GraphVisualizationDemo:
    """Demo class for the TradingAgents Graph Visualizer"""
    
    def __init__(self):
        self.analyzer = GraphAnalyzer()
        self.generator = GraphGenerator()
        self.results = []
    
    def run_comprehensive_demo(self):
        """Run comprehensive demonstration of graph visualization capabilities"""
        
        print("🚀 TradingAgents Graph Visualizer - Challenge Demo")
        print("=" * 60)
        
        # Generate challenge graphs
        challenge_graphs = self.generator.generate_challenge_graphs()
        
        print(f"📊 Generated {len(challenge_graphs)} challenge graphs")
        print()
        
        # Test each graph
        for i, (name, graph) in enumerate(challenge_graphs, 1):
            print(f"🔍 Testing Graph {i}/{len(challenge_graphs)}: {name}")
            print(f"   Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
            
            # Analyze graph
            start_time = time.time()
            metrics = self.analyzer.analyze_graph(graph)
            analysis_time = time.time() - start_time
            
            # Get visualization strategy
            strategy = self.analyzer.get_visualization_strategy(metrics)
            
            # Generate visualization
            start_time = time.time()
            fig = self.analyzer.visualize_graph(graph, strategy)
            visualization_time = time.time() - start_time
            
            # Generate insights
            insights = self.analyzer.generate_insights(graph, metrics)
            
            # Store results
            result = {
                "name": name,
                "nodes": metrics.node_count,
                "edges": metrics.edge_count,
                "density": metrics.density,
                "complexity_score": metrics.complexity_score,
                "strategy": strategy,
                "analysis_time": analysis_time,
                "visualization_time": visualization_time,
                "total_time": analysis_time + visualization_time,
                "insights": insights
            }
            self.results.append(result)
            
            print(f"   ✅ Strategy: {strategy}")
            print(f"   ⏱️  Analysis: {analysis_time:.2f}s, Visualization: {visualization_time:.2f}s")
            print(f"   📈 Complexity: {metrics.complexity_score:.3f}")
            print()
        
        # Generate summary report
        self._generate_summary_report()
        
        # Create performance dashboard
        self._create_performance_dashboard()
        
        print("🎉 Demo completed successfully!")
        print("📁 Check the generated files for detailed results")
    
    def run_specific_test(self, graph_name: str):
        """Run test on a specific graph type"""
        
        print(f"🔍 Testing specific graph: {graph_name}")
        
        # Generate the specific graph
        if graph_name == "dense":
            graph = self.generator.generate_graph(GraphConfig("erdos_renyi", 1000, density=0.3))
        elif graph_name == "sparse":
            graph = self.generator.generate_graph(GraphConfig("erdos_renyi", 1000, density=0.001))
        elif graph_name == "scale_free":
            graph = self.generator.generate_graph(GraphConfig("barabasi_albert", 1000))
        elif graph_name == "small_world":
            graph = self.generator.generate_graph(GraphConfig("watts_strogatz", 1000))
        elif graph_name == "communities":
            graph = self.generator.generate_graph(GraphConfig("stochastic_block", 1000, communities=10))
        elif graph_name == "financial":
            graph = self.generator.generate_graph(GraphConfig("financial", 1000))
        elif graph_name == "social":
            graph = self.generator.generate_graph(GraphConfig("social", 1000))
        elif graph_name == "biological":
            graph = self.generator.generate_graph(GraphConfig("biological", 1000))
        else:
            print(f"❌ Unknown graph type: {graph_name}")
            return
        
        # Analyze and visualize
        metrics = self.analyzer.analyze_graph(graph)
        strategy = self.analyzer.get_visualization_strategy(metrics)
        fig = self.analyzer.visualize_graph(graph, strategy)
        
        # Generate insights
        insights = self.analyzer.generate_insights(graph, metrics)
        
        # Display results
        print(f"📊 Graph Analysis Results:")
        print(f"   Nodes: {metrics.node_count}")
        print(f"   Edges: {metrics.edge_count}")
        print(f"   Density: {metrics.density:.3f}")
        print(f"   Complexity: {metrics.complexity_score:.3f}")
        print(f"   Strategy: {strategy}")
        print(f"   Communities: {metrics.communities}")
        print()
        print(f"💡 AI Insights:")
        print(insights)
        print()
        
        # Save visualization
        filename = f"visualization_{graph_name}.html"
        self.analyzer.export_visualization(fig, filename)
        
        return fig, metrics, insights
    
    def run_performance_benchmark(self):
        """Run performance benchmark on various graph sizes"""
        
        print("⚡ Performance Benchmark")
        print("=" * 40)
        
        # Test different graph sizes
        sizes = [100, 500, 1000, 2000, 5000, 10000]
        benchmark_results = []
        
        for size in sizes:
            print(f"🔍 Testing graph with {size} nodes...")
            
            # Generate graph
            graph = self.generator.generate_graph(GraphConfig("erdos_renyi", size, density=0.01))
            
            # Measure performance
            start_time = time.time()
            metrics = self.analyzer.analyze_graph(graph)
            analysis_time = time.time() - start_time
            
            start_time = time.time()
            strategy = self.analyzer.get_visualization_strategy(metrics)
            strategy_time = time.time() - start_time
            
            start_time = time.time()
            fig = self.analyzer.visualize_graph(graph, strategy)
            visualization_time = time.time() - start_time
            
            total_time = analysis_time + strategy_time + visualization_time
            
            result = {
                "nodes": size,
                "edges": graph.number_of_edges(),
                "analysis_time": analysis_time,
                "strategy_time": strategy_time,
                "visualization_time": visualization_time,
                "total_time": total_time,
                "complexity_score": metrics.complexity_score,
                "strategy": strategy
            }
            benchmark_results.append(result)
            
            print(f"   ⏱️  Total time: {total_time:.2f}s")
            print(f"   📈 Complexity: {metrics.complexity_score:.3f}")
            print(f"   🎨 Strategy: {strategy}")
            print()
        
        # Create performance chart
        self._create_performance_chart(benchmark_results)
        
        return benchmark_results
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        
        print("📋 Generating Summary Report...")
        
        # Create summary DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate summary statistics
        summary = {
            "total_graphs_tested": len(self.results),
            "average_nodes": df["nodes"].mean(),
            "average_edges": df["edges"].mean(),
            "average_density": df["density"].mean(),
            "average_complexity": df["complexity_score"].mean(),
            "average_analysis_time": df["analysis_time"].mean(),
            "average_visualization_time": df["visualization_time"].mean(),
            "average_total_time": df["total_time"].mean(),
            "strategy_distribution": df["strategy"].value_counts().to_dict(),
            "max_nodes_handled": df["nodes"].max(),
            "max_edges_handled": df["edges"].max(),
            "max_complexity": df["complexity_score"].max()
        }
        
        # Save summary
        with open("graph_visualization_summary.json", "w") as f:
            import json
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        df.to_csv("graph_visualization_results.csv", index=False)
        
        print("✅ Summary report saved to graph_visualization_summary.json")
        print("✅ Detailed results saved to graph_visualization_results.csv")
    
    def _create_performance_dashboard(self):
        """Create interactive performance dashboard"""
        
        print("📊 Creating Performance Dashboard...")
        
        df = pd.DataFrame(self.results)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Nodes vs Analysis Time", "Complexity vs Visualization Time", 
                          "Strategy Distribution", "Performance by Graph Type"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Nodes vs Analysis Time
        fig.add_trace(
            go.Scatter(x=df["nodes"], y=df["analysis_time"], mode="markers+lines",
                      name="Analysis Time", marker=dict(color="blue")),
            row=1, col=1
        )
        
        # Complexity vs Visualization Time
        fig.add_trace(
            go.Scatter(x=df["complexity_score"], y=df["visualization_time"], mode="markers+lines",
                      name="Visualization Time", marker=dict(color="red")),
            row=1, col=2
        )
        
        # Strategy Distribution
        strategy_counts = df["strategy"].value_counts()
        fig.add_trace(
            go.Pie(labels=strategy_counts.index, values=strategy_counts.values,
                  name="Strategy Distribution"),
            row=2, col=1
        )
        
        # Performance by Graph Type
        fig.add_trace(
            go.Bar(x=df["name"], y=df["total_time"], name="Total Time",
                  marker=dict(color="green")),
            row=2, col=2
        )
        
        fig.update_layout(
            title="TradingAgents Graph Visualizer Performance Dashboard",
            showlegend=False,
            height=800
        )
        
        # Save dashboard
        fig.write_html("performance_dashboard.html")
        print("✅ Performance dashboard saved to performance_dashboard.html")
    
    def _create_performance_chart(self, benchmark_results: List[Dict]):
        """Create performance chart for benchmark results"""
        
        df = pd.DataFrame(benchmark_results)
        
        fig = go.Figure()
        
        # Add performance lines
        fig.add_trace(go.Scatter(
            x=df["nodes"], y=df["analysis_time"],
            mode="lines+markers", name="Analysis Time",
            line=dict(color="blue", width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df["nodes"], y=df["visualization_time"],
            mode="lines+markers", name="Visualization Time",
            line=dict(color="red", width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df["nodes"], y=df["total_time"],
            mode="lines+markers", name="Total Time",
            line=dict(color="green", width=3)
        ))
        
        fig.update_layout(
            title="Graph Visualization Performance Benchmark",
            xaxis_title="Number of Nodes",
            yaxis_title="Time (seconds)",
            hovermode="x unified",
            height=500
        )
        
        fig.write_html("performance_benchmark.html")
        print("✅ Performance benchmark chart saved to performance_benchmark.html")
    
    def demonstrate_ai_insights(self):
        """Demonstrate AI-powered graph insights"""
        
        print("🤖 AI-Powered Graph Insights Demo")
        print("=" * 40)
        
        # Test different graph types
        test_cases = [
            ("Financial Network", GraphConfig("financial", 500)),
            ("Social Network", GraphConfig("social", 500)),
            ("Biological Network", GraphConfig("biological", 500)),
            ("Scale-Free Network", GraphConfig("barabasi_albert", 500)),
            ("Small-World Network", GraphConfig("watts_strogatz", 500))
        ]
        
        for name, config in test_cases:
            print(f"\n🔍 Analyzing {name}...")
            
            # Generate graph
            graph = self.generator.generate_graph(config)
            
            # Analyze
            metrics = self.analyzer.analyze_graph(graph)
            insights = self.analyzer.generate_insights(graph, metrics)
            
            print(f"📊 Metrics: {metrics.node_count} nodes, {metrics.edge_count} edges")
            print(f"📈 Complexity: {metrics.complexity_score:.3f}")
            print(f"🎨 Strategy: {self.analyzer.get_visualization_strategy(metrics)}")
            print(f"💡 AI Insights:")
            print(insights)
            print("-" * 40)
    
    def run_challenge_simulation(self):
        """Simulate the actual challenge scenario"""
        
        print("🏆 Challenge Simulation")
        print("=" * 30)
        
        # Simulate challenge graphs
        challenge_scenarios = [
            ("Dense Financial Network", GraphConfig("financial", 2000)),
            ("Sparse Social Network", GraphConfig("social", 2000)),
            ("High Clustering Biological", GraphConfig("biological", 2000)),
            ("Many Communities", GraphConfig("stochastic_block", 2000, communities=20)),
            ("Very Large Scale-Free", GraphConfig("barabasi_albert", 5000))
        ]
        
        challenge_results = []
        
        for name, config in challenge_scenarios:
            print(f"\n🎯 Challenge: {name}")
            
            # Generate graph
            graph = self.generator.generate_graph(config)
            
            # Measure performance
            start_time = time.time()
            metrics = self.analyzer.analyze_graph(graph)
            analysis_time = time.time() - start_time
            
            start_time = time.time()
            strategy = self.analyzer.get_visualization_strategy(metrics)
            strategy_time = time.time() - start_time
            
            start_time = time.time()
            fig = self.analyzer.visualize_graph(graph, strategy)
            visualization_time = time.time() - start_time
            
            total_time = analysis_time + strategy_time + visualization_time
            
            # Generate insights
            insights = self.analyzer.generate_insights(graph, metrics)
            
            result = {
                "challenge": name,
                "nodes": metrics.node_count,
                "edges": metrics.edge_count,
                "complexity": metrics.complexity_score,
                "strategy": strategy,
                "total_time": total_time,
                "success": total_time < 30,  # 30 second limit
                "insights": insights
            }
            challenge_results.append(result)
            
            print(f"   📊 Nodes: {metrics.node_count}, Edges: {metrics.edge_count}")
            print(f"   ⏱️  Time: {total_time:.2f}s")
            print(f"   🎨 Strategy: {strategy}")
            print(f"   ✅ Success: {'Yes' if result['success'] else 'No'}")
            print(f"   📈 Complexity: {metrics.complexity_score:.3f}")
        
        # Calculate success rate
        success_rate = sum(1 for r in challenge_results if r["success"]) / len(challenge_results)
        print(f"\n🏆 Challenge Success Rate: {success_rate:.1%}")
        
        return challenge_results

def main():
    """Main demo function"""
    
    demo = GraphVisualizationDemo()
    
    print("🚀 TradingAgents Graph Visualizer Demo")
    print("Choose an option:")
    print("1. Run comprehensive demo")
    print("2. Run specific test")
    print("3. Run performance benchmark")
    print("4. Demonstrate AI insights")
    print("5. Run challenge simulation")
    print("6. Run all demos")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        demo.run_comprehensive_demo()
    elif choice == "2":
        graph_name = input("Enter graph type (dense, sparse, scale_free, small_world, communities, financial, social, biological): ").strip()
        demo.run_specific_test(graph_name)
    elif choice == "3":
        demo.run_performance_benchmark()
    elif choice == "4":
        demo.demonstrate_ai_insights()
    elif choice == "5":
        demo.run_challenge_simulation()
    elif choice == "6":
        demo.run_comprehensive_demo()
        demo.run_performance_benchmark()
        demo.demonstrate_ai_insights()
        demo.run_challenge_simulation()
    else:
        print("Invalid choice. Running comprehensive demo...")
        demo.run_comprehensive_demo()

if __name__ == "__main__":
    main()
