"""
Enhanced Visualization Agent with Advanced Chart Selection and LLM Reasoning
Integrates all graph enhancement components for sophisticated financial visualization
"""

import asyncio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Any, Optional
import logging
from anthropic import Anthropic

from .advanced_chart_selector import AdvancedChartSelector, ChartRecommendation
from .enhanced_reasoning_engine import EnhancedLLMReasoning, ReasoningResult
from .exploratory_prompt_engine import ExploratoryPromptEngine, ExplorationIntent

class EnhancedVisualizationAgent:
    """Enhanced visualization agent with advanced chart selection and reasoning"""
    
    def __init__(self, claude_client: Optional[Anthropic] = None):
        self.claude = claude_client or Anthropic()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.chart_selector = AdvancedChartSelector(claude_client)
        self.reasoning_engine = EnhancedLLMReasoning(claude_client)
        self.exploration_engine = ExploratoryPromptEngine(claude_client)
        
        # Chart generation registry
        self.chart_generators = {
            'candlestick': self._generate_candlestick_chart,
            'line_chart': self._generate_line_chart,
            'multi_line_chart': self._generate_multi_line_chart,
            'bollinger_bands': self._generate_bollinger_bands,
            'correlation_heatmap': self._generate_correlation_heatmap,
            'scatter_plot': self._generate_scatter_plot,
            'histogram': self._generate_histogram,
            'rsi_chart': self._generate_rsi_chart,
            'volume_profile': self._generate_volume_profile,
            'trend_analysis': self._generate_trend_analysis,
            'volatility_analysis': self._generate_volatility_analysis
        }
        
        self.logger.info("Enhanced Visualization Agent initialized with advanced capabilities")
    
    async def generate_comprehensive_visualization_report(
        self,
        data: pd.DataFrame,
        symbol: str,
        context: Dict[str, Any],
        user_query: str = ""
    ) -> Dict[str, Any]:
        """Generate comprehensive visualization report with advanced features"""
        
        try:
            self.logger.info(f"Generating comprehensive visualization report for {symbol}")
            
            # Step 1: Advanced chart selection
            chart_recommendations = await self.chart_selector.select_optimal_charts(
                data, context, max_charts=4
            )
            
            self.logger.info(f"Selected {len(chart_recommendations)} optimal charts")
            
            # Step 2: Generate enhanced reasoning
            reasoning_result = await self.reasoning_engine.generate_chart_reasoning(
                chart_recommendations, 
                self._create_data_summary(data),
                context.get('market_context', {}),
                user_query,
                context.get('agent_context', {})
            )
            
            # Step 3: Generate visualizations
            visualizations = []
            for recommendation in chart_recommendations:
                try:
                    chart = await self._generate_chart_from_recommendation(
                        recommendation, data, symbol, context
                    )
                    if chart:
                        visualizations.append(chart)
                except Exception as e:
                    self.logger.warning(f"Failed to generate {recommendation.chart_type}: {e}")
            
            # Step 4: Generate exploration suggestions if user query provided
            exploration_results = None
            if user_query:
                exploration_results = await self.exploration_engine.process_exploration_request(
                    user_query, data, visualizations, context.get('agent_context', {})
                )
            
            # Step 5: Create comprehensive report
            report = await self._create_comprehensive_report(
                symbol, chart_recommendations, reasoning_result, 
                visualizations, exploration_results, context
            )
            
            self.logger.info(f"Generated comprehensive visualization report with {len(visualizations)} charts")
            return report
            
        except Exception as e:
            self.logger.error(f"Comprehensive visualization generation failed: {e}")
            return await self._generate_fallback_report(data, symbol, context)
    
    async def _generate_chart_from_recommendation(
        self,
        recommendation: ChartRecommendation,
        data: pd.DataFrame,
        symbol: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate chart from recommendation"""
        
        chart_type = recommendation.chart_type
        generator = self.chart_generators.get(chart_type)
        
        if not generator:
            self.logger.warning(f"No generator found for chart type: {chart_type}")
            return None
        
        try:
            chart_figure = await generator(data, recommendation, symbol, context)
            
            return {
                'type': chart_type,
                'figure': chart_figure,
                'recommendation': recommendation,
                'title': f"{symbol} - {chart_type.replace('_', ' ').title()}",
                'description': recommendation.reasoning,
                'confidence': recommendation.confidence_score,
                'priority': recommendation.priority
            }
            
        except Exception as e:
            self.logger.error(f"Chart generation failed for {chart_type}: {e}")
            return None
    
    async def _generate_candlestick_chart(
        self,
        data: pd.DataFrame,
        recommendation: ChartRecommendation,
        symbol: str,
        context: Dict[str, Any]
    ) -> go.Figure:
        """Generate candlestick chart"""
        
        # Find OHLC columns
        ohlc_cols = {}
        for col in ['open', 'high', 'low', 'close']:
            matching_cols = [c for c in data.columns if c.lower() == col]
            if matching_cols:
                ohlc_cols[col] = matching_cols[0]
        
        if not all(col in ohlc_cols for col in ['open', 'high', 'low', 'close']):
            raise ValueError("OHLC data not found for candlestick chart")
        
        # Find date column
        date_col = self._find_date_column(data)
        
        # Create candlestick chart
        fig = go.Figure(data=go.Candlestick(
            x=data[date_col] if date_col else data.index,
            open=data[ohlc_cols['open']],
            high=data[ohlc_cols['high']],
            low=data[ohlc_cols['low']],
            close=data[ohlc_cols['close']],
            name=symbol
        ))
        
        # Add volume if available
        volume_col = self._find_volume_column(data)
        if volume_col and recommendation.parameters.get('include_volume', False):
            # Create subplot with volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{symbol} Price', 'Volume'),
                row_width=[0.7, 0.3]
            )
            
            # Add candlestick to first subplot
            fig.add_trace(go.Candlestick(
                x=data[date_col] if date_col else data.index,
                open=data[ohlc_cols['open']],
                high=data[ohlc_cols['high']],
                low=data[ohlc_cols['low']],
                close=data[ohlc_cols['close']],
                name=symbol
            ), row=1, col=1)
            
            # Add volume to second subplot
            fig.add_trace(go.Bar(
                x=data[date_col] if date_col else data.index,
                y=data[volume_col],
                name='Volume',
                marker_color='lightblue'
            ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            showlegend=True
        )
        
        return fig
    
    async def _generate_line_chart(
        self,
        data: pd.DataFrame,
        recommendation: ChartRecommendation,
        symbol: str,
        context: Dict[str, Any]
    ) -> go.Figure:
        """Generate line chart"""
        
        # Find primary metric
        primary_metric = recommendation.data_columns[0] if recommendation.data_columns else None
        if not primary_metric:
            numeric_cols = data.select_dtypes(include=['number']).columns
            primary_metric = numeric_cols[0] if len(numeric_cols) > 0 else None
        
        if not primary_metric:
            raise ValueError("No numeric column found for line chart")
        
        # Find date column
        date_col = self._find_date_column(data)
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data[date_col] if date_col else data.index,
            y=data[primary_metric],
            mode='lines+markers' if recommendation.parameters.get('include_markers', False) else 'lines',
            name=primary_metric,
            line=dict(width=2)
        ))
        
        # Add trend line if requested
        if recommendation.parameters.get('show_trend', False):
            # Simple linear trend
            x_vals = range(len(data))
            y_vals = data[primary_metric].values
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            trend_line = p(x_vals)
            
            fig.add_trace(go.Scatter(
                x=data[date_col] if date_col else data.index,
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(dash='dash', color='red')
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - {primary_metric.title()} Trend",
            xaxis_title="Date",
            yaxis_title=primary_metric.title(),
            height=500,
            showlegend=True
        )
        
        return fig
    
    async def _generate_multi_line_chart(
        self,
        data: pd.DataFrame,
        recommendation: ChartRecommendation,
        symbol: str,
        context: Dict[str, Any]
    ) -> go.Figure:
        """Generate multi-line chart"""
        
        # Find date column
        date_col = self._find_date_column(data)
        
        # Create multi-line chart
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, column in enumerate(recommendation.data_columns):
            if column in data.columns:
                fig.add_trace(go.Scatter(
                    x=data[date_col] if date_col else data.index,
                    y=data[column],
                    mode='lines',
                    name=column,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Multi-Metric Comparison",
            xaxis_title="Date",
            yaxis_title="Value",
            height=500,
            showlegend=True
        )
        
        return fig
    
    async def _generate_correlation_heatmap(
        self,
        data: pd.DataFrame,
        recommendation: ChartRecommendation,
        symbol: str,
        context: Dict[str, Any]
    ) -> go.Figure:
        """Generate correlation heatmap"""
        
        # Select numeric columns
        numeric_data = data[recommendation.data_columns].select_dtypes(include=['number'])
        
        if numeric_data.empty:
            raise ValueError("No numeric data found for correlation heatmap")
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values if recommendation.parameters.get('annotate_values', False) else None,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Correlation Heatmap",
            height=500,
            width=600
        )
        
        return fig
    
    async def _generate_scatter_plot(
        self,
        data: pd.DataFrame,
        recommendation: ChartRecommendation,
        symbol: str,
        context: Dict[str, Any]
    ) -> go.Figure:
        """Generate scatter plot"""
        
        if len(recommendation.data_columns) < 2:
            raise ValueError("Scatter plot requires at least 2 variables")
        
        x_col, y_col = recommendation.data_columns[0], recommendation.data_columns[1]
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode='markers',
            name=f'{x_col} vs {y_col}',
            marker=dict(size=8, opacity=0.6)
        ))
        
        # Add trend line if requested
        if recommendation.parameters.get('show_trend_line', False):
            # Calculate linear regression
            x_vals = data[x_col].values
            y_vals = data[y_col].values
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            trend_line = p(x_vals)
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=trend_line,
                mode='lines',
                name='Trend Line',
                line=dict(dash='dash', color='red')
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - {x_col} vs {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500,
            showlegend=True
        )
        
        return fig
    
    async def _generate_histogram(
        self,
        data: pd.DataFrame,
        recommendation: ChartRecommendation,
        symbol: str,
        context: Dict[str, Any]
    ) -> go.Figure:
        """Generate histogram"""
        
        column = recommendation.data_columns[0]
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data[column],
            nbinsx=30,  # Fixed number of bins instead of 'auto'
            name=column,
            opacity=0.7
        ))
        
        # Add normal curve if requested
        if recommendation.parameters.get('show_normal_curve', False):
            mean = data[column].mean()
            std = data[column].std()
            x_range = np.linspace(data[column].min(), data[column].max(), 100)
            normal_curve = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
            normal_curve = normal_curve * len(data) * (data[column].max() - data[column].min()) / 100
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=normal_curve,
                mode='lines',
                name='Normal Distribution',
                line=dict(dash='dash', color='red')
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - {column} Distribution",
            xaxis_title=column,
            yaxis_title="Frequency",
            height=500,
            showlegend=True
        )
        
        return fig
    
    async def _generate_bollinger_bands(
        self,
        data: pd.DataFrame,
        recommendation: ChartRecommendation,
        symbol: str,
        context: Dict[str, Any]
    ) -> go.Figure:
        """Generate Bollinger Bands chart"""
        
        close_col = recommendation.data_columns[0]
        period = recommendation.parameters.get('period', 20)
        std_dev = recommendation.parameters.get('std_dev', 2)
        
        # Calculate Bollinger Bands
        sma = data[close_col].rolling(window=period).mean()
        std = data[close_col].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Find date column
        date_col = self._find_date_column(data)
        
        # Create Bollinger Bands chart
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=data[date_col] if date_col else data.index,
            y=data[close_col],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add moving average
        fig.add_trace(go.Scatter(
            x=data[date_col] if date_col else data.index,
            y=sma,
            mode='lines',
            name=f'SMA {period}',
            line=dict(color='orange', width=1)
        ))
        
        # Add upper band
        fig.add_trace(go.Scatter(
            x=data[date_col] if date_col else data.index,
            y=upper_band,
            mode='lines',
            name='Upper Band',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        # Add lower band
        fig.add_trace(go.Scatter(
            x=data[date_col] if date_col else data.index,
            y=lower_band,
            mode='lines',
            name='Lower Band',
            line=dict(color='red', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Bollinger Bands ({period} period, {std_dev}σ)",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
            showlegend=True
        )
        
        return fig
    
    async def _generate_rsi_chart(
        self,
        data: pd.DataFrame,
        recommendation: ChartRecommendation,
        symbol: str,
        context: Dict[str, Any]
    ) -> go.Figure:
        """Generate RSI chart"""
        
        close_col = recommendation.data_columns[0]
        period = recommendation.parameters.get('period', 14)
        
        # Calculate RSI
        delta = data[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Find date column
        date_col = self._find_date_column(data)
        
        # Create RSI chart
        fig = go.Figure()
        
        # Add RSI line
        fig.add_trace(go.Scatter(
            x=data[date_col] if date_col else data.index,
            y=rsi,
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ))
        
        # Add overbought/oversold lines if requested
        if recommendation.parameters.get('highlight_zones', False):
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - RSI ({period} period)",
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
            height=400,
            showlegend=True
        )
        
        return fig
    
    async def _generate_volume_profile(
        self,
        data: pd.DataFrame,
        recommendation: ChartRecommendation,
        symbol: str,
        context: Dict[str, Any]
    ) -> go.Figure:
        """Generate volume profile chart"""
        
        volume_col = recommendation.data_columns[0]
        price_col = recommendation.data_columns[1] if len(recommendation.data_columns) > 1 else None
        
        if not price_col:
            raise ValueError("Volume profile requires both volume and price data")
        
        # Create volume profile
        fig = go.Figure()
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=data.index,
            y=data[volume_col],
            name='Volume',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Volume Profile",
            xaxis_title="Date",
            yaxis_title="Volume",
            height=400,
            showlegend=True
        )
        
        return fig
    
    async def _generate_trend_analysis(
        self,
        data: pd.DataFrame,
        recommendation: ChartRecommendation,
        symbol: str,
        context: Dict[str, Any]
    ) -> go.Figure:
        """Generate trend analysis chart"""
        
        # This would implement comprehensive trend analysis
        # For now, return a simple line chart with trend indicators
        return await self._generate_line_chart(data, recommendation, symbol, context)
    
    async def _generate_volatility_analysis(
        self,
        data: pd.DataFrame,
        recommendation: ChartRecommendation,
        symbol: str,
        context: Dict[str, Any]
    ) -> go.Figure:
        """Generate volatility analysis chart"""
        
        # This would implement comprehensive volatility analysis
        # For now, return Bollinger Bands as volatility indicator
        return await self._generate_bollinger_bands(data, recommendation, symbol, context)
    
    def _find_date_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find the primary date column in the dataframe"""
        date_cols = data.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            return date_cols[0]
        
        # Check for common date column names
        common_date_names = ['date', 'timestamp', 'time', 'datetime']
        for col in data.columns:
            if col.lower() in common_date_names:
                return col
        
        return None
    
    def _find_volume_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find the volume column in the dataframe"""
        for col in data.columns:
            if 'volume' in col.lower():
                return col
        return None
    
    def _create_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create data summary for reasoning"""
        return {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'numeric_columns': data.select_dtypes(include=['number']).columns.tolist(),
            'datetime_columns': data.select_dtypes(include=['datetime64']).columns.tolist(),
            'has_ohlc': self._has_ohlc_data(data),
            'has_volume': self._find_volume_column(data) is not None,
            'missing_data_ratio': data.isnull().sum() / len(data)
        }
    
    def _has_ohlc_data(self, data: pd.DataFrame) -> bool:
        """Check if dataframe contains OHLC data"""
        ohlc_columns = {'open', 'high', 'low', 'close'}
        df_columns_lower = {col.lower() for col in data.columns}
        return ohlc_columns.issubset(df_columns_lower)
    
    async def _create_comprehensive_report(
        self,
        symbol: str,
        chart_recommendations: List[ChartRecommendation],
        reasoning_result: ReasoningResult,
        visualizations: List[Dict[str, Any]],
        exploration_results: Optional[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive visualization report"""
        
        # Save visualizations to files
        saved_files = []
        for i, viz in enumerate(visualizations, 1):
            try:
                filename = f"enhanced_visualization_{symbol}_{viz['type']}_{i}.html"
                viz['figure'].write_html(filename)
                saved_files.append(filename)
                self.logger.info(f"Saved visualization: {filename}")
            except Exception as e:
                self.logger.warning(f"Failed to save {viz['type']}: {e}")
        
        # Create comprehensive report
        report = f"""
# Enhanced Financial Visualization Report - {symbol}

## Executive Summary
This report provides advanced financial visualization analysis for {symbol} using sophisticated chart selection algorithms and LLM-powered reasoning.

## Chart Selection Analysis
**Total Charts Generated**: {len(visualizations)}
**Selection Confidence**: {sum(rec.confidence_score for rec in chart_recommendations) / len(chart_recommendations):.2f}

### Selected Chart Types:
{chr(10).join([f"- **{rec.chart_type}**: {rec.reasoning} (Confidence: {rec.confidence_score:.2f})" for rec in chart_recommendations])}

## LLM Reasoning Analysis
**Overall Confidence**: {reasoning_result.confidence_score:.2f}

### Chart Justification:
{reasoning_result.chart_justification}

### Market Insights:
{chr(10).join([f"- {insight}" for insight in reasoning_result.market_insights])}

### Trading Implications:
{chr(10).join([f"- {implication}" for implication in reasoning_result.trading_implications])}

### Risk Considerations:
{chr(10).join([f"- {risk}" for risk in reasoning_result.risk_considerations])}

### Suggested Actions:
{chr(10).join([f"- {action}" for action in reasoning_result.suggested_actions])}

## Generated Visualizations
{chr(10).join([f"### {viz['title']}\n**Type**: {viz['type']}\n**Description**: {viz['description']}\n**Confidence**: {viz['confidence']:.2f}\n" for viz in visualizations])}

## Interactive Files
{chr(10).join([f"- **{file}**: Interactive HTML visualization" for file in saved_files])}

## Exploration Results
"""
        
        if exploration_results:
            report += f"""
### User Query Analysis:
{exploration_results.get('explanation', 'No explanation available')}

### Follow-up Suggestions:
{chr(10).join([f"- {suggestion.description}" for suggestion in exploration_results.get('follow_up_suggestions', [])])}
"""
        else:
            report += "No specific exploration query provided."
        
        report += f"""

## Technical Details
- **Data Points**: {context.get('data_points', 'Unknown')}
- **Analysis Date**: {context.get('analysis_date', 'Unknown')}
- **Market Context**: {context.get('market_context', {})}
- **Agent Context**: {context.get('agent_context', {})}

## Files Generated
{chr(10).join([f"- {file}" for file in saved_files])}

---
*Report generated by Enhanced Visualization Agent with Advanced Chart Selection and LLM Reasoning*
"""
        
        return {
            'report': report,
            'visualizations': visualizations,
            'chart_recommendations': chart_recommendations,
            'reasoning_result': reasoning_result,
            'exploration_results': exploration_results,
            'saved_files': saved_files,
            'summary': {
                'symbol': symbol,
                'total_charts': len(visualizations),
                'avg_confidence': sum(rec.confidence_score for rec in chart_recommendations) / len(chart_recommendations) if chart_recommendations else 0,
                'reasoning_confidence': reasoning_result.confidence_score,
                'files_generated': len(saved_files)
            }
        }
    
    async def _generate_fallback_report(
        self,
        data: pd.DataFrame,
        symbol: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate fallback report when main generation fails"""
        
        # Create simple line chart as fallback
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            primary_col = numeric_cols[0]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[primary_col],
                mode='lines',
                name=primary_col
            ))
            
            fig.update_layout(
                title=f"{symbol} - {primary_col} (Fallback)",
                height=400
            )
            
            # Save fallback chart
            filename = f"fallback_visualization_{symbol}.html"
            fig.write_html(filename)
            
            return {
                'report': f"# Fallback Visualization Report - {symbol}\n\nBasic visualization generated due to processing error.",
                'visualizations': [{
                    'type': 'line_chart',
                    'figure': fig,
                    'title': f"{symbol} - {primary_col}",
                    'description': "Fallback visualization",
                    'confidence': 0.3
                }],
                'saved_files': [filename],
                'summary': {
                    'symbol': symbol,
                    'total_charts': 1,
                    'avg_confidence': 0.3,
                    'reasoning_confidence': 0.3,
                    'files_generated': 1
                }
            }
        
        return {
            'report': f"# Error Report - {symbol}\n\nUnable to generate visualizations due to data issues.",
            'visualizations': [],
            'saved_files': [],
            'summary': {
                'symbol': symbol,
                'total_charts': 0,
                'avg_confidence': 0.0,
                'reasoning_confidence': 0.0,
                'files_generated': 0
            }
        }
