"""
Advanced Chart Type Selection Algorithm for TradingAgents
Implements sophisticated chart selection based on data characteristics and financial context
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import asyncio
from anthropic import Anthropic

@dataclass
class ChartRecommendation:
    """Data class for chart recommendations with confidence scoring"""
    chart_type: str
    confidence_score: float
    data_columns: List[str]
    parameters: Dict[str, Any]
    reasoning: str
    priority: int

class AdvancedChartSelector:
    """Advanced algorithm for selecting optimal chart types based on data characteristics"""
    
    def __init__(self, claude_client: Optional[Anthropic] = None):
        self.claude = claude_client or Anthropic()
        self.logger = logging.getLogger(__name__)
        
        # Chart type weights for selection algorithm
        self.chart_weights = {
            'candlestick': 0.9,
            'line_chart': 0.7,
            'volume_chart': 0.8,
            'correlation_heatmap': 0.6,
            'distribution_chart': 0.5,
            'technical_indicators': 0.8,
            'scatter_plot': 0.4,
            'multi_timeframe': 0.7
        }
        
        # Chart selection algorithms
        self.chart_algorithms = {
            'temporal_analysis': self._analyze_temporal_patterns,
            'volatility_analysis': self._analyze_volatility_patterns,
            'correlation_analysis': self._analyze_correlation_patterns,
            'distribution_analysis': self._analyze_distribution_patterns,
            'technical_analysis': self._analyze_technical_patterns,
            'volume_analysis': self._analyze_volume_patterns
        }
        
    async def select_optimal_charts(
        self, 
        df: pd.DataFrame, 
        context: Dict[str, Any],
        max_charts: int = 4
    ) -> List[ChartRecommendation]:
        """
        Main method to select optimal chart types based on data analysis
        """
        try:
            self.logger.info(f"Starting chart selection for {len(df)} rows of data")
            
            # Analyze data characteristics
            data_profile = await self._profile_data(df, context)
            
            # Generate chart recommendations from each algorithm
            all_recommendations = []
            
            for algorithm_name, algorithm_func in self.chart_algorithms.items():
                try:
                    recommendations = await algorithm_func(df, data_profile, context)
                    all_recommendations.extend(recommendations)
                    self.logger.debug(f"Algorithm {algorithm_name} generated {len(recommendations)} recommendations")
                except Exception as e:
                    self.logger.warning(f"Algorithm {algorithm_name} failed: {e}")
            
            # Score and rank recommendations
            ranked_recommendations = self._rank_recommendations(all_recommendations, data_profile)
            
            # Apply diversity filter to avoid redundant charts
            diverse_recommendations = self._apply_diversity_filter(ranked_recommendations)
            
            # Enhance with LLM reasoning
            enhanced_recommendations = await self._enhance_with_llm_reasoning(
                diverse_recommendations[:max_charts], data_profile, context
            )
            
            self.logger.info(f"Selected {len(enhanced_recommendations)} optimal charts")
            return enhanced_recommendations
            
        except Exception as e:
            self.logger.error(f"Chart selection failed: {e}")
            return self._get_fallback_charts(df)
    
    async def _profile_data(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data profiling for chart selection"""
        profile = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_data_ratio': df.isnull().sum() / len(df),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
        }
        
        # Financial data specific profiling
        profile['has_ohlc'] = self._has_ohlc_data(df)
        profile['has_volume'] = 'volume' in df.columns.str.lower()
        profile['timespan_days'] = self._calculate_timespan_days(df)
        profile['data_frequency'] = self._detect_data_frequency(df)
        
        # Statistical profiling
        if profile['numeric_columns']:
            numeric_df = df[profile['numeric_columns']]
            profile['volatility_metrics'] = self._calculate_volatility_metrics(numeric_df)
            profile['trend_metrics'] = self._calculate_trend_metrics(numeric_df)
            profile['correlation_matrix'] = numeric_df.corr().abs()
            profile['outlier_ratios'] = self._detect_outliers(numeric_df)
        
        # Context integration
        profile['user_intent'] = context.get('user_query', '').lower()
        profile['market_conditions'] = context.get('market_context', {})
        profile['agent_context'] = context.get('agent_analysis', {})
        
        return profile
    
    async def _analyze_temporal_patterns(
        self, 
        df: pd.DataFrame, 
        profile: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[ChartRecommendation]:
        """Analyze temporal patterns and recommend appropriate time series charts"""
        recommendations = []
        
        # Candlestick chart for OHLC data
        if profile['has_ohlc'] and profile['timespan_days'] > 5:
            confidence = 0.9 if profile['timespan_days'] > 30 else 0.7
            recommendations.append(ChartRecommendation(
                chart_type='candlestick',
                confidence_score=confidence,
                data_columns=['open', 'high', 'low', 'close'],
                parameters={'include_volume': profile['has_volume']},
                reasoning="OHLC data with sufficient history ideal for candlestick visualization",
                priority=1
            ))
        
        # Line chart for single metric time series
        if len(profile['numeric_columns']) >= 1:
            primary_metric = self._identify_primary_metric(df, profile)
            if primary_metric:
                recommendations.append(ChartRecommendation(
                    chart_type='line_chart',
                    confidence_score=0.8,
                    data_columns=[primary_metric],
                    parameters={'show_trend': True, 'include_markers': profile['shape'][0] < 100},
                    reasoning=f"Clear temporal trend visualization for {primary_metric}",
                    priority=2
                ))
        
        # Multi-line chart for multiple metrics
        if len(profile['numeric_columns']) > 1:
            correlated_metrics = self._find_correlated_metrics(profile['correlation_matrix'])
            if len(correlated_metrics) > 1:
                recommendations.append(ChartRecommendation(
                    chart_type='multi_line_chart',
                    confidence_score=0.7,
                    data_columns=correlated_metrics[:4],  # Limit to 4 lines for readability
                    parameters={'normalize_scales': True},
                    reasoning="Multiple correlated metrics benefit from comparative visualization",
                    priority=3
                ))
        
        return recommendations
    
    async def _analyze_volatility_patterns(
        self, 
        df: pd.DataFrame, 
        profile: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[ChartRecommendation]:
        """Analyze volatility patterns and recommend volatility-focused charts"""
        recommendations = []
        
        if not profile['has_ohlc']:
            return recommendations
        
        volatility_score = np.mean(list(profile['volatility_metrics'].values()))
        
        # Bollinger Bands for high volatility
        if volatility_score > 0.02:  # 2% daily volatility threshold
            recommendations.append(ChartRecommendation(
                chart_type='bollinger_bands',
                confidence_score=min(0.9, volatility_score * 30),
                data_columns=['close'],
                parameters={'period': 20, 'std_dev': 2},
                reasoning="High volatility detected - Bollinger Bands will show price boundaries",
                priority=2
            ))
        
        return recommendations
    
    async def _analyze_correlation_patterns(
        self, 
        df: pd.DataFrame, 
        profile: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[ChartRecommendation]:
        """Analyze correlation patterns and recommend correlation visualizations"""
        recommendations = []
        
        if len(profile['numeric_columns']) < 2:
            return recommendations
        
        corr_matrix = profile['correlation_matrix']
        
        # Heatmap for multiple variables
        if len(profile['numeric_columns']) >= 3:
            max_correlation = corr_matrix.abs().max().max()
            recommendations.append(ChartRecommendation(
                chart_type='correlation_heatmap',
                confidence_score=min(0.8, max_correlation + 0.2),
                data_columns=profile['numeric_columns'][:10],  # Limit for readability
                parameters={'annotate_values': True, 'cluster_order': True},
                reasoning="Multiple variables with correlations benefit from heatmap visualization",
                priority=3
            ))
        
        # Scatter plot for strong pairwise correlations
        strong_pairs = self._find_strong_correlation_pairs(corr_matrix, threshold=0.7)
        for pair, correlation in strong_pairs[:2]:  # Limit to top 2 pairs
            recommendations.append(ChartRecommendation(
                chart_type='scatter_plot',
                confidence_score=abs(correlation),
                data_columns=list(pair),
                parameters={'show_trend_line': True, 'color_by_time': True},
                reasoning=f"Strong correlation ({correlation:.2f}) between {pair[0]} and {pair[1]}",
                priority=4
            ))
        
        return recommendations
    
    async def _analyze_distribution_patterns(
        self, 
        df: pd.DataFrame, 
        profile: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[ChartRecommendation]:
        """Analyze distribution patterns and recommend distribution charts"""
        recommendations = []
        
        for col in profile['numeric_columns'][:3]:  # Analyze top 3 numeric columns
            col_data = df[col].dropna()
            
            if len(col_data) < 50:
                continue
            
            # Calculate distribution metrics
            skewness = col_data.skew()
            kurtosis = col_data.kurtosis()
            outlier_ratio = profile['outlier_ratios'].get(col, 0)
            
            # Histogram for interesting distributions
            if abs(skewness) > 0.5 or abs(kurtosis) > 1 or outlier_ratio > 0.05:
                confidence = min(0.7, abs(skewness) * 0.3 + abs(kurtosis) * 0.1 + outlier_ratio * 2)
                recommendations.append(ChartRecommendation(
                    chart_type='histogram',
                    confidence_score=confidence,
                    data_columns=[col],
                    parameters={
                        'bins': 'auto',
                        'show_normal_curve': True,
                        'show_statistics': True
                    },
                    reasoning=f"Interesting distribution pattern in {col} (skew: {skewness:.2f})",
                    priority=5
                ))
        
        return recommendations
    
    async def _analyze_technical_patterns(
        self, 
        df: pd.DataFrame, 
        profile: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[ChartRecommendation]:
        """Analyze technical patterns and recommend technical analysis charts"""
        recommendations = []
        
        if not profile['has_ohlc'] or profile['timespan_days'] < 20:
            return recommendations
        
        try:
            # Calculate basic technical indicators
            close_prices = df['close'] if 'close' in df.columns else df[df.select_dtypes(include=[np.number]).columns[0]]
            
            # Simple RSI calculation
            if len(close_prices) > 14:
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_current = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
                
                if rsi_current < 30 or rsi_current > 70:
                    recommendations.append(ChartRecommendation(
                        chart_type='rsi_chart',
                        confidence_score=0.8,
                        data_columns=['close'],
                        parameters={'period': 14, 'highlight_zones': True},
                        reasoning=f"RSI at {rsi_current:.1f} indicates potential overbought/oversold condition",
                        priority=2
                    ))
            
        except Exception as e:
            self.logger.warning(f"Technical analysis failed: {e}")
        
        return recommendations
    
    async def _analyze_volume_patterns(
        self, 
        df: pd.DataFrame, 
        profile: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[ChartRecommendation]:
        """Analyze volume patterns and recommend volume-based charts"""
        recommendations = []
        
        if not profile['has_volume']:
            return recommendations
        
        volume_col = next((col for col in df.columns if 'volume' in col.lower()), None)
        if not volume_col:
            return recommendations
        
        volume_data = df[volume_col].dropna()
        
        # Volume profile analysis
        volume_variability = volume_data.std() / volume_data.mean() if volume_data.mean() > 0 else 0
        
        if volume_variability > 0.5:  # High volume variability
            recommendations.append(ChartRecommendation(
                chart_type='volume_profile',
                confidence_score=min(0.8, volume_variability),
                data_columns=[volume_col, 'close'] if 'close' in df.columns else [volume_col],
                parameters={'show_poc': True, 'show_value_area': True},
                reasoning="High volume variability indicates significant volume patterns",
                priority=3
            ))
        
        return recommendations
    
    def _rank_recommendations(
        self, 
        recommendations: List[ChartRecommendation], 
        profile: Dict[str, Any]
    ) -> List[ChartRecommendation]:
        """Rank recommendations based on confidence scores and context"""
        
        for rec in recommendations:
            # Apply chart type weights
            weight = self.chart_weights.get(rec.chart_type, 0.5)
            rec.confidence_score *= weight
            
            # Boost score based on user intent
            if profile['user_intent']:
                if any(keyword in profile['user_intent'] for keyword in ['volatility', 'risk']) and 'volatility' in rec.chart_type:
                    rec.confidence_score *= 1.2
                elif any(keyword in profile['user_intent'] for keyword in ['trend', 'momentum']) and rec.chart_type in ['candlestick', 'line_chart']:
                    rec.confidence_score *= 1.15
                elif 'correlation' in profile['user_intent'] and 'correlation' in rec.chart_type:
                    rec.confidence_score *= 1.25
        
        # Sort by priority first, then by confidence score
        return sorted(recommendations, key=lambda x: (-x.priority, -x.confidence_score))
    
    def _apply_diversity_filter(self, recommendations: List[ChartRecommendation]) -> List[ChartRecommendation]:
        """Apply diversity filter to ensure variety in chart types"""
        diverse_recommendations = []
        used_chart_families = set()
        
        # Define chart families to avoid redundancy
        chart_families = {
            'candlestick': 'price_action',
            'line_chart': 'trend',
            'multi_line_chart': 'trend',
            'bollinger_bands': 'volatility',
            'correlation_heatmap': 'correlation',
            'scatter_plot': 'correlation',
            'histogram': 'distribution',
            'rsi_chart': 'technical',
            'volume_profile': 'volume'
        }
        
        for rec in recommendations:
            chart_family = chart_families.get(rec.chart_type, 'other')
            
            # Always include high-priority charts
            if rec.priority <= 2 or chart_family not in used_chart_families:
                diverse_recommendations.append(rec)
                used_chart_families.add(chart_family)
        
        return diverse_recommendations
    
    async def _enhance_with_llm_reasoning(
        self,
        recommendations: List[ChartRecommendation],
        data_profile: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[ChartRecommendation]:
        """Enhance recommendations with LLM-generated reasoning"""
        
        for recommendation in recommendations:
            try:
                reasoning_prompt = self._build_reasoning_prompt(
                    recommendation, data_profile, context
                )
                
                # Use synchronous call in async context
                import asyncio
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.claude.messages.create(
                        model="claude-sonnet-4-0",
                        max_tokens=500,
                        messages=[{"role": "user", "content": reasoning_prompt}]
                    )
                )
                
                recommendation.reasoning = response.content[0].text.strip()
                
            except Exception as e:
                self.logger.error(f"Error generating LLM reasoning: {e}")
                recommendation.reasoning = f"Recommended based on data characteristics: {recommendation.chart_type}"
        
        return recommendations
    
    def _build_reasoning_prompt(
        self,
        recommendation: ChartRecommendation,
        data_profile: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM reasoning generation"""
        
        prompt = f"""
As a financial data visualization expert, provide a concise explanation for why a {recommendation.chart_type}
is recommended for this dataset.

Dataset Characteristics:
- Data Shape: {data_profile.get('shape', {})}
- Data Types: {data_profile.get('data_types', {})}
- Has OHLC: {data_profile.get('has_ohlc', False)}
- Has Volume: {data_profile.get('has_volume', False)}
- Timespan: {data_profile.get('timespan_days', 0)} days

Context:
- Analysis Type: {context.get('analysis_type', 'general')}
- User Goal: {context.get('user_goal', 'analysis')}

Chart Confidence: {recommendation.confidence_score:.2f}
Chart Priority: {recommendation.priority}

Provide a 2-3 sentence explanation focusing on:
1. Why this chart type suits the data characteristics
2. What insights it will reveal
3. How it serves the user's analytical goals

Keep the explanation concise and professional.
"""
        return prompt
    
    # Helper methods
    def _has_ohlc_data(self, df: pd.DataFrame) -> bool:
        """Check if dataframe contains OHLC data"""
        ohlc_columns = {'open', 'high', 'low', 'close'}
        df_columns_lower = {col.lower() for col in df.columns}
        return ohlc_columns.issubset(df_columns_lower)
    
    def _calculate_timespan_days(self, df: pd.DataFrame) -> int:
        """Calculate timespan of data in days"""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            date_col = datetime_cols[0]
            return (df[date_col].max() - df[date_col].min()).days
        return len(df)  # Fallback to row count
    
    def _detect_data_frequency(self, df: pd.DataFrame) -> str:
        """Detect the frequency of the data (daily, hourly, etc.)"""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) == 0:
            return 'unknown'
        
        date_col = datetime_cols[0]
        if len(df) < 2:
            return 'unknown'
        
        time_diff = (df[date_col].iloc[1] - df[date_col].iloc[0]).total_seconds()
        
        if time_diff <= 3600:  # 1 hour or less
            return 'intraday'
        elif time_diff <= 86400:  # 1 day or less
            return 'daily'
        elif time_diff <= 604800:  # 1 week or less
            return 'weekly'
        else:
            return 'monthly'
    
    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility metrics for numeric columns"""
        volatility_metrics = {}
        
        for col in df.columns:
            try:
                returns = df[col].pct_change().dropna()
                if len(returns) > 1:
                    volatility_metrics[col] = returns.std()
            except:
                volatility_metrics[col] = 0.0
        
        return volatility_metrics
    
    def _calculate_trend_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trend metrics for numeric columns"""
        trend_metrics = {}
        
        for col in df.columns:
            try:
                values = df[col].dropna()
                if len(values) > 1:
                    # Simple trend calculation using linear regression slope
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    trend_metrics[col] = slope / values.mean() if values.mean() != 0 else 0
            except:
                trend_metrics[col] = 0.0
        
        return trend_metrics
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, float]:
        """Detect outliers using IQR method"""
        outlier_ratios = {}
        
        for col in df.columns:
            try:
                values = df[col].dropna()
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                outlier_ratios[col] = len(outliers) / len(values)
            except:
                outlier_ratios[col] = 0.0
        
        return outlier_ratios
    
    def _identify_primary_metric(self, df: pd.DataFrame, profile: Dict[str, Any]) -> Optional[str]:
        """Identify the primary metric for single-line charts"""
        if profile['has_ohlc']:
            return 'close'
        
        numeric_cols = profile['numeric_columns']
        if not numeric_cols:
            return None
        
        # Prefer commonly used financial metrics
        preferred_metrics = ['price', 'close', 'value', 'amount', 'return']
        for metric in preferred_metrics:
            matching_cols = [col for col in numeric_cols if metric in col.lower()]
            if matching_cols:
                return matching_cols[0]
        
        return numeric_cols[0]  # Fallback to first numeric column
    
    def _find_correlated_metrics(self, correlation_matrix: pd.DataFrame, threshold: float = 0.3) -> List[str]:
        """Find metrics that are correlated above threshold"""
        if correlation_matrix.empty:
            return []
        
        # Get average correlation for each metric
        avg_correlations = correlation_matrix.abs().mean().sort_values(ascending=False)
        return avg_correlations[avg_correlations > threshold].index.tolist()
    
    def _find_strong_correlation_pairs(self, correlation_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Tuple[Tuple[str, str], float]]:
        """Find pairs of variables with strong correlations"""
        strong_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                    strong_pairs.append(((col1, col2), corr_value))
        
        return sorted(strong_pairs, key=lambda x: abs(x[1]), reverse=True)
    
    def _get_fallback_charts(self, df: pd.DataFrame) -> List[ChartRecommendation]:
        """Provide fallback charts when analysis fails"""
        fallback_charts = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Simple line chart for first numeric column
            fallback_charts.append(ChartRecommendation(
                chart_type='line_chart',
                confidence_score=0.5,
                data_columns=[numeric_cols[0]],
                parameters={},
                reasoning="Fallback visualization for primary numeric data",
                priority=1
            ))
        
        return fallback_charts
