"""
Exploratory Prompt Support for User Interaction
Enables natural language exploration of financial data and visualizations
"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import logging
from anthropic import Anthropic
from .advanced_chart_selector import ChartRecommendation

@dataclass
class ExplorationIntent:
    """Parsed user exploration intent"""
    primary_intent: str
    target_variables: List[str]
    time_constraints: Dict[str, Any]
    chart_preferences: List[str]
    business_questions: List[str]
    confidence_score: float

@dataclass
class ExplorationSuggestion:
    """Suggestion for data exploration"""
    suggestion_type: str
    description: str
    chart_type: str
    rationale: str
    priority: int

class ExploratoryPromptEngine:
    """Engine for processing natural language exploration requests"""
    
    def __init__(self, claude_client: Optional[Anthropic] = None):
        self.claude = claude_client or Anthropic()
        self.logger = logging.getLogger(__name__)
        
        # Exploration patterns for intent detection
        self.exploration_patterns = {
            'drill_down': ['show me details', 'zoom in', 'break down by', 'drill down', 'more detail'],
            'comparison': ['compare with', 'versus', 'relative to', 'compared to', 'vs'],
            'time_shift': ['historical', 'trend over', 'year over year', 'monthly', 'weekly', 'daily'],
            'correlation': ['relationship between', 'how does x affect y', 'correlation', 'related to'],
            'outlier': ['anomalies', 'unusual patterns', 'outliers', 'spikes', 'dips'],
            'distribution': ['spread', 'distribution', 'variance', 'range', 'statistics'],
            'volatility': ['volatility', 'risk', 'uncertainty', 'stability'],
            'trend': ['trend', 'direction', 'momentum', 'slope', 'change over time'],
            'volume': ['volume', 'trading volume', 'liquidity', 'activity']
        }
        
        # Chart alternatives for exploration
        self.chart_alternatives = {
            'candlestick': ['line_chart', 'area_chart', 'heikin_ashi', 'renko'],
            'line_chart': ['candlestick', 'area_chart', 'multi_line', 'scatter'],
            'scatter': ['regression_plot', 'bubble_chart', 'correlation_matrix', 'heatmap'],
            'histogram': ['density_plot', 'box_plot', 'violin_plot', 'qq_plot'],
            'correlation_heatmap': ['scatter_matrix', 'parallel_coordinates', 'network_graph']
        }
        
        # Business question templates
        self.business_question_templates = {
            'risk_assessment': [
                "What are the main risk factors?",
                "How volatile is this asset?",
                "What are the downside scenarios?"
            ],
            'trend_analysis': [
                "What is the overall trend?",
                "Are there any trend reversals?",
                "How strong is the momentum?"
            ],
            'correlation_analysis': [
                "What factors are most correlated?",
                "How do different metrics relate?",
                "What drives the performance?"
            ],
            'performance_analysis': [
                "How has performance changed?",
                "What are the key performance drivers?",
                "Where are the opportunities?"
            ]
        }
    
    async def process_exploration_request(
        self,
        user_prompt: str,
        current_data: pd.DataFrame,
        current_visualizations: List[Dict[str, Any]],
        agent_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process natural language exploration request"""
        
        try:
            self.logger.info(f"Processing exploration request: {user_prompt[:100]}...")
            
            # Parse user intent
            intent = await self._parse_user_intent(user_prompt, current_data, agent_context)
            
            # Generate exploration plan
            exploration_plan = await self._generate_exploration_plan(
                intent, current_data, current_visualizations, agent_context
            )
            
            # Create interactive visualizations
            interactive_charts = await self._create_interactive_visualizations(
                exploration_plan, current_data
            )
            
            # Generate follow-up suggestions
            follow_up_suggestions = await self._generate_follow_up_suggestions(
                intent, exploration_plan, current_data, agent_context
            )
            
            return {
                'intent': intent,
                'exploration_plan': exploration_plan,
                'interactive_charts': interactive_charts,
                'follow_up_suggestions': follow_up_suggestions,
                'explanation': self._generate_explanation(intent, exploration_plan)
            }
            
        except Exception as e:
            self.logger.error(f"Exploration request processing failed: {e}")
            return self._generate_fallback_exploration(user_prompt, current_data)
    
    async def _parse_user_intent(
        self,
        user_prompt: str,
        current_data: pd.DataFrame,
        agent_context: Dict[str, Any]
    ) -> ExplorationIntent:
        """Parse user intent using LLM and pattern matching"""
        
        try:
            # Use LLM for intent analysis
            intent_prompt = self._build_intent_analysis_prompt(
                user_prompt, current_data, agent_context
            )
            
            llm_response = await self._query_claude(intent_prompt)
            parsed_intent = await self._parse_intent_response(llm_response)
            
            # Enhance with pattern matching
            enhanced_intent = self._enhance_with_pattern_matching(
                parsed_intent, user_prompt, current_data
            )
            
            return enhanced_intent
            
        except Exception as e:
            self.logger.error(f"Intent parsing failed: {e}")
            return self._get_fallback_intent(user_prompt, current_data)
    
    def _build_intent_analysis_prompt(
        self,
        user_prompt: str,
        current_data: pd.DataFrame,
        agent_context: Dict[str, Any]
    ) -> str:
        """Build prompt for intent analysis"""
        
        data_summary = {
            'shape': current_data.shape,
            'columns': list(current_data.columns),
            'numeric_columns': current_data.select_dtypes(include=['number']).columns.tolist(),
            'datetime_columns': current_data.select_dtypes(include=['datetime64']).columns.tolist(),
            'sample_data': current_data.head(2).to_dict('records') if len(current_data) > 0 else []
        }
        
        prompt = f"""
Analyze this user's data exploration request in the context of financial trading analysis.

## User Request:
"{user_prompt}"

## Available Data:
{data_summary}

## Agent Context:
{agent_context}

## Required Analysis:
Extract the following from the user's request:

1. **Primary Intent**: What is the main goal? (drill_down, comparison, correlation, trend_analysis, risk_assessment, etc.)
2. **Target Variables**: Which specific data columns or metrics are mentioned?
3. **Time Constraints**: Any time-related filters or periods mentioned?
4. **Chart Preferences**: Any specific visualization types requested?
5. **Business Questions**: What trading or investment questions are implied?
6. **Confidence Score**: How clear is the user's intent? (0.0 to 1.0)

## Response Format:
Return JSON with these exact keys:
{{
    "primary_intent": "intent_category",
    "target_variables": ["variable1", "variable2"],
    "time_constraints": {{"start_date": null, "end_date": null, "period": "daily"}},
    "chart_preferences": ["chart_type1", "chart_type2"],
    "business_questions": ["question1", "question2"],
    "confidence_score": 0.85
}}

Focus on financial trading context and be specific about what the user wants to explore.
"""
        return prompt
    
    async def _parse_intent_response(self, llm_response: str) -> ExplorationIntent:
        """Parse LLM response into structured intent"""
        
        try:
            import json
            if llm_response.strip().startswith('{'):
                parsed = json.loads(llm_response)
                return ExplorationIntent(
                    primary_intent=parsed.get('primary_intent', 'general_exploration'),
                    target_variables=parsed.get('target_variables', []),
                    time_constraints=parsed.get('time_constraints', {}),
                    chart_preferences=parsed.get('chart_preferences', []),
                    business_questions=parsed.get('business_questions', []),
                    confidence_score=parsed.get('confidence_score', 0.7)
                )
            else:
                return self._parse_text_intent_response(llm_response)
                
        except json.JSONDecodeError:
            return self._parse_text_intent_response(llm_response)
    
    def _parse_text_intent_response(self, llm_response: str) -> ExplorationIntent:
        """Parse text response when JSON parsing fails"""
        
        # Extract intent from text response
        primary_intent = 'general_exploration'
        target_variables = []
        business_questions = []
        
        lines = llm_response.split('\n')
        for line in lines:
            line = line.strip().lower()
            if 'intent' in line or 'goal' in line:
                for intent_type in self.exploration_patterns.keys():
                    if intent_type in line:
                        primary_intent = intent_type
                        break
        
        return ExplorationIntent(
            primary_intent=primary_intent,
            target_variables=target_variables,
            time_constraints={},
            chart_preferences=[],
            business_questions=business_questions,
            confidence_score=0.6
        )
    
    def _enhance_with_pattern_matching(
        self,
        parsed_intent: ExplorationIntent,
        user_prompt: str,
        current_data: pd.DataFrame
    ) -> ExplorationIntent:
        """Enhance parsed intent with pattern matching"""
        
        prompt_lower = user_prompt.lower()
        
        # Detect additional intents from patterns
        detected_intents = []
        for intent_type, patterns in self.exploration_patterns.items():
            if any(pattern in prompt_lower for pattern in patterns):
                detected_intents.append(intent_type)
        
        # Update primary intent if pattern matching found something
        if detected_intents and parsed_intent.primary_intent == 'general_exploration':
            parsed_intent.primary_intent = detected_intents[0]
        
        # Extract target variables from prompt
        data_columns = [col.lower() for col in current_data.columns]
        for col in data_columns:
            if col in prompt_lower:
                parsed_intent.target_variables.append(col)
        
        # Extract time constraints
        time_patterns = {
            'daily': ['daily', 'day', 'intraday'],
            'weekly': ['weekly', 'week'],
            'monthly': ['monthly', 'month'],
            'yearly': ['yearly', 'year', 'annual']
        }
        
        for period, patterns in time_patterns.items():
            if any(pattern in prompt_lower for pattern in patterns):
                parsed_intent.time_constraints['period'] = period
                break
        
        return parsed_intent
    
    async def _generate_exploration_plan(
        self,
        intent: ExplorationIntent,
        current_data: pd.DataFrame,
        current_visualizations: List[Dict[str, Any]],
        agent_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate exploration plan based on intent"""
        
        plan = {
            'primary_visualizations': [],
            'secondary_visualizations': [],
            'data_filters': {},
            'analysis_focus': [],
            'interaction_options': []
        }
        
        # Generate visualizations based on intent
        if intent.primary_intent == 'drill_down':
            plan['primary_visualizations'] = await self._plan_drill_down_exploration(
                intent, current_data, current_visualizations
            )
        elif intent.primary_intent == 'comparison':
            plan['primary_visualizations'] = await self._plan_comparison_exploration(
                intent, current_data, current_visualizations
            )
        elif intent.primary_intent == 'correlation':
            plan['primary_visualizations'] = await self._plan_correlation_exploration(
                intent, current_data, current_visualizations
            )
        elif intent.primary_intent == 'trend':
            plan['primary_visualizations'] = await self._plan_trend_exploration(
                intent, current_data, current_visualizations
            )
        elif intent.primary_intent == 'volatility':
            plan['primary_visualizations'] = await self._plan_volatility_exploration(
                intent, current_data, current_visualizations
            )
        else:
            plan['primary_visualizations'] = await self._plan_general_exploration(
                intent, current_data, current_visualizations
            )
        
        # Add data filters based on time constraints
        if intent.time_constraints:
            plan['data_filters'] = intent.time_constraints
        
        # Add analysis focus based on business questions
        plan['analysis_focus'] = intent.business_questions
        
        return plan
    
    async def _plan_drill_down_exploration(
        self,
        intent: ExplorationIntent,
        current_data: pd.DataFrame,
        current_visualizations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Plan drill-down exploration"""
        
        visualizations = []
        
        # Add detailed time series if temporal data available
        if current_data.select_dtypes(include=['datetime64']).columns.any():
            visualizations.append({
                'type': 'detailed_time_series',
                'description': 'Detailed view of time series data',
                'parameters': {'zoom_level': 'high', 'show_annotations': True}
            })
        
        # Add distribution analysis for target variables
        for var in intent.target_variables:
            if var in current_data.columns:
                visualizations.append({
                    'type': 'distribution_analysis',
                    'description': f'Distribution analysis for {var}',
                    'parameters': {'variable': var, 'show_statistics': True}
                })
        
        return visualizations
    
    async def _plan_comparison_exploration(
        self,
        intent: ExplorationIntent,
        current_data: pd.DataFrame,
        current_visualizations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Plan comparison exploration"""
        
        visualizations = []
        
        # Multi-line comparison chart
        if len(intent.target_variables) > 1:
            visualizations.append({
                'type': 'multi_line_comparison',
                'description': 'Compare multiple variables over time',
                'parameters': {'variables': intent.target_variables, 'normalize': True}
            })
        
        # Side-by-side comparison
        visualizations.append({
            'type': 'side_by_side_comparison',
            'description': 'Side-by-side comparison of metrics',
            'parameters': {'layout': 'horizontal', 'show_differences': True}
        })
        
        return visualizations
    
    async def _plan_correlation_exploration(
        self,
        intent: ExplorationIntent,
        current_data: pd.DataFrame,
        current_visualizations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Plan correlation exploration"""
        
        visualizations = []
        
        # Correlation heatmap
        numeric_cols = current_data.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 2:
            visualizations.append({
                'type': 'correlation_heatmap',
                'description': 'Correlation matrix heatmap',
                'parameters': {'variables': numeric_cols, 'annotate': True}
            })
        
        # Scatter plot matrix
        if len(intent.target_variables) >= 2:
            visualizations.append({
                'type': 'scatter_matrix',
                'description': 'Scatter plot matrix for correlation analysis',
                'parameters': {'variables': intent.target_variables[:4]}
            })
        
        return visualizations
    
    async def _plan_trend_exploration(
        self,
        intent: ExplorationIntent,
        current_data: pd.DataFrame,
        current_visualizations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Plan trend exploration"""
        
        visualizations = []
        
        # Trend analysis chart
        visualizations.append({
            'type': 'trend_analysis',
            'description': 'Comprehensive trend analysis',
            'parameters': {'show_trend_lines': True, 'show_breakpoints': True}
        })
        
        # Moving averages
        visualizations.append({
            'type': 'moving_averages',
            'description': 'Moving averages for trend confirmation',
            'parameters': {'periods': [20, 50, 200], 'show_signals': True}
        })
        
        return visualizations
    
    async def _plan_volatility_exploration(
        self,
        intent: ExplorationIntent,
        current_data: pd.DataFrame,
        current_visualizations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Plan volatility exploration"""
        
        visualizations = []
        
        # Volatility analysis
        visualizations.append({
            'type': 'volatility_analysis',
            'description': 'Volatility and risk analysis',
            'parameters': {'rolling_window': 20, 'show_bands': True}
        })
        
        # Risk metrics
        visualizations.append({
            'type': 'risk_metrics',
            'description': 'Risk metrics dashboard',
            'parameters': {'metrics': ['var', 'cvar', 'volatility', 'sharpe']}
        })
        
        return visualizations
    
    async def _plan_general_exploration(
        self,
        intent: ExplorationIntent,
        current_data: pd.DataFrame,
        current_visualizations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Plan general exploration"""
        
        visualizations = []
        
        # Data overview
        visualizations.append({
            'type': 'data_overview',
            'description': 'Comprehensive data overview',
            'parameters': {'show_summary': True, 'show_patterns': True}
        })
        
        # Key insights
        visualizations.append({
            'type': 'key_insights',
            'description': 'Key insights and patterns',
            'parameters': {'highlight_anomalies': True, 'show_trends': True}
        })
        
        return visualizations
    
    async def _create_interactive_visualizations(
        self,
        exploration_plan: Dict[str, Any],
        current_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Create interactive visualizations based on exploration plan"""
        
        interactive_charts = []
        
        for viz_config in exploration_plan['primary_visualizations']:
            try:
                chart = await self._create_interactive_chart(viz_config, current_data)
                if chart:
                    interactive_charts.append(chart)
            except Exception as e:
                self.logger.warning(f"Failed to create chart {viz_config['type']}: {e}")
        
        return interactive_charts
    
    async def _create_interactive_chart(
        self,
        viz_config: Dict[str, Any],
        current_data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Create a single interactive chart"""
        
        chart_type = viz_config['type']
        
        # This would integrate with the actual chart generation system
        # For now, return a placeholder structure
        return {
            'type': chart_type,
            'title': viz_config['description'],
            'data': current_data.head(100).to_dict('records'),  # Sample data
            'parameters': viz_config['parameters'],
            'interactive_features': ['zoom', 'pan', 'hover', 'select'],
            'chart_id': f"exploration_{chart_type}_{len(current_data)}"
        }
    
    async def _generate_follow_up_suggestions(
        self,
        intent: ExplorationIntent,
        exploration_plan: Dict[str, Any],
        current_data: pd.DataFrame,
        agent_context: Dict[str, Any]
    ) -> List[ExplorationSuggestion]:
        """Generate follow-up exploration suggestions"""
        
        suggestions = []
        
        # Based on current intent, suggest related explorations
        if intent.primary_intent == 'correlation':
            suggestions.append(ExplorationSuggestion(
                suggestion_type='related_analysis',
                description='Analyze the strongest correlations in detail',
                chart_type='detailed_correlation',
                rationale='Strong correlations often indicate important relationships',
                priority=1
            ))
        
        elif intent.primary_intent == 'trend':
            suggestions.append(ExplorationSuggestion(
                suggestion_type='related_analysis',
                description='Examine trend reversals and breakpoints',
                chart_type='trend_reversal_analysis',
                rationale='Trend reversals are critical for trading decisions',
                priority=1
            ))
        
        # Suggest alternative chart types
        for viz in exploration_plan['primary_visualizations']:
            chart_type = viz['type']
            alternatives = self.chart_alternatives.get(chart_type, [])
            
            for alt_chart in alternatives[:2]:  # Limit to 2 alternatives
                suggestions.append(ExplorationSuggestion(
                    suggestion_type='alternative_view',
                    description=f'Try {alt_chart} view for different perspective',
                    chart_type=alt_chart,
                    rationale=f'Alternative visualization might reveal different insights',
                    priority=2
                ))
        
        # Suggest business-focused explorations
        if intent.business_questions:
            suggestions.append(ExplorationSuggestion(
                suggestion_type='business_analysis',
                description='Focus on trading implications and risk assessment',
                chart_type='trading_dashboard',
                rationale='Business questions require focused trading analysis',
                priority=1
            ))
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _generate_explanation(
        self,
        intent: ExplorationIntent,
        exploration_plan: Dict[str, Any]
    ) -> str:
        """Generate explanation for the exploration"""
        
        explanation = f"I'll help you explore the data focusing on {intent.primary_intent.replace('_', ' ')}. "
        
        if intent.target_variables:
            explanation += f"I'll analyze {', '.join(intent.target_variables)} specifically. "
        
        if exploration_plan['primary_visualizations']:
            viz_types = [viz['type'] for viz in exploration_plan['primary_visualizations']]
            explanation += f"I'll create {', '.join(viz_types)} visualizations to answer your questions. "
        
        if intent.business_questions:
            explanation += f"This analysis will help answer: {'; '.join(intent.business_questions[:2])}."
        
        return explanation
    
    async def _query_claude(self, prompt: str) -> str:
        """Query Claude API with error handling"""
        try:
            # Use synchronous call in async context
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.claude.messages.create(
                    model="claude-sonnet-4-0",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"Claude API query failed: {e}")
            raise
    
    def _get_fallback_intent(self, user_prompt: str, current_data: pd.DataFrame) -> ExplorationIntent:
        """Get fallback intent when parsing fails"""
        
        return ExplorationIntent(
            primary_intent='general_exploration',
            target_variables=[],
            time_constraints={},
            chart_preferences=[],
            business_questions=['What patterns can we see in this data?'],
            confidence_score=0.5
        )
    
    def _generate_fallback_exploration(
        self,
        user_prompt: str,
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate fallback exploration when processing fails"""
        
        return {
            'intent': self._get_fallback_intent(user_prompt, current_data),
            'exploration_plan': {
                'primary_visualizations': [{
                    'type': 'data_overview',
                    'description': 'Basic data overview',
                    'parameters': {}
                }],
                'secondary_visualizations': [],
                'data_filters': {},
                'analysis_focus': ['General data analysis'],
                'interaction_options': []
            },
            'interactive_charts': [],
            'follow_up_suggestions': [],
            'explanation': f"I'll provide a general analysis of your data to help answer: {user_prompt}"
        }
