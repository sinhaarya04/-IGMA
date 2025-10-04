"""
Enhanced LLM Reasoning Engine for Chart Justification
Provides comprehensive reasoning for chart selection and financial analysis
"""

import asyncio
from typing import Dict, List, Any, Optional
from anthropic import Anthropic
import json
import pandas as pd
from dataclasses import dataclass
import logging
from .advanced_chart_selector import ChartRecommendation

@dataclass
class ReasoningResult:
    """Result from LLM reasoning analysis"""
    chart_justification: str
    market_insights: List[str]
    trading_implications: List[str]
    risk_considerations: List[str]
    confidence_score: float
    suggested_actions: List[str]

class EnhancedLLMReasoning:
    """Enhanced LLM reasoning for chart justification and analysis"""
    
    def __init__(self, claude_client: Optional[Anthropic] = None):
        self.claude = claude_client or Anthropic()
        self.logger = logging.getLogger(__name__)
        self.reasoning_cache = {}
        
    async def generate_chart_reasoning(
        self,
        chart_recommendations: List[ChartRecommendation],
        data_summary: Dict[str, Any],
        market_context: Dict[str, Any],
        user_query: str = "",
        agent_context: Dict[str, Any] = None
    ) -> ReasoningResult:
        """Generate comprehensive reasoning for chart recommendations"""
        
        try:
            # Create cache key for performance
            cache_key = self._create_cache_key(chart_recommendations, data_summary, user_query)
            if cache_key in self.reasoning_cache:
                return self.reasoning_cache[cache_key]
            
            # Prepare comprehensive prompt
            reasoning_prompt = await self._build_reasoning_prompt(
                chart_recommendations, data_summary, market_context, user_query, agent_context
            )
            
            # Get LLM analysis
            llm_response = await self._query_claude(reasoning_prompt)
            
            # Parse and structure the response
            reasoning_result = await self._parse_reasoning_response(llm_response, chart_recommendations)
            
            # Cache result
            self.reasoning_cache[cache_key] = reasoning_result
            
            return reasoning_result
            
        except Exception as e:
            self.logger.error(f"Chart reasoning generation failed: {e}")
            return self._generate_fallback_reasoning(chart_recommendations, data_summary)
    
    async def generate_exploratory_insights(
        self,
        df: pd.DataFrame,
        user_query: str,
        agent_analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate exploratory insights for user interaction"""
        
        try:
            # Analyze data patterns for exploration
            exploration_prompt = await self._build_exploration_prompt(df, user_query, agent_analyses)
            
            llm_response = await self._query_claude(exploration_prompt)
            
            exploration_results = await self._parse_exploration_response(llm_response)
            
            return exploration_results
            
        except Exception as e:
            self.logger.error(f"Exploratory insights generation failed: {e}")
            return self._generate_fallback_exploration(df, user_query)
    
    async def justify_agent_workflow_decisions(
        self,
        workflow_state: Dict[str, Any],
        agent_communications: List[Dict[str, Any]],
        current_visualizations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate reasoning for multi-agent workflow decisions"""
        
        try:
            workflow_prompt = await self._build_workflow_justification_prompt(
                workflow_state, agent_communications, current_visualizations
            )
            
            llm_response = await self._query_claude(workflow_prompt)
            
            workflow_reasoning = await self._parse_workflow_response(llm_response)
            
            return workflow_reasoning
            
        except Exception as e:
            self.logger.error(f"Workflow reasoning generation failed: {e}")
            return self._generate_fallback_workflow_reasoning(workflow_state)
    
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
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"Claude API query failed: {e}")
            raise
    
    def _create_cache_key(
        self, 
        chart_recommendations: List[ChartRecommendation], 
        data_summary: Dict[str, Any], 
        user_query: str
    ) -> str:
        """Create cache key for reasoning results"""
        chart_types = [rec.chart_type for rec in chart_recommendations]
        data_shape = data_summary.get('shape', (0, 0))
        return f"{chart_types}_{data_shape}_{user_query[:50]}"
    
    async def _build_reasoning_prompt(
        self,
        chart_recommendations: List[ChartRecommendation],
        data_summary: Dict[str, Any],
        market_context: Dict[str, Any],
        user_query: str,
        agent_context: Dict[str, Any]
    ) -> str:
        """Build comprehensive reasoning prompt"""
        
        chart_summary = []
        for rec in chart_recommendations:
            chart_summary.append(f"- {rec.chart_type}: {rec.reasoning} (confidence: {rec.confidence_score:.2f})")
        
        prompt = f"""
As a senior financial data visualization expert, analyze these chart recommendations and provide comprehensive reasoning.

## Chart Recommendations:
{chr(10).join(chart_summary)}

## Data Summary:
- Shape: {data_summary.get('shape', 'Unknown')}
- Columns: {data_summary.get('columns', [])}
- Data Types: {data_summary.get('dtypes', {})}
- Has OHLC: {data_summary.get('has_ohlc', False)}
- Has Volume: {data_summary.get('has_volume', False)}
- Timespan: {data_summary.get('timespan_days', 0)} days

## Market Context:
{json.dumps(market_context, indent=2)}

## User Query:
{user_query}

## Agent Context:
{json.dumps(agent_context or {}, indent=2)}

## Required Analysis:
Provide a structured analysis covering:

1. **Chart Justification**: Why these specific chart types are optimal for this data and context
2. **Market Insights**: What key insights these visualizations will reveal about market conditions
3. **Trading Implications**: How these charts support trading decisions and risk management
4. **Risk Considerations**: What risks or limitations should be considered when interpreting these charts
5. **Suggested Actions**: Specific actions traders should take based on these visualizations

Format your response as JSON with these exact keys:
{{
    "chart_justification": "detailed explanation",
    "market_insights": ["insight1", "insight2", "insight3"],
    "trading_implications": ["implication1", "implication2", "implication3"],
    "risk_considerations": ["risk1", "risk2", "risk3"],
    "confidence_score": 0.85,
    "suggested_actions": ["action1", "action2", "action3"]
}}

Be specific, actionable, and focus on financial trading context.
"""
        return prompt
    
    async def _build_exploration_prompt(
        self,
        df: pd.DataFrame,
        user_query: str,
        agent_analyses: Dict[str, Any]
    ) -> str:
        """Build prompt for exploratory insights"""
        
        data_summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
        }
        
        prompt = f"""
As a financial data exploration expert, analyze this user query and provide insights for interactive exploration.

## User Query:
{user_query}

## Available Data:
{json.dumps(data_summary, indent=2, default=str)}

## Agent Analyses:
{json.dumps(agent_analyses, indent=2, default=str)}

## Required Analysis:
Provide exploration insights covering:

1. **Query Intent**: What is the user trying to understand or explore?
2. **Relevant Data Patterns**: What patterns in the data are most relevant to this query?
3. **Suggested Visualizations**: What additional charts or views would help answer this query?
4. **Exploration Path**: What logical next steps should the user consider?
5. **Key Insights**: What insights can be derived from the current data and agent analyses?

Format your response as JSON:
{{
    "query_intent": "description of what user wants to explore",
    "relevant_patterns": ["pattern1", "pattern2"],
    "suggested_visualizations": ["chart1", "chart2"],
    "exploration_path": ["step1", "step2", "step3"],
    "key_insights": ["insight1", "insight2", "insight3"]
}}
"""
        return prompt
    
    async def _build_workflow_justification_prompt(
        self,
        workflow_state: Dict[str, Any],
        agent_communications: List[Dict[str, Any]],
        current_visualizations: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for workflow justification"""
        
        prompt = f"""
As a multi-agent system expert, analyze this trading workflow and provide justification for the current state.

## Workflow State:
{json.dumps(workflow_state, indent=2, default=str)}

## Agent Communications:
{json.dumps(agent_communications, indent=2, default=str)}

## Current Visualizations:
{json.dumps(current_visualizations, indent=2, default=str)}

## Required Analysis:
Provide workflow justification covering:

1. **Workflow Progress**: How well is the multi-agent workflow progressing?
2. **Agent Coordination**: Are agents effectively communicating and building on each other's insights?
3. **Visualization Integration**: How well are visualizations supporting agent decision-making?
4. **Decision Quality**: What is the quality of decisions being made based on available information?
5. **Next Steps**: What should happen next in the workflow?

Format your response as JSON:
{{
    "workflow_progress": "assessment of current progress",
    "agent_coordination": "assessment of agent communication",
    "visualization_integration": "assessment of visual support",
    "decision_quality": "assessment of decision quality",
    "next_steps": ["step1", "step2", "step3"]
}}
"""
        return prompt
    
    async def _parse_reasoning_response(
        self, 
        llm_response: str, 
        chart_recommendations: List[ChartRecommendation]
    ) -> ReasoningResult:
        """Parse LLM response into structured reasoning result"""
        
        try:
            # Try to parse as JSON first
            if llm_response.strip().startswith('{'):
                parsed = json.loads(llm_response)
                return ReasoningResult(
                    chart_justification=parsed.get('chart_justification', ''),
                    market_insights=parsed.get('market_insights', []),
                    trading_implications=parsed.get('trading_implications', []),
                    risk_considerations=parsed.get('risk_considerations', []),
                    confidence_score=parsed.get('confidence_score', 0.7),
                    suggested_actions=parsed.get('suggested_actions', [])
                )
            else:
                # Fallback to text parsing
                return self._parse_text_response(llm_response, chart_recommendations)
                
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON response, using text parsing")
            return self._parse_text_response(llm_response, chart_recommendations)
    
    def _parse_text_response(
        self, 
        llm_response: str, 
        chart_recommendations: List[ChartRecommendation]
    ) -> ReasoningResult:
        """Parse text response when JSON parsing fails"""
        
        # Extract key sections from text response
        lines = llm_response.split('\n')
        
        justification = ""
        insights = []
        implications = []
        risks = []
        actions = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if 'justification' in line.lower() or 'why' in line.lower():
                current_section = 'justification'
                justification += line + " "
            elif 'insight' in line.lower():
                current_section = 'insights'
                if line.startswith('-') or line.startswith('•'):
                    insights.append(line[1:].strip())
            elif 'implication' in line.lower() or 'trading' in line.lower():
                current_section = 'implications'
                if line.startswith('-') or line.startswith('•'):
                    implications.append(line[1:].strip())
            elif 'risk' in line.lower():
                current_section = 'risks'
                if line.startswith('-') or line.startswith('•'):
                    risks.append(line[1:].strip())
            elif 'action' in line.lower() or 'suggest' in line.lower():
                current_section = 'actions'
                if line.startswith('-') or line.startswith('•'):
                    actions.append(line[1:].strip())
            elif current_section == 'justification':
                justification += line + " "
            elif current_section == 'insights' and (line.startswith('-') or line.startswith('•')):
                insights.append(line[1:].strip())
            elif current_section == 'implications' and (line.startswith('-') or line.startswith('•')):
                implications.append(line[1:].strip())
            elif current_section == 'risks' and (line.startswith('-') or line.startswith('•')):
                risks.append(line[1:].strip())
            elif current_section == 'actions' and (line.startswith('-') or line.startswith('•')):
                actions.append(line[1:].strip())
        
        # Calculate confidence score based on chart recommendations
        avg_confidence = sum(rec.confidence_score for rec in chart_recommendations) / len(chart_recommendations) if chart_recommendations else 0.5
        
        return ReasoningResult(
            chart_justification=justification.strip() or f"Charts selected based on data characteristics: {[rec.chart_type for rec in chart_recommendations]}",
            market_insights=insights[:3] or ["Market patterns require further analysis"],
            trading_implications=implications[:3] or ["Trading decisions should consider multiple factors"],
            risk_considerations=risks[:3] or ["Standard market risks apply"],
            confidence_score=avg_confidence,
            suggested_actions=actions[:3] or ["Monitor market conditions closely"]
        )
    
    async def _parse_exploration_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse exploration response"""
        try:
            if llm_response.strip().startswith('{'):
                return json.loads(llm_response)
            else:
                return self._parse_text_exploration_response(llm_response)
        except json.JSONDecodeError:
            return self._parse_text_exploration_response(llm_response)
    
    def _parse_text_exploration_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse text exploration response"""
        return {
            "query_intent": "User wants to explore data patterns",
            "relevant_patterns": ["Data patterns require analysis"],
            "suggested_visualizations": ["Additional charts may be helpful"],
            "exploration_path": ["Continue analysis", "Review findings"],
            "key_insights": ["Insights will emerge from analysis"]
        }
    
    async def _parse_workflow_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse workflow response"""
        try:
            if llm_response.strip().startswith('{'):
                return json.loads(llm_response)
            else:
                return self._parse_text_workflow_response(llm_response)
        except json.JSONDecodeError:
            return self._parse_text_workflow_response(llm_response)
    
    def _parse_text_workflow_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse text workflow response"""
        return {
            "workflow_progress": "Workflow progressing normally",
            "agent_coordination": "Agents communicating effectively",
            "visualization_integration": "Visualizations supporting analysis",
            "decision_quality": "Decisions based on available information",
            "next_steps": ["Continue analysis", "Review results"]
        }
    
    def _generate_fallback_reasoning(
        self, 
        chart_recommendations: List[ChartRecommendation], 
        data_summary: Dict[str, Any]
    ) -> ReasoningResult:
        """Generate fallback reasoning when LLM fails"""
        
        chart_types = [rec.chart_type for rec in chart_recommendations]
        avg_confidence = sum(rec.confidence_score for rec in chart_recommendations) / len(chart_recommendations) if chart_recommendations else 0.5
        
        return ReasoningResult(
            chart_justification=f"Selected charts {chart_types} based on data characteristics and financial analysis requirements.",
            market_insights=["Market data shows various patterns requiring analysis"],
            trading_implications=["Charts provide insights for trading decisions"],
            risk_considerations=["Consider market volatility and data limitations"],
            confidence_score=avg_confidence,
            suggested_actions=["Monitor market conditions", "Review chart patterns", "Consider risk management"]
        )
    
    def _generate_fallback_exploration(self, df: pd.DataFrame, user_query: str) -> Dict[str, Any]:
        """Generate fallback exploration when LLM fails"""
        return {
            "query_intent": f"User wants to explore: {user_query}",
            "relevant_patterns": ["Data patterns in the dataset"],
            "suggested_visualizations": ["Additional charts for exploration"],
            "exploration_path": ["Analyze current data", "Generate insights"],
            "key_insights": ["Insights will emerge from data analysis"]
        }
    
    def _generate_fallback_workflow_reasoning(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback workflow reasoning when LLM fails"""
        return {
            "workflow_progress": "Workflow in progress",
            "agent_coordination": "Agents working together",
            "visualization_integration": "Visualizations integrated",
            "decision_quality": "Decisions being made",
            "next_steps": ["Continue workflow", "Complete analysis"]
        }
