"""
LLM Integration for Demand Interpretation and Decision Explanation
Uses Ollama API with Llama 3.2 for local inference
"""

import requests
import json
from typing import Dict, List, Optional


class OllamaLLM:
    """Wrapper for Ollama API (local LLM inference)"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"

    def _call_api(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Make API call to Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }

            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=120,  # Increased timeout for slower LLM responses
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"Error: API returned status {response.status_code}"

        except requests.exceptions.RequestException as e:
            return f"Error: Could not connect to Ollama - {str(e)}"

    def predict_demand_adjustment(
        self,
        product: str,
        current_inventory: int,
        day_of_week: int,
        recent_sales: List[int],
        weather_signal: float,
    ) -> Dict:
        """
        Use LLM to interpret demand signals and suggest adjustment

        Returns:
            dict with 'adjustment_factor' (float) and 'reasoning' (str)
        """

        # Build concise prompt
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        avg_sales = sum(recent_sales[-7:]) / len(recent_sales[-7:]) if recent_sales else 0

        prompt = f"""Given the following grocery inventory context, predict demand adjustment:

Product: {product}
Current Inventory: {current_inventory} units
Day: {day_names[day_of_week]}
Avg Daily Sales (last 7 days): {avg_sales:.1f} units
Weather Index: {weather_signal:.2f} (0=bad, 1=good)

Provide a demand adjustment factor (0.5 to 1.5, where 1.0 is normal) and brief reasoning.
Format: FACTOR: X.XX | REASON: brief explanation

Response:"""

        response = self._call_api(prompt, max_tokens=100, temperature=0.5)

        # Parse response
        try:
            parts = response.split('|')
            factor_part = parts[0].replace('FACTOR:', '').strip()
            reason_part = parts[1].replace('REASON:', '').strip() if len(parts) > 1 else "No reason provided"

            factor = float(factor_part)
            factor = max(0.5, min(1.5, factor))  # Clamp to valid range

            return {
                'adjustment_factor': factor,
                'reasoning': reason_part,
                'raw_response': response,
            }

        except (ValueError, IndexError):
            # Fallback if parsing fails
            return {
                'adjustment_factor': 1.0,
                'reasoning': 'LLM parsing failed, using default',
                'raw_response': response,
            }

    def explain_decision(
        self,
        product: str,
        state_info: Dict,
        action_taken: float,
        reward_components: Dict,
    ) -> str:
        """
        Generate human-readable explanation for RL agent's decision

        Args:
            product: Product name
            state_info: Current state information
            action_taken: Order quantity decided by agent
            reward_components: Revenue, spoilage, stockout breakdown

        Returns:
            Explanation string
        """

        prompt = f"""Explain this grocery ordering decision in 2-3 sentences:

Product: {product}
Inventory Level: {state_info.get('inventory', 0)} units
Order Quantity: {action_taken:.0f} units
Day: {state_info.get('day', 'Unknown')}

Performance:
- Revenue: ${reward_components.get('revenue', 0):.2f}
- Spoilage Cost: ${reward_components.get('spoilage_cost', 0):.2f}
- Stockout Penalty: ${reward_components.get('stockout_penalty', 0):.2f}

Explain why this order quantity makes sense given the inventory and performance metrics.

Explanation:"""

        response = self._call_api(prompt, max_tokens=150, temperature=0.7)
        return response

    def generate_insights(
        self,
        total_stats: Dict,
        product_breakdown: Dict,
    ) -> str:
        """
        Generate high-level insights about inventory performance

        Args:
            total_stats: Overall statistics (revenue, waste, stockouts)
            product_breakdown: Per-product performance

        Returns:
            Insights string
        """

        prompt = f"""Analyze this grocery inventory performance and provide 3 key insights:

Overall Performance (30 days):
- Total Revenue: ${total_stats.get('revenue', 0):.2f}
- Total Waste: ${total_stats.get('spoilage', 0):.2f}
- Waste %: {total_stats.get('waste_pct', 0):.1f}%
- Stockouts: {total_stats.get('stockouts', 0)} units

Top Issues:
{json.dumps(product_breakdown, indent=2)[:300]}

Provide 3 actionable insights to improve performance.

Insights:"""

        response = self._call_api(prompt, max_tokens=250, temperature=0.7)
        return response

    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


class DemandPredictor:
    """High-level demand prediction with LLM assistance"""

    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def predict_with_context(
        self,
        product: str,
        base_demand: float,
        context: Dict,
    ) -> Dict:
        """
        Predict demand using base model + LLM adjustment

        Args:
            product: Product name
            base_demand: Base demand from historical data
            context: Environmental context (inventory, day, weather, etc.)

        Returns:
            dict with 'predicted_demand', 'confidence', 'explanation'
        """

        # Get LLM adjustment
        llm_result = self.llm.predict_demand_adjustment(
            product=product,
            current_inventory=context.get('inventory', 0),
            day_of_week=context.get('day_of_week', 0),
            recent_sales=context.get('recent_sales', []),
            weather_signal=context.get('weather_signal', 0.5),
        )

        # Apply adjustment
        adjusted_demand = base_demand * llm_result['adjustment_factor']

        return {
            'predicted_demand': adjusted_demand,
            'base_demand': base_demand,
            'adjustment_factor': llm_result['adjustment_factor'],
            'confidence': 0.7,  # Placeholder
            'explanation': llm_result['reasoning'],
        }


if __name__ == "__main__":
    # Test LLM integration
    print("Testing Ollama LLM integration...")

    llm = OllamaLLM()

    # Check connection
    if llm.check_connection():
        print("✓ Ollama is running and accessible")
    else:
        print("✗ Ollama not accessible. Make sure it's running: ollama serve")
        exit(1)

    # Test demand adjustment
    print("\nTesting demand adjustment...")
    result = llm.predict_demand_adjustment(
        product="berries",
        current_inventory=15,
        day_of_week=5,  # Saturday
        recent_sales=[12, 14, 11, 13, 15, 18, 20],
        weather_signal=0.8,
    )
    print(f"Adjustment Factor: {result['adjustment_factor']}")
    print(f"Reasoning: {result['reasoning']}")

    # Test decision explanation
    print("\nTesting decision explanation...")
    explanation = llm.explain_decision(
        product="milk",
        state_info={'inventory': 25, 'day': 'Monday'},
        action_taken=30,
        reward_components={'revenue': 87.25, 'spoilage_cost': 5.00, 'stockout_penalty': 0},
    )
    print(f"Explanation: {explanation}")

    print("\nLLM integration test complete!")
