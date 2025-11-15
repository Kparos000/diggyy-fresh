"""
Streamlit Dashboard for Diggyy Fresh
Live simulation, RL vs Baseline comparison, and LLM explanations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment import GroceryInventoryEnv
from src.agent import GroceryAgent, BaselinePolicy
from src.llm_integration import OllamaLLM
from src.utils import (
    run_simulation,
    calculate_waste_percentage,
    format_currency,
    format_percentage,
    get_performance_grade,
)


# Page configuration
st.set_page_config(
    page_title="Diggyy Fresh - RL Grocery Inventory",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_agent():
    """Load trained RL agent (NO CACHE - always fresh)"""
    env = GroceryInventoryEnv(num_products=5, episode_length=30)
    agent = GroceryAgent(env)

    try:
        agent.load_model()
        return agent, env, True
    except FileNotFoundError:
        st.warning("No trained model found. Please train the agent first.")
        return agent, env, False


@st.cache_resource
def load_llm():
    """Load LLM (cached)"""
    llm = OllamaLLM()
    is_connected = llm.check_connection()
    return llm, is_connected


def run_live_simulation(env, policy, policy_name: str):
    """Run live simulation and collect step-by-step data"""

    obs, info = env.reset()
    simulation_data = []
    done = False
    step = 0

    while not done:
        # Get action
        if hasattr(policy, 'predict'):
            action, *_ = policy.predict(obs, deterministic=True)
        else:
            action, *_ = policy.predict(obs)

        # CRITICAL FIX: Ensure action is always an array
        import numpy as np
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 0:
            # Scalar detected - convert to array with 5 products
            action = np.full(5, action, dtype=np.float32)

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record data
        step_data = {
            'Day': step + 1,
            'Revenue': info.get('revenue', 0),
            'Spoilage': info.get('spoilage_cost', 0),
            'Stockouts': info.get('stockout_penalty', 0),
            'Reward': reward,
            'Total_Inventory': info.get('total_inventory', 0),
            'Cumulative_Revenue': info['total_revenue'],
            'Cumulative_Spoilage': info['total_spoilage'],
        }

        simulation_data.append(step_data)
        step += 1

    df = pd.DataFrame(simulation_data)
    return df, info


def plot_comparison_metrics(rl_df, baseline_df):
    """Create comparison plots"""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cumulative Revenue', 'Cumulative Spoilage',
                       'Daily Reward', 'Inventory Level'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Cumulative Revenue
    fig.add_trace(
        go.Scatter(x=rl_df['Day'], y=rl_df['Cumulative_Revenue'],
                  name='RL Agent', line=dict(color='green', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=baseline_df['Day'], y=baseline_df['Cumulative_Revenue'],
                  name='Baseline', line=dict(color='gray', width=2, dash='dash')),
        row=1, col=1
    )

    # Cumulative Spoilage
    fig.add_trace(
        go.Scatter(x=rl_df['Day'], y=rl_df['Cumulative_Spoilage'],
                  name='RL Agent', line=dict(color='red', width=2),
                  showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=baseline_df['Day'], y=baseline_df['Cumulative_Spoilage'],
                  name='Baseline', line=dict(color='gray', width=2, dash='dash'),
                  showlegend=False),
        row=1, col=2
    )

    # Daily Reward
    fig.add_trace(
        go.Scatter(x=rl_df['Day'], y=rl_df['Reward'],
                  name='RL Agent', line=dict(color='blue', width=2),
                  showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=baseline_df['Day'], y=baseline_df['Reward'],
                  name='Baseline', line=dict(color='gray', width=2, dash='dash'),
                  showlegend=False),
        row=2, col=1
    )

    # Inventory Level
    fig.add_trace(
        go.Scatter(x=rl_df['Day'], y=rl_df['Total_Inventory'],
                  name='RL Agent', line=dict(color='purple', width=2),
                  showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=baseline_df['Day'], y=baseline_df['Total_Inventory'],
                  name='Baseline', line=dict(color='gray', width=2, dash='dash'),
                  showlegend=False),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Day", row=2, col=1)
    fig.update_xaxes(title_text="Day", row=2, col=2)
    fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
    fig.update_yaxes(title_text="Spoilage ($)", row=1, col=2)
    fig.update_yaxes(title_text="Reward", row=2, col=1)
    fig.update_yaxes(title_text="Units", row=2, col=2)

    fig.update_layout(height=600, showlegend=True, legend=dict(x=0.7, y=1.15, orientation='h'))

    return fig


def main():
    st.title("🥬 Diggyy Fresh - RL Grocery Inventory Agent")
    st.markdown("### Reducing food waste with Reinforcement Learning + LLM")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Load resources
    agent, env, agent_loaded = load_agent()
    llm, llm_connected = load_llm()

    # Status indicators
    st.sidebar.markdown("**System Status:**")
    st.sidebar.markdown(f"{'✅' if agent_loaded else '❌'} RL Agent")
    st.sidebar.markdown(f"{'✅' if llm_connected else '❌'} LLM (Ollama)")

    # Simulation settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Settings")

    baseline_order_qty = st.sidebar.slider(
        "Baseline Order Quantity",
        min_value=10,
        max_value=40,
        value=20,
        help="Fixed order quantity for baseline policy"
    )

    demand_shock = st.sidebar.selectbox(
        "Demand Shock",
        ["None", "Weekend Spike", "Supply Drop", "Volatile"],
        help="Simulate demand shocks"
    )

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Simulation",
        "⚖️ RL vs Baseline",
        "🤖 LLM Insights",
        "📈 Training Stats"
    ])

    # Tab 1: Live Simulation
    with tab1:
        st.header("Live 30-Day Simulation")

        if not agent_loaded:
            st.error("Please train the RL agent first by running: `python src/agent.py`")
            return

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 Run RL Agent Simulation", type="primary"):
                with st.spinner("Running RL agent simulation..."):
                    try:
                        rl_df, rl_info = run_live_simulation(env, agent, "RL Agent")
                        st.session_state['rl_df'] = rl_df
                        st.session_state['rl_info'] = rl_info
                    except Exception as e:
                        st.error(f"RL Agent Error: {str(e)}")
                        st.warning("⚠️ Model needs retraining. Run: `python src/agent.py`")
                        st.info("In the meantime, try the Baseline simulation or RL vs Baseline comparison tab.")

        with col2:
            if st.button("📊 Run Baseline Simulation"):
                with st.spinner("Running baseline simulation..."):
                    baseline = BaselinePolicy(order_quantity=baseline_order_qty)
                    baseline_df, baseline_info = run_live_simulation(env, baseline, "Baseline")
                    st.session_state['baseline_df'] = baseline_df
                    st.session_state['baseline_info'] = baseline_info

        # Display results
        if 'rl_df' in st.session_state:
            st.subheader("RL Agent Performance")

            rl_df = st.session_state['rl_df']
            rl_info = st.session_state['rl_info']

            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Revenue",
                    format_currency(rl_info['total_revenue']),
                )

            with col2:
                waste_pct = calculate_waste_percentage(
                    rl_info['total_spoilage'],
                    rl_info['total_revenue']
                )
                st.metric(
                    "Waste %",
                    format_percentage(waste_pct),
                )

            with col3:
                st.metric(
                    "Total Spoilage",
                    format_currency(rl_info['total_spoilage']),
                )

            with col4:
                st.metric(
                    "Stockouts",
                    f"{rl_info['total_stockouts']} units",
                )

            # Performance grade
            grade, description = get_performance_grade(waste_pct)
            st.success(f"**Performance Grade: {grade}** - {description}")

            # Plot
            fig = px.line(
                rl_df,
                x='Day',
                y=['Cumulative_Revenue', 'Cumulative_Spoilage'],
                title='RL Agent: Revenue vs Spoilage Over Time',
                labels={'value': 'Amount ($)', 'variable': 'Metric'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Data table
            with st.expander("📋 View Daily Data"):
                st.dataframe(rl_df, use_container_width=True)

    # Tab 2: RL vs Baseline Comparison
    with tab2:
        st.header("RL Agent vs Baseline Comparison")

        if not agent_loaded:
            st.error("Please train the RL agent first.")
            return

        if st.button("🔄 Run Comparison", type="primary"):
            with st.spinner("Running both simulations..."):
                try:
                    # Run RL agent
                    rl_df, rl_info = run_live_simulation(env, agent, "RL Agent")

                    # Run baseline
                    baseline = BaselinePolicy(order_quantity=baseline_order_qty)
                    baseline_df, baseline_info = run_live_simulation(env, baseline, "Baseline")

                    st.session_state['comparison_rl_df'] = rl_df
                    st.session_state['comparison_rl_info'] = rl_info
                    st.session_state['comparison_baseline_df'] = baseline_df
                    st.session_state['comparison_baseline_info'] = baseline_info
                except Exception as e:
                    st.error(f"Simulation Error: {str(e)}")
                    st.warning("⚠️ Model needs retraining. Run: `python src/agent.py`")

        if 'comparison_rl_df' in st.session_state:
            rl_df = st.session_state['comparison_rl_df']
            rl_info = st.session_state['comparison_rl_info']
            baseline_df = st.session_state['comparison_baseline_df']
            baseline_info = st.session_state['comparison_baseline_info']

            # Comparison metrics
            st.subheader("Performance Metrics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**RL Agent**")
                rl_waste_pct = calculate_waste_percentage(
                    rl_info['total_spoilage'],
                    rl_info['total_revenue']
                )
                st.metric("Revenue", format_currency(rl_info['total_revenue']))
                st.metric("Waste %", format_percentage(rl_waste_pct))
                st.metric("Spoilage", format_currency(rl_info['total_spoilage']))
                st.metric("Stockouts", f"{rl_info['total_stockouts']} units")

            with col2:
                st.markdown("**Baseline**")
                baseline_waste_pct = calculate_waste_percentage(
                    baseline_info['total_spoilage'],
                    baseline_info['total_revenue']
                )
                st.metric("Revenue", format_currency(baseline_info['total_revenue']))
                st.metric("Waste %", format_percentage(baseline_waste_pct))
                st.metric("Spoilage", format_currency(baseline_info['total_spoilage']))
                st.metric("Stockouts", f"{baseline_info['total_stockouts']} units")

            # Calculate improvement
            waste_reduction = (baseline_waste_pct - rl_waste_pct) / baseline_waste_pct * 100
            revenue_improvement = (rl_info['total_revenue'] - baseline_info['total_revenue']) / baseline_info['total_revenue'] * 100

            st.markdown("---")
            st.subheader("🎯 RL Agent Improvements")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Waste Reduction",
                    format_percentage(waste_reduction),
                    delta=f"{waste_reduction:.1f}%",
                    delta_color="normal"
                )

            with col2:
                st.metric(
                    "Revenue Improvement",
                    format_percentage(revenue_improvement),
                    delta=f"{revenue_improvement:.1f}%",
                    delta_color="normal"
                )

            with col3:
                profit_diff = (rl_info['total_revenue'] - rl_info['total_spoilage']) - \
                             (baseline_info['total_revenue'] - baseline_info['total_spoilage'])
                st.metric(
                    "Profit Gain",
                    format_currency(profit_diff),
                    delta=format_currency(profit_diff),
                    delta_color="normal"
                )

            # Comparison plot
            st.subheader("Detailed Comparison")
            fig = plot_comparison_metrics(rl_df, baseline_df)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 3: LLM Insights
    with tab3:
        st.header("🤖 LLM-Powered Insights")

        if not llm_connected:
            st.warning("Ollama is not running. Start it with: `ollama serve`")
            st.info("Install Ollama from https://ollama.ai and pull llama3.2")
            return

        st.success("✅ LLM connected and ready")

        if 'rl_df' in st.session_state and 'rl_info' in st.session_state:
            rl_info = st.session_state['rl_info']

            if st.button("🧠 Generate Insights", type="primary"):
                with st.spinner("LLM is analyzing performance (may take 30-60 seconds)..."):
                    try:
                        # Prepare stats
                        total_stats = {
                            'revenue': rl_info['total_revenue'],
                            'spoilage': rl_info['total_spoilage'],
                            'waste_pct': calculate_waste_percentage(
                                rl_info['total_spoilage'],
                                rl_info['total_revenue']
                            ),
                            'stockouts': rl_info['total_stockouts'],
                        }

                        product_breakdown = {
                            'berries': {'waste': 'high', 'stockouts': 'low'},
                            'milk': {'waste': 'medium', 'stockouts': 'medium'},
                            'bread': {'waste': 'low', 'stockouts': 'low'},
                        }

                        # Generate insights
                        insights = llm.generate_insights(total_stats, product_breakdown)

                        # Check if error message
                        if insights.startswith("Error:"):
                            st.error(f"LLM Error: {insights}")
                            st.info("💡 Try: 1) Check Ollama is running (`ollama serve`), 2) Model is loaded (`ollama pull llama3.2`)")
                        else:
                            st.session_state['insights'] = insights
                    except Exception as e:
                        st.error(f"Failed to generate insights: {str(e)}")
                        st.info("Make sure Ollama is running and llama3.2 model is available")

            if 'insights' in st.session_state:
                st.subheader("Key Insights")
                st.markdown(st.session_state['insights'])

            # Demand prediction demo
            st.markdown("---")
            st.subheader("Demand Adjustment Prediction")

            product_select = st.selectbox(
                "Select Product",
                ["berries", "milk", "bread", "leafy_greens", "tomatoes"]
            )

            if st.button("Predict Demand Adjustment"):
                with st.spinner("Asking LLM..."):
                    result = llm.predict_demand_adjustment(
                        product=product_select,
                        current_inventory=20,
                        day_of_week=5,  # Saturday
                        recent_sales=[12, 14, 15, 16, 18, 22, 25],
                        weather_signal=0.8,
                    )

                    st.metric("Adjustment Factor", f"{result['adjustment_factor']:.2f}x")
                    st.info(f"**Reasoning:** {result['reasoning']}")

        else:
            st.info("Run a simulation first to generate insights")

    # Tab 4: Training Stats
    with tab4:
        st.header("📈 Training Statistics")

        if agent_loaded:
            st.success("Model loaded successfully")
            st.markdown("""
            **Model Details:**
            - Algorithm: PPO (Proximal Policy Optimization)
            - Training steps: 50,000
            - Policy network: MLP (Multi-Layer Perceptron)
            - Environment: 5 products, 30-day episodes
            """)

            st.info("Training metrics will be added after first training run")

        else:
            st.warning("No trained model found")

            st.markdown("""
            **To train the model:**

            ```bash
            python src/agent.py
            ```

            This will:
            1. Create the environment
            2. Train PPO agent for 50k timesteps (~10 min)
            3. Save model to `models/ppo_grocery.zip`
            4. Run baseline comparison
            """)


if __name__ == "__main__":
    main()
