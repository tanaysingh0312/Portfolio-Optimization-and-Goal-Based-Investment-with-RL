import streamlit as st
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import sys
from datetime import datetime

# --- SETTINGS ---
SAVED_OUTPUTS_DIR = "saved_outputs"

st.set_page_config(
    page_title="Goal-Based RL Portfolio",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for UI styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

def get_latest_checkpoint():
    """Finds the most recent timestamped folder in saved_outputs."""
    if not os.path.exists(SAVED_OUTPUTS_DIR):
        return None
    folders = [f for f in glob.glob(os.path.join(SAVED_OUTPUTS_DIR, "*")) if os.path.isdir(f)]
    if not folders:
        return None
    # Sort by modification time, newest first
    latest = max(folders, key=os.path.getmtime)
    # Check if this folder has trained networks
    if os.path.exists(os.path.join(latest, "networks", "actor")):
        return latest
    return None

def get_latest_checkpoint_with_networks():
    """Find the latest checkpoint that actually has trained networks."""
    base = "saved_outputs"
    if not os.path.exists(base):
        return None
    runs = sorted(
        [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
        key=os.path.getmtime,
        reverse=True
    )
    for r in runs:
        if os.path.exists(os.path.join(r, "networks", "actor")):
            return r
    return None

# --- SIDEBAR CONFIG ---
st.sidebar.header("🛠️ Run Configuration")
mode = st.sidebar.selectbox("Execution Mode", ["train", "test"], index=0)
episodes = st.sidebar.number_input("Episodes", min_value=1, max_value=5000, value=10)
horizon = st.sidebar.number_input("Time Horizon (days)", min_value=10, max_value=500, value=20)
target_rate = st.sidebar.slider("Target Return Rate (%)", 5, 50, 25) / 100

# --- MAIN UI ---
st.title("📈 Goal-Based Portfolio Optimization")
st.markdown(f"**Status**: Ready to run RL Agent in `{mode}` mode.")

col_ctrl, col_viz = st.columns([1, 2])

with col_ctrl:
    st.header("Control Panel")
    run_btn = st.button("🚀 Start RL Agent", use_container_width=True)
    
    if run_btn:
        st.info(f"Initiating `{mode}`...")
        
        # --- ROBUST PATH FIX ---
        # 1. Identify where ui.py is. This is the PROJECT ROOT.
        # It should contain the 'src' folder.
        project_root = os.path.abspath(os.path.dirname(__file__))
        
        # 2. Setup the environment
        env_vars = os.environ.copy()
        # Prepend project root to PYTHONPATH so 'src' is discoverable as a package
        env_vars["PYTHONPATH"] = project_root + os.pathsep + env_vars.get("PYTHONPATH", "")
        
        # 3. Get Python executable
        python_exe = sys.executable 
        
        cmd = [
            python_exe, "-m", "src.main",
            "--mode", mode,
            "--n_episodes", str(episodes),
            "--time_horizon", str(horizon),
            "--target_return_rate", str(target_rate),
            "--assets_to_trade", "portfolios_and_tickers/tickers_S&P500_subset.txt",
            "--initial_portfolio", "portfolios_and_tickers/initial_portfolio_subset.json"
        ]
        
        # Auto-load latest trained checkpoint when in test mode
        if mode == "test":
            checkpoint_dir = get_latest_checkpoint_with_networks()
            if checkpoint_dir:
                cmd.append("--checkpoint_directory")
                cmd.append(checkpoint_dir)
                full_log = f"Using checkpoint: {checkpoint_dir}\n"
            else:
                full_log = "Warning: No trained networks found for test mode\n"
        else:
            full_log = ""
        
        log_area = st.empty()
        full_log += f"Detected Project Root: {project_root}\n"
        full_log += f"Using Python: {python_exe}\n"
        full_log += "-"*30 + "\n"
        
        # Run subprocess forced to the project root
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1,
            env=env_vars,
            cwd=project_root  # CRITICAL: Ensures we are NOT inside 'src' when running
        )
        
        for line in process.stdout:
            full_log += line
            log_area.code("\n".join(full_log.splitlines()[-15:]))
            
        process.wait()
        
        if process.returncode == 0:
            st.success("Execution Completed!")
            st.rerun()
        else:
            st.error(f"Execution Failed (Code {process.returncode}). Check if 'src/__init__.py' exists.")

with col_viz:
    st.header("📊 Performance Analysis")
    
    latest_run = get_latest_checkpoint_with_networks()
    
    if latest_run:
        st.caption(f"📂 Source: `{os.path.basename(latest_run)}`")
        
        logs_dir = os.path.join(latest_run, "logs")
        reward_file = os.path.join(logs_dir, f"{mode}_reward_history.npy")
        portfolio_file = os.path.join(logs_dir, f"{mode}_portfolio_value_history.npy")
        content_file = os.path.join(logs_dir, f"{mode}_portfolio_content_history.npy")
        
        tab_train, tab_test, tab_assets = st.tabs(["🎯 Rewards", "💰 Portfolio", "📋 Assets"])
        
        with tab_train:
            if os.path.exists(reward_file):
                rewards = np.load(reward_file)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(rewards, alpha=0.3, label="Reward")
                if len(rewards) > 5:
                    ma = pd.Series(rewards).rolling(5).mean()
                    ax.plot(ma, linewidth=2.5, label="MA(5)")
                ax.set_title("Learning Progress")
                ax.legend()
                st.pyplot(fig)
                st.caption(
                    "Note: Goal-based rewards are sparse and discontinuous. "
                    "Fluctuations are expected during early training."
                )
            else:
                st.warning("No reward logs found.")

        with tab_test:
            if os.path.exists(portfolio_file):
                pv_data = np.load(portfolio_file)
                fig, ax = plt.subplots(figsize=(10, 4))
                plot_val = pv_data[-1] if len(pv_data.shape) > 1 else pv_data
                ax.plot(plot_val, color='#0068C9', linewidth=2)
                
                goal_val = plot_val[0] * (1 + target_rate)
                ax.axhline(y=goal_val, color='green', linestyle='--', label="Target")
                ax.set_title("Portfolio Value")
                st.pyplot(fig)
                
                ret = ((plot_val[-1] - plot_val[0]) / plot_val[0]) * 100
                st.metric("Total Return", f"{ret:.2f}%")
            else:
                st.info("No portfolio history found.")

        with tab_assets:
            if os.path.exists(content_file):
                contents = np.load(content_file)
                last_alloc = contents[-1] if len(contents.shape) > 1 else contents
                if len(last_alloc.shape) > 1: last_alloc = last_alloc[-1]
                
                try:
                    with open("portfolios_and_tickers/tickers_S&P500_subset.txt", 'r') as f:
                        tickers = f.read().splitlines()
                    st.bar_chart(pd.DataFrame(last_alloc, index=tickers[:len(last_alloc)]))
                except:
                    st.bar_chart(last_alloc)
            else:
                st.info("No asset data found.")
    else:
        st.info("No results to display. Click 'Run' to start.")

st.divider()
st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")