Deep Reinforcement Learning for Goal-Based Portfolio Optimization

Repository:
https://github.com/tanaysingh0312/Portfolio-Optimization-and-Goal-Based-Investment-with-RL

Overview

This project implements a Deep Reinforcement Learning (DRL) system for goal-based portfolio optimization using the Soft Actor-Critic (SAC) algorithm.
The agent learns how to dynamically rebalance a multi-asset portfolio while considering market uncertainty, transaction costs, cash constraints, and long-term investment goals.

Unlike traditional portfolio optimization methods, this approach directly learns optimal strategies from market data using continuous control and adaptive decision-making.

Key Features

Soft Actor-Critic (SAC) based reinforcement learning agent

Goal-based reward structure for long-term wealth targeting

Dynamic portfolio rebalancing with continuous actions

Realistic constraints such as transaction costs and cash limits

Optional use of asset correlation features for improved learning

Modular and extensible codebase built with PyTorch

Project Structure
Portfolio-Optimization-and-Goal-Based-Investment-with-RL/
├── data/                     # Market data
├── portfolios_and_tickers/   # Asset lists
├── src/                      # Core source code
│   ├── environment.py        # Custom investment environment
│   ├── agents.py             # SAC agent
│   ├── networks.py           # Neural networks
│   ├── buffer.py             # Replay buffer
│   └── main.py               # Training and testing entry point
├── postprocess.ipynb         # Result analysis
├── requirements.txt
└── LICENSE

Installation
git clone https://github.com/tanaysingh0312/Portfolio-Optimization-and-Goal-Based-Investment-with-RL.git
cd Portfolio-Optimization-and-Goal-Based-Investment-with-RL

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

Usage
Train the Agent
python -m src.main --mode train --n_episodes 1000 --goal 120000 --auto_alpha

Test a Trained Model
python -m src.main --mode test --checkpoint_directory saved_outputs/YOUR_TIMESTAMP --plot

Applications

Goal-based investment planning

Portfolio optimization research

Financial reinforcement learning experiments

License

Apache License 2.0

Author

Tanay Singh

GitHub: https://github.com/tanaysingh0312

LinkedIn: https://www.linkedin.com/in/stanay657/
