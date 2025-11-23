Deep Reinforcement Learning for Dynamic Portfolio Optimization and Goal-Based Investment

This repository presents an end-to-end Deep Reinforcement Learning (DRL) system for multi-asset portfolio management. The project integrates the Soft Actor-Critic (SAC) algorithm with a custom-designed financial environment to learn dynamic rebalancing strategies under realistic market conditions. The objective is to optimize long-term portfolio returns while managing risk and supporting investor-defined financial goals.

The environment models portfolio management as a continuous Markov Decision Process (MDP). At each step, the agent observes market data, portfolio configuration, and historical price information. Based on this information, it outputs continuous rebalancing actions that are converted into tradable quantities. The reward function combines risk-adjusted performance metrics (Sharpe ratio) with optional goal-based incentives to guide the learning process.

This implementation includes a modular architecture for agents, neural networks, replay buffers, data processing, logging, and evaluation. It is suitable for research, experimentation, and pedagogical use in reinforcement learning, quantitative finance, and algorithmic trading.

Key Features

Soft Actor-Critic (SAC) agent for continuous trading decisions

Custom financial environment with realistic constraints

Risk-adjusted reward design using a Sharpe-based formulation

Optional terminal reward for goal-based investment

Modular codebase for extensibility and experimentation

Integrated data processing and evaluation workflows

Unit tests for major components

Project Structure
Portfolio-Optimization-and-Goal-Based-Investment-with-RL/
│
├── data/                        # Market data samples (CSV)
├── portfolios_and_tickers/      # Portfolio templates and asset lists
├── src/
│   ├── environment.py           # Custom RL environment
│   ├── agents.py                # SAC agent implementation
│   ├── networks.py              # Policy and critic network models
│   ├── buffer.py                # Replay buffer
│   ├── get_data.py              # Data loading utilities
│   ├── logger.py                # Training logs
│   ├── run.py                   # Main training script
│   ├── main.py                  # Execution entry point
│   └── utilities.py             # Additional helper functions
│
├── tests/                       # Test suite for components
├── requirements.txt             # Dependency specifications
├── README.md                    # Project documentation
└── LICENSE                      # Project license

Methodology
State Representation

The agent receives a structured state vector composed of:

Cash position

Current asset prices

Asset holdings

Rolling historical price windows

Normalized market indicators

Action Space

A continuous vector in the range [-1, 1]^n, later scaled and discretized to represent buy and sell quantities for each asset.

Reward Function

The primary reward is based on a risk-adjusted Sharpe formulation:

r_t = λ * (R_t − R_f) / σ_t


An optional terminal reward component supports goal-based investment scenarios.

Training and Execution

Install dependencies:

pip install -r requirements.txt


Run the training process:

python src/run.py


Execute the main program:

python src/main.py

Evaluation

The repository includes a Jupyter notebook for analyzing:

Portfolio value curves

Return and volatility metrics

Sharpe ratio evolution

Action patterns and trading behavior

Risk characteristics and drawdowns

Applications

This project serves as a practical benchmark for:

Reinforcement learning research in finance

Portfolio management experimentation

Academic coursework involving quantitative methods

Comparative evaluation of DRL algorithms in continuous control settings

License

The repository is distributed under the license included in the project.
