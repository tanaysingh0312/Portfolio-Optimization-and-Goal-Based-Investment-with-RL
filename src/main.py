# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from argparse import ArgumentParser
import json 
import numpy as np
import os
import time
import torch

from src.agents import instanciate_agent
from src.environment import Environment
from src.get_data import load_data
from src.run import Run
from src.utilities import create_directory_tree, instanciate_scaler, prepare_initial_portfolio

def main(args):

    # specifying the hardware
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initializing the random seeds for reproducibility
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # creating all the necessary directory tree structure for efficient logging
    checkpoint_directory = create_directory_tree(
        checkpoint_directory=args.checkpoint_directory,
        simple=args.simple
    )

    # instanciating the scaler
    scaler = instanciate_scaler(use_scaler=args.use_scaler)

    # loading the data
    stock_market_history = load_data(
        initial_date=args.initial_date,
        final_date=args.final_date,
        tickers_subset=args.tickers_subset,
        mode=args.mode,
        scaler=scaler
    )

    # preparing the initial portfolio
    initial_portfolio = prepare_initial_portfolio(
        initial_portfolio_value=args.initial_portfolio_value,
        stock_market_history=stock_market_history
    )

    # instanciating the environment
    env = Environment(
        stock_market_history=stock_market_history,
        initial_portfolio=initial_portfolio,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        bank_rate=args.bank_rate,
        limit_n_stocks=args.limit_n_stocks,
        buy_rule=args.buy_rule,
        # --- NEW GBI PARAMETERS PASSED TO ENVIRONMENT ---
        target_return_rate=args.target_return_rate,
        max_steps_per_episode=args.time_horizon,
        # ----------------------------------------------
        use_corr_matrix=args.use_corr_matrix,
        use_corr_eigenvalues=args.use_corr_eigenvalues,
        window=args.window,
        number_of_eigenvalues=args.number_of_eigenvalues
    )

    # instanciating the agent
    agent = instanciate_agent(
        args=args,
        env=env,
        device=device,
        checkpoint_directory=checkpoint_directory
    )

    # instanciating the run
    run = Run(
        env=env,
        agent=agent,
        n_episodes=args.n_episodes,
        agent_type=args.agent_type,
        scaler=scaler,
        checkpoint_directory=checkpoint_directory,
        sac_temperature=args.sac_temperature,
        mode=args.mode,
        plot=args.plot
    )

    # logging initial portfolio value
    run.logger._store_initial_value_portfolio(initial_portfolio['portfolio_value'])

    # running the training/testing loop
    run.logger.set_time_stamp(1)
    for _ in range(args.n_episodes):
        run.run_episode()
    run.logger.set_time_stamp(2)
    run.logger.print_status()

    # save the logs and plot the results
    run.logger.save_logs()
    if args.plot:
        run.logger.plot_results()


if __name__ == '__main__':

    parser = ArgumentParser()

    # parameters concerning the run itself (train or test)
    parser.add_argument('--mode',             type=str,            default='train', help='Train or test mode')
    parser.add_argument('--n_episodes',       type=int,            default=50000,   help='Number of episodes to run')
    parser.add_argument('--sac_temperature',  type=float,          default=1.0,     help='The temperature (alpha) parameter of the SAC algorithm')

    # parameters concerning data preprocessing
    parser.add_argument('--initial_date',     type=str,            default='2010-01-01', help='Start date for data fetching')
    parser.add_argument('--final_date',       type=str,            default='2020-12-31', help='End date for data fetching')
    parser.add_argument('--tickers_subset',   type=str,            default='portfolios_and_tickers/subset_tickers_S&P500.txt', help='Path to the file containing the tickers to use')

    # parameters concerning the environment and the learning process
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--use_scaler',             action='store_true', default=False, help='Whether to normalize the environment state (except for the correlation matrix) or not')
    parser.add_argument('--initial_portfolio_value', type=float,          default=10000.0, help='Starting value of the portfolio')
    parser.add_argument('--buy_cost',                type=float,          default=0.001,   help='Transaction cost when buying, expressed in fraction')
    parser.add_argument('--sell_cost',               type=float,          default=0.001,   help='Transaction cost when selling, expressed in fraction')
    parser.add_argument('--bank_rate',               type=float,          default=0.5,     help='The rate at which the bank account is growing (or shrinking)')
    parser.add_argument('--limit_n_stocks',          type=float,          default=200,     help='Upper limit to the number of stocks we can own')
    parser.add_argument('--buy_rule',                type=str,            default='most_first', help='Rule according to which stocks are bought, either most_first or least_first')
    
    # --- NEW GBI PARAMETERS START ---
    # These parameters explicitly define the Goal-Based Investment objective
    parser.add_argument('--target_return_rate',      type=float,          default=0.25,    help='Target return rate for the investment goal (e.g., 0.25 for 25% goal)')
    parser.add_argument('--time_horizon',            type=int,            default=252,     help='Number of steps (trading days) to reach the goal (e.g., 252 for 1 year)')
    # --- NEW GBI PARAMETERS END ---

    # parameters concerning the agent type (SAC variants)
    parser.add_argument('--agent_type',       type=str,            default='regular', help='SAC agent type: regular, distributional or manual_temperature')
    parser.add_argument('--lr_Q',             type=float,          default=0.0003,    help='Learning rate for the Q-networks')
    parser.add_argument('--lr_pi',            type=float,          default=0.0003,    help='Learning rate for the Actor network')
    parser.add_argument('--lr_alpha',         type=float,          default=0.0003,    help='Learning rate for the temperature (alpha) network')
    parser.add_argument('--tau',              type=float,          default=0.005,     help='Interpolation factor for the target networks')
    parser.add_argument('--memory_size',      type=int,            default=1000000,   help='Size of the Replay Buffer')
    parser.add_argument('--layer_size',       type=int,            default=256,       help='Number of neurons in the hidden layers')
    parser.add_argument('--batch_size',       type=int,            default=256,       help='Batch size for learning')
    parser.add_argument('--delay',            type=int,            default=1,         help='Delay factor for target networks update')
    parser.add_argument('--grad_clip',        type=float,          default=1.0,       help='Clip the gradients to this value')

    # miscellaneous parameters
    parser.add_argument('--simple',               action='store_true', default=False,        help='Whether to save the outputs in an overwritten directory, used simple experiments and tuning')
    
    # random seed, logs information and hardware
    parser.add_argument('--checkpoint_directory', type=str,            default=None,         help='In test mode, specify the directory in which to find the weights of the trained networks')
    parser.add_argument('--plot',                 action='store_true', default=False,        help='Whether to automatically generate plots or not')
    parser.add_argument('--seed',                 type=int,            default='42',         help='Random seed for reproducibility')
    parser.add_argument('--gpu_devices',          type=int, nargs='+', default=[0, 1, 2, 3], help='Specify the GPUs if any')
    
    # parameters concerning data preprocessing
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--use_corr_matrix',       action='store_true', default=False, help='To append the sliding correlation matrix to the time series')
    group2.add_argument('--use_corr_eigenvalues',  action='store_true', default=False, help='To append the eigenvalues of the correlation matrix to the time series')
    parser.add_argument('--window',                type=int,            default=20,    help='Window for correlation matrix computation')
    parser.add_argument('--number_of_eigenvalues', type=int,            default=10,    help='Number of largest eigenvalues to append to the close prices time series')
     
    args = parser.parse_args()
    main(args)