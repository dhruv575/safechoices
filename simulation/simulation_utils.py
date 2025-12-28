"""
Simulation utilities for Safe Choices prediction market trading simulations.

This module contains shared functions for running Monte Carlo simulations
of different trading strategies on prediction markets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List, Dict, Any

def load_and_filter_data(csv_path: str, start_date: str = '2025-01-01') -> pd.DataFrame:
    """
    Load the market data and filter for simulation period.
    
    Args:
        csv_path: Path to the CSV file containing market data
        start_date: Start date for simulation (markets must close after this date)
    
    Returns:
        Filtered DataFrame ready for simulation
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Convert dates - handle timezone awareness
    df['closingDate'] = pd.to_datetime(df['closingDate'], format='mixed', errors='coerce', utc=True)
    start_dt = pd.to_datetime(start_date, utc=True)
    
    # Filter for markets that close after start date and have complete data
    mask = (
        (df['closingDate'] >= start_dt) &
        (df['outcome'].notna()) &
        (df['probability7d'].notna()) &
        (df['probability6d'].notna()) &
        (df['probability5d'].notna()) &
        (df['probability4d'].notna()) &
        (df['probability3d'].notna()) &
        (df['probability2d'].notna()) &
        (df['probability1d'].notna())
    )
    
    filtered_df = df[mask].copy().reset_index(drop=True)
    
    # Vectorized outcome conversion for speed
    outcome_map = {'True': 1, 'true': 1, 'FALSE': 0, 'false': 0, True: 1, False: 0}
    filtered_df['outcome_int'] = filtered_df['outcome'].map(outcome_map)
    
    # Fill any remaining NaN outcomes with proper conversion
    remaining_mask = filtered_df['outcome_int'].isna()
    if remaining_mask.any():
        def convert_outcome(value):
            if pd.isna(value):
                return None
            if isinstance(value, (int, float)):
                return int(value)
            return 1 if str(value).lower() == 'true' else 0
        
        filtered_df.loc[remaining_mask, 'outcome_int'] = filtered_df.loc[remaining_mask, 'outcome'].apply(convert_outcome)
    
    # Sort by closing date for better performance in simulations
    filtered_df = filtered_df.sort_values('closingDate').reset_index(drop=True)
    
    return filtered_df

def check_market_eligibility(market_row: pd.Series, days_before: int, 
                           min_prob_7d: float, min_prob_current: float) -> bool:
    """
    Check if a market meets the probability thresholds for investment.
    
    Args:
        market_row: Row from the DataFrame containing market data
        days_before: Number of days before resolution to check (1-7)
        min_prob_7d: Minimum probability threshold at 7 days before
        min_prob_current: Minimum probability threshold at current day
    
    Returns:
        True if market meets criteria, False otherwise
    """
    prob_col = f'probability{days_before}d'
    
    # Check if required columns exist and have valid data
    if pd.isna(market_row.get('probability7d')) or pd.isna(market_row.get(prob_col)):
        return False
    
    # Check probability thresholds
    prob_7d = market_row['probability7d']
    prob_current = market_row[prob_col]
    
    return prob_7d >= min_prob_7d and prob_current >= min_prob_current

def calculate_days_until_resolution(current_date: datetime, closing_date: datetime) -> int:
    """
    Calculate days until market resolution.
    
    Args:
        current_date: Current simulation date
        closing_date: Market closing date
    
    Returns:
        Number of days until resolution
    """
    return max(0, (closing_date - current_date).days)

def get_available_markets(df: pd.DataFrame, current_date: datetime, days_before: int,
                         min_prob_7d: float, min_prob_current: float) -> pd.DataFrame:
    """
    Get markets available for investment at current date.
    
    Args:
        df: DataFrame containing market data
        current_date: Current simulation date
        days_before: Days before resolution to invest
        min_prob_7d: Minimum probability at 7 days before
        min_prob_current: Minimum probability at current day
    
    Returns:
        DataFrame of available markets with their days until resolution
    """
    # Vectorized approach - much faster than iterating
    prob_col = f'probability{days_before}d'
    
    # Calculate days until resolution for all markets at once
    days_until = (df['closingDate'] - current_date).dt.days
    
    # Create boolean mask for all conditions at once
    mask = (
        (days_until >= days_before) &  # Market resolves in future
        (df['probability7d'] >= min_prob_7d) &  # 7d probability threshold
        (df[prob_col] >= min_prob_current) &  # Current day probability threshold
        (df['probability7d'].notna()) &  # Valid 7d data
        (df[prob_col].notna()) &  # Valid current day data
        (df['outcome_int'].notna())  # Valid outcome data
    )
    
    if not mask.any():
        return pd.DataFrame()
    
    # Filter and add days_until_resolution column
    available_markets = df[mask].copy()
    available_markets['days_until_resolution'] = days_until[mask]
    
    return available_markets

def select_next_market(available_markets: pd.DataFrame, random_state: np.random.RandomState, 
                      skew_factor: float = 0.1) -> Optional[pd.Series]:
    """
    Select the next market to invest in using a left-skewed exponential distribution.
    
    Args:
        available_markets: DataFrame of markets available for investment
        random_state: Random state for reproducible results
        skew_factor: Controls the left skew (higher = more skew toward closer markets)
    
    Returns:
        Selected market as Series, or None if no markets available
    """
    if len(available_markets) == 0:
        return None
    
    # Configurable market selection with adjustable skew
    days_until = available_markets['days_until_resolution'].values
    
    # Use exponential decay with configurable skew factor
    # Closer markets (lower days) get exponentially higher weights
    weights = np.exp(-days_until * skew_factor)
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Select market based on weights
    selected_idx = random_state.choice(len(available_markets), p=weights)
    
    return available_markets.iloc[selected_idx]

def calculate_investment_return(market: pd.Series, days_before: int, capital: float) -> float:
    """
    Calculate return from investing in a market.
    
    Args:
        market: Market data as Series
        days_before: Days before resolution when investment was made
        capital: Amount invested
    
    Returns:
        Final capital after resolution (capital * (1/probability) if win, 0 if loss)
    """
    prob_col = f'probability{days_before}d'
    probability = market[prob_col]
    outcome = market['outcome_int']
    
    if outcome == 1:  # Market resolved True
        # Return is capital * (1 / probability)
        return capital / probability
    else:  # Market resolved False
        return 0.0

def run_single_fund_simulation(df: pd.DataFrame, 
                              starting_capital: float = 10000,
                              start_date: str = '2025-01-01',
                              max_duration_days: int = 365,
                              days_before: int = 1,
                              min_prob_7d: float = 0.90,
                              min_prob_current: float = 0.90,
                              ending_factor_start: float = 0.05,
                              ending_factor_increment: float = 0.001,
                              skew_factor: float = 0.1,
                              target_return: Optional[float] = None,
                              random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run a single fund simulation.
    
    Args:
        df: Market data DataFrame
        starting_capital: Initial capital
        start_date: Simulation start date
        max_duration_days: Maximum simulation duration
        days_before: Days before resolution to invest
        min_prob_7d: Minimum probability at 7 days
        min_prob_current: Minimum probability at investment day
        ending_factor_start: Starting ending factor
        ending_factor_increment: Daily increment to ending factor
        skew_factor: Controls left skew in market selection (higher = more skew)
        target_return: Target return threshold to stop trading (None = no threshold)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing simulation results
    """
    random_state = np.random.RandomState(random_seed)
    
    current_date = pd.to_datetime(start_date, utc=True)
    end_date = current_date + timedelta(days=max_duration_days)
    capital = starting_capital
    
    trades = []
    daily_capital = []
    
    sim_day = 0
    
    # Track daily capital less frequently to improve performance
    last_capital_record = 0
    
    while current_date <= end_date and capital > 0:
        # Check if we've reached target return threshold
        if target_return is not None:
            current_return = (capital - starting_capital) / starting_capital
            if current_return >= target_return:
                break
        
        # Calculate current ending factor
        ending_factor = ending_factor_start + (ending_factor_increment * sim_day)
        
        # Check if we hit ending factor
        if random_state.random() < ending_factor:
            break
        
        # Get available markets
        available_markets = get_available_markets(
            df, current_date, days_before, min_prob_7d, min_prob_current
        )
        
        if len(available_markets) == 0:
            # No markets available, advance day by larger steps to speed up
            current_date += timedelta(days=7)  # Skip a week instead of one day
            sim_day = (current_date - pd.to_datetime(start_date, utc=True)).days
            continue
        
        # Select market
        selected_market = select_next_market(available_markets, random_state, skew_factor)
        
        if selected_market is None:
            current_date += timedelta(days=7)
            sim_day = (current_date - pd.to_datetime(start_date, utc=True)).days
            continue
        
        # Calculate investment date and resolution date
        investment_date = current_date
        resolution_date = selected_market['closingDate']
        
        # Calculate return
        new_capital = calculate_investment_return(selected_market, days_before, capital)
        
        # Record trade (only essential info for speed)
        trades.append({
            'trade_number': len(trades) + 1,
            'investment_date': investment_date,
            'resolution_date': resolution_date,
            'probability': selected_market[f'probability{days_before}d'],
            'capital_invested': capital,
            'outcome': selected_market['outcome_int'],
            'capital_after': new_capital,
            'return': (new_capital - capital) / capital if capital > 0 else 0,
            'sim_day': sim_day
        })
        
        capital = new_capital
        
        # Advance to resolution date + 1 day
        current_date = resolution_date + timedelta(days=1)
        sim_day = (current_date - pd.to_datetime(start_date, utc=True)).days
        
        # Record daily capital less frequently (every 10th day or trade)
        if sim_day - last_capital_record >= 10 or len(trades) % 10 == 0:
            daily_capital.append({
                'date': current_date,
                'capital': capital,
                'day': sim_day
            })
            last_capital_record = sim_day
    
    # Calculate final statistics
    total_return = (capital - starting_capital) / starting_capital if starting_capital > 0 else 0
    num_trades = len(trades)
    went_bust = capital == 0
    reached_target = target_return is not None and total_return >= target_return
    
    # Determine ending reason
    if went_bust:
        ending_reason = 'bust'
    elif reached_target:
        ending_reason = 'target_reached'
    elif current_date <= end_date:
        ending_reason = 'ending_factor'
    else:
        ending_reason = 'max_duration'
    
    return {
        'final_capital': capital,
        'total_return': total_return,
        'num_trades': num_trades,
        'went_bust': went_bust,
        'reached_target': reached_target,
        'ending_reason': ending_reason,
        'simulation_days': sim_day,
        'trades': trades,
        'daily_capital': daily_capital,
        'parameters': {
            'starting_capital': starting_capital,
            'start_date': start_date,
            'max_duration_days': max_duration_days,
            'days_before': days_before,
            'min_prob_7d': min_prob_7d,
            'min_prob_current': min_prob_current,
            'ending_factor_start': ending_factor_start,
            'ending_factor_increment': ending_factor_increment,
            'skew_factor': skew_factor,
            'target_return': target_return,
            'random_seed': random_seed
        }
    }

def plot_simulation_results(results_list: List[Dict[str, Any]], title: str = "Simulation Results"):
    """
    Plot results from multiple simulation runs.
    
    Args:
        results_list: List of simulation result dictionaries
        title: Plot title
    """
    if not results_list:
        print("No results to plot")
        return
    
    # Extract data
    final_capitals = [r['final_capital'] for r in results_list]
    total_returns = [r['total_return'] for r in results_list]
    num_trades = [r['num_trades'] for r in results_list]
    bust_rate = sum(1 for r in results_list if r['went_bust']) / len(results_list)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Final capital distribution
    axes[0, 0].hist(final_capitals, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(final_capitals), color='red', linestyle='--', 
                       label=f'Mean: ${np.mean(final_capitals):,.0f}')
    axes[0, 0].axvline(np.median(final_capitals), color='green', linestyle='--',
                       label=f'Median: ${np.median(final_capitals):,.0f}')
    axes[0, 0].set_xlabel('Final Capital ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Final Capital Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Return distribution
    return_pct = [r * 100 for r in total_returns]
    axes[0, 1].hist(return_pct, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(return_pct), color='red', linestyle='--',
                       label=f'Mean: {np.mean(return_pct):.1f}%')
    axes[0, 1].axvline(np.median(return_pct), color='green', linestyle='--',
                       label=f'Median: {np.median(return_pct):.1f}%')
    axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.5, label='Break-even')
    axes[0, 1].set_xlabel('Total Return (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Return Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Number of trades distribution
    axes[1, 0].hist(num_trades, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(num_trades), color='red', linestyle='--',
                       label=f'Mean: {np.mean(num_trades):.1f}')
    axes[1, 0].set_xlabel('Number of Trades')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Number of Trades Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    Summary Statistics:
    
    Total Simulations: {len(results_list):,}
    Bust Rate: {bust_rate:.1%}
    
    Final Capital:
    Mean: ${np.mean(final_capitals):,.0f}
    Median: ${np.median(final_capitals):,.0f}
    Min: ${np.min(final_capitals):,.0f}
    Max: ${np.max(final_capitals):,.0f}
    
    Total Return:
    Mean: {np.mean(total_returns):.1%}
    Median: {np.median(total_returns):.1%}
    Min: {np.min(total_returns):.1%}
    Max: {np.max(total_returns):.1%}
    
    Trades per Simulation:
    Mean: {np.mean(num_trades):.1f}
    Median: {np.median(num_trades):.1f}
    """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def print_simulation_summary(results_list: List[Dict[str, Any]]):
    """
    Print detailed summary statistics for simulation results.
    
    Args:
        results_list: List of simulation result dictionaries
    """
    if not results_list:
        print("No results to summarize")
        return
    
    # Extract data
    final_capitals = np.array([r['final_capital'] for r in results_list])
    total_returns = np.array([r['total_return'] for r in results_list])
    num_trades = np.array([r['num_trades'] for r in results_list])
    
    # Calculate statistics
    bust_count = sum(1 for r in results_list if r['went_bust'])
    bust_rate = bust_count / len(results_list)
    
    target_reached_count = sum(1 for r in results_list if r.get('reached_target', False))
    target_reached_rate = target_reached_count / len(results_list)
    
    positive_return_count = sum(1 for r in total_returns if r > 0)
    positive_return_rate = positive_return_count / len(results_list)
    
    # Check if target return was used
    target_return = results_list[0]['parameters'].get('target_return', None)
    
    print("=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total Simulations: {len(results_list):,}")
    print(f"Went Bust: {bust_count:,} ({bust_rate:.1%})")
    if target_return is not None:
        print(f"Reached Target ({target_return:.1%}): {target_reached_count:,} ({target_reached_rate:.1%})")
    print(f"Positive Returns: {positive_return_count:,} ({positive_return_rate:.1%})")
    
    print(f"\nFINAL CAPITAL STATISTICS:")
    print(f"Mean: ${final_capitals.mean():,.2f}")
    print(f"Median: ${np.median(final_capitals):,.2f}")
    print(f"Std Dev: ${final_capitals.std():,.2f}")
    print(f"Min: ${final_capitals.min():,.2f}")
    print(f"Max: ${final_capitals.max():,.2f}")
    
    print(f"\nRETURN STATISTICS:")
    print(f"Mean: {total_returns.mean():.1%}")
    print(f"Median: {np.median(total_returns):.1%}")
    print(f"Std Dev: {total_returns.std():.1%}")
    print(f"Min: {total_returns.min():.1%}")
    print(f"Max: {total_returns.max():.1%}")
    
    print(f"\nTRADE STATISTICS:")
    print(f"Mean Trades: {num_trades.mean():.1f}")
    print(f"Median Trades: {np.median(num_trades):.1f}")
    print(f"Min Trades: {num_trades.min()}")
    print(f"Max Trades: {num_trades.max()}")
    
    # Percentiles
    percentiles = [5, 10, 25, 75, 90, 95]
    print(f"\nRETURN PERCENTILES:")
    for p in percentiles:
        value = np.percentile(total_returns, p)
        print(f"{p}th percentile: {value:.1%}")

def run_multi_fund_simulation(df: pd.DataFrame,
                             n_funds: int = 5,
                             starting_capital: float = 10000,
                             start_date: str = '2025-01-01',
                             max_duration_days: int = 365,
                             days_before: int = 1,
                             min_prob_7d: float = 0.90,
                             min_prob_current: float = 0.90,
                             ending_factor_start: float = 0.05,
                             ending_factor_increment: float = 0.001,
                             skew_factor: float = 0.1,
                             target_return: Optional[float] = None,
                             random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run a multi-fund simulation where capital is divided into independent funds.
    
    Args:
        df: Market data DataFrame
        n_funds: Number of independent funds to create
        starting_capital: Total initial capital (divided among funds)
        start_date: Simulation start date
        max_duration_days: Maximum simulation duration
        days_before: Days before resolution to invest
        min_prob_7d: Minimum probability at 7 days
        min_prob_current: Minimum probability at investment day
        ending_factor_start: Starting ending factor
        ending_factor_increment: Daily increment to ending factor
        skew_factor: Controls left skew in market selection
        target_return: Target return threshold per fund (None = no threshold)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing multi-fund simulation results
    """
    # Set up random state
    random_state = np.random.RandomState(random_seed)
    
    # Calculate capital per fund
    capital_per_fund = starting_capital / n_funds
    
    # Run simulation for each fund independently
    fund_results = []
    all_trades = []
    
    for fund_id in range(n_funds):
        # Use different seed for each fund to ensure independence
        fund_seed = random_state.randint(0, 1000000)
        
        # Run single fund simulation for this fund
        fund_result = run_single_fund_simulation(
            df=df,
            starting_capital=capital_per_fund,
            start_date=start_date,
            max_duration_days=max_duration_days,
            days_before=days_before,
            min_prob_7d=min_prob_7d,
            min_prob_current=min_prob_current,
            ending_factor_start=ending_factor_start,
            ending_factor_increment=ending_factor_increment,
            skew_factor=skew_factor,
            target_return=target_return,
            random_seed=fund_seed
        )
        
        # Add fund ID to result and trades
        fund_result['fund_id'] = fund_id
        for trade in fund_result['trades']:
            trade['fund_id'] = fund_id
            all_trades.append(trade)
        
        fund_results.append(fund_result)
    
    # Calculate portfolio-level statistics
    surviving_funds = sum(1 for fund in fund_results if not fund['went_bust'])
    total_final_capital = sum(fund['final_capital'] for fund in fund_results)
    total_portfolio_return = (total_final_capital - starting_capital) / starting_capital if starting_capital > 0 else 0
    
    # Calculate average metrics across surviving funds
    if surviving_funds > 0:
        avg_capital_per_surviving_fund = sum(fund['final_capital'] for fund in fund_results if not fund['went_bust']) / surviving_funds
        avg_return_per_surviving_fund = sum(fund['total_return'] for fund in fund_results if not fund['went_bust']) / surviving_funds
    else:
        avg_capital_per_surviving_fund = 0
        avg_return_per_surviving_fund = -1  # All funds went bust
    
    # Target achievement stats
    funds_reached_target = sum(1 for fund in fund_results if fund.get('reached_target', False))
    target_achievement_rate = funds_reached_target / n_funds
    
    # Trading activity stats
    total_trades = len(all_trades)
    avg_trades_per_fund = total_trades / n_funds
    
    # Survivorship and diversification metrics
    survivorship_rate = surviving_funds / n_funds
    bust_rate = 1 - survivorship_rate
    
    return {
        'portfolio_final_capital': total_final_capital,
        'portfolio_total_return': total_portfolio_return,
        'n_funds': n_funds,
        'surviving_funds': surviving_funds,
        'survivorship_rate': survivorship_rate,
        'bust_rate': bust_rate,
        'avg_capital_per_surviving_fund': avg_capital_per_surviving_fund,
        'avg_return_per_surviving_fund': avg_return_per_surviving_fund,
        'funds_reached_target': funds_reached_target,
        'target_achievement_rate': target_achievement_rate,
        'total_trades': total_trades,
        'avg_trades_per_fund': avg_trades_per_fund,
        'fund_results': fund_results,
        'all_trades': all_trades,
        'parameters': {
            'n_funds': n_funds,
            'starting_capital': starting_capital,
            'capital_per_fund': capital_per_fund,
            'start_date': start_date,
            'max_duration_days': max_duration_days,
            'days_before': days_before,
            'min_prob_7d': min_prob_7d,
            'min_prob_current': min_prob_current,
            'ending_factor_start': ending_factor_start,
            'ending_factor_increment': ending_factor_increment,
            'skew_factor': skew_factor,
            'target_return': target_return,
            'random_seed': random_seed
        }
    }

def plot_multi_fund_results(results_list: List[Dict[str, Any]], title: str = "Multi-Fund Simulation Results"):
    """
    Plot results from multiple multi-fund simulation runs.
    
    Args:
        results_list: List of multi-fund simulation result dictionaries
        title: Plot title
    """
    if not results_list:
        print("No results to plot")
        return
    
    # Extract portfolio-level data
    portfolio_final_capitals = [r['portfolio_final_capital'] for r in results_list]
    portfolio_returns = [r['portfolio_total_return'] for r in results_list]
    surviving_funds = [r['surviving_funds'] for r in results_list]
    survivorship_rates = [r['survivorship_rate'] for r in results_list]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Portfolio final capital distribution
    axes[0, 0].hist(portfolio_final_capitals, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 0].axvline(np.mean(portfolio_final_capitals), color='red', linestyle='--', 
                       label=f'Mean: ${np.mean(portfolio_final_capitals):,.0f}')
    axes[0, 0].axvline(np.median(portfolio_final_capitals), color='green', linestyle='--',
                       label=f'Median: ${np.median(portfolio_final_capitals):,.0f}')
    axes[0, 0].set_xlabel('Portfolio Final Capital ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Portfolio Final Capital Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Portfolio return distribution
    return_pct = [r * 100 for r in portfolio_returns]
    axes[0, 1].hist(return_pct, bins=30, alpha=0.7, edgecolor='black', color='green')
    axes[0, 1].axvline(np.mean(return_pct), color='red', linestyle='--',
                       label=f'Mean: {np.mean(return_pct):.1f}%')
    axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.5, label='Break-even')
    axes[0, 1].set_xlabel('Portfolio Total Return (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Portfolio Return Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Number of surviving funds distribution
    n_funds = results_list[0]['n_funds']
    axes[1, 0].hist(surviving_funds, bins=range(n_funds + 2), alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].axvline(np.mean(surviving_funds), color='red', linestyle='--',
                       label=f'Mean: {np.mean(surviving_funds):.1f}')
    axes[1, 0].set_xlabel('Number of Surviving Funds')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Surviving Funds Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(range(n_funds + 1))
    
    # Summary statistics
    axes[1, 1].axis('off')
    
    # Calculate additional stats
    total_bust_rate = sum(1 for r in results_list if r['surviving_funds'] == 0) / len(results_list)
    avg_survivorship = np.mean(survivorship_rates)
    
    stats_text = f"""
    Multi-Fund Summary Statistics:
    
    Total Simulations: {len(results_list):,}
    Funds per Portfolio: {n_funds}
    Total Bust Rate: {total_bust_rate:.1%}
    
    Portfolio Capital:
    Mean: ${np.mean(portfolio_final_capitals):,.0f}
    Median: ${np.median(portfolio_final_capitals):,.0f}
    Min: ${np.min(portfolio_final_capitals):,.0f}
    Max: ${np.max(portfolio_final_capitals):,.0f}
    
    Portfolio Return:
    Mean: {np.mean(portfolio_returns):.1%}
    Median: {np.median(portfolio_returns):.1%}
    
    Fund Survivorship:
    Avg Surviving: {np.mean(surviving_funds):.1f} / {n_funds}
    Avg Survivorship: {avg_survivorship:.1%}
    """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def print_multi_fund_summary(results_list: List[Dict[str, Any]]):
    """
    Print detailed summary statistics for multi-fund simulation results.
    
    Args:
        results_list: List of multi-fund simulation result dictionaries
    """
    if not results_list:
        print("No results to summarize")
        return
    
    # Extract data
    n_funds = results_list[0]['n_funds']
    portfolio_capitals = np.array([r['portfolio_final_capital'] for r in results_list])
    portfolio_returns = np.array([r['portfolio_total_return'] for r in results_list])
    surviving_funds = np.array([r['surviving_funds'] for r in results_list])
    survivorship_rates = np.array([r['survivorship_rate'] for r in results_list])
    
    # Calculate portfolio-level statistics
    total_bust_count = sum(1 for r in results_list if r['surviving_funds'] == 0)
    total_bust_rate = total_bust_count / len(results_list)
    
    positive_return_count = sum(1 for r in portfolio_returns if r > 0)
    positive_return_rate = positive_return_count / len(results_list)
    
    # Check if target return was used
    target_return = results_list[0]['parameters'].get('target_return', None)
    
    print("=" * 80)
    print("MULTI-FUND SIMULATION SUMMARY")
    print("=" * 80)
    print(f"Total Simulations: {len(results_list):,}")
    print(f"Funds per Portfolio: {n_funds}")
    print(f"Starting Capital per Fund: ${results_list[0]['parameters']['capital_per_fund']:,.0f}")
    print(f"Total Starting Capital: ${results_list[0]['parameters']['starting_capital']:,.0f}")
    
    print(f"\nPORTFOLIO SURVIVORSHIP:")
    print(f"Total Portfolio Bust Rate: {total_bust_rate:.1%} ({total_bust_count:,} portfolios)")
    print(f"Average Surviving Funds: {surviving_funds.mean():.1f} / {n_funds}")
    print(f"Average Survivorship Rate: {survivorship_rates.mean():.1%}")
    print(f"Portfolios with All Funds Surviving: {sum(1 for s in surviving_funds if s == n_funds)} ({sum(1 for s in surviving_funds if s == n_funds)/len(results_list):.1%})")
    
    if target_return is not None:
        target_achieved_portfolios = sum(1 for r in results_list if r['funds_reached_target'] > 0)
        avg_funds_reaching_target = np.mean([r['funds_reached_target'] for r in results_list])
        print(f"\nTARGET ACHIEVEMENT ({target_return:.1%}):")
        print(f"Portfolios with ≥1 Fund Reaching Target: {target_achieved_portfolios:,} ({target_achieved_portfolios/len(results_list):.1%})")
        print(f"Average Funds Reaching Target: {avg_funds_reaching_target:.1f} / {n_funds}")
    
    print(f"\nPORTFOLIO PERFORMANCE:")
    print(f"Positive Returns: {positive_return_count:,} ({positive_return_rate:.1%})")
    
    print(f"\nPORTFOLIO CAPITAL STATISTICS:")
    print(f"Mean: ${portfolio_capitals.mean():,.2f}")
    print(f"Median: ${np.median(portfolio_capitals):,.2f}")
    print(f"Std Dev: ${portfolio_capitals.std():,.2f}")
    print(f"Min: ${portfolio_capitals.min():,.2f}")
    print(f"Max: ${portfolio_capitals.max():,.2f}")
    
    print(f"\nPORTFOLIO RETURN STATISTICS:")
    print(f"Mean: {portfolio_returns.mean():.1%}")
    print(f"Median: {np.median(portfolio_returns):.1%}")
    print(f"Std Dev: {portfolio_returns.std():.1%}")
    print(f"Min: {portfolio_returns.min():.1%}")
    print(f"Max: {portfolio_returns.max():.1%}")
    
    # Compare to single fund equivalent
    print(f"\nDIVERSIFICATION ANALYSIS:")
    single_fund_equivalent = results_list[0]['parameters']['starting_capital']
    avg_portfolio_capital = portfolio_capitals.mean()
    diversification_benefit = (avg_portfolio_capital - single_fund_equivalent) / single_fund_equivalent
    print(f"Diversification Benefit: {diversification_benefit:+.1%} vs single fund baseline")
    
    # Risk metrics
    portfolio_volatility = portfolio_returns.std()
    print(f"Portfolio Return Volatility: {portfolio_volatility:.1%}")
    
    # Percentiles
    percentiles = [5, 10, 25, 75, 90, 95]
    print(f"\nPORTFOLIO RETURN PERCENTILES:")
    for p in percentiles:
        value = np.percentile(portfolio_returns, p)
        print(f"{p}th percentile: {value:.1%}")