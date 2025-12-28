# Safe Choices
## _An effort to assess the safety of "safe choices" on prediction markets, and by consequence the efficiency of said markets_

Let's conduct an experiment. 50 hedge funds swindle investors into inundating their coffers with $100 in deployable assets each, operating something they call the _State Earthquake Trade_. Each chooses a different state and in our fictional setting, the probability that each state is hit by an earthquake over the course of a year is 10%. The hedgefunds deploy all of their capital every year on a contract that will return a 1.11x multiple if their associated state is not hit by an earthquake. A python simulation of the scenario above is provided below.

```python
import random

def simulate(n_funds=50, years=10, sims=100000, p_eq=0.1):
    results = []
    for _ in range(sims):
        alive = [True] * n_funds
        for _ in range(years):
            for i in range(n_funds):
                if alive[i] and random.random() < p_eq:
                    alive[i] = False
        results.append(sum(alive))
    return results

tally = simulate()
print(sum(tally) / len(tally), min(tally), max(tally))
```
Over 100,000 simulations of 50 funds, the average, minimum, and maximum number of surviving hedge funds after 10 years were
```
17.42233 4 32
```
These 17 hedge funds will have produced 11% returns year on year for 10 year, an excellent outcome by all standards. In this fictional realm where market probabilities are strictly equivalant to the odds they trade at (i.e. a world where the strong form of Fama's _Efficient Market Hypothesis_ holds true), luck becomes the differentiating factor between visionaries of the investing world and going bust. 

We aim to explore the efficiency of "safe choices" (which we define as markets trading at a price of >= 90 cents 24 hours before resolution) and "very safe choices" (markets trading at  aprice of >= 95 cents at 24 hours before resolution). How often will traders making random walks across these markets end up with nothing? How often will they beat the market? If they were to stop at a certain return treshold (i.e. 12%) how often will they make it there? Were they to deploy their capital amongst several smaller funds, would they then have a higher survivorship rate?

These questioins drive us to our overarching hypothesis: inefficiencies in "safe" markets provide opportunities for positive expected value trades and ROI greater than investing in traditional benchmarks such as the risk-free rate.

---

## Methodology

### Dataset Collection

We area able to collect the necessary infromation through a combination of usage of the Polymarket Gamma API, which allows us to filter markets by end date, and the CLOB API, which lets us fetch price history for a given market via its CLOB token id.

An example of the Gamma API request is included here, where we search for markets that have been alive for greater than a week, closed on January 1st 2025, and have a minimum of $100,000 in trading volume
```
curl --request GET \
  --url 'https://gamma-api.polymarket.com/markets?ascending=true&closed=true&end_date_min=2025-01-01T00%3A00%3A00Z&end_date_max=2025-01-02T00%3A00%3A00Z&volume_num_min=500000&start_date_max=2024-12-30T00%3A00%3A00Z&limit=100&offset=0&order=volume'
```
From here, we then filter out any markets with an inherent level of variability to them, such as sports, crypto, or weather markets. 

The exact list of filter words we use is listed below
```
sports_keywords = ['vs.', ' vs ', 'NFL', 'NBA', 'MLB', 'NHL', 'UFC', 'MMA',
                    'Soccer', 'Football', 'Basketball', 'Baseball', 'Hockey',
                    'Tennis', 'Golf', 'Boxing', 'Premier League', 'Champions League',
                    'ucl-', 'mlb-', 'nba-', 'nfl-', 'nhl-', 'wnba-', 'Serie A',
                    'La Liga', 'Bundesliga', 'Ligue 1', 'UEFA', 'FIFA', 'World Cup',
                    'Cavaliers', 'Lakers', 'Warriors', 'Celtics', 'Knicks', 'Nets',
                    'Yankees', 'Dodgers', 'Astros', 'Heisman', 'Davey O\'Brien',
                    'Doak Walker', 'Biletnikoff', 'Award Winner', 'cfb-', 'ncaa',
                    'Bowl Game', 'Championship Game', 'playoffs', 'tournament']

esports_keywords = ['LoL:', 'Dota', 'CS:GO', 'Valorant', 'Overwatch', 'Rocket League',
                    'Fortnite', 'PUBG', 'Apex Legends', 'Rainbow Six', 'Call of Duty',
                    'esports', 'e-sports', '(BO3)', '(BO5)', 'Gen.G', 'T1', 'TSM',
                    'Team Liquid', 'Cloud9', 'FaZe', 'NaVi', 'Fnatic', 'G2',
                    'Mobile Legends', 'MLBB', 'Honor of Kings', 'Arena of Valor',
                    'League of Legends:', 'StarCraft', 'Hearthstone', 'Overwatch League']

crypto_keywords = ['bitcoin', 'ethereum', 'btc', 'eth', 'solana', 'sol', 'xrp',
                    'crypto', 'coin', 'token', 'above', 'below', 'hit',
                    'multistrike', '4pm et', '8pm et', '12pm et', 'trading',
                    'market cap', 'defi', 'nft', 'blockchain', '3:00pm', '3:15pm',
                    '3:30pm', '3:45pm', 'price -', 'above ___', 'below ___',
                    'price on october', 'price on november', 'price on december',
                    'price on january', 'price on february', 'price on march',
                    'what price will', 'binance', 'coinbase', 'doge', 'shib',
                    'cardano', 'ada', 'bnb', 'polygon', 'matic', 'avalanche',
                    'avax', 'polkadot', 'dot', 'chainlink', 'link']

weather_keywords = ['temperature', 'degrees', 'rain', 'snow', 'weather', 'storm',
                    'hurricane', 'tornado', 'hotter', 'colder', 'warmest', 'coldest',
                    'precipitation', 'humidity', 'forecast', 'climate']
``` 

For the remaining markets, we log their price history every 24 hours from 7 days before their end date to 1 day before their end date (for experiments hinged on time horizon). An example of the CLOB API request is included here
```
curl --request GET \
  --url 'https://clob.polymarket.com/prices-history?fidelity=1440&market=41248677391516436501520443748383894699563681344034127905029783553952611928088&startTs=1735130728&endTs=1735706728'
```
The surviving markets are then committed to our dataset. This procedure is repeated for every day in the years of 2024 and 2025. 

Our full data collection Python Notebook can be seen in the dataCollection folder of the Github for this project, along with examples of the responses from the API requests shown above. Our dataset has been opersourced and uploaded to Kaggle at [https://www.kaggle.com/datasets/dhruvgup/polymarket-closed-2025-markets-7-day-price-history/data](https://www.kaggle.com/datasets/dhruvgup/polymarket-closed-2025-markets-7-day-price-history/data). 

### Dataset Exploration

With the dataset in hand, we can try and address the questions mentioned earlier and explore some statistics.

**Key Questions Addressed:**

1. Probability Calibration: What percentage of markets at different probability thresholds (90-95%, 95-98%, 98-99%, 99-100%) actually resolve to the expected outcome? This tests whether market probabilities are well-calibrated or systematically biased.

2. Confidence Intervals: For each probability band, we calculate:
   - Win rate (percentage of safe bets that won)
   - 95% confidence intervals using binomial proportion confidence intervals
   - Comparison to expected win rate based on market probability

3. Time Horizon Analysis: Comparison of outcomes based on probability at 48 hours vs. 7 days before resolution to understand how market efficiency changes as resolution approaches.

4. Distribution Analysis: 
   - Distribution of probabilities across the dataset
   - Volume-weighted vs. unweighted win rates


5. Expected Value Calculation: For each probability band, calculate the expected return assuming:
   - Investment at the market price (probability = price)
   - Binary outcome (1.0 if win, 0.0 if loss)
   - Expected return = (probability × 1.0) - (1 - probability) × 0.0 = probability

The analysis reveals whether markets are overconfident (actual win rate < expected), underconfident (actual win rate > expected), or well-calibrated.

### Simulations - Single Fund, Single Fund Threshold, Multi Fund

We implement three types of Monte Carlo simulations to explore different trading strategies and risk management approaches, starting from 2025-01-01:

#### 1. Single Fund Simulation

A single trader deploys all capital sequentially across safe markets, reinvesting all winnings. This represents the baseline "all-in" strategy.

**Parameters:**
- `starting_capital`: Initial investment ($10,000)
- `days_before`: Days before resolution to invest (1-7 days)
- `min_prob_7d`: Minimum probability threshold at 7 days before resolution (e.g., 0.90)
- `min_prob_current`: Minimum probability threshold at investment day (e.g., 0.90)
- `skew_factor`: Controls left skew in market selection (higher = prefer closer markets)
- `ending_factor`: Probability of ending simulation that increases daily (starts 5%, +0.1% per day)
- `max_duration_days`: Maximum simulation duration (365 days)

**Market Selection:**
- Markets selected using exponential decay favoring closer resolution dates
- Must meet probability thresholds at both 7 days and investment day
- Uses actual historical market probabilities and outcomes

**Key Metrics:**
- Final capital distribution and bust rate
- Return distribution and percentiles
- Number of trades per simulation
- Average time to completion

#### 2. Single Fund Threshold Simulation

Identical to Single Fund but with target return thresholds - trading stops when target is reached.

**Additional Parameters:**
- `target_return`: Target return percentage (4.14% Treasury rate or 10.56% NASDAQ average)

**Benchmark Comparisons:**
- **Treasury Rate (4.14%)**: Tests beating risk-free returns
- **NASDAQ Average (10.56%)**: Tests beating equity market returns

**Key Metrics:**
- Success rate (percentage reaching target)
- Average time to reach target
- Comparison to "never stop" baseline strategy
- Risk-adjusted performance metrics

#### 3. Multi-Fund Simulation

Capital is divided into multiple independent funds, each operating as separate single funds. Tests diversification benefits of splitting capital.

**Parameters:**
- `n_funds`: Number of independent funds (tested: 1, 3, 5, 10 funds)
- `starting_capital`: Total capital divided equally among funds
- Same market selection and probability parameters as Single Fund

**Independence Features:**
- Each fund uses different random seed
- Funds can invest in same markets simultaneously (no coordination)
- Each fund operates completely independently

**Key Metrics:**
- Portfolio-level survivorship rate vs single fund baseline
- Average number of surviving funds per portfolio
- Portfolio return volatility and risk reduction
- Diversification benefit analysis

#### Simulation Implementation

**Technical Details:**
- Uses actual Polymarket data filtered for 2025+ resolution dates
- Vectorized operations for performance (1000 simulations in ~10-20 seconds)
- Left-skewed exponential market selection using `skew_factor`
- Binary outcomes: win = 1/probability return, loss = total loss
- Tracks daily capital evolution and complete trade history

**Quality Assurance:**
- Reproducible results via random seeds
- Performance timing and optimization
- Comprehensive statistical analysis and visualization
- Export capabilities for further research

## Limitations and Ackwnoledgments

1. Volume and market movement; If we count the starting capital as $10,000 distributed over 5 funds, we may not need to worry about available liquidity. However, were we to scale this experiment up, this becomes a real concern.