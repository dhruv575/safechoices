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

1. Develop a dataset by pulling all closed markets of 2025 that are in reasonable categories (i.e. no sports, no 15 minute tweets, etc) and reference their price history at 24 hours delta from their closing time. This dataset should eventually have 4 columns: market, closingDate, outcome, probability24h
2. Do some exploratory data analysis on the markets to answer questions such as what percent of 95% probability graphs resolve to the expected outcome; use some statsitics to come up with confidence ranges for these values
3. Develop the simulations mentioned above. Create some cool graphs and stuff. Write about it
4. Create a frontend interface where anyone can divide their starting capital into as many smaller funds as they like, have whatever target returns they'd like, and run however many simulations they'd like. 

## Limitations and Ackwnoledgments

1. Volume and market movement; If we count the starting capital as $10,000 distributed over 5 funds, we may not need to worry about available liquidity. However, were we to scale this experiment up, this becomes a real concern.