import numpy as np


def calculate_position_size(current_price, capital=100000, risk_per_trade=0.01):
    """
    Calculate the number of shares to buy and total investment amount
    based on current price, total capital, and risk per trade.

    Parameters:
    - current_price_series (pd.Series): Series containing the current price (e.g., last closing price)
    - capital (float): Total capital available for investment
    - risk_per_trade (float): Fraction of capital to risk on a single trade (e.g., 0.02 for 2%)

    Returns:
    - quantity (int): Number of shares to buy
    - investment_amount (float): Total investment amount
    """

    quantity = int((risk_per_trade * capital) / current_price)
    investment_amount = quantity * current_price

    return quantity, investment_amount



def fractional_kelly(current_price, mu, sigma, r_f, b, capital=100000):
    """
    Calculate the optimal number of shares and investment amount using Fractional Kelly Criterion.

    Parameters:
    - current_price (float or array): Current price(s) of the stock(s)
    - mu (float or array): Expected return(s)
    - sigma (float or array): Volatility(ies)
    - r_f (float): Risk-free rate
    - b (float or array): Probability of winning (or edge)
    - capital (float): Total capital

    Returns:
    - quantity (array): Number of shares to buy
    - investment_amount (array): Amount to invest in each stock
    """
    mu = np.array(mu, dtype=float)
    sigma = np.array(sigma, dtype=float)
    b = np.array(b, dtype=float)
    current_price = np.array(current_price, dtype=float)

    # Basic Kelly formula adjusted conservatively
    f_star = b * (mu - r_f) / (sigma**2)
    f_fractional = f_star / (1 + f_star)
    f_fractional = np.clip(f_fractional, 0, 1)  # avoid negative or overleveraged bets

    # Investment and quantity
    investment_amount = f_fractional * capital
    quantity = np.floor(investment_amount / current_price).astype(int)

    return quantity, investment_amount


def fractional_kelly(current_price, mu, sigma, r_f, b, capital=100000):
    """
    Calculate the optimal fraction of bankroll to allocate to each stock
    using the Fractional Kelly Criterion.

    Parameters:
    mu (array): expected returns of each stock
    sigma (array): volatilities of each stock
    r_f (float): risk-free rate
    b (array): odds or probabilities of each stock going up

    Returns:
    f (array): optimal fractions of bankroll to allocate to each stock
    """
    f = (b * (mu - r_f) / (sigma**2)) / (1 + (b * (mu - r_f) / (sigma**2)))
    return (f * capital) // current_price, f * capital