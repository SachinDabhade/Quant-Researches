import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from scipy.stats import t, norm
import matplotlib.pyplot as plt



def forecast_tgarch_risk(returns):
    """Returns tomorrow's volatility and risk metrics using T-GARCH(1,1)."""
    try:
        model = arch_model(returns, mean='Zero', vol='GARCH', p=1, q=1, dist='StudentsT')
        results = model.fit(disp='off')

        forecast = results.forecast(horizon=1)
        sigma_tomorrow = np.sqrt(forecast.variance.iloc[-1, 0])
        nu = results.params['nu']
        mu = np.mean(returns)

        prob_down = t.cdf(0, df=nu, loc=mu, scale=sigma_tomorrow)
        prob_up = 1 - prob_down

        var_95 = t.ppf(0.05, df=nu, loc=mu, scale=sigma_tomorrow)
        ci_lower = t.ppf(0.025, df=nu, loc=mu, scale=sigma_tomorrow)
        ci_upper = t.ppf(0.975, df=nu, loc=mu, scale=sigma_tomorrow)

        max_return = mu + 3 * sigma_tomorrow
        min_return = mu - 3 * sigma_tomorrow

        return {
            "forecast_volatility": sigma_tomorrow,
            "prob_up": prob_up,
            "prob_down": prob_down,
            "expected_return": mu,
            "max_return": max_return,
            "min_return": min_return,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "var_95": var_95
        }

    except Exception as e:
        print(f"T-GARCH forecast failed for Stock: {e}")
        return {
            "forecast_volatility": np.nan,
            "prob_up": np.nan,
            "prob_down": np.nan,
            "expected_return": np.nan,
            "max_return": np.nan,
            "min_return": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "var_95": np.nan
        }
    


def forecast_egarch_risk(returns):
    """
    Perform EGARCH(1,1) analysis and return volatility metrics, probabilities, and position sizing.

    Parameters:
        ticker (str): Stock ticker (e.g., 'INFY.NS')
        capital (float): Total capital available (default ₹100,000)
        risk_pct (float): Risk per trade as fraction (e.g., 0.02 for 2%)
        plot (bool): Whether to display a volatility plot

    Returns:
        dict: Dictionary of forecast metrics and position sizing
    """
    try:
        # Fit EGARCH(1,1) model
        model = arch_model(returns, mean='Constant', vol='EGARCH', p=1, q=1, dist='normal')
        results = model.fit(update_freq=5, disp='off')

        forecast = results.forecast(horizon=1)
        sigma_tomorrow = np.sqrt(forecast.variance.iloc[-1, 0])
        mu = results.params.get('mu', 0)

        # Probability estimates
        prob_down = norm.cdf(0, loc=mu, scale=sigma_tomorrow)
        prob_up = 1 - prob_down

        # Return bounds and risk
        max_return = mu + 3 * sigma_tomorrow
        min_return = mu - 3 * sigma_tomorrow
        ci_lower = norm.ppf(0.025, loc=mu, scale=sigma_tomorrow)
        ci_upper = norm.ppf(0.975, loc=mu, scale=sigma_tomorrow)
        var_95 = norm.ppf(0.05, loc=mu, scale=sigma_tomorrow)

        # Results dictionary
        return {
            "expected_return": mu,
            "forecast_volatility": sigma_tomorrow,
            "prob_up": prob_up,
            "prob_down": prob_down,
            "max_return": max_return,
            "min_return": min_return,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "var_95": var_95,
        }
    except Exception as e:
        print(f"E-GARCH forecast failed for Stock: {e}")
        return {
            "forecast_volatility": np.nan,
            "prob_up": np.nan,
            "prob_down": np.nan,
            "expected_return": np.nan,
            "max_return": np.nan,
            "min_return": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "var_95": np.nan
        }