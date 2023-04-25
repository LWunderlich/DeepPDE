import numpy as np


class transform:
    def __init__(self, t_min, t_max, strike_price, volatility_min, volatility_max, normalise_min, normalise_max, r_min,
                 r_max, rho_min, rho_max):
        self.normalise_min = normalise_min
        self.normalise_max = normalise_max
        self.correlation_min = rho_min
        self.correlation_max = rho_max
        self.riskfree_rate_min = r_min
        self.riskfree_rate_max = r_max
        self.strike_price = strike_price
        self.volatility_min = volatility_min
        self.volatility_max = volatility_max
        self.t_min = t_min
        self.t_max = t_max

        self.s_max = self.strike_price * (1 + 3 * self.volatility_max * t_max)
        self.x_max = np.log(self.s_max)
        self.x_min = 2 * np.log(self.strike_price) - self.x_max

    def transform_ab_to_cd(self, x, a, b, c, d):
        """
         Perform a linear transformation of a scalar from the souce interval
         to the target interval.

         Keyword arguments:
         x -- scalar point(s) to transform
         a, b -- interval to transform from
         c, d -- interval to transform to 
            """
        return c + (x - a) * (d - c) / (b - a)

    def transform_to_logprice(self, x, normalised_min, normalised_max):
        """ Transform normalised variable to the log-price. """

        return self.transform_ab_to_cd(x, normalised_min, normalised_max, self.x_min, self.x_max)

    def transform_to_time(self, t):
        """ Transform normalised variable to the time variable. """
        return self.transform_ab_to_cd(t, self.normalise_min, self.normalise_max, self.t_min, self.t_max)

    def normalise_logprice(self, x):
        """ Transform log-price to its corresponding normalised variable. """
        return self.transform_ab_to_cd(x, self.x_min, self.x_max, self.normalise_min, self.normalise_max)

    def normalise_time(self, t):
        """ Transform time to its corresponding normalised variable. """
        return self.transform_ab_to_cd(t, self.t_min, self.t_max, self.normalise_min, self.normalise_max)

    def transform_to_riskfree_rate(self, mu_1):
        """ Transform normalised variable to the risk-free rate. """
        return self.transform_ab_to_cd(mu_1, self.normalise_min, self.normalise_max, self.riskfree_rate_max,
                                       self.riskfree_rate_max)

    def transform_to_volatility(self, mu_2):
        """ Transform normalised variable to the volatility. """
        return self.transform_ab_to_cd(mu_2, self.normalise_min, self.normalise_max, self.volatility_min,
                                       self.volatility_max)

    def transform_to_correlation(self, mu_3):
        """ Transform normalised variable to the correlation. """
        return self.transform_ab_to_cd(mu_3, self.normalise_min, self.normalise_max, self.correlation_min,
                                       self.correlation_max)

    def normalise_riskfree_rate(self, riskfree_rate: float) -> float:
        """ Transform risk-free rate to its corresponding normalised variable. """
        return self.transform_ab_to_cd(riskfree_rate, self.riskfree_rate_min, self.riskfree_rate_max,
                                       self.normalise_min, self.normalise_max)

    def normalise_volatility(self, volatility):
        """ Transform volatility to its corresponding normalised variable. """
        return self.transform_ab_to_cd(volatility, self.volatility_min, self.volatility_max, self.normalise_min,
                                       self.normalise_max)

    def normalise_correlation(self, correlation):
        """ Transform correlation to its corresponding normalised variable. """
        return self.transform_ab_to_cd( correlation, self.correlation_min, self.correlation_max,
                                       self.normalise_min, self.normalise_max)
