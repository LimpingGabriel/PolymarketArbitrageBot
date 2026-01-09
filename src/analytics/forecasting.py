import numpy as np
from src.analytics.tgr_math import TGRPhysics

class SeismicForecaster:
    def __init__(self, beta: float = 0.678, corner_magnitude: float = 9.0):
        self.beta = beta
        self.corner_moment = TGRPhysics.magnitude_to_moment(corner_magnitude)

    def probability_at_least_one(self, 
                                 alpha_threshold: float, 
                                 mag_threshold: float, 
                                 target_mag: float, 
                                 forecast_duration_years: float) -> float:
        """
        Calculates P(at least one quake >= target_mag) in the next time window.
        
        Args:
            alpha_threshold: The observed yearly rate of quakes >= mag_threshold.
            mag_threshold: The magnitude used to calculate alpha (e.g., 5.0).
            target_mag: The magnitude we are betting on (e.g., 7.0).
            forecast_duration_years: Duration of the prediction market (e.g., 30/365).
        """
        # Convert magnitudes to moments
        mt = TGRPhysics.magnitude_to_moment(mag_threshold)
        mx = TGRPhysics.magnitude_to_moment(target_mag)

        # 1. Calculate the fraction of the catalog expected to exceed target_mag
        # G(target) / G(threshold)
        survivor_prob = TGRPhysics.survivor_function(mx, mt, self.beta, self.corner_moment)
        
        # 2. Extrapolate the rate for the target magnitude
        # lambda_target = alpha_threshold * (Surv(target) / Surv(threshold))
        # Note: TGRPhysics.survivor_function assumes Surv(threshold) is roughly 1 if Mt=M
        lambda_target = alpha_threshold * survivor_prob

        # 3. Poisson Probability
        # P(N >= 1) = 1 - exp(-lambda * t)
        expected_events = lambda_target * forecast_duration_years
        probability = 1.0 - np.exp(-expected_events)
        
        return probability