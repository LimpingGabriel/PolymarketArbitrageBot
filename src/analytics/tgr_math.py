import numpy as np
import pandas as pd
from scipy.special import gammainc, gamma

# Constants from Kagan & Jackson (Standard Seismology)
C_MOMENT = 9.0  # Constant for Nm conversion
BETA_DEFAULT = 0.678
CORNER_MAG_DEFAULT = 9.0

class TGRPhysics:
    """
    Implements the Physics/Math of the Tapered Gutenberg-Richter (TGR) distribution.
    References: Kagan & Jackson (2000), Bird & Kagan (2004).
    """

    @staticmethod
    def magnitude_to_moment(mag: float | np.ndarray) -> float | np.ndarray:
        """
        Converts Moment Magnitude (Mw) to Scalar Seismic Moment (M0) in Newton-meters.
        Formula: log10(M0) = 1.5 * Mw + 9.0
        """
        return 10.0 ** (1.5 * np.array(mag) + C_MOMENT)

    @staticmethod
    def moment_to_magnitude(moment: float | np.ndarray) -> float | np.ndarray:
        """Inverse of magnitude_to_moment."""
        return (np.log10(moment) - C_MOMENT) / 1.5

    @staticmethod
    def survivor_function(moment: float, 
                          moment_threshold: float, 
                          beta: float = BETA_DEFAULT, 
                          corner_moment: float = None) -> float:
        """
        The TGR Survivor Function G(M).
        G(M) = (Mt / M)^beta * exp((Mt - M) / Mcm)
        
        Args:
            moment: The moment M to evaluate.
            moment_threshold: The threshold moment Mt (lower bound).
            beta: The spectral slope.
            corner_moment: The corner moment Mcm (roll-off).
        """
        if corner_moment is None:
            # Default to M=9.0 equivalent if not provided
            corner_moment = TGRPhysics.magnitude_to_moment(CORNER_MAG_DEFAULT)

        term1 = (moment_threshold / moment) ** beta
        term2 = np.exp((moment_threshold - moment) / corner_moment)
        return term1 * term2

class SeismicityRate:
    """
    Handles the calculation of Earthquake Rates (Alpha).
    """
    
    @staticmethod
    def calculate_alpha0(df_catalog: pd.DataFrame, 
                         mag_threshold: float, 
                         time_window_years: float) -> float:
        """
        Calculates the observed rate of events >= mag_threshold per year.
        
        Alpha = N_events / Time_Window
        """
        count = len(df_catalog[df_catalog['magnitude'] >= mag_threshold])
        if time_window_years <= 0:
            return 0.0
        return count / time_window_years