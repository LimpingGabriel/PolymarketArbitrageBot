import time
import numpy as np
import math
import datetime
from scipy.special import gammainc, gamma, betainc, gammaln
from scipy.special import comb as scipy_comb

from utilities import load_to_numpy, parse_text_file


def magnitude_to_moment(mag):
    """Convert Mw to scalar seismic moment (N·m).
       M = 10^(1.5*m + C) where C=9.0 for M in N·m
       Reference: Eq. (1) in Kagan & Jackson papers
    """
    mag = np.asarray(mag, dtype=float)
    return 10.0 ** (1.5 * mag + C_MOMENT)

def moment_to_magnitude(M):
    """Inverse: convert moment (N·m) to magnitude."""
    M = np.asarray(M, dtype=float)
    return (np.log10(M) - C_MOMENT) / 1.5

# ---------- Tapered Gutenberg–Richter (TGR) survivor ----------
def tgr_survivor(mag, mt=5.6, beta=0.678, mcm=9.0):
    """
    TGR survivor function G(M; Mt, beta, Mcm) from Eq. (2):
    G(M, Mt, beta, Mcm) = (Mt/M)^beta * exp[(Mt - M)/Mcm]
    
    This gives the fraction of earthquakes with moment >= M
    relative to the threshold Mt.
    
    Parameters:
      mag : magnitude(s) at which to evaluate
      mt  : threshold magnitude (default 5.6 for GCMT)
      beta: slope parameter (b-value = 1.5*beta)
      mcm : corner magnitude for TGR distribution
    
    Returns:
      G : survivor fraction (should be <= 1 for mag >= mt)
    """
    mag = np.asarray(mag, dtype=float)
    
    # Convert to moments
    M = magnitude_to_moment(mag)
    Mt = magnitude_to_moment(mt)
    Mcm = magnitude_to_moment(mcm)
    
    # Eq. (2): G = (Mt/M)^beta * exp[(Mt - M)/Mcm]
    G = (Mt / M) ** beta * np.exp((Mt - M) / Mcm)
    
    return G

# ---------- Compute alpha0 from catalog ----------
def compute_alpha0_from_catalog(events, mt, fixed_start_year=1980):
    """
    Compute alpha0 = annual rate of events with Mw >= mt
    
    alpha0 = number_of_events / time_span_years
    """
    if events.size == 0:
        return 0.0
    
    # Count events >= mt
    n_events = np.sum(events['magnitude'] >= mt)
    
    if n_events == 0:
        return 0.0
    
    # Calculate time span
    start_date_np = np.datetime64(f'{fixed_start_year}-01-01T00:00:00')
    max_date_np = events['date'].max()
    duration_delta_np = max_date_np - start_date_np
    
    # Convert to years
    total_seconds = duration_delta_np / np.timedelta64(1, 's')
    SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
    years = total_seconds / SECONDS_PER_YEAR
    
    if years <= 0:
        return 0.0
    
    alpha0 = n_events / years
    return alpha0

# ---------- Expected rate ----------
def expected_annual_rate_tgr(mag, alpha0, mt=5.6, beta=0.678, mcm=9.0):
    """
    Expected annual number of earthquakes with Mw >= mag.
    
    N(Mw >= mag) = alpha0 * G(mag; mt, beta, mcm)
    
    where alpha0 is the annual rate at threshold mt.
    """
    G = tgr_survivor(mag, mt=mt, beta=beta, mcm=mcm)
    return alpha0 * G

# ---------- Poisson probability ----------
def poisson_prob_at_least_one(lam):
    """P(K >= 1) = 1 - P(K=0) = 1 - exp(-lambda)"""
    lam = np.maximum(lam, 0)  # Ensure non-negative
    return 1.0 - np.exp(-lam)

def poisson_pmf(k, lam):
    """Poisson PMF: P(K=k) = exp(-λ) * λ^k / k!
       Using log-space for numerical stability.
    """
    lam = np.asarray(lam, dtype=float)
    lam = np.maximum(lam, 1e-10)  # Avoid log(0)
    # log P(k) = -lam + k*log(lam) - log(k!)
    log_prob = -lam + k * np.log(lam) - gammaln(k + 1)
    return np.exp(log_prob)

# ---------- Negative Binomial Distribution ----------
def nbd_pmf(k, theta, tau):
    """
    Negative-binomial PMF from Eq. (8):
    f(k) = comb(tau + k - 1, k) * theta^tau * (1-theta)^k
    
    Parameters from Table 4 for global PDE m>=5.0:
      theta = 0.0145, tau = 18.87
    
    This parameterization has:
      mean = tau*(1-theta)/theta
      variance = tau*(1-theta)/theta^2
    """
    # Use log-space for numerical stability
    theta = np.clip(theta, 1e-10, 1 - 1e-10)  # Avoid edge cases
    log_comb = gammaln(tau + k) - gammaln(k + 1) - gammaln(tau)
    log_prob = log_comb + tau * np.log(theta) + k * np.log(1 - theta)
    return np.exp(log_prob)

def nbd_prob_at_least_one(theta, tau):
    """P(K >= 1) = 1 - P(K=0) for NBD"""
    return 1.0 - nbd_pmf(0, theta, tau)

# ---------- NBD parameter estimation ----------
def estimate_nbd_parameters_from_catalog(events, mt, time_interval_years=1.0, 
                                        fixed_start_year=1980):
    """
    Estimate NBD parameters (theta, tau) for a given magnitude threshold.
    
    This follows the methodology from Table 4 in the papers, where they
    compute annual earthquake counts and fit NBD parameters.
    
    Parameters:
      events : structured numpy array with 'magnitude' and 'date' fields
      mt : magnitude threshold
      time_interval_years : interval size for counting (default 1.0 year)
      fixed_start_year : start year for analysis
    
    Returns:
      dict with keys: 'theta', 'tau', 'lambda', 'n_intervals', 'counts', 'success'
      Returns None values if estimation fails
      
    Method:
      1. Divide catalog into equal time intervals
      2. Count earthquakes >= mt in each interval
      3. Compute sample mean and variance
      4. Use method of moments: mean = tau*(1-theta)/theta
                                 var = tau*(1-theta)/theta^2
         Solving: theta = mean/var, tau = mean^2/(var - mean)
    """
    result = {
        'theta': None,
        'tau': None,
        'lambda': 0.0,
        'n_intervals': 0,
        'counts': None,
        'success': False
    }
    
    if events.size == 0:
        return result
    
    # Filter events >= mt
    mask = events['magnitude'] >= mt
    filtered_events = events[mask]
    
    if filtered_events.size == 0:
        return result
    
    # Determine time span
    start_date = np.datetime64(f'{fixed_start_year}-01-01T00:00:00')
    
    # Get latest date in filtered events
    if filtered_events.size > 0:
        end_date = filtered_events['date'].max()
    else:
        return result
    
    # Convert to days
    total_days = (end_date - start_date) / np.timedelta64(1, 'D')
    interval_days = time_interval_years * 365.25
    
    # Number of complete intervals
    n_intervals = int(np.floor(total_days / interval_days))
    
    if n_intervals < 2:
        result['n_intervals'] = n_intervals
        return result
    
    # Count events in each interval
    counts = []
    for i in range(n_intervals):
        interval_start = start_date + np.timedelta64(int(i * interval_days), 'D')
        interval_end = start_date + np.timedelta64(int((i + 1) * interval_days), 'D')
        
        interval_mask = (filtered_events['date'] >= interval_start) & \
                       (filtered_events['date'] < interval_end)
        counts.append(np.sum(interval_mask))
    
    counts = np.array(counts)
    result['counts'] = counts
    result['n_intervals'] = n_intervals
    
    # Compute sample statistics
    lambda_est = np.mean(counts)
    result['lambda'] = lambda_est
    
    if lambda_est <= 0:
        return result
    
    variance = np.var(counts, ddof=1)  # unbiased estimator
    
    # Method of moments estimation
    # From NBD properties: mean = tau*(1-theta)/theta, var = tau*(1-theta)/theta^2
    # Therefore: var/mean = 1/theta, so theta = mean/var
    # And: tau = mean^2 / (var - mean)
    
    if variance <= lambda_est or variance <= 0:
        # Under-dispersed or Poisson-like: return theta close to 1
        # This happens for large magnitude thresholds
        theta = 0.99
        tau = lambda_est
    else:
        theta = lambda_est / variance
        tau = (lambda_est ** 2) / (variance - lambda_est)
        
        # Constrain theta to valid range
        theta = np.clip(theta, 1e-6, 0.9999)
        tau = max(tau, 0.1)
    
    result['theta'] = theta
    result['tau'] = tau
    result['success'] = True
    
    return result

def get_nbd_parameters_for_magnitude(mt, method='interpolate'):
    """
    Get NBD parameters for a given magnitude threshold using values from Table 4.
    
    Parameters:
      mt : magnitude threshold (between 5.0 and 7.0)
      method : 'interpolate' (log-linear interpolation) or 'lookup' (nearest value)
    
    Returns:
      theta, tau : NBD parameters
      
    Reference: Table 4, Global ('G') rows for different magnitude thresholds
    
    Notes:
      - For mt >= 6.5, theta approaches 1.0 (Poisson is adequate)
      - For mt < 6.5, NBD is necessary due to clustering
      - Values are from PDE catalog 1969-2014 with annual intervals
    """
    # Table 4 values for Global catalog (row 'G')
    table_values = {
        5.0: {'theta': 0.0145, 'tau': 18.87},
        5.5: {'theta': 0.0568, 'tau': 21.81},
        6.0: {'theta': 0.1989, 'tau': 26.06},
        6.5: {'theta': 0.6088, 'tau': 55.35},
        7.0: {'theta': 0.8634, 'tau': 76.94}
    }
    
    # Exact match
    if mt in table_values:
        return table_values[mt]['theta'], table_values[mt]['tau']
    
    # Out of range
    if mt < 5.0:
        print(f"Warning: mt={mt} below tabulated range. Using mt=5.0 values.")
        return table_values[5.0]['theta'], table_values[5.0]['tau']
    
    if mt > 7.0:
        # For very large magnitudes, approach Poisson
        print(f"Warning: mt={mt} above tabulated range. Using theta≈1 (Poisson).")
        return 0.95, 100.0
    
    if method == 'lookup':
        # Find nearest magnitude
        mags = np.array(list(table_values.keys()))
        idx = np.argmin(np.abs(mags - mt))
        nearest_mag = mags[idx]
        return table_values[nearest_mag]['theta'], table_values[nearest_mag]['tau']
    
    elif method == 'interpolate':
        # Log-linear interpolation
        # Find bracketing magnitudes
        mags = sorted(table_values.keys())
        
        # Find where mt falls
        for i in range(len(mags) - 1):
            if mags[i] <= mt <= mags[i+1]:
                m1, m2 = mags[i], mags[i+1]
                theta1 = table_values[m1]['theta']
                theta2 = table_values[m2]['theta']
                tau1 = table_values[m1]['tau']
                tau2 = table_values[m2]['tau']
                
                # Linear interpolation in log(1-theta) space (since theta -> 1 exponentially)
                # and linear in tau
                weight = (mt - m1) / (m2 - m1)
                
                # Interpolate log(1-theta) for better behavior near 1
                log_one_minus_theta = (1 - weight) * np.log(1 - theta1) + \
                                     weight * np.log(1 - theta2)
                theta_interp = 1 - np.exp(log_one_minus_theta)
                
                # Linear interpolation for tau
                tau_interp = (1 - weight) * tau1 + weight * tau2
                
                return theta_interp, tau_interp
        
        # Shouldn't reach here if mt in range
        return table_values[5.5]['theta'], table_values[5.5]['tau']
    
    else:
        raise ValueError(f"Unknown method: {method}")

# ---------- High-level forecast functions ----------
def probability_at_least_one_tgr(a0, mag, start_date, end_date,
                                mt=5.6, beta=0.678, mcm=9.0,
                                use_nbd=False, nbd_theta=None, nbd_tau=None,
                                auto_nbd=True):
    """
    Probability of >= 1 earthquake with Mw >= mag in time interval.
    
    Parameters:
      a0 : alpha0, annual rate at threshold mt (events/year)
      mag : target magnitude threshold
      start_date, end_date : ISO format 'YYYY-MM-DD'
      mt, beta, mcm : TGR parameters
      use_nbd : if True, use NBD instead of Poisson
      nbd_theta, nbd_tau : NBD parameters (if None and use_nbd=True, will auto-lookup)
      auto_nbd : if True, automatically use NBD for mag < 6.5
    
    Notes:
      - For mag >= 6.5, Poisson is adequate (Table 4 shows theta -> 1)
      - For mag < 6.5, NBD is more accurate due to clustering
      - NBD parameters depend on magnitude threshold (see Table 4)
    """
    # Calculate time interval in years
    start = start_date
    end = end_date
    years = (end - start).total_seconds() / 31556952.0 
    
    # Expected annual rate at target magnitude
    N_annual = expected_annual_rate_tgr(mag, a0, mt=mt, beta=beta, mcm=mcm)
    
    # Expected number in time interval
    lam = N_annual * years
    
    # Auto-decide NBD vs Poisson
    if auto_nbd and mag < 6.5:
        use_nbd = True
    
    if not use_nbd:
        # Poisson: adequate for mag >= 6.5
        return poisson_prob_at_least_one(lam)
    else:
        # NBD: required for mag < 6.5
        if nbd_theta is None or nbd_tau is None:
            # Auto-lookup from Table 4
            nbd_theta, nbd_tau = get_nbd_parameters_for_magnitude(mag)
        
        # Scale NBD parameters for the time interval
        # The mean of NBD is tau*(1-theta)/theta
        # We need to scale tau to match expected count lam
        mean_annual = nbd_tau * (1 - nbd_theta) / nbd_theta
        if mean_annual > 0:
            tau_scaled = nbd_tau * (lam / (mean_annual * years))
        else:
            tau_scaled = nbd_tau
        
        return nbd_prob_at_least_one(nbd_theta, tau_scaled)

def probability_exact_k_tgr(a0, mag, start_date, end_date, k,
                            mt=5.6, beta=0.678, mcm=9.0,
                            use_nbd=False, nbd_theta=None, nbd_tau=None,
                            auto_nbd=True):
    """
    Probability of exactly k events in the interval.
    
    For Poisson: P(K=k) from Eq. (7)
    For NBD: P(K=k) from Eq. (8)
    """
    start = start_date
    end = end_date
    years = (end - start).total_seconds() / 31556952.0 
    
    N_annual = expected_annual_rate_tgr(mag, a0, mt=mt, beta=beta, mcm=mcm)
    lam = N_annual * years
    
    # Auto-decide NBD vs Poisson
    if auto_nbd and mag < 6.5:
        use_nbd = True
    
    if not use_nbd:
        return poisson_pmf(k, lam)
    else:
        if nbd_theta is None or nbd_tau is None:
            nbd_theta, nbd_tau = get_nbd_parameters_for_magnitude(mag)
        
        # Scale tau as above
        mean_annual = nbd_tau * (1 - nbd_theta) / nbd_theta
        if mean_annual > 0:
            tau_scaled = nbd_tau * (lam / (mean_annual * years))
        else:
            tau_scaled = nbd_tau
        
        return nbd_pmf(k, nbd_theta, tau_scaled)

def compute_kelly(true_probability, market_probability, kelly_factor=1.0):
    return ((true_probability - market_probability) / (1 - market_probability))*kelly_factor

class EarthquakeEvent():
    earthquakes = load_to_numpy(parse_text_file("filtered_earthquakes.txt"))
    def __init__(self, magnitude, start_date, end_date):
        self.magnitude = magnitude
        self.start_date = start_date
        self.end_date = end_date

        self.time_delta = self.end_date-self.start_date


class EarthquakeOccurrenceEvent(EarthquakeEvent):
    def __init__(self, magnitude, start_date, end_date, *args, **kwargs):
        super().__init__(magnitude, start_date, end_date, *args, **kwargs)

    def compute_probabilities(self):
        
        if self.magnitude >= 7.0:
            print(f"\n===Forecast For M>{self.magnitude} in [{self.start_date}, {self.end_date}]===")
            # Compute alpha0 for m >= 7.0
            # The number of annual events of this size
            a0 = compute_alpha0_from_catalog(self.earthquakes, self.magnitude)
            print(f"Alpha0 (m>={self.magnitude}): {a0:.2f} events/year")
            
            if a0 > 0:
                # Example: probability of m>=7.0 in November 2025
                self.prob = probability_at_least_one_tgr(a0, self.magnitude, self.start_date, self.end_date,
                                                    mt=self.magnitude, beta=0.678, mcm=9.0)
                print(f"P(at least one earthquake m>={self.magnitude} between {self.start_date} and {self.end_date}): {self.prob:.6f}")
        else:
            pass # TODO
        # Return dictionary of optimal values according to statistical model
        return self.prob

class EarthquakeCountEvent(EarthquakeEvent):
    def __init__(self, magnitude, max_n, start_date, end_date, *args, **kwargs):
        super().__init__(magnitude, start_date, end_date, *args, **kwargs)
        self.max_n = max_n

    def compute_probabilities(self):
        
        if self.magnitude >= 7.0:

            print(f"\n===Forecast For M>{self.magnitude} in [{self.start_date}, {self.end_date}]===")
            # Compute alpha0 for m >= 7.0
            # The number of annual events of this size
            a0 = compute_alpha0_from_catalog(self.earthquakes, self.magnitude)
            print(f"Alpha0 (m>={self.magnitude}): {a0:.2f} events/year")


            
            if a0 > 0:
                probs = []
                for k in range(self.max_n+1):
                    # Example: probability of m>=7.0 in November 2025
                    probs.append(probability_exact_k_tgr(a0, self.magnitude, self.start_date, self.end_date, k,
                                                        mt=self.magnitude, beta=0.678, mcm=9.0))
                    print(f"P(Exactly {k} earthquake(s) m>={self.magnitude} between {self.start_date} and {self.end_date}): {probs[-1]*100:.6f}%")
                print(f"P(> {k} earthquake(s) m>={self.magnitude} between {self.start_date} and {self.end_date}):       {(1-sum(probs))*100:.6f}%")
                return {k: float(probs[k]) for k in range(len(probs))} | {-1: 1 - sum(probs)}
        else:
            pass # TODO
        # Return dictionary of optimal values according to statistical model




# ---------- Physical conversions (used in the papers) ----------
C_MOMENT = 9.0  # constant for M in N·m: log10(M) = 1.5*m + C


if __name__ == "__main__":
    """
    print(f"Total earthquakes in catalog: {earthquakes.size}")
    print(f"Date range: {earthquakes['date'].min()} to {earthquakes['date'].max()}")
    print(f"Magnitude range: {earthquakes['magnitude'].min():.1f} to {earthquakes['magnitude'].max():.1f}")
    
    print("\n=== NBD Parameter Estimation ===")
    for mt in [5.0, 5.5, 6.0, 6.5, 7.0]:
        result = estimate_nbd_parameters_from_catalog(earthquakes, mt)
        theta_table, tau_table = get_nbd_parameters_for_magnitude(mt)
        
        print(f"\nm >= {mt}:")
        if result['success']:
            print(f"  Estimated from catalog: theta={result['theta']:.4f}, tau={result['tau']:.2f}, λ={result['lambda']:.2f}")
            print(f"  Number of intervals: {result['n_intervals']}")
        else:
            print(f"  Estimation failed: n_intervals={result['n_intervals']}, n_events={np.sum(earthquakes['magnitude'] >= mt)}")
        print(f"  Table 4 values:         theta={theta_table:.4f}, tau={tau_table:.2f}")
    
    print("\n=== Interpolated Values ===")
    for mt in [5.25, 5.75, 6.25, 6.75]:
        theta, tau = get_nbd_parameters_for_magnitude(mt, method='interpolate')
        print(f"m >= {mt}: theta={theta:.4f}, tau={tau:.2f}")
    
    
    print("\n=== Forecast Example ===")
    # Compute alpha0 for m >= 7.0
    a0 = compute_alpha0_from_catalog(earthquakes, 7.0)
    print(f"Alpha0 (m>=7.0): {a0:.2f} events/year")
    
    if a0 > 0:
        # Example: probability of m>=7.0 in November 2025
        prob = probability_at_least_one_tgr(a0, 7.0, "2025-11-01", "2025-11-30",
                                            mt=7.0, beta=0.678, mcm=9.0)
        print(f"P(at least one m>=7.0 in Nov 2025): {prob:.6f}")
        
        # Example with NBD for smaller magnitudes
        a0_6 = compute_alpha0_from_catalog(earthquakes, 6.0)
        if a0_6 > 0:
            prob_nbd = probability_at_least_one_tgr(a0_6, 6.0, "2025-11-01", "2025-11-30",
                                                   mt=6.0, beta=0.678, mcm=9.0,
                                                   auto_nbd=True)
            print(f"P(at least one m>=6.0 in Nov 2025, NBD): {prob_nbd:.6f}")

    """


    start_time = datetime.datetime.fromtimestamp(time.time())
    end_time = datetime.datetime(2025, 11, 30, 23, 59, 59)
    e = EarthquakeOccurrenceEvent(7.0, start_time, end_time)
    prob = e.compute_probabilities()

    kelly_fraction = compute_kelly(prob, int(input("Bid (cents): "))/100, 1)
    print(f"You should wager {kelly_fraction*100:.2f}% of your portfolio on this bet.")