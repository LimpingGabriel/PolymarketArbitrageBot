import pandas as pd
from obspy import read_events
from obspy.core.event import Catalog
from typing import Optional

class SeismicDataLoader:
    """
    Handles ingestion of historical earthquake data (NDK/CMT)
    using ObsPy and normalization into Quant-ready DataFrames.
    """

    @staticmethod
    def load_ndk(file_path: str) -> pd.DataFrame:
        """
        Parses a Global CMT (NDK) file and returns a standardized DataFrame.
        
        Args:
            file_path: Path to the .ndk file.

        Returns:
            pd.DataFrame: Columns ['timestamp', 'magnitude', 'lat', 'lon', 'depth']
        """
        print(f"Reading NDK file: {file_path}...")
        try:
            # ObsPy automatically detects NDK/CMT format
            catalog: Catalog = read_events(file_path)
        except Exception as e:
            raise IOError(f"Failed to read NDK file via ObsPy: {e}")

        # Vectorize the extraction (more efficient than looping python objects for massive files)
        events_data = []
        
        for event in catalog:
            try:
                # Extract primary origin and magnitude
                origin = event.preferred_origin() or event.origins[0]
                mag = event.preferred_magnitude() or event.magnitudes[0]
                
                events_data.append({
                    'timestamp': origin.time.datetime,
                    'magnitude': mag.mag,
                    'lat': origin.latitude,
                    'lon': origin.longitude,
                    'depth': origin.depth
                })
            except IndexError:
                # Skip malformed events without origin/mag
                continue

        df = pd.DataFrame(events_data)
        
        # Sort by time for time-series analysis
        df.sort_values(by='timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        print(f"✅ Loaded {len(df)} events successfully.")
        return df

    @staticmethod
    def filter_catalog(df: pd.DataFrame, 
                       min_mag: float = 0.0, 
                       start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Applies standard filters to the seismic dataframe.
        """
        mask = (df['magnitude'] >= min_mag)
        
        if start_date:
            mask &= (df['timestamp'] >= pd.to_datetime(start_date))
            
        return df.loc[mask].copy()