#!/usr/bin/env python3
"""
Utilities for reading and filtering earthquake data from text files.
Replacement for NDK parsing - reads simple date/magnitude text format.
"""

import numpy as np
from datetime import datetime


def parse_text_file(filename):
    """
    Reads a text file with date and magnitude and returns a list of 
    dictionaries with 'date' and 'magnitude' keys.
    
    Expected format: 
        YYYY/MM/DD magnitude
    or
        Date    Magnitude  (with header)
        YYYY-MM-DD HH:MM:SS    magnitude
    """
    events = []
    
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and header lines
                if not line or line.startswith('Date') or line.startswith('#'):
                    continue
                
                try:
                    # Split by whitespace or tab
                    parts = line.split()
                    
                    if len(parts) < 2:
                        continue
                    
                    # Parse date - handle both YYYY/MM/DD and YYYY-MM-DD HH:MM:SS
                    date_str = parts[0]
                    if len(parts) >= 3 and ':' in parts[1]:
                        # Format: YYYY-MM-DD HH:MM:SS
                        datetime_str = f"{parts[0]} {parts[1]}"
                        event_date = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                        magnitude = float(parts[2])
                    else:
                        # Format: YYYY/MM/DD or YYYY-MM-DD
                        if '/' in date_str:
                            event_date = datetime.strptime(date_str, '%Y/%m/%d')
                        else:
                            event_date = datetime.strptime(date_str, '%Y-%m-%d')
                        magnitude = float(parts[1])
                    
                    events.append({
                        'date': event_date,
                        'magnitude': magnitude
                    })
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line {line_num}: {line}")
                    continue

    except FileNotFoundError:
        print(f"ERROR: File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"ERROR: Failed to read the file '{filename}'.")
        print(f"Details: {e}")
        return []
        
    return events


def filter_earthquakes(events, min_magnitude, min_year):
    """
    Filters a list of earthquake event dictionaries based on magnitude and year.
    """
    min_date = datetime(min_year, 1, 1)
    
    filtered_events = [
        event for event in events
        if event['magnitude'] > min_magnitude and event['date'] >= min_date
    ]
    
    return filtered_events


def load_to_numpy(events):
    """
    Loads filtered event data (date and magnitude) into a 
    structured NumPy array for manipulation.
    """
    if not events:
        # Define the structure for an empty array
        dtype = np.dtype([
            ('date', 'datetime64[s]'),
            ('magnitude', 'f4')
        ])
        return np.empty(0, dtype=dtype)
        
    # Define the data type for the structured array
    dtype = np.dtype([
        ('date', 'datetime64[s]'),  # A 64-bit datetime
        ('magnitude', 'f4')         # A 32-bit float
    ])
    
    # Prepare the data for the array
    data = [(event['date'], event['magnitude']) for event in events]
        
    # Create and return the structured NumPy array
    return np.array(data, dtype=dtype)


def count_events_in_range(numpy_array, lower_mag, upper_mag):
    """
    Calculates the total number of earthquakes in the structured NumPy array 
    that fall between a specified lower_mag (exclusive) and upper_mag (inclusive).

    Args:
        numpy_array (np.ndarray): The structured array containing 'magnitude'.
        lower_mag (float): The lower bound magnitude (exclusive, i.e., > lower_mag).
        upper_mag (float): The upper bound magnitude (inclusive, i.e., <= upper_mag).

    Returns:
        int: The total count of earthquakes in the specified range.
    """
    if numpy_array.size == 0:
        return 0
        
    # Extract the 'magnitude' column
    magnitudes = numpy_array['magnitude']
    
    # Create boolean masks and combine them
    mask = (magnitudes > lower_mag) & (magnitudes <= upper_mag)
    
    return int(np.sum(mask))


def write_text_file(events, output_filename):
    """
    Writes filtered earthquakes to a simple text file with date and magnitude.
    
    Format: Each line contains date (YYYY-MM-DD HH:MM:SS) and magnitude separated by tab.
    """
    try:
        with open(output_filename, 'w') as f:
            # Write header
            f.write("Date\tMagnitude\n")
            
            # Write each event
            for event in events:
                date_str = event['date'].strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{date_str}\t{event['magnitude']:.2f}\n")
        
        print(f"\nSuccessfully wrote {len(events)} filtered events to '{output_filename}'.")
    except Exception as e:
        print(f"Error writing to file '{output_filename}': {e}")


def filter_text_file_script(input_filename, output_filename, min_mag, min_year):
    """
    Main function to orchestrate the text file filtering process.
    
    Args:
        input_filename: Path to input text file with date and magnitude
        output_filename: Path to output text file
        min_mag: Minimum magnitude threshold (exclusive)
        min_year: Minimum year threshold
        
    Returns:
        Structured NumPy array with filtered events
    """
    print(f"--- 🌎 Starting Earthquake Filter Script ---")
    print(f"Filtering '{input_filename}' for events with Magnitude > {min_mag} since {min_year}...")
    
    # Parse the file
    events = parse_text_file(input_filename)
    if not events:
        print("Script finished with no events parsed.")
        return None

    print(f"Parsed {len(events)} total events from the file.")
    
    # Filter the events
    filtered_events = filter_earthquakes(events, min_mag, min_year)
    print(f"Found {len(filtered_events)} events matching the criteria.")
    
    # Load the filtered data into a NumPy structure
    numpy_array = load_to_numpy(filtered_events)
    print(f"Loaded {len(numpy_array)} events into a structured NumPy array.")
    
    # Write the filtered events to a text file
    write_text_file(filtered_events, output_filename)
    
    # Print some statistics
    if len(numpy_array) > 0:
        print(f"\n--- Statistics ---")
        print(f"Magnitude range: {numpy_array['magnitude'].min():.2f} to {numpy_array['magnitude'].max():.2f}")
        print(f"Date range: {numpy_array['date'].min()} to {numpy_array['date'].max()}")
        print(f"Average magnitude: {numpy_array['magnitude'].mean():.2f}")
    
    print("\n--- ✅ Script Execution Complete ---")
    
    return numpy_array


# Example Usage
if __name__ == "__main__":
    INPUT_FILE = 'earthquake_data.txt'  # Output from your NDK parser
    OUTPUT_FILE = 'filtered_earthquakes.txt'
    MIN_MAGNITUDE = 5.5
    MIN_YEAR = 1980

    filtered_data_np = filter_text_file_script(INPUT_FILE, OUTPUT_FILE, MIN_MAGNITUDE, MIN_YEAR)
    
    # Example: Count events in a magnitude range
    if filtered_data_np is not None and len(filtered_data_np) > 0:
        count_6_to_7 = count_events_in_range(filtered_data_np, 6.0, 7.0)
        print(f"\nEvents between M6.0 and M7.0: {count_6_to_7}")