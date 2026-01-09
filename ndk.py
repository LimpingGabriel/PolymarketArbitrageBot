"""
Script to extract moment magnitude and date from NDK earthquake data files.
NDK format uses 5 lines per event.
Moment magnitude is calculated from scalar moment using: Mw = (2/3) * log10(M0) - 10.7
"""

import math

def parse_ndk_file(input_file, output_file):
    """
    Parse NDK file and extract date and moment magnitude.
    
    Args:
        input_file: Path to input NDK file
        output_file: Path to output text file
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        
        # NDK format has 5 lines per event
        num_events = len(lines) // 5
        
        for i in range(num_events):
            # Line 1 contains the date information
            line1 = lines[i * 5]
            
            # Extract date (format: YYYY/MM/DD)
            # Date is in columns 6-10 (year), 11-12 (month), 13-14 (day)
            year = line1[5:9]
            month = line1[10:12]
            day = line1[13:15]
            date = f"{year}/{month}/{day}"
            
            # Line 4 contains the exponent for moment values (columns 1-2)
            line4 = lines[i * 5 + 3]
            exponent = int(line4[0:2].strip())
            
            # Line 5 contains scalar moment in columns 50-56
            line5 = lines[i * 5 + 4]
            scalar_moment_str = line5[49:56].strip()
            
            try:
                # Calculate scalar moment in dyne-cm
                scalar_moment = float(scalar_moment_str) * (10 ** exponent)
                
                # Calculate moment magnitude: Mw = (2/3) * log10(M0) - 10.7
                magnitude = (2.0 / 3.0) * math.log10(scalar_moment) - 10.7
                magnitude = f"{magnitude:.1f}"
            except (ValueError, ZeroDivisionError):
                magnitude = "N/A"
            
            # Write to output file
            outfile.write(f"{date} {magnitude}\n")
    
    print(f"Processed {num_events} events")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    input_ndk = "jan76_dec20.ndk"  # Change to your NDK file path
    output_txt = "earthquake_data.txt"  # Output file name
    
    try:
        parse_ndk_file(input_ndk, output_txt)
    except FileNotFoundError:
        print(f"Error: Input file '{input_ndk}' not found")
    except Exception as e:
        print(f"Error processing file: {e}")