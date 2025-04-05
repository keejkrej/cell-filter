import argparse
import json
from pathlib import Path
from typing import List, Tuple

from cell_counter.utils.parse_labels import parse_labels_from_file

def display_labels(pairs: List[Tuple[str, str]]) -> None:
    """
    Display image-label pairs in a formatted table.
    
    Args:
        pairs (List[Tuple[str, str]]): List of (filename, label) tuples
    """
    # Find the maximum filename length for formatting
    max_filename_len = max(len(filename) for filename, _ in pairs)
    
    # Print header
    print(f"{'Filename':<{max_filename_len}} | Label")
    print("-" * (max_filename_len + 3 + 5))  # 3 for " | ", 5 for "Label"
    
    # Print each pair
    for filename, label in pairs:
        print(f"{filename:<{max_filename_len}} | {label}")

def main():
    parser = argparse.ArgumentParser(description="Display image-label pairs from a JSON file")
    parser.add_argument("json_file", type=str, help="Path to the JSON file containing annotations")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.json_file).exists():
        print(f"Error: File '{args.json_file}' does not exist")
        return
    
    try:
        # Parse the labels
        pairs = parse_labels_from_file(args.json_file)
        
        if not pairs:
            print("No image-label pairs found in the JSON file")
            return
            
        # Display the results
        display_labels(pairs)
        
    except json.JSONDecodeError:
        print(f"Error: '{args.json_file}' is not a valid JSON file")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 