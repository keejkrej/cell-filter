"""
Display image-label pairs from a JSON file.
"""

import argparse
import json
from pathlib import Path

from ..core.show_labels import display_labels, get_labels_from_file


def main():
    parser = argparse.ArgumentParser(description="Display image-label pairs from a JSON file")
    parser.add_argument("json_file", type=str, help="Path to the JSON file containing annotations")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.json_file).exists():
        print(f"Error: File '{args.json_file}' does not exist")
        return
    
    try:
        # Get the labels
        pairs = get_labels_from_file(args.json_file)
        
        # Display the results
        display_labels(pairs)
        
    except json.JSONDecodeError:
        print(f"Error: '{args.json_file}' is not a valid JSON file")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 