"""
Core show_labels functionality for cell-counter.
"""

from typing import List, Tuple

from ..utils.parse_labels import parse_labels_from_file


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


def get_labels_from_file(json_file: str) -> List[Tuple[str, str]]:
    """
    Get image-label pairs from a JSON file.
    
    Args:
        json_file (str): Path to the JSON file containing annotations
        
    Returns:
        List[Tuple[str, str]]: List of (filename, label) tuples
    """
    # Parse the labels
    pairs = parse_labels_from_file(json_file)
    
    if not pairs:
        raise ValueError("No image-label pairs found in the JSON file")
        
    return pairs 