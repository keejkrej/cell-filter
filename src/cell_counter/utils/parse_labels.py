import json
from typing import Dict, List, Tuple

def parse_labels(json_data: Dict) -> List[Tuple[str, str]]:
    """
    Parse the JSON data and extract file names and their corresponding labels.
    
    Args:
        json_data (Dict): The JSON data containing annotations
        
    Returns:
        List[Tuple[str, str]]: List of tuples containing (filename, label)
    """
    results = []
    
    # Extract the filename from data.image
    filename = json_data["data"]["image"]
    
    # Get the annotations
    annotations = json_data["annotations"]
    
    # For each annotation, extract the label
    for annotation in annotations:
        for result in annotation["result"]:
            if result["type"] == "choices":
                label = result["value"]["choices"][0]  # Get the first choice
                results.append((filename, label))
    
    return results

def parse_labels_from_file(json_file_path: str) -> List[Tuple[str, str]]:
    """
    Parse labels from a JSON file containing a list of annotation objects.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        List[Tuple[str, str]]: List of tuples containing (filename, label)
    """
    with open(json_file_path, 'r') as f:
        json_data_list = json.load(f)
    
    all_results = []
    for json_data in json_data_list:
        results = parse_labels(json_data)
        all_results.extend(results)
    
    return all_results 