"""
Time series analysis command for cell-counter.
"""

"""
Usage:
    After installing the package with `pip install -e .`, run:
    
    # Basic usage
    python -m cell_counter.cli.time_series --patterns <patterns_path> --nuclei <nuclei_path> --output <output_path>
    
    # With custom parameters
    python -m cell_counter.cli.time_series --patterns <patterns_path> --nuclei <nuclei_path> --output <output_path> --wanted 3 --no-cellpose --diameter 20
    
    Optional arguments:
    --wanted: Number of nuclei to look for (default: 3)
    --no-cellpose: Use simple thresholding instead of Cellpose
    --no-gpu: Don't use GPU for Cellpose
    --diameter: Expected diameter of cells in pixels (default: 15)
    --channels: Channel indices for Cellpose (default: "0,0")
    --model: Type of Cellpose model to use (default: "cyto3")
"""

import argparse
from pathlib import Path
from ..core.time_series import analyze_time_series, save_time_series

def parse_channels(channels_str: str) -> list:
    """Parse channel string into a list of integers."""
    try:
        return [int(x) for x in channels_str.split(',')]
    except ValueError:
        raise ValueError("Channels must be comma-separated integers")

def main():
    """Main entry point for time series command."""
    parser = argparse.ArgumentParser(description='Analyze time series data to find valid frames for each contour')
    
    # Required arguments
    parser.add_argument('--patterns', type=str, required=True, help='Path to the patterns image file')
    parser.add_argument('--nuclei', type=str, required=True, help='Path to the nuclei image file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    
    # Optional arguments
    parser.add_argument('--wanted', type=int, default=3, help='Desired number of nuclei per contour (default: 3)')
    parser.add_argument('--no-cellpose', action='store_true', help='Use simple thresholding instead of Cellpose')
    parser.add_argument('--no-gpu', action='store_true', help='Don\'t use GPU for Cellpose')
    parser.add_argument('--diameter', type=int, default=15, help='Expected diameter of cells in pixels (default: 15)')
    parser.add_argument('--channels', type=str, default='0,0', help='Channel indices for Cellpose (default: "0,0")')
    parser.add_argument('--model', type=str, default='cyto3', help='Type of Cellpose model to use (default: "cyto3")')
    parser.add_argument('--min-intensity', type=int, default=10, help='Minimum average intensity for valid regions (default: 10)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--save-frames', type=str, help='Directory to save problematic frames for debugging')
    
    args = parser.parse_args()
    
    # Validate paths
    patterns_path = Path(args.patterns)
    nuclei_path = Path(args.nuclei)
    output_path = Path(args.output)
    
    if not patterns_path.exists():
        raise FileNotFoundError(f"Patterns file not found: {patterns_path}")
    if not nuclei_path.exists():
        raise FileNotFoundError(f"Nuclei file not found: {nuclei_path}")
    
    # Create debug directory if needed
    if args.save_frames:
        debug_dir = Path(args.save_frames)
        debug_dir.mkdir(parents=True, exist_ok=True)
    else:
        debug_dir = None
    
    # Parse channels
    try:
        channels = parse_channels(args.channels)
    except ValueError as e:
        parser.error(f"Invalid channels format: {str(e)}")
    
    # Run analysis
    time_lapse = analyze_time_series(
        patterns_path=str(patterns_path),
        nuclei_path=str(nuclei_path),
        wanted_nuclei=args.wanted,
        use_cellpose=not args.no_cellpose,
        use_gpu=not args.no_gpu,
        diameter=args.diameter,
        channels=channels,
        model_type=args.model,
        min_intensity=args.min_intensity,
        debug_dir=debug_dir,
        debug=args.debug
    )
    
    # Save results
    save_time_series(
        time_lapse=time_lapse,
        output_path=str(output_path),
        patterns_path=str(patterns_path),
        nuclei_path=str(nuclei_path),
        wanted_nuclei=args.wanted,
        use_cellpose=not args.no_cellpose,
        use_gpu=not args.no_gpu,
        diameter=args.diameter,
        channels=channels,
        model_type=args.model
    )
    
    print(f"\nAnalysis complete. Results saved to: {output_path}")

if __name__ == '__main__':
    main() 