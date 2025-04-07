import argparse
from cell_counter.analyzer import Analyzer

def main():
    parser = argparse.ArgumentParser(description='Analyze time series data for cell counting')
    parser.add_argument('--patterns', type=str, required=True, help='Path to patterns image')
    parser.add_argument('--nuclei', type=str, required=True, help='Path to nuclei image')
    parser.add_argument('--wanted', type=int, default=3, help='Number of nuclei to look for')
    parser.add_argument('--use-cellpose', action='store_true', help='Use Cellpose for counting')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for Cellpose')
    parser.add_argument('--diameter', type=int, default=15, help='Expected diameter of cells in pixels')
    parser.add_argument('--channels', type=str, default="0,0", help='Channel indices for Cellpose')
    parser.add_argument('--model-type', type=str, default="cyto3", help='Type of Cellpose model to use')
    parser.add_argument('--grid-size', type=int, default=20, help='Size of the grid for snapping pattern centers')
    parser.add_argument('--threshold', type=int, help='Threshold value for nuclei extraction')
    parser.add_argument('--output', type=str, required=True, help='Path to save results')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = Analyzer(
        patterns_path=args.patterns,
        nuclei_path=args.nuclei,
        wanted=args.wanted,
        use_cellpose=args.use_cellpose,
        use_gpu=args.use_gpu,
        diameter=args.diameter,
        channels=args.channels,
        model_type=args.model_type,
        grid_size=args.grid_size,
        threshold=args.threshold
    )
    
    # Analyze time series
    results = analyzer.analyze_time_series()
    
    # Save results
    analyzer.save_time_series(results, args.output) 