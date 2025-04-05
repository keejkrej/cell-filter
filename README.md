# Cell Counter

A Python package for counting cells in microscopy images.

## Installation

```bash
git clone https://github.com/yourusername/cell-counter.git
cd cell-counter
pip install -e .
```

## Usage

### Command Line Interface

#### Generate Command

Generate nuclei and/or cytoplasm data from images:

```bash
# Generate nuclei data
python -m cell_counter.cli.generate --patterns <patterns_path> --nuclei <nuclei_path> --output <output_dir>

# Generate cytoplasm data
python -m cell_counter.cli.generate --patterns <patterns_path> --cyto <cyto_path> --output <output_dir>

# Generate both
python -m cell_counter.cli.generate --patterns <patterns_path> --nuclei <nuclei_path> --cyto <cyto_path> --output <output_dir>
```

Optional arguments:
- `--frames`: Range of frames to process (e.g., "0-5" for frames 0 to 5, "0,2,4" for specific frames)
- `--contours`: Range of contours to process (e.g., "0-5" for contours 0 to 5, "0,2,4" for specific contours)

Example:
```bash
python -m cell_counter.cli.generate --patterns /path/to/patterns.tif --nuclei /path/to/nuclei.tif --cyto /path/to/cyto.tif --output /path/to/output_dir --frames 0-5 --contours 0-10
```

#### Info Command

Display information about the image stacks:

```bash
# Show info for patterns and nuclei
python -m cell_counter.cli.info --patterns <patterns_path> --nuclei <nuclei_path>

# Show info for patterns and cytoplasm
python -m cell_counter.cli.info --patterns <patterns_path> --cyto <cyto_path>

# Show info for all
python -m cell_counter.cli.info --patterns <patterns_path> --nuclei <nuclei_path> --cyto <cyto_path>
```

The info command displays:
- Patterns image dimensions
- Number of frames in nuclei/cytoplasm stacks
- Dimensions of each frame
- Total number of contours found
- Number of contours after filtering (contours touching edges are filtered out)
- Number of contours filtered out (if any)

#### Count Command

Count nuclei in extracted regions:

```bash
python -m cell_counter.cli.count --input <input_dir> --output <output_dir>
```

Optional arguments:
- `--wanted`: Desired number of nuclei per contour (default: 3)
- `--no-cellpose`: Use simple thresholding instead of Cellpose
- `--no-gpu`: Disable GPU usage for Cellpose
- `--diameter`: Expected diameter of cells in pixels (default: 15)
- `--channels`: Channel indices for Cellpose (default: "0,0")
- `--model`: Type of Cellpose model to use (default: "cyto3")

Example:
```bash
python -m cell_counter.cli.count --input /path/to/input_dir --output /path/to/output_dir --wanted 3 --diameter 20 --channels "0,0" --model "cyto3"
```

#### Time Series Command

Analyze time series data and track nuclei counts:

```bash
python -m cell_counter.cli.time_series --patterns <patterns_path> --nuclei <nuclei_path> --output <output_json>
```

Optional arguments:
- `--wanted`: Desired number of nuclei per contour (default: 3)
- `--no-cellpose`: Use simple thresholding instead of Cellpose
- `--no-gpu`: Disable GPU usage for Cellpose
- `--diameter`: Expected diameter of cells in pixels (default: 15)
- `--channels`: Channel indices for Cellpose (default: "0,0")
- `--model`: Type of Cellpose model to use (default: "cyto3")
- `--min-intensity`: Minimum average intensity for valid regions (default: 10)
- `--debug`: Enable debug output
- `--save-frames`: Directory to save problematic frames for debugging

Example:
```bash
python -m cell_counter.cli.time_series --patterns /path/to/patterns.tif --nuclei /path/to/nuclei.tif --output results.json --wanted 3 --debug --save-frames debug_frames
```

#### Extract Command

Extract valid frames for each contour based on time series analysis:

```bash
python -m cell_counter.cli.extract --patterns <patterns_path> --nuclei <nuclei_path> --time-series <time_series_path> --output <output_dir>
```

Optional arguments:
- `--min-frames`: Minimum number of valid frames required (default: 10)

Example:
```bash
python -m cell_counter.cli.extract --patterns /path/to/patterns.tif --nuclei /path/to/nuclei.tif --time-series results.json --output extracted_frames --min-frames 20
```

#### Load Data Command

Load and process data from image files:

```bash
python -m cell_counter.cli.load_data --patterns <patterns_path> --nuclei <nuclei_path> --cyto <cyto_path>
```

Optional arguments:
- `--frames`: Range of frames to process
- `--contours`: Range of contours to process

#### Show Labels Command

Display labeled regions from image files:

```bash
python -m cell_counter.cli.show_labels --patterns <patterns_path> --nuclei <nuclei_path> --cyto <cyto_path>
```

Optional arguments:
- `--frame`: Specific frame to display
- `--contour`: Specific contour to display

#### Train Command

Train a model for cell detection:

```bash
python -m cell_counter.cli.train --input <input_dir> --output <output_dir>
```

Optional arguments:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 8)

#### Inference Command

Run inference using a trained model:

```bash
python -m cell_counter.cli.inference --model <model_path> --input <input_dir> --output <output_dir>
```

Optional arguments:
- `--batch-size`: Batch size for inference (default: 8)
- `--threshold`: Confidence threshold for detection (default: 0.5)

## Features

- Process both nuclei and cytoplasm data
- Extract regions of interest using contours
- Filter out contours that touch or are too close to image edges (5-pixel margin)
- Support for processing specific ranges of frames and contours
- Normalize image intensity for better visualization
- Save extracted regions as PNG files with frame and contour information in filenames
- Time series analysis with Cellpose integration
- Debugging tools for problematic frames
- Flexible extraction of valid frames based on analysis results

## Output

The generate command creates a directory structure like this:
```
output_dir/
  frame_000/
    nuclei_000_000.png
    nuclei_000_001.png
    cyto_000_000.png
    cyto_000_001.png
  frame_001/
    nuclei_001_000.png
    nuclei_001_001.png
    cyto_001_000.png
    cyto_001_001.png
```

Where:
- `frame_XXX`: Directory for each frame
- `nuclei_XXX_YYY.png`: Extracted nuclei region (XXX = frame number, YYY = contour number)
- `cyto_XXX_YYY.png`: Extracted cytoplasm region (XXX = frame number, YYY = contour number)

## Development

This project uses:
- `black` for code formatting
- `isort` for import sorting