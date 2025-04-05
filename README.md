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

## Features

- Process both nuclei and cytoplasm data
- Extract regions of interest using contours
- Filter out contours that touch or are too close to image edges (5-pixel margin)
- Support for processing specific ranges of frames and contours
- Normalize image intensity for better visualization
- Save extracted regions as PNG files with frame and contour information in filenames

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
- `