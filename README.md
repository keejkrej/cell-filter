# Cell Counter

A Python tool for counting cells in images.

## Installation

```bash
pip install .
```

For development installation:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from cell_counter import count_cells
import cv2

# Load an image
image = cv2.imread("path_to_image.jpg")
count, contours = count_cells(image)
print(f"Found {count} cells")
```

## Development

This project uses:
- `black` for code formatting
- `isort` for import sorting
- `mypy` for type checking
- `pytest` for testing

## License

MIT 