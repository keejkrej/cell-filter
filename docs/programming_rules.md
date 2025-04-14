# Programming Rules and Conventions

## Code Organization

### 1. File Structure
- Use clear section headers with consistent formatting:
  ```python
  # =====================================================================
  # Section Name
  # =====================================================================
  ```

### 2. Class Organization
Classes should be organized in the following order:
1. Class docstring
2. Constructor and Initialization section
3. Private Methods section
4. Public Methods section

### 3. Method Organization
- Group related methods together
- Place initialization methods in the constructor section
- Place helper methods in the private methods section
- Place API methods in the public methods section

## Naming Conventions

### 1. Private Methods
- All private methods must start with an underscore: `_method_name`
- Private methods should be placed in the private methods section
- Examples:
  ```python
  def _init_patterns(self) -> None:
  def _validate_files(self) -> None:
  def _extract_region(self, frame: np.ndarray, pattern_idx: int) -> np.ndarray:
  ```

### 2. Public Methods
- Public methods should not start with an underscore
- Public methods should be placed in the public methods section
- Examples:
  ```python
  def load_view(self, view_idx: int) -> None:
  def process_patterns(self) -> None:
  def extract_nuclei(self, pattern_idx: int) -> np.ndarray:
  ```

## Documentation

### 1. Class Documentation
- Include a comprehensive class docstring
- Document all attributes
- Include usage examples if helpful
- Example:
  ```python
  class CellGenerator:
      """
      A class for generating and processing cell data from images.
      
      This class handles the loading and processing of pattern and cell images from ND2 files,
      providing methods to extract and analyze cell regions.
      
      Attributes:
          patterns_path (str): Path to the patterns ND2 file
          cells_path (str): Path to the cell ND2 file
          ...
      """
  ```

### 2. Method Documentation
- Include docstrings for all methods
- Document parameters, return values, and exceptions
- Use consistent formatting
- Example:
  ```python
  def method_name(self, param1: type) -> return_type:
      """
      Brief description of what the method does.
      
      Args:
          param1 (type): Description of param1
          
      Returns:
          return_type: Description of return value
          
      Raises:
          ValueError: Description of when this error occurs
      """
  ```

## Error Handling

### 1. Input Validation
- Validate inputs at the start of methods
- Raise appropriate exceptions with descriptive messages
- Example:
  ```python
  if frame is None:
        raise ValueError("Frame not provided")
  if pattern_idx >= self.n_patterns or pattern_idx < 0:
      raise ValueError(f"Pattern index {pattern_idx} out of range (0-{self.n_patterns-1})")
  ```

### 2. Exception Handling
- Use try-except blocks for operations that might fail
- Log errors before raising exceptions
- Example:
  ```python
  try:
      # Operation that might fail
  except Exception as e:
      logger.error(f"Error message: {e}")
      raise ValueError(f"Error message: {e}")
  ```

## Logging

### 1. Logging Configuration
- Configure logging only at the application's entry point (CLI scripts)
- Set root logger to INFO to suppress third-party debug messages
- Add a --debug flag to control package logger level
- Example:
  ```python
  # In CLI script (entry point)
  logging.basicConfig(level=logging.INFO)  # Root logger at INFO to suppress third-party debug messages
  
  # Set package logger level based on --debug flag
  package_logger = logging.getLogger("cell_counter")
  package_logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
  
  # Create module-specific logger
  logger = logging.getLogger(__name__)
  ```

### 2. Logging Levels
- Use appropriate logging levels:
  - DEBUG: Detailed information for debugging
  - INFO: General information about program execution
  - ERROR: Error conditions that might still allow the program to continue
  - CRITICAL: Error conditions that prevent the program from continuing

### 3. Logging Messages
- Include relevant context in log messages
- Use f-strings for dynamic content
- Example:
  ```python
  logger.debug(f"Found {len(contours)} contours in image")
  logger.info(f"Processed {self.n_patterns} patterns")
  logger.error(f"Error loading patterns: {e}")
  ```

## Type Hints

### 1. Method Signatures
- Use type hints for all method parameters and return values
- Use typing module for complex types
- Example:
  ```python
  from typing import List, Tuple
  
  def method_name(self, param1: np.ndarray) -> List[Tuple[int, int]]:
  ```

### 2. Variable Types
- Use type hints for class attributes in docstrings
- Use Optional for nullable values
- Example:
  ```python
  Attributes:
      patterns (Optional[np.ndarray]): Current patterns image
      n_patterns (int): Number of detected patterns
  ```

## Constants

### 1. Constant Definition
- Define constants at the module level
- Use uppercase with underscores
- Group related constants together
- Example:
  ```python
  # Constants
  DEFAULT_GRID_SIZE = 20
  AREA_STD_DEVIATIONS = 2
  GAUSSIAN_BLUR_SIZE = (5, 5)
  ```

### 2. Constant Usage
- Use constants instead of magic numbers
- Reference constants in method docstrings if relevant
- Example:
  ```python
  if abs(area - mean_area) > AREA_STD_DEVIATIONS * std_area:
      continue
  ``` 