"""
Abstract base class for configuration handling functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any, List
from pathlib import Path

class ConfigHandlerABC(ABC):
    """
    Abstract base class defining the interface for configuration handling.
    """
    
    @abstractmethod
    def __init__(
        self,
        config_path: Union[str, Path]
    ) -> None:
        """
        Initialize the config handler with a configuration file path.
        
        Args:
            config_path (Union[str, Path]): Path to the configuration file
        """
        pass
    
    @abstractmethod
    def get_value(
        self,
        key: str,
        required: bool = True,
        default: Any = None,
        expected_type: Optional[type] = None
    ) -> Any:
        """
        Get a configuration value.
        
        Args:
            key (str): Configuration key
            required (bool): Whether the key is required
            default (Any): Default value if key is not found
            expected_type (Optional[type]): Expected type of the value
            
        Returns:
            Any: Configuration value
            
        Raises:
            ValueError: If required key is missing or value has wrong type
        """
        pass
    
    @abstractmethod
    def get_path(
        self,
        key: str,
        required: bool = True,
        default: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Get a configuration value as a Path.
        
        Args:
            key (str): Configuration key
            required (bool): Whether the key is required
            default (Optional[Union[str, Path]]): Default path if key is not found
            
        Returns:
            Path: Configuration value as Path
            
        Raises:
            ValueError: If required key is missing
        """
        pass
    
    @abstractmethod
    def get_int(
        self,
        key: str,
        required: bool = True,
        default: Optional[int] = None
    ) -> int:
        """
        Get a configuration value as an integer.
        
        Args:
            key (str): Configuration key
            required (bool): Whether the key is required
            default (Optional[int]): Default value if key is not found
            
        Returns:
            int: Configuration value as integer
            
        Raises:
            ValueError: If required key is missing or value is not an integer
        """
        pass
    
    @abstractmethod
    def get_float(
        self,
        key: str,
        required: bool = True,
        default: Optional[float] = None
    ) -> float:
        """
        Get a configuration value as a float.
        
        Args:
            key (str): Configuration key
            required (bool): Whether the key is required
            default (Optional[float]): Default value if key is not found
            
        Returns:
            float: Configuration value as float
            
        Raises:
            ValueError: If required key is missing or value is not a float
        """
        pass
    
    @abstractmethod
    def get_bool(
        self,
        key: str,
        required: bool = True,
        default: Optional[bool] = None
    ) -> bool:
        """
        Get a configuration value as a boolean.
        
        Args:
            key (str): Configuration key
            required (bool): Whether the key is required
            default (Optional[bool]): Default value if key is not found
            
        Returns:
            bool: Configuration value as boolean
            
        Raises:
            ValueError: If required key is missing or value is not a boolean
        """
        pass
    
    @abstractmethod
    def get_list(
        self,
        key: str,
        required: bool = True,
        default: Optional[List[Any]] = None
    ) -> List[Any]:
        """
        Get a configuration value as a list.
        
        Args:
            key (str): Configuration key
            required (bool): Whether the key is required
            default (Optional[List[Any]]): Default value if key is not found
            
        Returns:
            List[Any]: Configuration value as list
            
        Raises:
            ValueError: If required key is missing or value is not a list
        """
        pass
    
    @abstractmethod
    def get_dict(
        self,
        key: str,
        required: bool = True,
        default: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get a configuration value as a dictionary.
        
        Args:
            key (str): Configuration key
            required (bool): Whether the key is required
            default (Optional[Dict[str, Any]]): Default value if key is not found
            
        Returns:
            Dict[str, Any]: Configuration value as dictionary
            
        Raises:
            ValueError: If required key is missing or value is not a dictionary
        """
        pass
    
    @abstractmethod
    def save_config(
        self,
        config: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Save configuration to a file.
        
        Args:
            config (Dict[str, Any]): Configuration to save
            output_path (Optional[Union[str, Path]]): Path to save the configuration
        """
        pass 