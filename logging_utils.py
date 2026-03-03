import logging
import sys
from typing import Dict

class Logger:
    """A centralized logger for the cognitive robotics project."""
    
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance with the specified name.
        
        Args:
            name (str): Name of the logger (typically the module name)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        if name in self._loggers:
            return self._loggers[name]
            
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # Prevent adding handlers multiple times
        if not logger.handlers:
            # Create formatter with source identification
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(console_handler)
            
            # Prevent propagation to root logger
            logger.propagate = False
        
        self._loggers[name] = logger
        return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name (str): Name of the logger (typically the module name)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger_instance = Logger()
    return logger_instance.get_logger(name)