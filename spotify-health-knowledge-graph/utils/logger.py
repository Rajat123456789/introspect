import os
import logging
from datetime import datetime
from typing import Optional

class HealthLogger:
    def __init__(self, log_dir: str = 'logs', log_level: int = logging.INFO):
        """
        Initialize the HealthLogger.
        
        Args:
            log_dir (str): Directory to store log files
            log_level (int): Logging level (default: INFO)
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a timestamp for the log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'health_metrics_{timestamp}.log')
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('HealthMetrics')
        self.logger.info(f"Initialized logger. Log file: {log_file}")
    
    def log_metric(self, metric_name: str, value: float, additional_info: Optional[dict] = None):
        """
        Log a health metric with optional additional information.
        
        Args:
            metric_name (str): Name of the metric
            value (float): Value of the metric
            additional_info (dict, optional): Additional information to log
        """
        message = f"Metric: {metric_name} = {value}"
        if additional_info:
            message += f" | Additional Info: {additional_info}"
        self.logger.info(message)
    
    def log_error(self, error_message: str, exception: Optional[Exception] = None):
        """
        Log an error message with optional exception details.
        
        Args:
            error_message (str): Error message to log
            exception (Exception, optional): Exception object
        """
        if exception:
            self.logger.error(f"{error_message} | Exception: {str(exception)}")
        else:
            self.logger.error(error_message)
    
    def log_warning(self, warning_message: str):
        """
        Log a warning message.
        
        Args:
            warning_message (str): Warning message to log
        """
        self.logger.warning(warning_message)
    
    def log_info(self, info_message: str):
        """
        Log an information message.
        
        Args:
            info_message (str): Information message to log
        """
        self.logger.info(info_message)
    
    def log_debug(self, debug_message: str):
        """
        Log a debug message.
        
        Args:
            debug_message (str): Debug message to log
        """
        self.logger.debug(debug_message) 