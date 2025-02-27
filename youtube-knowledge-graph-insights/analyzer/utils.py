"""
Utility functions for data processing and file operations
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def save_dataframe_to_files(result_data, analysis_name, formats=None):
    """Save analysis results to files in various formats"""
    # Default to CSV and JSON if not specified
    if formats is None:
        formats = ['csv', 'json']
    
    # Create reports directory if it doesn't exist
    reports_dir = 'analysis_reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Generate filename base
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{reports_dir}/{analysis_name}_{timestamp}"
    
    # Custom JSON encoder to handle pandas Timestamp objects
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (pd.Timestamp, pd.Period, datetime)):
                return obj.isoformat()
            elif pd.isna(obj):
                return None
            try:
                return json.JSONEncoder.default(self, obj)
            except TypeError:
                return str(obj)
    
    # Handle different data types
    if isinstance(result_data, pd.DataFrame):
        if 'csv' in formats:
            csv_path = f"{base_filename}.csv"
            result_data.to_csv(csv_path, index=True)
            print(f"Saved {analysis_name} to {csv_path}")
            
        if 'json' in formats:
            json_path = f"{base_filename}.json"
            # Convert to dict first, then use custom encoder
            if isinstance(result_data.index, pd.MultiIndex):
                result_data_dict = result_data.reset_index().to_dict(orient='records')
            else:
                result_data_dict = result_data.to_dict(orient='records')
                
            with open(json_path, 'w') as f:
                json.dump(result_data_dict, f, indent=2, cls=DateTimeEncoder)
            print(f"Saved {analysis_name} to {json_path}")
            
    elif isinstance(result_data, dict):
        if 'json' in formats:
            json_path = f"{base_filename}.json"
            
            # Process dict contents for JSON serialization
            serializable_data = {}
            for key, value in result_data.items():
                if isinstance(value, pd.DataFrame):
                    if isinstance(value.index, pd.MultiIndex):
                        serializable_data[key] = value.reset_index().to_dict(orient='records')
                    else:
                        serializable_data[key] = value.to_dict(orient='records')
                elif isinstance(value, pd.Series):
                    if isinstance(value.index, pd.MultiIndex):
                        serializable_data[key] = value.reset_index().to_dict(orient='records')
                    else:
                        serializable_data[key] = value.to_dict()
                else:
                    serializable_data[key] = value
            
            with open(json_path, 'w') as f:
                json.dump(serializable_data, f, indent=2, cls=DateTimeEncoder)
                
            print(f"Saved {analysis_name} to {json_path}")
    
    return base_filename 