import os
import glob
import pandas as pd
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('merge_script.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get all files recursively from the Fitbit directory
def get_unique_prefixes(root_dir='../'):
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.csv', '.json')):  # Only process CSV and JSON files
                all_files.append(os.path.join(root, file))

    # Extract prefixes (everything before the first '-')
    prefixes = set()
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        if '-' in file_name:
            prefix = file_name.split('-')[0].strip()
            prefixes.add(prefix)

    return sorted(list(prefixes))

# Get and display unique prefixes
prefixes = get_unique_prefixes()
print("Unique file prefixes found:")
for prefix in prefixes:
    print(f"- {prefix}")

def standardize_timestamp(df, timestamp_col):
    """
    Standardize timestamp format across all datasets
    """
    try:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Handle timezone if present
        if df[timestamp_col].dt.tz is not None:
            df[timestamp_col] = df[timestamp_col].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Round to nearest minute to ensure consistency
        df[timestamp_col] = df[timestamp_col].dt.floor('T')
        
        return df
    except Exception as e:
        logger.error(f"Error standardizing timestamp: {str(e)}")
        return None

def find_timestamp_column(df):
    """
    Find the timestamp column in a DataFrame
    """
    timestamp_cols = ['timestamp', 'dateTime', 'Date', 'datetime', 'date_time', 'date', 'Watched At']
    found_col = next((col for col in timestamp_cols if col in df.columns), None)
    return found_col

def process_steps_data(root_dir='../'):
    """Process and merge steps data"""
    logger.info("Processing steps data...")
    steps_files = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith('steps') and (file.endswith('.json') or file.endswith('.csv')):
                steps_files.append(os.path.join(root, file))
    
    if not steps_files:
        logger.warning("No steps files found")
        return None

    all_steps = pd.DataFrame()
    for file in steps_files:
        try:
            if file.endswith('.json'):
                with open(file, 'r') as f:
                    data = json.load(f)
                    df = pd.DataFrame(data)
            else:
                df = pd.read_csv(file)
            
            # Find and standardize timestamp column
            timestamp_col = find_timestamp_column(df)
            if timestamp_col:
                df = df.rename(columns={timestamp_col: 'timestamp'})
                df = standardize_timestamp(df, 'timestamp')
                
                if df is not None:
                    # Standardize value column
                    value_col = 'value' if 'value' in df.columns else 'Value'
                    if value_col in df.columns:
                        df = df.rename(columns={value_col: 'steps'})
                    
                    all_steps = pd.concat([all_steps, df[['timestamp', 'steps']]])
                    logger.info(f"Processed {file}")
            else:
                logger.warning(f"No timestamp column found in {file}")
        
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}")
    
    if not all_steps.empty:
        all_steps = all_steps.sort_values('timestamp')
        all_steps = all_steps.drop_duplicates(subset=['timestamp'])
        
        # Resample to 10-minute intervals
        all_steps = all_steps.set_index('timestamp')
        all_steps = all_steps.resample('10T').sum()
        
        all_steps.to_csv('merged_steps_data.csv')
        logger.info(f"Saved merged steps data with {len(all_steps)} records")
    
    return all_steps

def process_heart_rate_data(root_dir='../'):
    """Process and merge heart rate data"""
    logger.info("Processing heart rate data...")
    hr_files = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if 'heart_rate' in file.lower() and file.endswith('.json'):
                hr_files.append(os.path.join(root, file))
    
    if not hr_files:
        logger.warning("No heart rate files found")
        return None

    all_hr = pd.DataFrame()
    for file in hr_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                records = []
                for record in data:
                    records.append({
                        'timestamp': record['dateTime'],
                        'bpm': record['value']['bpm']
                    })
                df = pd.DataFrame(records)
            
            df = standardize_timestamp(df, 'timestamp')
            if df is not None:
                all_hr = pd.concat([all_hr, df])
                logger.info(f"Processed {file}")
        
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}")
    
    if not all_hr.empty:
        all_hr = all_hr.sort_values('timestamp')
        all_hr = all_hr.drop_duplicates(subset=['timestamp'])
        
        # Resample to 10-minute intervals
        all_hr = all_hr.set_index('timestamp')
        all_hr = all_hr.resample('10T').mean()
        
        all_hr.to_csv('merged_heart_rate_10min.csv')
        logger.info(f"Saved merged heart rate data with {len(all_hr)} records")
    
    return all_hr

def process_spo2_data(root_dir='../'):
    """Process and merge SpO2 data"""
    logger.info("Processing SpO2 data...")
    spo2_files = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if ('spo2' in file.lower() or 'oxygen' in file.lower()) and file.endswith('.csv'):
                spo2_files.append(os.path.join(root, file))
    
    if not spo2_files:
        logger.warning("No SpO2 files found")
        return None

    all_spo2 = pd.DataFrame()
    for file in spo2_files:
        try:
            df = pd.read_csv(file)
            
            # Find and standardize timestamp column
            timestamp_col = find_timestamp_column(df)
            if timestamp_col:
                df = df.rename(columns={timestamp_col: 'timestamp'})
                df = standardize_timestamp(df, 'timestamp')
                
                if df is not None:
                    # Find SpO2 value column
                    value_col = next((col for col in df.columns if 'value' in col.lower() or 'spo2' in col.lower()), None)
                    if value_col:
                        df = df.rename(columns={value_col: 'spo2'})
                        all_spo2 = pd.concat([all_spo2, df[['timestamp', 'spo2']]])
                        logger.info(f"Processed {file}")
            else:
                logger.warning(f"No timestamp column found in {file}")
        
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}")
    
    if not all_spo2.empty:
        all_spo2 = all_spo2.sort_values('timestamp')
        all_spo2 = all_spo2.drop_duplicates(subset=['timestamp'])
        
        # Resample to 10-minute intervals
        all_spo2 = all_spo2.set_index('timestamp')
        all_spo2 = all_spo2.resample('10T').mean()
        
        all_spo2.to_csv('merged_spo2_data.csv')
        logger.info(f"Saved merged SpO2 data with {len(all_spo2)} records")
    
    return all_spo2

def process_hrv_data(root_dir='../'):
    """Process and merge HRV data"""
    logger.info("Processing HRV data...")
    hrv_files = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if 'Heart Rate Variability' in file and file.endswith('.csv'):
                hrv_files.append(os.path.join(root, file))
    
    if not hrv_files:
        logger.warning("No HRV files found")
        return None

    all_hrv = pd.DataFrame()
    for file in hrv_files:
        try:
            df = pd.read_csv(file)
            timestamp_col = find_timestamp_column(df)
            if timestamp_col:
                df = df.rename(columns={timestamp_col: 'timestamp'})
                df = standardize_timestamp(df, 'timestamp')
                
                if df is not None:
                    # Keep relevant columns
                    hrv_cols = ['rmssd', 'nremhr', 'entropy', 'coverage', 'low_frequency', 'high_frequency']
                    cols_to_keep = ['timestamp'] + [col for col in hrv_cols if col in df.columns]
                    all_hrv = pd.concat([all_hrv, df[cols_to_keep]])
                    logger.info(f"Processed {file}")
            else:
                logger.warning(f"No timestamp column found in {file}")
        
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}")
    
    if not all_hrv.empty:
        all_hrv = all_hrv.sort_values('timestamp')
        all_hrv = all_hrv.drop_duplicates(subset=['timestamp'])
        all_hrv = all_hrv.set_index('timestamp')
        all_hrv = all_hrv.resample('10T').mean()
        all_hrv.to_csv('merged_hrv_data.csv')
        logger.info(f"Saved merged HRV data with {len(all_hrv)} records")
    
    return all_hrv

def process_respiratory_data(root_dir='../'):
    """Process and merge respiratory rate data"""
    logger.info("Processing respiratory rate data...")
    resp_files = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if 'Respiratory Rate' in file and file.endswith('.csv'):
                resp_files.append(os.path.join(root, file))
    
    if not resp_files:
        logger.warning("No respiratory rate files found")
        return None

    all_resp = pd.DataFrame()
    for file in resp_files:
        try:
            df = pd.read_csv(file)
            timestamp_col = find_timestamp_column(df)
            if timestamp_col:
                df = df.rename(columns={timestamp_col: 'timestamp'})
                df = standardize_timestamp(df, 'timestamp')
                
                if df is not None:
                    value_col = next((col for col in df.columns if 'value' in col.lower() or 'rate' in col.lower()), None)
                    if value_col:
                        df = df.rename(columns={value_col: 'respiratory_rate'})
                        all_resp = pd.concat([all_resp, df[['timestamp', 'respiratory_rate']]])
                        logger.info(f"Processed {file}")
            else:
                logger.warning(f"No timestamp column found in {file}")
        
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}")
    
    if not all_resp.empty:
        all_resp = all_resp.sort_values('timestamp')
        all_resp = all_resp.drop_duplicates(subset=['timestamp'])
        all_resp = all_resp.set_index('timestamp')
        all_resp = all_resp.resample('10T').mean()
        all_resp.to_csv('merged_respiratory_data.csv')
        logger.info(f"Saved merged respiratory data with {len(all_resp)} records")
    
    return all_resp

def process_azm_data(root_dir='../'):
    """Process and merge Active Zone Minutes data"""
    logger.info("Processing AZM data...")
    azm_files = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if ('Active Zone Minutes' in file or 'active_zone_minutes' in file) and file.endswith('.csv'):
                azm_files.append(os.path.join(root, file))
    
    if not azm_files:
        logger.warning("No AZM files found")
        return None

    all_azm = pd.DataFrame()
    for file in azm_files:
        try:
            df = pd.read_csv(file)
            timestamp_col = find_timestamp_column(df)
            if timestamp_col:
                df = df.rename(columns={timestamp_col: 'timestamp'})
                df = standardize_timestamp(df, 'timestamp')
                
                if df is not None:
                    # Process zone information
                    if 'zone_id' in df.columns and 'minutes' in df.columns:
                        # Pivot the data to get zones as columns
                        df_pivot = df.pivot(index='timestamp', columns='zone_id', values='minutes')
                        df_pivot.columns = ['below_zones', 'fat_burn', 'cardio', 'peak']
                        all_azm = pd.concat([all_azm, df_pivot])
                    logger.info(f"Processed {file}")
            else:
                logger.warning(f"No timestamp column found in {file}")
        
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}")
    
    if not all_azm.empty:
        all_azm = all_azm.sort_index()
        all_azm = all_azm.resample('10T').sum()
        all_azm.to_csv('merged_azm_data.csv')
        logger.info(f"Saved merged AZM data with {len(all_azm)} records")
    
    return all_azm

def final_merge():
    """Merge all processed data files into final dataset"""
    logger.info("Performing final merge of all health metrics...")
    
    # Process all data types
    dfs = {
        'steps': process_steps_data(),
        'heart_rate': process_heart_rate_data(),
        'spo2': process_spo2_data(),
        'hrv': process_hrv_data(),
        'respiratory': process_respiratory_data(),
        'azm': process_azm_data()
    }
    
    # Initialize with first non-None DataFrame
    final_df = next((df for df in dfs.values() if df is not None), pd.DataFrame())
    
    # Merge remaining DataFrames
    for name, df in dfs.items():
        if df is not None and not df.equals(final_df):
            final_df = final_df.join(df, how='outer')
            logger.info(f"Merged {name} data")
    
    if not final_df.empty:
        # Ensure final sorting
        final_df = final_df.sort_index()
        
        # Save final merged dataset
        final_df.to_csv('final_health_metrics.csv')
        logger.info(f"Saved final merged dataset with {len(final_df)} records")
        
        # Log time range
        logger.info(f"Time range: {final_df.index.min()} to {final_df.index.max()}")
        
        # Log data completeness
        completeness = (final_df.count() / len(final_df) * 100).sort_values(ascending=False)
        logger.info("\nData completeness (%):")
        for col, pct in completeness.items():
            logger.info(f"{col:<20} {pct:>6.2f}%")
    else:
        logger.error("No data was successfully merged")
    
    return final_df

if __name__ == "__main__":
    final_merge()

