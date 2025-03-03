# Fitbit Data Processing Scripts

These scripts process Fitbit exported data to combine steps, heart rate (BPM), and SpO2 measurements.

## Setup

1. Place these Python notebooks in your Fitbit Global Export Data folder:
   ```
   D:/Takeout_2Feb/Fit_Takeoutdata/Fitbit/Global Export Data/
   ├── 1merge_script.ipynb    # Initial data processing
   └── 2_joinDS.ipynb        # Combines metrics and analyzes overlaps
   ```

2. Required input files in the same folder:
   - heart_rate-*.json files (BPM data)
   - estimated_oxygen_variation-*.csv (SpO2 data)
   - Other Fitbit exported data files

## Output Files

The scripts will generate these files:
1. `health_data_10min_intervals.csv` - All metrics with 10-minute intervals
2. `health_data_no_empty_rows.csv` - Data with empty rows removed
3. `health_data_no_empty_rows_2plus_metrics.csv` - Only rows where 2+ metrics are present

## Running Order

1. Run `1merge_script.ipynb` first to process raw data
2. Then run `2_joinDS.ipynb` to analyze metric combinations

Each script will display statistics about the processed data and save the results to CSV files.