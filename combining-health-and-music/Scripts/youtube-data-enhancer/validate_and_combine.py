import os
import pandas as pd
import glob
import re
from datetime import datetime

def validate_and_combine_files(output_dir, file_type='main'):
    """
    Validate and combine all output files of a specific type into a single consolidated file.
    
    Args:
        output_dir: Directory containing the output files
        file_type: Type of file to combine ('main', 'patterns', 'mental_health', 'engagement')
    
    Returns:
        Tuple of (good_file_path, bad_file_path)
    """
    print(f"\nValidating and combining all '{file_type}' files...")
    
    # Define expected columns for each file type
    expected_columns = {
        'main': ['video_id', 'title', 'watched_at', 'primary_category', 'detailed_type', 
                 'sentiment', 'sentiment_score', 'primary_format', 'primary_purpose', 
                 'style', 'confidence'],
        'patterns': ['video_id', 'pattern_type', 'pattern', 'timestamp', 'category'],
        'mental_health': ['video_id', 'category', 'score', 'timestamp', 'sentiment', 'sentiment_score'],
        'engagement': ['video_id', 'timestamp', 'content_type', 'audience_engagement', 
                      'production_quality', 'content_format', 'content_purpose']
    }
    
    # Find all files matching the pattern
    pattern = os.path.join(output_dir, f'youtube_analysis_*_{file_type}.csv')
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"No '{file_type}' files found in {output_dir}")
        return None, None
    
    print(f"Found {len(all_files)} {file_type} files:")
    for file in all_files:
        file_size = os.path.getsize(file) / 1024  # KB
        print(f"  - {os.path.basename(file)} ({file_size:.1f} KB)")
    
    # Initialize containers for good and bad rows
    good_rows = []
    bad_rows = []
    empty_files = []
    
    # Track columns for consistency
    column_sets = set()
    
    # Process all files
    for file in all_files:
        try:
            print(f"\nProcessing {os.path.basename(file)}...")
            
            # Check if file is empty
            if os.path.getsize(file) <= 5:  # Practically empty
                print(f"  - SKIPPING: File is empty")
                empty_files.append(os.path.basename(file))
                continue
                
            # Try different encodings
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
            success = False
            
            for encoding in encodings_to_try:
                try:
                    # Read the file with specific encoding
                    print(f"  Trying to read with {encoding} encoding...")
                    df = pd.read_csv(file, encoding=encoding, on_bad_lines='warn')
                    
                    # If we got here, the encoding worked
                    success = True
                    print(f"  Successfully read with {encoding} encoding")
                    break
                    
                except Exception as e:
                    print(f"  Failed with {encoding} encoding: {str(e)}")
            
            if not success:
                print(f"  ERROR: Could not read file with any encoding")
                error_df = pd.DataFrame({
                    'video_id': ['ERROR'],
                    'source_file': [os.path.basename(file)],
                    'error': ["Encoding error - could not read file"]
                })
                bad_rows.append(error_df)
                continue
                
            # Check if file has only headers but no data
            if len(df) == 0:
                print(f"  - SKIPPING: File has headers but no data rows")
                empty_files.append(os.path.basename(file))
                continue
                
            print(f"  - Read {len(df)} rows")
            
            # Check column consistency
            file_columns = set(df.columns)
            column_sets.add(tuple(sorted(file_columns)))
            
            # Expected columns
            expected = set(expected_columns[file_type])
            
            # Check for missing or extra columns
            missing_cols = expected - file_columns
            extra_cols = file_columns - expected
            
            if missing_cols:
                print(f"  - WARNING: Missing columns: {missing_cols}")
            if extra_cols:
                print(f"  - INFO: Extra columns found: {extra_cols}")
                
            # Split into good and bad rows
            # First check if we have the columns we need to check
            columns_to_check = [col for col in expected_columns[file_type] if col in df.columns]
            
            if len(columns_to_check) == 0:
                print(f"  - ERROR: No valid columns to check")
                # Add the whole file to bad rows with source info
                df['source_file'] = os.path.basename(file)
                df['error'] = "No valid columns"
                bad_rows.append(df)
                continue
                
            # Good rows have values for all expected columns that exist in this file
            good_df = df.dropna(subset=columns_to_check)
            bad_df = df[~df.index.isin(good_df.index)]
            
            # Add file source info for tracing
            source_file = os.path.basename(file)
            good_df['source_file'] = source_file
            bad_df['source_file'] = source_file
            
            # Add good and bad rows to respective lists
            print(f"  - Good rows: {len(good_df)}")
            good_rows.append(good_df)
            
            if len(bad_df) > 0:
                print(f"  - Bad rows: {len(bad_df)}")
                bad_rows.append(bad_df)
                
        except Exception as e:
            print(f"  - ERROR processing {os.path.basename(file)}: {str(e)}")
            # Track the error
            try:
                # Try to read at least the header for error reporting
                header_df = pd.read_csv(file, nrows=0)
                error_df = pd.DataFrame(columns=header_df.columns)
                error_df.loc[0] = ['ERROR'] * len(error_df.columns)
                error_df['source_file'] = os.path.basename(file)
                error_df['error'] = str(e)
                bad_rows.append(error_df)
            except:
                # If even that fails, create minimal error record
                error_df = pd.DataFrame({
                    'video_id': ['ERROR'],
                    'source_file': [os.path.basename(file)],
                    'error': [str(e)]
                })
                bad_rows.append(error_df)
    
    # Report empty files
    if empty_files:
        print(f"\nFound {len(empty_files)} empty or header-only files:")
        for file in empty_files:
            print(f"  - {file}")
    
    # Report column consistency
    if len(column_sets) > 1:
        print(f"\nWARNING: Found {len(column_sets)} different column sets across files!")
        for i, cols in enumerate(column_sets):
            print(f"  Set {i+1}: {cols}")
    else:
        if column_sets:
            print(f"\nAll files have consistent columns: {list(column_sets)[0]}")
        else:
            print("\nNo column information available (all files empty or invalid)")
    
    # Combine good rows
    if good_rows:
        good_combined = pd.concat(good_rows, ignore_index=True)
        
        # Remove duplicates
        if file_type == 'main':
            # For main files, deduplicate on video_id
            if 'video_id' in good_combined.columns:
                before_dedup = len(good_combined)
                good_combined = good_combined.drop_duplicates(subset=['video_id'])
                after_dedup = len(good_combined)
                print(f"Removed {before_dedup - after_dedup} duplicate videos")
        elif file_type == 'patterns':
            # For patterns files, deduplicate on video_id and pattern
            if 'video_id' in good_combined.columns and 'pattern' in good_combined.columns:
                before_dedup = len(good_combined)
                good_combined = good_combined.drop_duplicates(subset=['video_id', 'pattern'])
                after_dedup = len(good_combined)
                print(f"Removed {before_dedup - after_dedup} duplicate patterns")
        elif file_type == 'mental_health':
            # For mental health files, deduplicate on video_id and category
            if 'video_id' in good_combined.columns and 'category' in good_combined.columns:
                before_dedup = len(good_combined)
                good_combined = good_combined.drop_duplicates(subset=['video_id', 'category'])
                after_dedup = len(good_combined)
                print(f"Removed {before_dedup - after_dedup} duplicate mental health entries")
        elif file_type == 'engagement':
            # For engagement files, deduplicate on video_id
            if 'video_id' in good_combined.columns:
                before_dedup = len(good_combined)
                good_combined = good_combined.drop_duplicates(subset=['video_id'])
                after_dedup = len(good_combined)
                print(f"Removed {before_dedup - after_dedup} duplicate engagement entries")
        
        # Sort by video_id numerically if available
        if 'video_id' in good_combined.columns:
            # Convert video_id to numeric for proper sorting
            try:
                # First try to convert directly to numeric
                good_combined['video_id'] = pd.to_numeric(good_combined['video_id'], errors='coerce')
                print("Converted video_id to numeric for proper sorting")
            except:
                # If direct conversion fails, make sure we're sorting numerically
                try:
                    good_combined['sort_id'] = pd.to_numeric(good_combined['video_id'], errors='coerce')
                    good_combined = good_combined.sort_values('sort_id')
                    good_combined = good_combined.drop('sort_id', axis=1)
                    print("Used temporary numeric column for sorting video_id")
                except Exception as e:
                    print(f"Warning: Could not convert video_id to numeric: {e}")
                    # Fall back to regular string sorting
                    good_combined = good_combined.sort_values('video_id')
                    print("Falling back to regular string sorting")
            
            # Do the actual sorting
            good_combined = good_combined.sort_values('video_id')
            print("Sorted data by video_id in ascending order")
        
        # Save the combined file
        good_file = os.path.join(output_dir, f'youtube_analysis_COMBINED_{file_type}.csv')
        good_combined.to_csv(good_file, index=False)
        print(f"Combined valid data saved to:\n{good_file}")
    else:
        good_file = None
        print("No valid rows found to combine")
    
    # Combine bad rows
    if bad_rows:
        bad_combined = pd.concat(bad_rows, ignore_index=True)
        bad_file = os.path.join(output_dir, f'youtube_analysis_INVALID_{file_type}.csv')
        bad_combined.to_csv(bad_file, index=False)
        print(f"Combined invalid data saved to:\n{bad_file}")
    else:
        bad_file = None
        print("No invalid rows found")
    
    return good_file, bad_file

def main():
    # Path to the output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    
    print("=" * 80)
    print(f"YouTube Analysis File Validator and Combiner")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Validate and combine different types of files
    file_types = ['main', 'patterns', 'mental_health', 'engagement']
    good_files = {}
    bad_files = {}
    
    for file_type in file_types:
        good_files[file_type], bad_files[file_type] = validate_and_combine_files(output_dir, file_type)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF VALIDATION AND COMBINATION:")
    
    print("\nValid combined files:")
    for file_type, file_path in good_files.items():
        if file_path:
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"- {file_type}: {os.path.basename(file_path)} ({file_size:.1f} KB)")
        else:
            print(f"- {file_type}: No valid data found")
    
    print("\nInvalid data files:")
    has_invalid = False
    for file_type, file_path in bad_files.items():
        if file_path:
            has_invalid = True
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"- {file_type}: {os.path.basename(file_path)} ({file_size:.1f} KB)")
    
    if not has_invalid:
        print("No invalid data found in any files! All data was combined successfully.")
    
    print("\nValidation and combination complete!")
    print("=" * 80)

if __name__ == "__main__":
    main() 