import os
import pandas as pd
import glob
import re
from datetime import datetime

def combine_output_files(output_dir, file_type='main'):
    """
    Combine all output files of a specific type into a single consolidated file.
    
    Args:
        output_dir: Directory containing the output files
        file_type: Type of file to combine ('main', 'patterns', 'mental_health', 'engagement')
    
    Returns:
        Path to the combined file
    """
    print(f"\nCombining all '{file_type}' files...")
    
    # Find all files matching the pattern
    pattern = os.path.join(output_dir, f'youtube_analysis_*_{file_type}.csv')
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"No '{file_type}' files found in {output_dir}")
        return None
    
    print(f"Found {len(all_files)} {file_type} files:")
    for file in all_files:
        file_size = os.path.getsize(file) / 1024  # KB
        print(f"  - {os.path.basename(file)} ({file_size:.1f} KB)")
    
    # Read and combine all files
    print(f"Reading and combining files...")
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            print(f"  - Read {len(df)} rows from {os.path.basename(file)}")
            dfs.append(df)
        except Exception as e:
            print(f"  - ERROR reading {os.path.basename(file)}: {str(e)}")
    
    if not dfs:
        print("No valid data found!")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataframe has {len(combined_df)} rows")
    
    # Remove duplicates
    if file_type == 'main':
        # For main files, deduplicate on video_id
        if 'video_id' in combined_df.columns:
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['video_id'])
            after_dedup = len(combined_df)
            print(f"Removed {before_dedup - after_dedup} duplicate videos")
    elif file_type == 'patterns':
        # For patterns files, deduplicate on video_id and pattern
        if 'video_id' in combined_df.columns and 'pattern' in combined_df.columns:
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['video_id', 'pattern'])
            after_dedup = len(combined_df)
            print(f"Removed {before_dedup - after_dedup} duplicate patterns")
    elif file_type == 'mental_health':
        # For mental health files, deduplicate on video_id and category
        if 'video_id' in combined_df.columns and 'category' in combined_df.columns:
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['video_id', 'category'])
            after_dedup = len(combined_df)
            print(f"Removed {before_dedup - after_dedup} duplicate mental health entries")
    elif file_type == 'engagement':
        # For engagement files, deduplicate on video_id
        if 'video_id' in combined_df.columns:
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['video_id'])
            after_dedup = len(combined_df)
            print(f"Removed {before_dedup - after_dedup} duplicate engagement entries")
    
    # Create output filename without timestamp
    output_file = os.path.join(output_dir, f'youtube_analysis_COMBINED_{file_type}.csv')
    
    # Sort by video_id numerically if available
    if 'video_id' in combined_df.columns:
        # Convert video_id to numeric for proper sorting
        try:
            # First try to convert directly to numeric
            combined_df['video_id'] = pd.to_numeric(combined_df['video_id'], errors='coerce')
            print("Converted video_id to numeric for proper sorting")
        except:
            # If direct conversion fails, make sure we're sorting numerically
            # (needed if video_id contains non-numeric characters)
            try:
                combined_df['sort_id'] = pd.to_numeric(combined_df['video_id'], errors='coerce')
                combined_df = combined_df.sort_values('sort_id')
                combined_df = combined_df.drop('sort_id', axis=1)
                print("Used temporary numeric column for sorting video_id")
            except Exception as e:
                print(f"Warning: Could not convert video_id to numeric: {e}")
                # Fall back to regular string sorting
                combined_df = combined_df.sort_values('video_id')
                print("Falling back to regular string sorting")
        
        # Do the actual sorting
        combined_df = combined_df.sort_values('video_id')
        print("Sorted data by video_id in ascending order")
    
    # Save the combined file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to:\n{output_file}")
    
    return output_file

def main():
    # Path to the output directory - make sure this points to where your files are
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    
    # Alternatively, you can hard-code the path if needed:
    # output_dir = 'combining-health-and-music/Scripts/output'
    
    print("=" * 80)
    print(f"YouTube Analysis File Combiner")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Combine different types of files
    file_types = ['main', 'patterns', 'mental_health', 'engagement']
    combined_files = {}
    
    for file_type in file_types:
        combined_files[file_type] = combine_output_files(output_dir, file_type)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary of combined files:")
    for file_type, file_path in combined_files.items():
        if file_path:
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"- {file_type}: {os.path.basename(file_path)} ({file_size:.1f} KB)")
        else:
            print(f"- {file_type}: No files combined")
    
    print("\nFile combination complete!")
    print("=" * 80)

if __name__ == "__main__":
    main() 