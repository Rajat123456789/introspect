import os
import pandas as pd
import glob
import csv
from datetime import datetime

def combine_all_files(output_dir):
    """
    Process and combine all YouTube analysis files, properly handling:
    - Empty/header-only files
    - Encoding issues
    - Extra columns (appending to the last column)
    """
    print("=" * 80)
    print("YouTube Data File Combiner")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
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
    
    file_types = ['main', 'patterns', 'mental_health', 'engagement']
    
    # Process each file type
    for file_type in file_types:
        print("\n" + "=" * 60)
        print(f"PROCESSING {file_type.upper()} FILES")
        print("=" * 60)
        
        # Find all files of this type
        pattern = os.path.join(output_dir, f'youtube_analysis_*_{file_type}.csv')
        all_files = glob.glob(pattern)
        
        if not all_files:
            print(f"No {file_type} files found!")
            continue
        
        print(f"Found {len(all_files)} {file_type} files")
        
        # Initialize combined dataframe with expected columns
        columns = expected_columns[file_type].copy()
        columns.append('source_file')  # Add source tracking
        combined_data = pd.DataFrame(columns=columns)
        
        # Track statistics
        processed_files = 0
        empty_files = []
        problematic_files = []
        total_rows = 0
        
        # Process each file
        for file in all_files:
            filename = os.path.basename(file)
            print(f"\nProcessing: {filename}")
            
            # Check for empty file
            if os.path.getsize(file) <= 5:
                print(f"  SKIPPING: File is empty")
                empty_files.append(filename)
                continue
                
            # Use custom CSV reading to handle extra columns
            rows = []
            try:
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(file, 'r', encoding=encoding, newline='') as csvfile:
                            print(f"  Trying {encoding} encoding...")
                            # Read as CSV
                            reader = csv.reader(csvfile)
                            header = next(reader)  # Get header row
                            
                            # Validate header matches expected columns
                            expected = expected_columns[file_type]
                            if len(header) < len(expected):
                                print(f"  WARNING: Header has fewer columns than expected ({len(header)} vs {len(expected)})")
                                problematic_files.append((filename, "Missing columns"))
                                # Still try to process
                            
                            # Process data rows
                            row_count = 0
                            for row in reader:
                                if not row or all(cell.strip() == '' for cell in row):
                                    continue  # Skip blank rows
                                
                                # Handle if row has more columns than header
                                if len(row) > len(header):
                                    print(f"  Row {row_count+1} has {len(row)} columns (expected {len(header)})")
                                    # Combine extra columns into the last column
                                    extra_content = ','.join(row[len(header):])
                                    new_row = row[:len(header)-1]
                                    # Append extra content to last column
                                    if len(header) > 0:
                                        last_col_content = row[len(header)-1] if len(header)-1 < len(row) else ""
                                        new_row.append(f"{last_col_content} {extra_content}")
                                    row = new_row
                                
                                # Handle if row has fewer columns than header
                                while len(row) < len(expected):
                                    row.append('')  # Pad with empty values
                                
                                # Add source file
                                row.append(filename)
                                rows.append(row)
                                row_count += 1
                            
                            print(f"  Successfully read {row_count} rows with {encoding}")
                            break  # Success, break out of encoding loop
                            
                    except UnicodeDecodeError:
                        print(f"  Failed with {encoding} encoding")
                        continue
                    except Exception as e:
                        print(f"  Error reading file: {str(e)}")
                        problematic_files.append((filename, str(e)))
                        break
            
            except Exception as e:
                print(f"  ERROR: Could not process file: {str(e)}")
                problematic_files.append((filename, str(e)))
                continue
            
            # If no rows were read, mark as empty
            if not rows:
                print(f"  File contains headers but no data rows")
                empty_files.append(filename)
                continue
            
            # Convert to dataframe and add to combined data
            file_df = pd.DataFrame(rows, columns=columns)
            combined_data = pd.concat([combined_data, file_df], ignore_index=True)
            total_rows += len(file_df)
            processed_files += 1
        
        # Report on processing
        print("\n" + "-" * 40)
        print(f"Processed {processed_files} files ({len(all_files) - processed_files} skipped)")
        print(f"Total rows gathered: {total_rows}")
        
        if empty_files:
            print(f"\nSkipped {len(empty_files)} empty files:")
            for f in empty_files[:5]:
                print(f"  - {f}")
            if len(empty_files) > 5:
                print(f"  ...and {len(empty_files) - 5} more")
        
        if problematic_files:
            print(f"\nEncountered issues with {len(problematic_files)} files:")
            for f, error in problematic_files[:5]:
                print(f"  - {f}: {error}")
            if len(problematic_files) > 5:
                print(f"  ...and {len(problematic_files) - 5} more")
        
        # If we have data, process and save it
        if not combined_data.empty:
            print(f"\nCombined data has {len(combined_data)} rows")
            
            # Remove duplicates
            if 'video_id' in combined_data.columns:
                before_dedup = len(combined_data)
                
                if file_type == 'main':
                    combined_data = combined_data.drop_duplicates(subset=['video_id'])
                elif file_type == 'patterns' and 'pattern' in combined_data.columns:
                    combined_data = combined_data.drop_duplicates(subset=['video_id', 'pattern'])
                elif file_type == 'mental_health' and 'category' in combined_data.columns:
                    combined_data = combined_data.drop_duplicates(subset=['video_id', 'category'])
                elif file_type == 'engagement':
                    combined_data = combined_data.drop_duplicates(subset=['video_id'])
                
                after_dedup = len(combined_data)
                print(f"Removed {before_dedup - after_dedup} duplicate entries")
            
            # Sort by video_id if present
            if 'video_id' in combined_data.columns:
                try:
                    # Convert to numeric for proper sorting
                    combined_data['sort_id'] = pd.to_numeric(combined_data['video_id'], errors='coerce')
                    combined_data = combined_data.sort_values('sort_id')
                    combined_data = combined_data.drop('sort_id', axis=1)
                    print("Sorted by video_id numerically")
                except:
                    # Fall back to string sorting
                    combined_data = combined_data.sort_values('video_id')
                    print("Sorted by video_id as string")
            
            # Save combined data
            output_file = os.path.join(output_dir, f'youtube_analysis_COMBINED_{file_type}.csv')
            combined_data.to_csv(output_file, index=False)
            print(f"Combined data saved to: {output_file}")
        else:
            print(f"No valid data found for {file_type}")
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    # Path to the output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    combine_all_files(output_dir) 