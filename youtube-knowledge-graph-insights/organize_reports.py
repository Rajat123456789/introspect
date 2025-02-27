import os
import re
import shutil
from collections import defaultdict

def organize_reports():
    """
    Organize analysis report files into timestamp-based folders.
    Files like 'analysis_reports/viewing_patterns_20250226_225748.csv' will be moved to
    'analysis_reports/20250226_225748/viewing_patterns.csv'
    """
    reports_dir = 'analysis_reports'
    
    # Skip if directory doesn't exist
    if not os.path.exists(reports_dir):
        print(f"Directory {reports_dir} does not exist.")
        return
    
    # Get all files in the directory (not subdirectories)
    files = [f for f in os.listdir(reports_dir) 
            if os.path.isfile(os.path.join(reports_dir, f))]
    
    # Group files by timestamp
    timestamp_pattern = r'(.+)_(\d{8}_\d{6})\.(csv|json|png|txt)$'
    timestamp_groups = defaultdict(list)
    
    # First pass: identify timestamp groups
    for file in files:
        match = re.match(timestamp_pattern, file)
        if match:
            base_name = match.group(1)
            timestamp = match.group(2)
            extension = match.group(3)
            
            timestamp_groups[timestamp].append({
                'file': file,
                'base_name': base_name,
                'extension': extension
            })
    
    # Second pass: create directories and move files
    for timestamp, file_infos in timestamp_groups.items():
        # Create timestamp directory if it doesn't exist
        timestamp_dir = os.path.join(reports_dir, timestamp)
        os.makedirs(timestamp_dir, exist_ok=True)
        
        # Move files to the timestamp directory
        for file_info in file_infos:
            original_path = os.path.join(reports_dir, file_info['file'])
            new_filename = f"{file_info['base_name']}.{file_info['extension']}"
            new_path = os.path.join(timestamp_dir, new_filename)
            
            print(f"Moving {file_info['file']} to {timestamp}/{new_filename}")
            shutil.move(original_path, new_path)
    
    print("\nOrganization complete!")
    print(f"Organized {sum(len(files) for files in timestamp_groups.values())} files into {len(timestamp_groups)} timestamp directories.")

if __name__ == "__main__":
    organize_reports() 