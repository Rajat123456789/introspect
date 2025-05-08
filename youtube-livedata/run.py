import os
import subprocess
import sys
from pathlib import Path

def print_header(message):
    """Print a formatted header message"""
    border = "=" * (len(message) + 4)
    print(f"\n{border}")
    print(f"| {message} |")
    print(f"{border}\n")

def run_script(script_name):
    """Run a Python script and return its exit code"""
    print_header(f"Running {script_name}")
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return e.returncode
    except Exception as e:
        print(f"Unexpected error running {script_name}: {e}")
        return 1

def main():
    print_header("YouTube Watch History and Transcript Fetcher")
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Define paths to the scripts
    history_script = script_dir / "fetch_history.py"
    transcript_script = script_dir / "fetch_transcripts.py"
    
    # Check if scripts exist
    if not history_script.exists():
        print(f"Error: {history_script} not found.")
        return 1
    
    if not transcript_script.exists():
        print(f"Error: {transcript_script} not found.")
        return 1
    
    # Run scripts
    history_result = run_script(history_script)
    
    if history_result != 0:
        print("History fetching failed or was not completed. Stopping.")
        return history_result
    
    continue_prompt = input("\nDo you want to continue with fetching transcripts? (y/n): ")
    
    if continue_prompt.lower() != 'y':
        print("Stopping after history fetch as requested.")
        return 0
    
    transcript_result = run_script(transcript_script)
    
    if transcript_result != 0:
        print("Transcript fetching failed.")
        return transcript_result
    
    print_header("All processing completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 