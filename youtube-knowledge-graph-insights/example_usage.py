#!/usr/bin/env python3
"""
Example usage of the YouTube viewing pattern analysis.
This file demonstrates different ways to run the analysis.
"""
import os
from datetime import datetime, timedelta
from date_range_analysis import (
    run_date_range_analysis,
    detect_doom_scrolling,
    detect_rabbit_holes,
    detect_addiction_pattern,
    detect_escapism,
    detect_negative_mood,
    detect_unhealthy_comparison
)

def example_full_analysis():
    """Run a complete analysis for the last 14 days"""
    # Set Neo4j connection details
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"  # Change this to your actual password
    
    # Set date range (last 14 days)
    end_date = datetime.now().isoformat()
    start_date = (datetime.now() - timedelta(days=14)).isoformat()
    
    # Set output directory
    output_dir = "example_analysis"
    
    print(f"Running full analysis for {start_date} to {end_date}...")
    
    # Run the analysis
    results = run_date_range_analysis(
        uri=uri,
        user=user,
        password=password,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
    
    print(f"Analysis complete. Results saved to {output_dir}/ directory.")

def example_customize_parameters():
    """Example showing how to customize pattern detection parameters"""
    # Set Neo4j connection details
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"  # Change this to your actual password
    
    # This example shows how to customize the parameters for more or less sensitive detection
    
    # Create a custom function that wraps the standard analysis with custom parameters
    def run_custom_analysis(uri, user, password, start_date, end_date, output_dir):
        """Run analysis with custom pattern detection parameters"""
        # Initialize analyzer and get video data
        from date_range_analysis import run_date_range_analysis
        
        # Import the actual implementation to avoid code repetition
        # This gets the raw dataframe with videos
        analyzer = MentalHealthAnalyzer(uri, user, password)
        
        # Get videos in date range
        query = f"""
        MATCH (u:User)-[:WATCHED]->(v:Video)
        WHERE v.timestamp >= datetime('{start_date}') AND v.timestamp <= datetime('{end_date}')
        RETURN v.id as id, v.title as title, v.category as category, 
               v.description as description, v.timestamp as timestamp
        ORDER BY v.timestamp
        """
        df = analyzer.execute_query(query)
        
        # Apply custom pattern detection
        # More sensitive doom scrolling detection (lower threshold, longer window)
        df = detect_doom_scrolling(df, threshold=10, time_window_hours=3)
        
        # More specific rabbit hole detection
        df = detect_rabbit_holes(
            df, 
            min_sequence=8,          # Require longer sequences
            max_time_gap=timedelta(minutes=15)  # Require videos closer together
        )
        
        # More sensitive addiction detection
        df = detect_addiction_pattern(
            df,
            daily_threshold=10,      # Lower daily threshold
            daily_consecutive_days=3  # Fewer consecutive days required
        )
        
        # Continue with the normal analysis
        # ... other analyses here
        
        return {
            "custom_parameter_results": "Analysis with custom parameters completed",
            "total_videos": len(df),
            "doom_scrolling_count": df["pattern_doom_scrolling"].sum(),
            "rabbit_holes_count": df["pattern_rabbit_holes"].sum(),
            "addiction_count": df["pattern_addiction"].sum()
        }
    
    print("This example demonstrates how to customize pattern detection parameters.")
    print("See the function code for implementation details.")

def example_individual_pattern_detection():
    """Example showing how to use individual pattern detection functions"""
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create some example data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
    sample_data = {
        'timestamp': dates,
        'title': [f'Video about {"travel" if i % 5 == 0 else "cooking" if i % 3 == 0 else "tech"}' for i in range(100)],
        'category': ['Travel' if i % 5 == 0 else 'Food' if i % 3 == 0 else 'Technology' for i in range(100)],
        'description': ['Exploring new places' if i % 5 == 0 else 'How to cook' if i % 3 == 0 else 'Tech review' for i in range(100)]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Sample data shape:", df.shape)
    
    # Apply pattern detection individually
    df = detect_doom_scrolling(df, threshold=20, time_window_hours=1)
    print(f"Doom scrolling count: {df['pattern_doom_scrolling'].sum()}")
    
    df = detect_rabbit_holes(df)
    print(f"Rabbit holes count: {df['pattern_rabbit_holes'].sum()}")
    
    df = detect_negative_mood(df)
    print(f"Negative mood count: {df['pattern_negative_mood'].sum()}")
    
    # You can apply multiple detections and then analyze overlaps
    pattern_columns = [col for col in df.columns if col.startswith('pattern_')]
    if pattern_columns:
        overlap_count = df[df[pattern_columns].sum(axis=1) > 1].shape[0]
        print(f"Videos with multiple patterns: {overlap_count}")

if __name__ == "__main__":
    print("YouTube Viewing Pattern Analysis Examples")
    print("----------------------------------------")
    print("1. Full analysis example (commented out - requires Neo4j)")
    print("2. Custom parameters example (code demonstration)")
    print("3. Individual pattern detection example")
    print()
    
    # Uncomment to run the full analysis (requires Neo4j database)
    # example_full_analysis()
    
    example_customize_parameters()
    print()
    
    example_individual_pattern_detection() 