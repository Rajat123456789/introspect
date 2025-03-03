import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import pandas as pd
import numpy as np
from transformers import pipeline, CLIPProcessor, CLIPModel
from textblob import TextBlob
import spacy
import logging
from collections import defaultdict
from tqdm import tqdm
import time
import sys
from datetime import datetime
import re
import json
from multiprocessing import Pool
import argparse
import gc

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging to both file and terminal with timestamps
class TeeLogger:
    def __init__(self):
        self.terminal = sys.stdout
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(OUTPUT_DIR, f'youtube_analysis_{timestamp}.log')
        self.log = open(log_file, 'a', encoding='utf-8')
        self.log.write("\n" + "="*80 + "\n")
        self.log.write(f"Analysis started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write("="*80 + "\n\n")

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        except UnicodeEncodeError:
            # Fall back to ASCII if Unicode fails
            clean_message = message.encode('ascii', 'replace').decode()
            self.terminal.write(clean_message)
            self.log.write(clean_message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# At the top of your script
gc.enable()

class YouTubeContentAnalyzer:
    def __init__(self):
        print("\nInitializing YouTube Content Analyzer...")
        
        try:
            # Use a real music classification model
            self.music_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",  # Using the same model for music classification
                framework="pt",
                device=-1
            )
            print("[OK] Music classifier loaded")
            
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                framework="pt",
                device=-1
            )
            print("[OK] Sentiment analyzer loaded")
            
            self.nlp = spacy.load("en_core_web_sm")
            print("[OK] SpaCy model loaded\n")
            
            # Add music-specific categories
            self.music_categories = {
                'mood': [
                    'happy', 'sad', 'energetic', 'calm', 'aggressive', 
                    'relaxed', 'dark', 'atmospheric'
                ],
                'genre': [
                    'electronic', 'rock', 'pop', 'hip hop', 'classical',
                    'jazz', 'metal', 'folk', 'ambient', 'indie'
                ],
                'characteristics': [
                    'instrumental', 'vocal', 'fast', 'slow', 'melodic',
                    'rhythmic', 'acoustic', 'synthetic'
                ],
                'context': [
                    'party', 'study', 'workout', 'sleep', 'meditation',
                    'dance', 'focus', 'background'
                ]
            }
            
            # Add CLIP model for advanced tagging
            print("Loading CLIP model for advanced tagging...")
            self.clip_model = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("[OK] CLIP model loaded")
            
        except Exception as e:
            print(f"[ERROR] Error loading models: {str(e)}")
            raise

        # Initialize keyword lists
        self.mental_health_keywords = {
            'anxiety': ['anxiety', 'stress', 'worried', 'panic'],
            'depression': ['depression', 'sad', 'lonely', 'hopeless'],
            'addiction': ['addiction', 'cant stop', 'binge', 'compulsive'],
            'motivation': ['motivation', 'inspire', 'goals', 'success'],
            'self_improvement': ['growth', 'learning', 'improve', 'better'],
            'mindfulness': ['mindful', 'meditation', 'peace', 'calm']
        }
        
        self.content_categories = [
            "educational",
            "entertainment",
            "gaming",
            "music",
            "self_help",
            "news",
            "technology"
        ]

    def analyze_youtube_history(self, csv_file, start_from=None, num_processes=4, max_runs=None):
        """Analyze YouTube history with resume capability and run limit"""
        print(f"\nReading data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Apply resume point if specified
        if start_from:
            df = df[start_from:]
            print(f"Resuming from row {start_from}")
        
        # Apply max_runs limit if specified (after resume point)
        if max_runs:
            df = df.head(max_runs)
            print(f"Test mode: Limited to {max_runs} videos from resume point")
        
        total_videos = len(df)
        print(f"Will process {total_videos} videos")
        
        # Split into chunks
        chunk_size = max(1, total_videos // num_processes)
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        print(f"Split into {len(chunks)} chunks of ~{chunk_size} videos each")
        
        # Create timestamp for output files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_output_file = os.path.join(OUTPUT_DIR, f'youtube_analysis_{timestamp}')
        
        # Save progress file
        progress_file = os.path.join(OUTPUT_DIR, 'analysis_progress.json')
        
        # Initialize summary stats
        summary_stats = {
            'total_processed': 0,
            'errors': 0,
            'content_types': defaultdict(int),
            'mental_health_mentions': defaultdict(int),
            'sentiment_distribution': defaultdict(int),
            'start_time': time.time()
        }
        
        start_time = time.time()
        processed_total = 0
        
        # Initialize CSV files with headers
        self._initialize_csv_files(base_output_file)
        
        with Pool(processes=num_processes) as pool:
            # Process chunks in parallel
            for i, chunk_stats in enumerate(pool.imap_unordered(self._process_chunk, chunks)):
                print(f"\nProcessing chunk {i+1}/{len(chunks)}")
                
                # Update progress
                processed_total += chunk_size  # Approximate for now
                elapsed = time.time() - start_time
                rate = processed_total / elapsed
                
                # Update summary stats
                for content_type, count in chunk_stats['content_types'].items():
                    summary_stats['content_types'][content_type] += count
                for sentiment, count in chunk_stats['sentiment_distribution'].items():
                    summary_stats['sentiment_distribution'][sentiment] += count
                for topic, count in chunk_stats['mental_health_mentions'].items():
                    summary_stats['mental_health_mentions'][topic] += count
                
                # Save progress
                self._save_progress(progress_file, processed_total)
                
                # Print progress
                print(f"\n[PROGRESS] Processed {processed_total}/{total_videos} videos")
                print(f"Processing rate: {rate:.2f} videos/second")
                print(f"Estimated time remaining: {((total_videos - processed_total) / rate) / 60:.1f} minutes")
                print(f"Content types so far: {dict(chunk_stats['content_types'])}")
                print(f"Mental health mentions: {dict(chunk_stats['mental_health_mentions'])}")
                print("-" * 80)
        
        print(f"\nAll results saved to:")
        print(f"- Main analysis: {base_output_file}_main.csv")
        print(f"- Mental health data: {base_output_file}_mental_health.csv")
        print(f"- Pattern analysis: {base_output_file}_patterns.csv")
        print(f"- Engagement metrics: {base_output_file}_engagement.csv")
        
        # Generate summary log
        summary_stats['total_processed'] = processed_total
        self._write_summary_log(summary_stats, f'{base_output_file}_summary.log', total_videos)
        
        # Clear progress file after successful completion
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        # Print final statistics
        elapsed_total = time.time() - start_time
        print(f"\nAnalysis Complete!")
        print(f"Total videos processed: {processed_total}/{total_videos}")
        print(f"Total time: {elapsed_total:.2f} seconds")
        print(f"Average processing rate: {processed_total/elapsed_total:.2f} videos/second")
        
        # Return a summary dictionary instead of results DataFrame
        return {
            'total_processed': processed_total,
            'total_time': elapsed_total,
            'output_files': {
                'main': f'{base_output_file}_main.csv',
                'mental_health': f'{base_output_file}_mental_health.csv',
                'patterns': f'{base_output_file}_patterns.csv',
                'engagement': f'{base_output_file}_engagement.csv'
            },
            'processing_rate': processed_total/elapsed_total
        }

    def _process_chunk(self, chunk_df, max_runs=None):
        """Process a chunk of videos with immediate saving and preserved row indices"""
        chunk_stats = {
            'errors': 0,
            'content_types': defaultdict(int),
            'mental_health_mentions': defaultdict(int),
            'sentiment_distribution': defaultdict(int)
        }
        
        # Limit the number of videos if max_runs is specified
        if max_runs:
            chunk_df = chunk_df.head(max_runs)
            print(f"\nLimiting processing to {max_runs} videos for testing")
        
        total_in_chunk = len(chunk_df)
        
        # Ensure we have a timestamp for file naming
        if not hasattr(self, 'output_timestamp'):
            self.output_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_output_file = os.path.join(OUTPUT_DIR, f'youtube_analysis_{self.output_timestamp}')
        
        print(f"\nProcessing chunk of {total_in_chunk} videos...")
        
        for idx, (original_idx, row) in enumerate(chunk_df.iterrows()):
            try:
                title = row['Title']
                print(f"\n[{idx + 1}/{total_in_chunk}] Analyzing: {title[:100]}")
                
                # Perform analysis
                analysis = self.analyze_title(title)
                
                # Save immediately to all CSV files
                try:
                    # Main data - using original_idx instead of idx
                    pd.DataFrame([{
                        'video_id': original_idx,  # Use original index from dataset
                        'title': title,
                        'watched_at': row['Watched At'],
                        'primary_category': analysis['video_type'].get('primary_category', 'unknown'),
                        'detailed_type': analysis['video_type'].get('detailed_type', ''),
                        'sentiment': analysis['sentiment'].get('label', 'NEUTRAL'),
                        'sentiment_score': analysis['sentiment'].get('score', 0.0),
                        'primary_format': analysis.get('content_analysis', {}).get('format', {}).get('primary', 'unknown'),
                        'primary_purpose': analysis.get('content_analysis', {}).get('purpose', {}).get('primary', 'unknown'),
                        'style': ','.join(analysis.get('content_analysis', {}).get('style', {}).keys()),
                        'confidence': analysis['video_type'].get('confidence', 0.0)
                    }]).to_csv(f'{base_output_file}_main.csv', 
                             mode='a', header=False, 
                             index=False, encoding='utf-8')
                    
                    # Mental health data - using original_idx
                    mental_health_data = []
                    for category, score in analysis.get('mental_health_tags', {}).items():
                        mental_health_data.append({
                            'video_id': original_idx,  # Use original index
                            'category': category,
                            'score': score,
                            'timestamp': row['Watched At'],
                            'sentiment': analysis['sentiment'].get('label', 'NEUTRAL'),
                            'sentiment_score': analysis['sentiment'].get('score', 0.0)
                        })
                    if mental_health_data:
                        pd.DataFrame(mental_health_data).to_csv(
                            f'{base_output_file}_mental_health.csv',
                            mode='a', header=False,
                            index=False, encoding='utf-8'
                        )
                    
                    # Pattern data - using original_idx
                    pattern_data = []
                    for pattern_type, patterns in analysis.get('patterns', {}).items():
                        for pattern in patterns:
                            pattern_data.append({
                                'video_id': original_idx,  # Use original index
                                'pattern_type': pattern_type,
                                'pattern': pattern,
                                'timestamp': row['Watched At'],
                                'category': analysis['video_type'].get('primary_category', 'unknown')
                            })
                    if pattern_data:
                        pd.DataFrame(pattern_data).to_csv(
                            f'{base_output_file}_patterns.csv',
                            mode='a', header=False,
                            index=False, encoding='utf-8'
                        )
                    
                    # Engagement data - using original_idx
                    pd.DataFrame([{
                        'video_id': original_idx,  # Use original index
                        'timestamp': row['Watched At'],
                        'content_type': analysis['video_type'].get('primary_category', 'unknown'),
                        'audience_engagement': ','.join(analysis.get('advanced_tags', {}).get('audience_engagement', {}).keys()),
                        'production_quality': ','.join(analysis.get('advanced_tags', {}).get('production_quality', {}).keys()),
                        'content_format': ','.join(analysis.get('advanced_tags', {}).get('content_format', {}).keys()),
                        'content_purpose': ','.join(analysis.get('advanced_tags', {}).get('content_purpose', {}).keys())
                    }]).to_csv(f'{base_output_file}_engagement.csv',
                              mode='a', header=False,
                              index=False, encoding='utf-8')
                    
                    print(f"Saved analysis for video {idx + 1} (ID: {original_idx})")
                    
                    # Update statistics
                    chunk_stats['content_types'][analysis['video_type']['primary_category']] += 1
                    chunk_stats['sentiment_distribution'][analysis['sentiment']['label']] += 1
                    for mh_tag in analysis.get('mental_health_tags', {}).keys():
                        chunk_stats['mental_health_mentions'][mh_tag] += 1
                    
                except Exception as e:
                    print(f"[ERROR] Failed to save results for video {idx + 1} (ID: {original_idx}): {str(e)}")
                    chunk_stats['errors'] += 1
                    continue
                
                # Force garbage collection after every 10 videos
                if idx % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"[ERROR] Failed to analyze video {idx + 1} (ID: {original_idx}): {str(e)}")
                chunk_stats['errors'] += 1
                continue
            
            # Print progress every 10 videos
            if (idx + 1) % 10 == 0:
                print(f"\nProgress: {idx + 1}/{total_in_chunk} videos processed")
                print(f"Content types so far: {dict(chunk_stats['content_types'])}")
                print(f"Errors: {chunk_stats['errors']}")
        
        return chunk_stats

    def _save_progress(self, progress_file, last_processed):
        """Save processing progress"""
        with open(progress_file, 'w') as f:
            json.dump({
                'last_processed_row': last_processed,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f)

    def _initialize_csv_files(self, base_output_file):
        """Initialize CSV files with headers"""
        # Main CSV headers
        pd.DataFrame(columns=[
            'video_id', 'title', 'watched_at', 'primary_category',
            'detailed_type', 'sentiment', 'sentiment_score',
            'primary_format', 'primary_purpose', 'style', 'confidence'
        ]).to_csv(f'{base_output_file}_main.csv', index=False, encoding='utf-8')
        
        # Mental health CSV headers
        pd.DataFrame(columns=[
            'video_id', 'category', 'score', 'timestamp',
            'sentiment', 'sentiment_score'
        ]).to_csv(f'{base_output_file}_mental_health.csv', index=False, encoding='utf-8')
        
        # Patterns CSV headers
        pd.DataFrame(columns=[
            'video_id', 'pattern_type', 'pattern',
            'timestamp', 'category'
        ]).to_csv(f'{base_output_file}_patterns.csv', index=False, encoding='utf-8')
        
        # Engagement CSV headers
        pd.DataFrame(columns=[
            'video_id', 'timestamp', 'content_type',
            'audience_engagement', 'production_quality',
            'content_format', 'content_purpose'
        ]).to_csv(f'{base_output_file}_engagement.csv', index=False, encoding='utf-8')

    def _save_chunk_results(self, chunk_results, base_output_file):
        """Save chunk results to CSV files incrementally"""
        # Prepare data for different files
        main_data = []
        mental_health_data = []
        pattern_data = []
        engagement_data = []
        
        print(f"\nSaving chunk of {len(chunk_results)} results...")
        
        for result in chunk_results:
            analysis = result['analysis']
            
            # Main data
            main_data.append({
                'video_id': result['video_id'],
                'title': result['title'],
                'watched_at': result['watched_at'],
                'primary_category': analysis['video_type']['primary_category'],
                'detailed_type': analysis['video_type'].get('detailed_type', ''),
                'sentiment': analysis['sentiment']['label'],
                'sentiment_score': analysis['sentiment']['score'],
                'primary_format': analysis['content_analysis']['format']['primary'],
                'primary_purpose': analysis['content_analysis']['purpose']['primary'],
                'style': ','.join(analysis['content_analysis'].get('style', {}).keys()),
                'confidence': analysis['video_type'].get('confidence', 0.0)
            })
            
            # Mental health data
            for category, score in analysis['mental_health_tags'].items():
                mental_health_data.append({
                    'video_id': result['video_id'],
                    'category': category,
                    'score': score,
                    'timestamp': result['watched_at'],
                    'sentiment': analysis['sentiment']['label'],
                    'sentiment_score': analysis['sentiment']['score']
                })
            
            # Pattern data
            for pattern_type, patterns in analysis['patterns'].items():
                for pattern in patterns:
                    pattern_data.append({
                        'video_id': result['video_id'],
                        'pattern_type': pattern_type,
                        'pattern': pattern,
                        'timestamp': result['watched_at'],
                        'category': analysis['video_type']['primary_category']
                    })
            
            # Engagement data
            engagement_data.append({
                'video_id': result['video_id'],
                'timestamp': result['watched_at'],
                'content_type': analysis['video_type']['primary_category'],
                'audience_engagement': ','.join(analysis['advanced_tags'].get('audience_engagement', {}).keys()),
                'production_quality': ','.join(analysis['advanced_tags'].get('production_quality', {}).keys()),
                'content_format': ','.join(analysis['advanced_tags'].get('content_format', {}).keys()),
                'content_purpose': ','.join(analysis['advanced_tags'].get('content_purpose', {}).keys())
            })
        
        # Append to files with verification
        if main_data:
            main_file = f'{base_output_file}_main.csv'
            try:
                # Verify file exists
                if not os.path.exists(main_file):
                    print(f"Warning: Main CSV file not found, creating new file: {main_file}")
                    self._initialize_csv_files(base_output_file)
                
                # Save data
                pd.DataFrame(main_data).to_csv(main_file, 
                                             mode='a', header=False, 
                                             index=False, encoding='utf-8')
                
                # Verify write
                file_size = os.path.getsize(main_file)
                print(f"Added {len(main_data)} rows to main CSV (file size: {file_size:,} bytes)")
                
            except Exception as e:
                print(f"Error writing to main CSV: {str(e)}")
        
        if mental_health_data:
            mh_file = f'{base_output_file}_mental_health.csv'
            try:
                pd.DataFrame(mental_health_data).to_csv(mh_file, 
                                                      mode='a', header=False, 
                                                      index=False, encoding='utf-8')
                file_size = os.path.getsize(mh_file)
                print(f"Added {len(mental_health_data)} rows to mental health CSV (file size: {file_size:,} bytes)")
            except Exception as e:
                print(f"Error writing to mental health CSV: {str(e)}")
        
        if pattern_data:
            p_file = f'{base_output_file}_patterns.csv'
            try:
                pd.DataFrame(pattern_data).to_csv(p_file, 
                                                mode='a', header=False, 
                                                index=False, encoding='utf-8')
                file_size = os.path.getsize(p_file)
                print(f"Added {len(pattern_data)} rows to patterns CSV (file size: {file_size:,} bytes)")
            except Exception as e:
                print(f"Error writing to patterns CSV: {str(e)}")
        
        if engagement_data:
            e_file = f'{base_output_file}_engagement.csv'
            try:
                pd.DataFrame(engagement_data).to_csv(e_file, 
                                                       mode='a', header=False, 
                                                       index=False, encoding='utf-8')
                file_size = os.path.getsize(e_file)
                print(f"Added {len(engagement_data)} rows to engagement CSV (file size: {file_size:,} bytes)")
            except Exception as e:
                print(f"Error writing to engagement CSV: {str(e)}")
        
        # Final verification
        print("\nVerifying files after chunk save:")
        for file_type in ['main', 'mental_health', 'patterns', 'engagement']:
            file_path = f'{base_output_file}_{file_type}.csv'
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"- {file_type}.csv: {size:,} bytes")
            else:
                print(f"Warning: {file_type}.csv not found!")
        
        print("Chunk save complete")

    def analyze_title(self, title, description=None, category=None, tags=None):
        """Comprehensive analysis of video title and metadata"""
        # Get all analyses with enhanced context
        video_type = self._detect_video_type(title)
        advanced_tags = self._get_advanced_tags(title)
        patterns = self._analyze_patterns(title)
        sentiment = self._get_sentiment(title)
        mental_health = self._get_mental_health_tags(
            title, 
            description=description,
            category=category,
            tags=tags
        )
        topics = self._extract_topics(title)
        
        # Get music-specific analysis if applicable
        if video_type['primary_category'] == 'Music':
            content_analysis = self._analyze_music_content(title)
        else:
            content_analysis = self._analyze_general_content(title)
        
        # Print sentiment and mental health analysis more prominently
        print("\nSentiment & Mental Health Analysis:")
        print(f"  Sentiment: {sentiment['label']} (confidence: {sentiment['score']:.2f})")
        if mental_health:
            print("  Mental Health Indicators:")
            for category, score in mental_health.items():
                print(f"    - {category}: {score}")
        else:
            print("  No mental health indicators found")
        
        # Print video type
        print(f"\nContent Type:")
        print(f"  Primary: {video_type['primary_category']} (confidence: {video_type['confidence']:.2f})")
        if 'detailed_type' in video_type:
            print(f"  Subtype: {video_type['detailed_type']} (confidence: {video_type['detailed_confidence']:.2f})")
        
        # Print advanced tags if available
        if advanced_tags:
            print("\nAdvanced Content Analysis:")
            for category, tags in advanced_tags.items():
                if tags:
                    print(f"  {category.replace('_', ' ').title()}:")
                    for tag, score in tags.items():
                        print(f"    - {tag.replace('_', ' ')} ({score:.2f})")
        
        # Print patterns if found
        if patterns:
            print("\nContent Patterns:")
            for category, matched_patterns in patterns.items():
                if matched_patterns:
                    print(f"  {category.title()}: {', '.join(matched_patterns)}")
        
        # Print content analysis based on type
        if video_type['primary_category'] == 'Music':
            print("\nMusic Content:")
            print(f"  Genre: {content_analysis['genre']['primary']}")
            print(f"  Mood: {content_analysis['mood']['primary']}")
            print(f"  Context: {content_analysis['context']['primary']}")
        else:
            print("\nContent Analysis:")
            print(f"  Format: {content_analysis['format']['primary']}")
            print(f"  Purpose: {content_analysis['purpose']['primary']}")
            if content_analysis['style']:
                print(f"  Style: {', '.join(content_analysis['style'].keys())}")
            if content_analysis['audience']:
                print(f"  Target Audience: {', '.join(content_analysis['audience'].keys())}")
        
        print("-" * 80)
        
        # Free some memory
        gc.collect()
        
        result = {
            "video_type": video_type,
            "advanced_tags": advanced_tags, 
            "patterns": patterns,
            "sentiment": sentiment,
            "mental_health_tags": mental_health,
            "topics": topics,
            "content_analysis": content_analysis
        }
        
        return result

    def _detect_video_type(self, title):
        """Detect video type using zero-shot classification with YouTube categories"""
        try:
            # Standard YouTube categories
            youtube_categories = [
                "Music",
                "Gaming",
                "Education",
                "Entertainment",
                "Sports",
                "News",
                "Technology",
                "Comedy",
                "Vlogs",
                "Tutorial",
                "Reviews",
                "Podcast",
                "Shorts"
            ]

            # First pass: Broad category classification
            results = self.music_classifier(
                title,
                youtube_categories,
                multi_label=True
            )

            # Get primary category and confidence
            primary_category = results['labels'][0]
            confidence = results['scores'][0]

            # Get detailed classification based on primary category
            detailed_categories = {
                "Music": [
                    "Official Music Video",
                    "Live Performance",
                    "Cover Song",
                    "Lyric Video",
                    "Music Mix",
                    "Concert Recording",
                    "Music Review",
                    "Behind the Scenes"
                ],
                "Gaming": [
                    "Gameplay",
                    "Tutorial",
                    "Review",
                    "Esports",
                    "Walkthrough",
                    "Gaming News",
                    "Stream Highlights"
                ],
                "Education": [
                    "Tutorial",
                    "Lecture",
                    "How-to Guide",
                    "Educational Series",
                    "Documentary",
                    "Course Material"
                ]
                # Add more detailed categories as needed
            }

            # If we have detailed categories for the primary category, do a second pass
            if primary_category in detailed_categories:
                detailed_results = self.music_classifier(
                    title,
                    detailed_categories[primary_category],
                    multi_label=True
                )
                
                return {
                    'primary_category': primary_category,
                    'confidence': confidence,
                    'detailed_type': detailed_results['labels'][0],
                    'detailed_confidence': detailed_results['scores'][0],
                    'all_categories': dict(zip(results['labels'], results['scores'])),
                    'all_detailed': dict(zip(detailed_results['labels'], detailed_results['scores']))
                }
            
            return {
                'primary_category': primary_category,
                'confidence': confidence,
                'all_categories': dict(zip(results['labels'], results['scores']))
            }

        except Exception as e:
            print(f"[ERROR] Video type detection failed: {str(e)}")
            return {'primary_category': 'unknown', 'confidence': 0.0}

    def _get_sentiment(self, text):
        """Get sentiment analysis with genre context"""
        try:
            base_sentiment = self.sentiment_analyzer(text)[0]
            
            return base_sentiment
            
        except Exception as e:
            print(f"[ERROR] Error in sentiment analysis: {str(e)}")
            return {'label': 'NEUTRAL', 'score': 0.5, 'context': 'error'}

    def _analyze_music_content(self, text):
        """Analyze music-specific content using specialized model"""
        try:
            # Classify across different music categories
            results = {}
            for category, labels in self.music_categories.items():
                classification = self.music_classifier(
                    text,
                    labels,
                    multi_label=True
                )
                
                # Get scores above threshold
                relevant_labels = {
                    label: score 
                    for label, score in zip(classification['labels'], classification['scores'])
                    if score > 0.2  # Adjust threshold as needed
                }
                
                results[category] = relevant_labels
            
            # Extract primary characteristics
            primary_mood = max(results['mood'].items(), key=lambda x: x[1])[0] if results['mood'] else 'unknown'
            primary_genre = max(results['genre'].items(), key=lambda x: x[1])[0] if results['genre'] else 'unknown'
            
            # Determine music context
            contexts = results['context']
            primary_context = max(contexts.items(), key=lambda x: x[1])[0] if contexts else 'general'
            
            return {
                'mood': {
                    'primary': primary_mood,
                    'distribution': results['mood']
                },
                'genre': {
                    'primary': primary_genre,
                    'distribution': results['genre']
                },
                'characteristics': results['characteristics'],
                'context': {
                    'primary': primary_context,
                    'distribution': contexts
                },
                'confidence': max([
                    max(category.values()) if category else 0 
                    for category in results.values()
                ])
            }
            
        except Exception as e:
            print(f"[ERROR] Music content analysis failed: {str(e)}")
            return {
                'mood': {'primary': 'unknown', 'distribution': {}},
                'genre': {'primary': 'unknown', 'distribution': {}},
                'characteristics': {},
                'context': {'primary': 'unknown', 'distribution': {}},
                'confidence': 0.0
            }

    def _get_mental_health_tags(self, text, description=None, category=None, tags=None):
        """Use zero-shot classification with enhanced context from YouTube metadata"""
        try:
            # Combine all available context
            full_context = [text]  # Start with title
            if description:
                full_context.append(description)
            if tags:
                full_context.append(' '.join(tags))
            
            # Join all context with proper spacing
            analysis_text = ' | '.join(full_context)
            
            # Enhanced categories based on video context
            mental_health_categories = [
                "anxiety and stress management",
                "depression and emotional support",
                "self-improvement and personal growth",
                "mindfulness and mental wellness",
                "emotional health discussion",
                "mental health education",
                "therapy and mental health support",
                "addiction awareness and recovery",
                "sleep and mental wellness",
                "social anxiety and relationships",
                "work-life balance and burnout",
                "trauma awareness and healing",
                "self-care practices",
                "motivation and mental strength",
                "mental health experiences"
            ]

            # Adjust hypothesis template based on content type
            if category and category.lower() in ['education', 'howto', 'science']:
                hypothesis_template = "This video teaches about {}"
            elif category and category.lower() in ['entertainment', 'people', 'blog']:
                hypothesis_template = "This video discusses or shares experiences about {}"
            else:
                hypothesis_template = "This content relates to {}"

            # Use zero-shot classification with enhanced context
            results = self.music_classifier(
                analysis_text,
                mental_health_categories,
                multi_label=True,
                hypothesis_template=hypothesis_template
            )

            # Adjust threshold based on available context
            base_threshold = 0.3
            context_bonus = 0.05 * (len(full_context) - 1)  # More context = lower threshold
            threshold = max(0.2, base_threshold - context_bonus)

            mental_health_tags = {
                label: score 
                for label, score in zip(results['labels'], results['scores'])
                if score > threshold
            }

            # Debug logging
            print("\nDebug: Mental Health Analysis")
            print(f"Analyzing content:")
            print(f"  Title: {text}")
            if description:
                print(f"  Description: {description[:200]}...")
            if tags:
                print(f"  Tags: {', '.join(tags)}")
            if mental_health_tags:
                print(f"Found mental health themes: {mental_health_tags}")
            else:
                print("No significant mental health themes detected")

            return mental_health_tags

        except Exception as e:
            print(f"[ERROR] Mental health classification failed: {str(e)}")
            return {}

    def _classify_content(self, text):
        """Enhanced content type classification"""
        text_lower = text.lower()
        content_types = {}
        
        # Define more specific content indicators
        content_indicators = {
            'music': ['music video', 'audio', 'lyrics', 'song', 'live performance'],
            'gaming': ['gameplay', 'playthrough', 'fortnite', 'gaming', 'game'],
            'educational': ['tutorial', 'how to', 'guide', 'learn', 'explained'],
            'entertainment': ['vlog', 'reaction', 'funny', 'comedy', 'meme'],
            'news': ['news', 'update', 'announcement', 'latest'],
            'technology': ['tech', 'review', 'unboxing', 'programming'],
            'sports': ['sport', 'workout', 'fitness', 'training'],
            'meditation': ['meditation', 'mindfulness', 'relaxation', 'yoga']
        }
        
        # Check for each content type
        for content_type, indicators in content_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                content_types[content_type] = 1.0
        
        return content_types

    def _extract_topics(self, text):
        """Extract key topics from text"""
        doc = self.nlp(text)
        topics = [token.text for token in doc if not token.is_stop]
        return {topic: 1 for topic in topics}

    def _analyze_engagement_patterns(self, text):
        """Analyze engagement patterns in text"""
        # Placeholder for engagement analysis
        return {}

    def _get_advanced_tags(self, title):
        """Use specialized video tagging model"""
        try:
            # Advanced tag categories
            tag_categories = {
                'content_format': [
                    'tutorial', 'review', 'reaction', 'compilation', 'highlights',
                    'commentary', 'analysis', 'walkthrough', 'guide', 'showcase',
                    'stream_highlights', 'podcast', 'interview', 'documentary'
                ],
                'production_quality': [
                    'professional', 'amateur', 'high_production', 'live_recording',
                    'studio_quality', 'raw_footage', 'edited', 'animated'
                ],
                'audience_engagement': [
                    'educational', 'entertaining', 'informative', 'controversial',
                    'clickbait', 'viral', 'trending', 'family_friendly', 'mature'
                ],
                'content_purpose': [
                    'skill_development', 'entertainment', 'news', 'analysis',
                    'social_commentary', 'product_review', 'storytelling',
                    'educational', 'promotional', 'personal_vlog'
                ]
            }
            
            results = {}
            for category, tags in tag_categories.items():
                classification = self.music_classifier(
                    title,
                    tags,
                    multi_label=True
                )
                results[category] = {
                    label: score 
                    for label, score in zip(classification['labels'], classification['scores'])
                    if score > 0.3
                }
            
            return results
        except Exception as e:
            print(f"[ERROR] Advanced tagging failed: {str(e)}")
            return {}

    def _analyze_patterns(self, title):
        """Analyze video patterns and trends"""
        try:
            pattern_indicators = {
                'series': [
                    r'part\s*\d+', r'episode\s*\d+', r'ep\.\s*\d+',
                    r'#\d+', r'\[\d+\]', r'series', r'season'
                ],
                'trending': [
                    r'trend(?:ing)?', r'viral', r'challenge', r'tiktok',
                    r'shorts', r'latest', r'new'
                ],
                'clickbait': [
                    r'you won\'?t believe', r'shocking', r'amazing',
                    r'\(\d+\s*shocking.*\)', r'gone wrong', r'mind blown'
                ],
                'monetization': [
                    r'sponsored', r'ad\b', r'#ad', r'partner',
                    r'affiliate', r'promo'
                ],
                'community': [
                    r'collab', r'ft\.', r'feat', r'featuring',
                    r'w/', r'with', r'vs'
                ]
            }
            
            patterns = defaultdict(list)
            for category, patterns_list in pattern_indicators.items():
                for pattern in patterns_list:
                    if re.search(pattern, title, re.IGNORECASE):
                        patterns[category].append(pattern)
            
            return dict(patterns)
        except Exception as e:
            print(f"[ERROR] Pattern analysis failed: {str(e)}")
            return {}

    def _analyze_general_content(self, text):
        """Analyze general (non-music) content"""
        try:
            # Use zero-shot classification for content analysis
            content_categories = {
                'format': [
                    'tutorial', 'review', 'gameplay', 'vlog', 
                    'reaction', 'commentary', 'guide', 'news'
                ],
                'purpose': [
                    'entertainment', 'education', 'information',
                    'comedy', 'storytelling', 'analysis'
                ],
                'style': [
                    'casual', 'professional', 'funny', 'serious',
                    'dramatic', 'informative', 'personal'
                ],
                'audience': [
                    'gamers', 'students', 'professionals', 'general',
                    'kids', 'teens', 'adults'
                ]
            }
            
            results = {}
            for category, labels in content_categories.items():
                classification = self.music_classifier(
                    text,
                    labels,
                    multi_label=True
                )
                
                # Get scores above threshold
                relevant_labels = {
                    label: score 
                    for label, score in zip(classification['labels'], classification['scores'])
                    if score > 0.2
                }
                
                results[category] = relevant_labels
            
            # Get primary characteristics
            primary_format = max(results['format'].items(), key=lambda x: x[1])[0] if results['format'] else 'unknown'
            primary_purpose = max(results['purpose'].items(), key=lambda x: x[1])[0] if results['purpose'] else 'unknown'
            
            return {
                'format': {
                    'primary': primary_format,
                    'distribution': results['format']
                },
                'purpose': {
                    'primary': primary_purpose,
                    'distribution': results['purpose']
                },
                'style': results['style'],
                'audience': results['audience'],
                'confidence': max([
                    max(category.values()) if category else 0 
                    for category in results.values()
                ])
            }
            
        except Exception as e:
            print(f"[ERROR] General content analysis failed: {str(e)}")
            return {
                'format': {'primary': 'unknown', 'distribution': {}},
                'purpose': {'primary': 'unknown', 'distribution': {}},
                'style': {},
                'audience': {},
                'confidence': 0.0
            }

    def _write_summary_log(self, stats, log_file, total_videos):
        """Write detailed summary log"""
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"YouTube Content Analysis Summary\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Processing Statistics
            f.write("Processing Statistics:\n")
            f.write(f"Total Videos Analyzed: {stats['total_processed']}/{total_videos}\n")
            f.write(f"Successful: {stats['total_processed'] - stats['errors']}\n")
            f.write(f"Errors: {stats['errors']}\n")
            
            elapsed_time = time.time() - stats['start_time']
            f.write(f"Total Processing Time: {elapsed_time:.2f} seconds\n")
            f.write(f"Average Processing Rate: {stats['total_processed']/elapsed_time:.2f} videos/second\n\n")

            # Content Type Distribution
            f.write("Content Type Distribution:\n")
            for content_type, count in stats['content_types'].items():
                percentage = (count / stats['total_processed']) * 100
                f.write(f"  {content_type}: {count} ({percentage:.1f}%)\n")
            f.write("\n")

            # Sentiment Distribution
            f.write("Sentiment Distribution:\n")
            for sentiment, count in stats['sentiment_distribution'].items():
                percentage = (count / stats['total_processed']) * 100
                f.write(f"  {sentiment}: {count} ({percentage:.1f}%)\n")
            f.write("\n")

            # Mental Health Mentions
            f.write("Mental Health Related Content:\n")
            for topic, count in stats['mental_health_mentions'].items():
                percentage = (count / stats['total_processed']) * 100
                f.write(f"  {topic}: {count} ({percentage:.1f}%)\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            print(f"\nSummary log written to: {log_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze YouTube watch history')
    parser.add_argument('--file', type=str, required=True, help='Input CSV file')
    parser.add_argument('--resume', type=int, help='Resume from specific row number')
    parser.add_argument('--processes', type=int, default=4, help='Number of parallel processes')
    parser.add_argument('--max-runs', type=int, help='Maximum number of videos to process')
    
    args = parser.parse_args()
    
    try:
        analyzer = YouTubeContentAnalyzer()
        analyses = analyzer.analyze_youtube_history(
            args.file,
            start_from=args.resume,
            num_processes=args.processes,
            max_runs=args.max_runs
        )
        
    except Exception as e:
        print(f"\n[ERROR] Fatal error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 