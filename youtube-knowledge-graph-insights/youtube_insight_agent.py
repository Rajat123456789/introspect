import json
import pandas as pd
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeInsightAgent:
    """Agent that provides personalized insights based on YouTube viewing and mental health data."""
    
    def __init__(self, insights_path='insights/youtube_mental_health_insights.json', 
                 visualizations_dir='analysis_reports'):
        """Initialize the agent with insights data and visualizations."""
        self.insights = self._load_insights(insights_path)
        self.visualizations_dir = visualizations_dir
        self.recommendations = {
            'content': [],
            'timing': [],
            'patterns_to_follow': [],
            'patterns_to_avoid': [],
            'music_impact': [],
            'sentiment_trajectory': [],
            'unhealthy_trends': []
        }
        # Track available visualizations
        self.available_visualizations = self._find_available_visualizations()
        
    def _load_insights(self, insights_path):
        """Load insights data from JSON."""
        try:
            with open(insights_path, 'r', encoding='utf-8') as f:
                insights = json.load(f)
                logger.info(f"Loaded insights data from {insights_path}")
                return insights
        except Exception as e:
            logger.error(f"Error loading insights data: {str(e)}")
            # Create empty insights structure if file doesn't exist
            return {
                'meta': {
                    'summary': "No insight data available. Please run analysis first.",
                    'total_videos_analyzed': 0,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
    def _find_available_visualizations(self):
        """Find all visualization files in the visualization directory."""
        available_vis = {}
        
        if not os.path.exists(self.visualizations_dir):
            logger.warning(f"Visualizations directory {self.visualizations_dir} does not exist.")
            return available_vis
            
        # Map visualization types to their file patterns
        vis_patterns = {
            'sentiment_trajectory': 'sentiment_trajectory_scatter.png',
            'mental_health_components': 'mental_health_components.png',
            'content_category_impact': 'content_category_impact.png',
            'music_genre_impact': 'music_genre_impact.png',
            'viewing_patterns': 'viewing_day_of_week.png',
            'late_night_viewing': 'late_night_viewing_pattern.png',
            'unhealthy_trends': 'unhealthy_viewing_trends.png',
            'mental_health_forecast': 'mental_health_forecast_simple.png'
        }
        
        # Find all visualization files
        for vis_type, pattern in vis_patterns.items():
            file_path = os.path.join(self.visualizations_dir, pattern)
            if os.path.exists(file_path):
                available_vis[vis_type] = file_path
                logger.info(f"Found visualization: {vis_type} at {file_path}")
                
        # Also check for report files
        report_patterns = {
            'music_impact_report': 'music_impact_report.md',
            'unhealthy_patterns_report': 'unhealthy_patterns_report.md'
        }
        
        for report_type, pattern in report_patterns.items():
            file_path = os.path.join(self.visualizations_dir, pattern)
            if os.path.exists(file_path):
                available_vis[report_type] = file_path
                logger.info(f"Found report: {report_type} at {file_path}")
                
        return available_vis
            
    def analyze_user_viewing_history(self, user_history_df):
        """Analyze a user's viewing history against the insights."""
        logger.info("Analyzing user viewing history...")
        
        # Convert user history to DataFrame if it's not already
        if not isinstance(user_history_df, pd.DataFrame):
            user_history_df = pd.DataFrame(user_history_df)
        
        # 1. Analyze content categories
        self._analyze_content_categories(user_history_df)
        
        # 2. Analyze viewing times
        self._analyze_viewing_times(user_history_df)
        
        # 3. Analyze patterns
        self._analyze_patterns(user_history_df)
        
        # 4. Analyze music impact if visualization exists
        self._analyze_music_impact()
        
        # 5. Analyze sentiment trajectory if visualization exists
        self._analyze_sentiment_trajectory()
        
        # 6. Analyze unhealthy trends if visualization exists
        self._analyze_unhealthy_trends()
        
        # 7. Generate personalized summary
        summary = self._generate_summary()
        
        return summary
        
    def _analyze_content_categories(self, user_history_df):
        """Analyze user's content category preferences against mental health impacts."""
        if 'primary_category' not in user_history_df.columns:
            logger.warning("User history doesn't contain 'primary_category' column. Skipping content analysis.")
            return
            
        # Get user's most watched categories
        user_categories = user_history_df['primary_category'].value_counts().head(5)
        
        # Get positive and negative impact categories from insights
        positive_categories = []
        negative_categories = []
        
        if 'category_correlations' in self.insights:
            for item in self.insights['category_correlations']:
                if item['avg_score'] > 0.6 and item['count'] >= 5:
                    positive_categories.append(item['content_category'])
                elif item['avg_score'] < 0.4 and item['count'] >= 5:
                    negative_categories.append(item['content_category'])
        
        # Check user's categories against positive/negative lists
        for category, count in user_categories.items():
            if category in positive_categories:
                self.recommendations['content'].append({
                    'category': category,
                    'impact': 'positive',
                    'frequency': int(count),
                    'recommendation': 'Continue watching this category as it correlates with positive mental health scores.'
                })
            elif category in negative_categories:
                self.recommendations['content'].append({
                    'category': category,
                    'impact': 'negative',
                    'frequency': int(count),
                    'recommendation': 'Consider reducing this category as it correlates with lower mental health scores.'
                })
            else:
                # Neutral category - no strong correlation
                self.recommendations['content'].append({
                    'category': category,
                    'impact': 'neutral',
                    'frequency': int(count),
                    'recommendation': 'This category shows no strong correlation with mental health indicators.'
                })
    
    def _analyze_viewing_times(self, user_history_df):
        """Analyze user's viewing times against optimal viewing hours."""
        if 'watched_at' not in user_history_df.columns:
            logger.warning("User history doesn't contain 'watched_at' column. Skipping time analysis.")
            return
            
        # Convert timestamp to datetime and extract hour
        user_history_df['timestamp'] = pd.to_datetime(user_history_df['watched_at'])
        user_history_df['hour'] = user_history_df['timestamp'].dt.hour
        
        # Get user's most common viewing hours
        user_hours = user_history_df['hour'].value_counts().head(5)
        
        # Get optimal and suboptimal hours from insights
        optimal_hours = {}
        suboptimal_hours = {}
        
        if 'viewing_time_patterns' in self.insights:
            for item in self.insights['viewing_time_patterns']:
                category = item['mental_health_category']
                hour = int(item['hour_of_day'])
                score = item['avg_score']
                count = item['count']
                
                if count >= 3:  # Only consider hours with enough data
                    if category not in optimal_hours or score > optimal_hours[category]['score']:
                        optimal_hours[category] = {'hour': hour, 'score': score}
                        
                    if category not in suboptimal_hours or score < suboptimal_hours[category]['score']:
                        suboptimal_hours[category] = {'hour': hour, 'score': score}
        
        # Check user's hours against optimal/suboptimal lists
        for hour, count in user_hours.items():
            hour = int(hour)
            
            # Check if this is an optimal hour for any mental health category
            optimal_for = []
            suboptimal_for = []
            
            for category, data in optimal_hours.items():
                if hour == data['hour']:
                    optimal_for.append(category)
                    
            for category, data in suboptimal_hours.items():
                if hour == data['hour']:
                    suboptimal_for.append(category)
            
            if optimal_for:
                self.recommendations['timing'].append({
                    'hour': hour,
                    'frequency': int(count),
                    'impact': 'positive',
                    'affects': optimal_for,
                    'recommendation': f'Continuing to watch at {hour:02d}:00 is beneficial for {", ".join(optimal_for)}.'
                })
            elif suboptimal_for:
                self.recommendations['timing'].append({
                    'hour': hour,
                    'frequency': int(count),
                    'impact': 'negative',
                    'affects': suboptimal_for,
                    'recommendation': f'Consider shifting viewing from {hour:02d}:00 to improve {", ".join(suboptimal_for)}.'
                })
            else:
                self.recommendations['timing'].append({
                    'hour': hour,
                    'frequency': int(count),
                    'impact': 'neutral',
                    'recommendation': f'No strong mental health correlation found for viewing at {hour:02d}:00.'
                })
    
    def _analyze_patterns(self, user_history_df):
        """Analyze recurring patterns in user's viewing habits."""
        if 'recurring_patterns' not in self.insights:
            logger.warning("No pattern data in insights. Skipping pattern analysis.")
            return
            
        # Get positive and negative patterns from insights
        positive_patterns = []
        negative_patterns = []
        
        for item in self.insights['recurring_patterns']:
            pattern_info = {
                'type': item['pattern_type'],
                'pattern': item['pattern'],
                'affects': item['mental_health_category'],
                'score': item['avg_score']
            }
            
            if item['avg_score'] > 0.6 and item['count'] >= 3:
                positive_patterns.append(pattern_info)
            elif item['avg_score'] < 0.4 and item['count'] >= 3:
                negative_patterns.append(pattern_info)
        
        # Add recommendations based on patterns
        for pattern in positive_patterns:
            self.recommendations['patterns_to_follow'].append({
                'pattern': pattern['pattern'],
                'type': pattern['type'],
                'impact': 'positive',
                'affects': pattern['affects'],
                'recommendation': f"Continue '{pattern['pattern']}' viewing pattern as it correlates with improved {pattern['affects']}."
            })
            
        for pattern in negative_patterns:
            self.recommendations['patterns_to_avoid'].append({
                'pattern': pattern['pattern'],
                'type': pattern['type'],
                'impact': 'negative',
                'affects': pattern['affects'],
                'recommendation': f"Consider breaking the '{pattern['pattern']}' viewing pattern as it correlates with decreased {pattern['affects']}."
            })
    
    def _analyze_music_impact(self):
        """Analyze music impact based on available visualizations."""
        if 'music_genre_impact' not in self.available_visualizations:
            logger.warning("No music genre impact visualization found. Skipping music impact analysis.")
            return
            
        # Try to load the music_impact JSON if available
        music_impact_path = os.path.join(self.visualizations_dir, 'music_impact.json')
        music_impact_data = {}
        
        try:
            if os.path.exists(music_impact_path):
                with open(music_impact_path, 'r', encoding='utf-8') as f:
                    music_impact_data = json.load(f)
                logger.info(f"Loaded music impact data from {music_impact_path}")
        except Exception as e:
            logger.warning(f"Could not load music impact data: {str(e)}")
        
        # Load music impact report if available
        music_report = ""
        if 'music_impact_report' in self.available_visualizations:
            try:
                with open(self.available_visualizations['music_impact_report'], 'r', encoding='utf-8') as f:
                    music_report = f.read()
            except Exception as e:
                logger.warning(f"Could not load music impact report: {str(e)}")
        
        # Generate recommendations based on music impact data
        avg_score = music_impact_data.get('avg_score', 0.5) if music_impact_data else 0.5
        total_count = music_impact_data.get('total_count', 0) if music_impact_data else 0
        
        impact_level = "positive" if avg_score > 0.6 else "negative" if avg_score < 0.4 else "neutral"
        
        self.recommendations['music_impact'].append({
            'impact': impact_level,
            'avg_score': avg_score,
            'total_count': total_count,
            'recommendation': self._generate_music_recommendation(avg_score, total_count),
            'visualization': self.available_visualizations.get('music_genre_impact', ""),
            'report': music_report
        })
    
    def _generate_music_recommendation(self, avg_score, total_count):
        """Generate a specific recommendation based on music impact score."""
        if total_count < 5:
            return "Not enough music content to provide meaningful recommendations."
            
        if avg_score > 0.8:
            return "Music content appears to have a very positive impact on your mental health. Consider increasing your music video consumption."
        elif avg_score > 0.6:
            return "Music content shows a positive correlation with your mental health indicators. Continue with your current music viewing habits."
        elif avg_score < 0.4:
            return "Music content appears to have a negative impact on your mental health. Consider reducing your music video consumption or changing the genres you watch."
        else:
            return "Music content shows neutral impact on your mental health indicators."
    
    def _analyze_sentiment_trajectory(self):
        """Analyze sentiment trajectory based on available visualizations."""
        if 'sentiment_trajectory' not in self.available_visualizations:
            logger.warning("No sentiment trajectory visualization found. Skipping sentiment analysis.")
            return
            
        # Try to load the sentiment trajectory data if available
        sentiment_path = os.path.join(self.visualizations_dir, 'sentiment_trajectory.json')
        sentiment_data = {}
        
        try:
            if os.path.exists(sentiment_path):
                with open(sentiment_path, 'r', encoding='utf-8') as f:
                    sentiment_data = json.load(f)
                logger.info(f"Loaded sentiment trajectory data from {sentiment_path}")
        except Exception as e:
            logger.warning(f"Could not load sentiment trajectory data: {str(e)}")
        
        # Generate recommendation based on the visualization
        self.recommendations['sentiment_trajectory'].append({
            'visualization': self.available_visualizations.get('sentiment_trajectory', ""),
            'recommendation': "This visualization shows how your mental health sentiment has changed over time across different categories. The smoothed trend lines make it easier to identify patterns and correlations."
        })
    
    def _analyze_unhealthy_trends(self):
        """Analyze unhealthy viewing trends based on available visualizations."""
        if 'unhealthy_trends' not in self.available_visualizations:
            logger.warning("No unhealthy trends visualization found. Skipping unhealthy trends analysis.")
            return
            
        # Try to load the unhealthy patterns report if available
        unhealthy_report = ""
        if 'unhealthy_patterns_report' in self.available_visualizations:
            try:
                with open(self.available_visualizations['unhealthy_patterns_report'], 'r', encoding='utf-8') as f:
                    unhealthy_report = f.read()
            except Exception as e:
                logger.warning(f"Could not load unhealthy patterns report: {str(e)}")
        
        # Generate recommendation based on the visualization and report
        self.recommendations['unhealthy_trends'].append({
            'visualization': self.available_visualizations.get('unhealthy_trends', ""),
            'report': unhealthy_report,
            'recommendation': "This analysis identifies potentially unhealthy viewing trends that may negatively impact your mental health. Pay attention to any significant increases in concerning content."
        })
    
    def _generate_summary(self):
        """Generate a personalized summary of recommendations."""
        summary = {
            'personalized_recommendations': self.recommendations,
            'timestamp': datetime.now().isoformat(),
            'text_summary': self._create_text_summary(),
            'visualizations': self.available_visualizations
        }
        
        return summary
        
    def _create_text_summary(self):
        """Create a human-readable text summary of the recommendations."""
        lines = ["# Personalized YouTube Viewing Recommendations", ""]
        
        # Content recommendations
        if self.recommendations['content']:
            lines.append("## Content Recommendations")
            for rec in self.recommendations['content']:
                impact_symbol = "✅" if rec['impact'] == 'positive' else "⚠️" if rec['impact'] == 'negative' else "➖"
                lines.append(f"{impact_symbol} **{rec['category']}**: {rec['recommendation']}")
            lines.append("")
            
        # Timing recommendations
        if self.recommendations['timing']:
            lines.append("## Viewing Time Recommendations")
            for rec in self.recommendations['timing']:
                impact_symbol = "✅" if rec['impact'] == 'positive' else "⚠️" if rec['impact'] == 'negative' else "➖"
                lines.append(f"{impact_symbol} **{rec['hour']:02d}:00 hour**: {rec['recommendation']}")
            lines.append("")
            
        # Music impact recommendations
        if self.recommendations['music_impact']:
            lines.append("## Music Content Impact")
            for rec in self.recommendations['music_impact']:
                impact_symbol = "✅" if rec['impact'] == 'positive' else "⚠️" if rec['impact'] == 'negative' else "➖"
                lines.append(f"{impact_symbol} **Music Impact Score: {rec['avg_score']:.2f}**: {rec['recommendation']}")
                
                # Add report content if available and not too long
                if rec['report'] and len(rec['report']) < 500:
                    lines.append("\n**Detailed Music Impact Analysis:**")
                    lines.append(rec['report'])
            lines.append("")
            
        # Sentiment trajectory insights
        if self.recommendations['sentiment_trajectory']:
            lines.append("## Sentiment Trajectory Insights")
            for rec in self.recommendations['sentiment_trajectory']:
                lines.append(f"📊 {rec['recommendation']}")
                lines.append("*View the visualization for detailed trends across different mental health categories.*")
            lines.append("")
            
        # Unhealthy viewing trends
        if self.recommendations['unhealthy_trends']:
            lines.append("## Unhealthy Viewing Trend Analysis")
            for rec in self.recommendations['unhealthy_trends']:
                lines.append(f"⚠️ {rec['recommendation']}")
                
                # Add report content if available
                if rec['report']:
                    lines.append("\n**Detected Concerns:**")
                    lines.append(rec['report'])
            lines.append("")
            
        # Patterns to follow
        if self.recommendations['patterns_to_follow']:
            lines.append("## Beneficial Viewing Patterns")
            for rec in self.recommendations['patterns_to_follow']:
                lines.append(f"✅ **{rec['pattern']}**: {rec['recommendation']}")
            lines.append("")
            
        # Patterns to avoid
        if self.recommendations['patterns_to_avoid']:
            lines.append("## Viewing Patterns to Reconsider")
            for rec in self.recommendations['patterns_to_avoid']:
                lines.append(f"⚠️ **{rec['pattern']}**: {rec['recommendation']}")
            lines.append("")
            
        # General advice
        lines.append("## General Advice")
        lines.append("Based on our comprehensive analysis of viewing patterns and mental health correlations:")
        lines.append("1. **Variety is beneficial**: Mix content types rather than binging on a single category")
        lines.append("2. **Be mindful of viewing times**: Your viewing hour can significantly impact mental wellness")
        lines.append("3. **Track your reactions**: Note how you feel after watching different content types")
        lines.append("4. **Take breaks**: Regular breaks during viewing sessions support better mental health")
        lines.append("5. **Consider music impact**: Music videos can have significant effects on your mental state")
        lines.append("6. **Monitor unhealthy trends**: Be aware of any increasing trends in concerning content")
        
        return "\n".join(lines)
        
    def generate_llm_prompt(self, user_history=None):
        """Generate a prompt for an LLM to provide personalized insights."""
        
        if user_history is not None:
            # Analyze the provided user history first
            summary = self.analyze_user_viewing_history(user_history)
        else:
            # Just use general insights
            summary = self._generate_summary()
            
        # Create a prompt that provides context to the LLM
        prompt = f"""
You are an AI assistant specialized in providing personalized YouTube viewing recommendations based on mental health impact analysis.

CONTEXT:
I have analyzed YouTube viewing patterns and their correlation with mental health metrics.
This analysis is based on a database of videos, mental health measurements, engagement metrics, and viewing patterns.

KEY INSIGHTS FROM THE DATASET:
{self.insights['meta']['summary']}

PERSONALIZED RECOMMENDATIONS:
{summary['text_summary']}

AVAILABLE VISUALIZATIONS:
{', '.join(summary['visualizations'].keys())}

Your task is to act as a helpful, conversational assistant that can discuss these findings with the user,
answer questions about the mental health impact of their viewing habits, and provide personalized recommendations.

Based on the data, help the user understand:
1. How their YouTube habits may be affecting their mental wellbeing
2. What content categories are most beneficial/harmful to them
3. Optimal viewing times for their mental health
4. Beneficial viewing patterns to adopt or harmful ones to break
5. The impact of music content on their mental state
6. How their sentiment trajectory has evolved over time
7. Any unhealthy viewing trends that may require attention

Remember to be empathetic, non-judgmental, and to frame recommendations in a positive, actionable way.
"""
        
        return prompt
    
    def create_interactive_report(self, user_id, recommendations, 
                                 output_dir='user_reports'):
        """Create an interactive HTML report with embedded visualizations."""
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # HTML template start - using triple braces {{ }} to escape curly braces in CSS
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>YouTube Insights - Personalized Mental Health Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f9f9f9;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                h1 {{
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    margin-top: 30px;
                    border-bottom: 1px solid #bdc3c7;
                    padding-bottom: 5px;
                }}
                .viz-container {{
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 20px;
                    margin: 20px 0;
                }}
                .viz-container img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                }}
                .recommendation {{
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                }}
                .positive {{
                    background-color: rgba(46, 204, 113, 0.15);
                    border-left: 4px solid #2ecc71;
                }}
                .negative {{
                    background-color: rgba(231, 76, 60, 0.15);
                    border-left: 4px solid #e74c3c;
                }}
                .neutral {{
                    background-color: rgba(52, 152, 219, 0.15);
                    border-left: 4px solid #3498db;
                }}
                .footer {{
                    margin-top: 50px;
                    text-align: center;
                    font-size: 0.9em;
                    color: #7f8c8d;
                }}
            </style>
        </head>
        <body>
            <h1>YouTube Viewing Insights: Mental Health Impact Analysis</h1>
            <p><strong>Generated for:</strong> User ID - {user_id}</p>
            <p><strong>Report Date:</strong> {date}</p>
        """
        
        # Fill in user ID and date
        html = html_template.format(
            user_id=user_id,
            date=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        
        # Add markdown content converted to HTML
        try:
            import markdown
            html_content = markdown.markdown(recommendations['text_summary'])
            html += f"""
                <div class="markdown-content">
                    {html_content}
                </div>
            """
        except ImportError:
            # Fallback if markdown module is not available
            logger.warning("Markdown module not installed. Using plain text for recommendations.")
            # Simple conversion of markdown to basic HTML
            text_content = recommendations['text_summary']
            # Convert headers
            for i in range(3, 0, -1):
                text_content = text_content.replace('#' * i + ' ', f'<h{i}>')
                # Add closing tags - simplistic approach, assumes headers are on their own lines
                lines = text_content.split('\n')
                for j in range(len(lines)):
                    if lines[j].startswith(f'<h{i}>'):
                        lines[j] = lines[j] + f'</h{i}>'
                text_content = '\n'.join(lines)
            
            # Convert basic markdown formatting
            text_content = text_content.replace('**', '<strong>').replace('**', '</strong>')
            text_content = text_content.replace('*', '<em>').replace('*', '</em>')
            
            # Convert newlines to <br> tags
            text_content = text_content.replace('\n', '<br>')
            
            html += f"""
                <div class="markdown-content">
                    {text_content}
                </div>
            """
        
        # Add visualizations
        html += "<h2>Detailed Visualizations</h2>"
        
        # Function to copy visualization file to output directory
        def copy_vis_to_output(vis_path, vis_type):
            if not vis_path or not os.path.exists(vis_path):
                return None
                
            # Create a filename for the copied visualization
            filename = f"{user_id}_{vis_type}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Copy the file
            try:
                shutil.copy2(vis_path, output_path)
                return filename
            except Exception as e:
                logger.warning(f"Could not copy visualization file: {str(e)}")
                return None
        
        # Add sentiment trajectory
        if 'sentiment_trajectory' in self.available_visualizations:
            vis_file = copy_vis_to_output(
                self.available_visualizations['sentiment_trajectory'], 
                'sentiment_trajectory'
            )
            if vis_file:
                html += f"""
                <div class="viz-container">
                    <h3>Sentiment Trajectory Analysis</h3>
                    <p>This visualization shows how your mental health sentiment has changed over time across different categories.</p>
                    <img src="{vis_file}" alt="Sentiment Trajectory">
                </div>
                """
        
        # Add music genre impact
        if 'music_genre_impact' in self.available_visualizations:
            vis_file = copy_vis_to_output(
                self.available_visualizations['music_genre_impact'], 
                'music_genre_impact'
            )
            if vis_file:
                html += f"""
                <div class="viz-container">
                    <h3>Music Genre Impact Analysis</h3>
                    <p>This visualization shows how different music genres impact your mental health metrics.</p>
                    <img src="{vis_file}" alt="Music Genre Impact">
                </div>
                """
        
        # Add unhealthy trends
        if 'unhealthy_trends' in self.available_visualizations:
            vis_file = copy_vis_to_output(
                self.available_visualizations['unhealthy_trends'], 
                'unhealthy_trends'
            )
            if vis_file:
                html += f"""
                <div class="viz-container">
                    <h3>Unhealthy Viewing Trend Analysis</h3>
                    <p>This visualization identifies potentially concerning trends in your viewing habits.</p>
                    <img src="{vis_file}" alt="Unhealthy Viewing Trends">
                </div>
                """
        
        # Add late night viewing
        if 'late_night_viewing' in self.available_visualizations:
            vis_file = copy_vis_to_output(
                self.available_visualizations['late_night_viewing'], 
                'late_night_viewing'
            )
            if vis_file:
                html += f"""
                <div class="viz-container">
                    <h3>Late Night Viewing Pattern Analysis</h3>
                    <p>This visualization shows your late night viewing patterns and their correlation with mental health impacts.</p>
                    <img src="{vis_file}" alt="Late Night Viewing Patterns">
                </div>
                """
        
        # Add viewing patterns
        if 'viewing_patterns' in self.available_visualizations:
            vis_file = copy_vis_to_output(
                self.available_visualizations['viewing_patterns'], 
                'viewing_patterns'
            )
            if vis_file:
                html += f"""
                <div class="viz-container">
                    <h3>Viewing Pattern Analysis by Day of Week</h3>
                    <p>This visualization shows your viewing patterns across different days of the week.</p>
                    <img src="{vis_file}" alt="Viewing Patterns">
                </div>
                """
        
        # Add mental health forecast
        if 'mental_health_forecast' in self.available_visualizations:
            vis_file = copy_vis_to_output(
                self.available_visualizations['mental_health_forecast'], 
                'mental_health_forecast'
            )
            if vis_file:
                html += f"""
                <div class="viz-container">
                    <h3>Mental Health Index Forecast</h3>
                    <p>This visualization shows a forecast of your mental health index based on historical data.</p>
                    <img src="{vis_file}" alt="Mental Health Forecast">
                </div>
                """
        
        # HTML template end
        html += """
            <div class="footer">
                <p>This report was generated by the YouTube Insight Agent for mental health analysis.</p>
                <p>© 2025 IntrospectAI - All visualizations are personalized based on your viewing history.</p>
            </div>
        </body>
        </html>
        """
        
        # Save HTML to file
        html_filename = f"{output_dir}/user_{user_id}_report.html"
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html)
            
        logger.info(f"Created interactive HTML report for user {user_id} at {html_filename}")
        
        return html_filename
    
    def save_recommendations(self, user_id, recommendations, output_dir='insights/user_recommendations'):
        """Save personalized recommendations to a file."""
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save as JSON
        filename = f"{output_dir}/user_{user_id}_recommendations.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2)
            
        # Also save the text summary as Markdown
        md_filename = f"{output_dir}/user_{user_id}_recommendations.md"
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(recommendations['text_summary'])
            
        # Create an interactive HTML report
        html_filename = self.create_interactive_report(user_id, recommendations)
            
        logger.info(f"Saved recommendations for user {user_id} to {filename}, {md_filename}, and {html_filename}")
        
        return {
            'json': filename,
            'markdown': md_filename,
            'html': html_filename
        }

# Example usage
if __name__ == "__main__":
    # Initialize the agent with insights data and visualizations directory
    agent = YouTubeInsightAgent(
        insights_path='insights/youtube_mental_health_insights.json',
        visualizations_dir='analysis_reports'
    )
    
    # Example user history (mocked data)
    example_user_history = [
        {"video_id": 1, "title": "Relaxing Music for Stress Relief", "watched_at": "2025-02-15T22:30:00", "primary_category": "Music"},
        {"video_id": 2, "title": "How to Meditate for Beginners", "watched_at": "2025-02-16T08:15:00", "primary_category": "Education"},
        {"video_id": 3, "title": "Gaming Livestream", "watched_at": "2025-02-16T23:45:00", "primary_category": "Gaming"},
        {"video_id": 4, "title": "News Update", "watched_at": "2025-02-17T07:30:00", "primary_category": "News"}
    ]
    
    # Generate personalized recommendations
    recommendations = agent.analyze_user_viewing_history(example_user_history)
    
    # Save recommendations and create interactive report
    output_files = agent.save_recommendations("example_user", recommendations)
    
    print(f"\nRecommendations saved to:")
    for file_type, file_path in output_files.items():
        print(f"- {file_type.capitalize()}: {file_path}")
    
    # Generate an LLM prompt
    prompt = agent.generate_llm_prompt(example_user_history)
    print("\nGenerated LLM Prompt:\n")
    print(prompt)
    
    print("\nVisualization Files Found:")
    for vis_type, vis_path in agent.available_visualizations.items():
        print(f"- {vis_type}: {vis_path}") 