"""
Image Analysis Agent - Uses OpenAI's Vision API to analyze images.
"""

import os
import base64
import time
import logging
import math
from pathlib import Path
from typing import Optional, Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("image_analysis_agent")

# Try to load OpenAI client, gracefully handle import error
try:
    from openai import OpenAI, RateLimitError
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not found. Please install it using: pip install openai")
    OPENAI_AVAILABLE = False

class ImageAnalysisAgent:
    """
    Agent responsible for analyzing images using OpenAI's Vision models.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the image analysis agent.
        
        Args:
            api_key: OpenAI API key. If None, will look for OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client if API key is available
        self.client = None
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    def _encode_image(self, image_path: str) -> Optional[str]:
        """
        Encode an image to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image or None if failed
        """
        try:
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
                return base64.b64encode(img_data).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None
    
    def analyze_image(self, image_path: str, analysis_type: str = "health", max_retries: int = 3, retry_delay: int = 5) -> Dict[str, Any]:
        """
        Analyze an image using OpenAI's Vision API.
        
        Args:
            image_path: Path to the image file
            analysis_type: Type of analysis to perform (health, youtube, general)
            max_retries: Maximum number of retries for rate limit errors
            retry_delay: Initial delay between retries (will be exponentially increased)
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        if not OPENAI_AVAILABLE or not self.client:
            logger.error("OpenAI client not available. Cannot perform image analysis.")
            return {"error": "OpenAI client not available"}
        
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return {"error": "Image file not found"}
        
        image_b64 = self._encode_image(image_path)
        if not image_b64:
            return {"error": "Failed to encode image"}
        
        # Select appropriate prompt based on analysis type
        analysis_prompt = self._get_analysis_prompt(analysis_type)
        
        logger.info(f"Analyzing image: {image_path} (Size: {os.path.getsize(image_path)/1024:.2f} KB)")
        start_time = time.time()
        
        # Implement retry logic with exponential backoff
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # Call the OpenAI API
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # Using the latest vision-capable model
                    messages=[
                        {"role": "system", "content": "You are a specialized data visualization analyst with expertise in interpreting metrics, trends and extracting insights from data. Focus only on the data shown without making suggestions or recommendations."},
                        {"role": "user", "content": [
                            {"type": "text", "text": analysis_prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }}
                        ]}
                    ],
                    max_tokens=2000
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                analysis_result = response.choices[0].message.content
                logger.info(f"Analysis completed in {duration:.2f} seconds")
                
                return {
                    "success": True,
                    "analysis": analysis_result,
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path),
                    "duration": duration,
                    "timestamp": time.time(),
                    "model": "gpt-4o",
                }
            
            except RateLimitError as e:
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                    logger.warning(f"Rate limit hit. Retrying in {wait_time} seconds... ({retry_count}/{max_retries})")
                    logger.warning(f"Rate limit error: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached for rate limit error: {str(e)}")
                    return {
                        "success": False,
                        "error": f"Rate limit error after {max_retries} retries: {str(e)}",
                        "image_path": image_path,
                        "image_name": os.path.basename(image_path)
                    }
            except Exception as e:
                logger.error(f"Error during image analysis: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path)
                }
    
    def _get_analysis_prompt(self, analysis_type: str) -> str:
        """
        Get the appropriate analysis prompt based on the analysis type.
        
        Args:
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis prompt string
        """
        if analysis_type == "health":
            return """
            Analyze this health data visualization in detail, focusing only on factual observations with specific numbers and patterns. Include:
            
            1. Type of visualization (chart type, graph format)
            2. Data being presented, including exact metrics, time periods, and categories
            3. Specific numerical values and statistics shown (report actual numbers wherever possible)
            4. Key patterns, such as differences between weekdays vs weekends, seasons, or time periods
            5. Notable trends (increases, decreases) with exact numerical values or percentages
            6. Data points that stand out from the norm (outliers)
            7. Observable correlations between different data points or categories
            
            For example, "Weekday steps average 5,430 while weekend steps average 7,820, indicating a 44% increase in activity during weekends" or "Heart rate readings show a consistent drop from an average of 75 BPM in January to 68 BPM in March."
            
            Only describe what is factually present in the data. Do not make recommendations or suggestions.
            
            Provide your analysis in a well-structured format with clear sections.
            """
        elif analysis_type == "youtube":
            return """
            Analyze this YouTube data visualization in detail, focusing only on factual observations with specific numbers and patterns. Include:
            
            1. Type of visualization (chart type, graph format)
            2. Data being presented, including exact metrics, time periods, and categories
            3. Specific numerical values and statistics shown (report actual numbers wherever possible)
            4. Key consumption patterns by time of day, day of week, or seasons
            5. Notable trends in viewing habits (increases, decreases) with exact numerical values or percentages
            6. Content categories or types that appear most frequently
            7. Correlations between consumption patterns and mental health indicators visible in the data
            
            For example, "Late night YouTube viewing (10PM-2AM) accounts for 45% of total watch time, with a concentration of content labeled as 'escapism'" or "Weekend consumption shows a 37% increase in 'educational' content compared to weekdays."
            
            Only describe what is factually present in the data. Do not make recommendations or suggestions.
            
            Provide your analysis in a well-structured format with clear sections.
            """
        else:
            return """
            Analyze this data visualization in detail, focusing only on factual observations with specific numbers and patterns. Include:
            
            1. Type of visualization (chart type, graph format)
            2. Data being presented, including exact metrics, time periods, and categories
            3. Specific numerical values and statistics shown (report actual numbers wherever possible)
            4. Key patterns by any time periods or categories shown
            5. Notable trends (increases, decreases) with exact numerical values or percentages
            6. Data points that stand out from the norm (outliers)
            7. Observable correlations between different data points or categories
            
            Only describe what is factually present in the data. Do not make recommendations or suggestions.
            
            Provide your analysis in a well-structured format with clear sections.
            """
    
    def process_directory(self, directory_path: str, output_dir: str, analysis_type: str = "general", skip_existing: bool = True) -> Dict[str, Any]:
        """
        Process all images in a directory and save analysis results.
        
        Args:
            directory_path: Path to directory containing images
            output_dir: Directory to save analysis results
            analysis_type: Type of analysis to perform
            skip_existing: Skip images that already have analysis files
            
        Returns:
            Dictionary with summary of processing results
        """
        # Ensure directories exist
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return {"error": "Directory not found"}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(Path(directory_path).glob(f'*{ext}')))
        
        if not image_files:
            logger.warning(f"No image files found in {directory_path}")
            return {"error": "No image files found"}
        
        logger.info(f"Found {len(image_files)} image files to process")
        
        # Check for existing analysis files if skip_existing is True
        skipped = 0
        to_process = []
        
        for image_path in image_files:
            output_file = os.path.join(output_dir, f"{Path(image_path).stem}_analysis.txt")
            if skip_existing and os.path.exists(output_file):
                logger.info(f"Skipping existing analysis for: {image_path}")
                skipped += 1
            else:
                to_process.append(image_path)
        
        if skip_existing and skipped > 0:
            logger.info(f"Skipping {skipped} images with existing analysis files")
            logger.info(f"Processing {len(to_process)} remaining images")
        
        # Process each image
        results = []
        successful = skipped  # Count skipped files as successful
        failed = 0
        
        for i, image_path in enumerate(to_process):
            logger.info(f"Processing image {i+1}/{len(to_process)}: {image_path}")
            
            result = self.analyze_image(str(image_path), analysis_type)
            
            # Save analysis to file
            if result.get("success", False):
                output_file = os.path.join(output_dir, f"{Path(image_path).stem}_analysis.txt")
                
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"# Analysis of {result['image_name']}\n\n")
                    f.write(f"Analysis timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp']))}\n")
                    f.write(f"Processing time: {result['duration']:.2f} seconds\n\n")
                    f.write(result['analysis'])
                
                logger.info(f"Analysis saved to: {output_file}")
                successful += 1
            else:
                logger.error(f"Analysis failed for: {image_path}")
                failed += 1
            
            # Add the result
            results.append(result)
            
            # Add a small delay to avoid rate limiting
            time.sleep(1.5)
        
        # Create a summary report
        summary_file = os.path.join(output_dir, "processing_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"# Image Analysis Processing Summary\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directory: {directory_path}\n")
            f.write(f"Analysis type: {analysis_type}\n")
            f.write(f"Total images: {len(image_files)}\n")
            f.write(f"Images processed: {len(to_process)}\n")
            f.write(f"Images skipped (existing analysis): {skipped}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n\n")
            
            f.write("## Processed Files\n\n")
            for i, result in enumerate(results):
                status = "✓ Success" if result.get("success", False) else "✗ Failed"
                f.write(f"{i+1}. {result['image_name']} - {status}\n")
        
        logger.info(f"Processing summary saved to: {summary_file}")
        
        return {
            "success": True,
            "total": len(image_files),
            "processed": len(to_process),
            "skipped": skipped,
            "successful": successful,
            "failed": failed,
            "directory": directory_path,
            "output_dir": output_dir,
            "summary_file": summary_file,
            "results": results
        }
        
    def generate_consolidated_report(self, analysis_dir: str, output_file: str, analysis_type: str = "general", max_retries: int = 3, batch_size: int = 15) -> Dict[str, Any]:
        """
        Generate a consolidated report from individual image analyses.
        
        Args:
            analysis_dir: Directory containing individual analyses
            output_file: Path to save the consolidated report
            analysis_type: Type of analysis that was performed
            max_retries: Maximum number of retries for rate limit errors
            batch_size: Number of analyses to process in each batch to avoid token limits
            
        Returns:
            Dictionary with report generation results
        """
        if not os.path.exists(analysis_dir):
            logger.error(f"Analysis directory not found: {analysis_dir}")
            return {"error": "Analysis directory not found"}
        
        # Check if report already exists
        if os.path.exists(output_file):
            logger.info(f"Consolidated report already exists at: {output_file}")
            return {
                "success": True,
                "message": "Report already exists",
                "output_file": output_file,
                "skipped": True
            }
        
        # Get all analysis files
        analysis_files = list(Path(analysis_dir).glob("*_analysis.txt"))
        
        if not analysis_files:
            logger.warning(f"No analysis files found in {analysis_dir}")
            return {"error": "No analysis files found"}
        
        logger.info(f"Found {len(analysis_files)} analysis files to consolidate")
        
        # Determine the appropriate system message based on analysis type
        if analysis_type == "health":
            system_message = "You are a specialized health data analyst with expertise in interpreting health metrics and data patterns. Your job is to summarize the collected findings without making suggestions or recommendations."
        elif analysis_type == "youtube":
            system_message = "You are a specialized digital media analyst with expertise in interpreting YouTube usage patterns and viewing data. Your job is to summarize the collected findings without making suggestions or recommendations."
        else:
            system_message = "You are a specialized data analyst with expertise in interpreting data visualizations. Your job is to summarize the collected findings without making suggestions or recommendations."
        
        # If there are too many files, process in batches
        if len(analysis_files) > batch_size:
            return self._generate_batched_report(
                analysis_files=analysis_files,
                output_file=output_file,
                analysis_type=analysis_type,
                system_message=system_message,
                batch_size=batch_size,
                max_retries=max_retries
            )
        else:
            # Process all files in one go for smaller sets
            return self._generate_single_report(
                analysis_files=analysis_files,
                output_file=output_file,
                analysis_type=analysis_type,
                system_message=system_message,
                max_retries=max_retries
            )
    
    def _generate_single_report(self, analysis_files: List[Path], output_file: str, 
                               analysis_type: str, system_message: str, max_retries: int = 3) -> Dict[str, Any]:
        """Generate a consolidated report from a single batch of analysis files."""
        # Read all analyses
        analyses = []
        for file_path in analysis_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                analyses.append({
                    "file_name": file_path.name,
                    "content": content
                })
        
        # Combine analyses for the consolidated report prompt
        combined_analyses = "\n\n===== NEXT ANALYSIS =====\n\n".join(
            [f"FILENAME: {a['file_name']}\n\n{a['content']}" for a in analyses]
        )
        
        # Create the appropriate prompt based on analysis type
        if analysis_type == "health":
            consolidated_prompt = f"""
            I'll provide you with multiple analyses of health data visualizations. Your task is to create a comprehensive consolidated report that:
            
            1. Summarizes all key findings across the visualizations, focusing on actual metrics and data patterns
            2. Highlights specific numerical values and statistics observed in the data
            3. Identifies patterns, correlations, and trends with exact numbers whenever possible
            4. Organizes findings by categories (steps, heart rate, activity types, etc.)
            
            IMPORTANT: Do NOT include any recommendations, advice, or suggestions. Strictly report what was found in the data.
            
            Organize the report in a clear, structured format with sections and subsections.
            
            Here are the individual analyses to consolidate:
            
            {combined_analyses}
            """
        elif analysis_type == "youtube":
            consolidated_prompt = f"""
            I'll provide you with multiple analyses of YouTube data visualizations. Your task is to create a comprehensive consolidated report that:
            
            1. Summarizes all key findings across the visualizations, focusing on actual metrics and viewing patterns
            2. Highlights specific numerical values and statistics observed in the data
            3. Identifies patterns in YouTube consumption and viewing habits with exact numbers whenever possible
            4. Notes any mental health correlations observed in the data
            
            IMPORTANT: Do NOT include any recommendations, advice, or suggestions. Strictly report what was found in the data.
            
            Organize the report in a clear, structured format with sections and subsections.
            
            Here are the individual analyses to consolidate:
            
            {combined_analyses}
            """
        else:
            consolidated_prompt = f"""
            I'll provide you with multiple analyses of data visualizations. Your task is to create a comprehensive consolidated report that:
            
            1. Summarizes all key findings across the visualizations, focusing on actual metrics and data patterns
            2. Highlights specific numerical values and statistics observed in the data
            3. Identifies patterns, correlations, and trends with exact numbers whenever possible
            
            IMPORTANT: Do NOT include any recommendations, advice, or suggestions. Strictly report what was found in the data.
            
            Organize the report in a clear, structured format with sections and subsections.
            
            Here are the individual analyses to consolidate:
            
            {combined_analyses}
            """
        
        if not OPENAI_AVAILABLE or not self.client:
            logger.error("OpenAI client not available. Cannot generate consolidated report.")
            return {"error": "OpenAI client not available"}
        
        logger.info(f"Generating consolidated report from {len(analyses)} analyses")
        start_time = time.time()
        
        # Implement retry logic with exponential backoff
        retry_count = 0
        retry_delay = 5  # Initial delay in seconds
        
        while retry_count <= max_retries:
            try:
                # Call the OpenAI API for the consolidated report
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": consolidated_prompt}
                    ],
                    max_tokens=4000
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                consolidated_report = response.choices[0].message.content
                logger.info(f"Consolidated report generated in {duration:.2f} seconds")
                
                # Save the consolidated report
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"# Consolidated {analysis_type.capitalize()} Data Analysis Report\n\n")
                    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Based on analysis of {len(analyses)} visualizations\n\n")
                    f.write(consolidated_report)
                
                logger.info(f"Consolidated report saved to: {output_file}")
                
                return {
                    "success": True,
                    "total_analyses": len(analyses),
                    "output_file": output_file,
                    "duration": duration
                }
                
            except RateLimitError as e:
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                    logger.warning(f"Rate limit hit. Retrying in {wait_time} seconds... ({retry_count}/{max_retries})")
                    logger.warning(f"Rate limit error: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached for rate limit error: {str(e)}")
                    return {
                        "success": False,
                        "error": f"Rate limit error after {max_retries} retries: {str(e)}",
                        "total_analyses": len(analyses)
                    }
            except Exception as e:
                logger.error(f"Error generating consolidated report: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "total_analyses": len(analyses)
                }
    
    def _generate_batched_report(self, analysis_files: List[Path], output_file: str, 
                                analysis_type: str, system_message: str, 
                                batch_size: int = 15, max_retries: int = 3) -> Dict[str, Any]:
        """Generate a consolidated report by processing analyses in batches to avoid token limits."""
        logger.info(f"Using batched processing for {len(analysis_files)} analysis files with batch size {batch_size}")
        
        # Calculate number of batches
        total_files = len(analysis_files)
        num_batches = math.ceil(total_files / batch_size)
        
        # Process each batch to create intermediate reports
        batch_reports = []
        batch_success = True
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total_files)
            
            batch_files = analysis_files[start_idx:end_idx]
            logger.info(f"Processing batch {i+1}/{num_batches} with {len(batch_files)} analyses")
            
            # Create temp file for batch report
            batch_output_file = f"{output_file}.batch_{i+1}.tmp"
            
            # Generate report for this batch
            batch_result = self._generate_single_report(
                analysis_files=batch_files,
                output_file=batch_output_file,
                analysis_type=analysis_type,
                system_message=system_message,
                max_retries=max_retries
            )
            
            if batch_result.get("success", False):
                batch_reports.append(batch_output_file)
                logger.info(f"Successfully generated batch report {i+1}/{num_batches}")
            else:
                batch_success = False
                logger.error(f"Failed to generate batch report {i+1}/{num_batches}: {batch_result.get('error', 'Unknown error')}")
                # Continue with other batches even if one fails
            
            # Add delay between batches to avoid rate limits
            if i < num_batches - 1:
                time.sleep(2)
        
        # If we have multiple batch reports, combine them
        if len(batch_reports) > 1:
            logger.info(f"Combining {len(batch_reports)} batch reports into final report")
            
            # Combine all batch reports
            combined_content = []
            
            for batch_file in batch_reports:
                try:
                    with open(batch_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Skip the header from intermediate reports
                        if "# Consolidated" in content:
                            lines = content.split("\n")
                            # Find where the actual content starts (after the header)
                            for j, line in enumerate(lines):
                                if j > 5 and line.strip() and not line.startswith("Date:") and not line.startswith("Based on"):
                                    content = "\n".join(lines[j:])
                                    break
                        combined_content.append(content)
                except Exception as e:
                    logger.error(f"Error reading batch file {batch_file}: {str(e)}")
                    batch_success = False
            
            final_report = "\n\n## Next Batch of Analyses\n\n".join(combined_content)
            
            # Save the final combined report
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"# Consolidated {analysis_type.capitalize()} Data Analysis Report\n\n")
                    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Based on analysis of {total_files} visualizations processed in {len(batch_reports)} batches\n\n")
                    f.write(final_report)
                
                logger.info(f"Final consolidated report saved to: {output_file}")
            except Exception as e:
                logger.error(f"Error saving final report: {str(e)}")
                batch_success = False
        
            # Clean up temp files
            for batch_file in batch_reports:
                try:
                    if os.path.exists(batch_file):
                        os.remove(batch_file)
                except Exception as e:
                    logger.warning(f"Error removing temp file {batch_file}: {str(e)}")
        
        elif len(batch_reports) == 1:
            # If we only have one batch report, rename it to the final output file
            try:
                os.rename(batch_reports[0], output_file)
                logger.info(f"Single batch report renamed to final report: {output_file}")
            except Exception as e:
                logger.error(f"Error renaming batch file to final report: {str(e)}")
                batch_success = False
        
        if batch_success:
            return {
                "success": True,
                "total_analyses": total_files,
                "batch_count": num_batches,
                "output_file": output_file
            }
        else:
            return {
                "success": False,
                "error": "One or more batches failed during processing",
                "total_analyses": total_files,
                "batch_count": num_batches
            } 