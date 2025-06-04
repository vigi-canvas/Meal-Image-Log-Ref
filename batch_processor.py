import pandas as pd
import numpy as np
import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from PIL import Image
import glob
import json
import csv
from dataclasses import asdict

from metrics import CGMMetricsCalculator
from gemini_api import GeminiAPI
from models import MealImpactAnalysis, ReferenceMealAnalysis

class MealTimingAnalyzer:
    """Analyzes CGM data to find optimal meal timing points"""
    
    def __init__(self):
        self.meal_time_windows = {
            'BREAKFAST': (7, 10),      # 7:00 AM - 10:00 AM
            'LUNCH': (12, 15),         # 12:00 PM - 3:00 PM  
            'DINNER': (18, 21)         # 6:00 PM - 9:00 PM
        }
        # Minimum time between meals (in hours)
        self.min_time_between_meals = 2  # At least 2 hours between any meals
        self.min_time_between_same_type = 24  # At least 24 hours between same meal types
        
    def load_cgm_data(self, csv_path: str) -> pd.DataFrame:
        """Load and clean CGM data"""
        df = pd.read_csv(csv_path)
        
        # Convert date and time to datetime
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Remove duplicates - keep last value for same timestamp
        df = df.drop_duplicates(subset=['datetime'], keep='last')
        
        return df
    
    def detect_glucose_rise_patterns(self, df: pd.DataFrame, min_rise: float = 20) -> List[Dict]:
        """Detect glucose rise patterns that indicate meal times"""
        patterns = []
        
        for i in range(1, len(df) - 5):  # Need at least 5 points ahead
            current_time = df.iloc[i]['datetime']
            current_glucose = float(df.iloc[i]['value'])
            
            # Check if within meal time windows
            hour = current_time.hour
            meal_type = None
            for meal, (start_h, end_h) in self.meal_time_windows.items():
                if start_h <= hour <= end_h:
                    meal_type = meal
                    break
            
            if not meal_type:
                continue
                
            # Look for glucose rise in next 60-90 minutes
            future_points = df[(df['datetime'] > current_time) & 
                             (df['datetime'] <= current_time + timedelta(minutes=90))]
            
            if len(future_points) < 3:
                continue
                
            max_future_glucose = future_points['value'].max()
            glucose_rise = float(max_future_glucose) - current_glucose
            
            # Check for significant rise
            if glucose_rise >= min_rise:
                # Ensure baseline is relatively stable (not already rising)
                baseline_points = df[(df['datetime'] >= current_time - timedelta(minutes=30)) & 
                                   (df['datetime'] <= current_time)]
                
                if len(baseline_points) >= 2:
                    baseline_trend = float(baseline_points['value'].iloc[-1]) - float(baseline_points['value'].iloc[0])
                    
                    # Good meal timing: stable or slightly declining baseline
                    if baseline_trend <= 10:  # Not rising too much
                        patterns.append({
                            'datetime': current_time,
                            'date': current_time.strftime('%B %d, %Y'),
                            'time': current_time.strftime('%H:%M'),
                            'meal_type': meal_type,
                            'baseline_glucose': current_glucose,
                            'peak_glucose': float(max_future_glucose),
                            'glucose_rise': glucose_rise,
                            'baseline_trend': baseline_trend
                        })
        
        return patterns
    
    def select_optimal_meal_points(self, patterns: List[Dict], target_count: int = 30) -> List[Dict]:
        """
        Select the best meal timing points with appropriate spacing between meals
        - No same meal type on same day
        - At least 2 hours between any meals
        - Balanced distribution of meal types (breakfast, lunch, dinner)
        - No more than 2 meals per day
        """
        # Sort by glucose rise (higher rises are more interesting)
        patterns.sort(key=lambda x: x['glucose_rise'], reverse=True)
        
        # Initialize selected points and tracking variables
        selected = []
        meal_type_counts = {'BREAKFAST': 0, 'LUNCH': 0, 'DINNER': 0}
        date_meal_types = {}  # Track which meal types are used per date
        
        # First pass: Get up to target_count/3 of each meal type with proper spacing
        for pattern in patterns:
            if len(selected) >= target_count:
                break
                
            meal_datetime = pattern['datetime']
            date_key = pattern['date']
            meal_type = pattern['meal_type']
            
            # Skip if we already have enough of this meal type
            if meal_type_counts[meal_type] >= target_count // 3:
                continue
                
            # Initialize date tracking if needed
            if date_key not in date_meal_types:
                date_meal_types[date_key] = set()
            
            # Skip if we already have this meal type for this date
            if meal_type in date_meal_types[date_key]:
                continue
                
            # Skip if we already have 2 meals for this date
            if len(date_meal_types[date_key]) >= 2:
                continue
            
            # Check time proximity to already selected meals
            too_close = False
            for selected_point in selected:
                selected_datetime = selected_point['datetime']
                time_diff = abs((meal_datetime - selected_datetime).total_seconds() / 3600)  # hours
                
                # Skip if too close to any meal
                if time_diff < self.min_time_between_meals:
                    too_close = True
                    break
                    
                # Skip if same meal type and within min_time_between_same_type
                if meal_type == selected_point['meal_type'] and time_diff < self.min_time_between_same_type:
                    too_close = True
                    break
            
            if too_close:
                continue
                
            # If we passed all checks, add this meal point
            selected.append(pattern)
            meal_type_counts[meal_type] += 1
            date_meal_types[date_key].add(meal_type)
            
            # Print selection for debugging
            print(f"Selected {meal_type} on {date_key} at {pattern['time']} (glucose rise: {pattern['glucose_rise']:.1f})")
        
        # Second pass: Fill remaining slots with best options that still maintain spacing
        target_per_type = target_count // 3
        remaining_slots = target_count - len(selected)
        
        if remaining_slots > 0:
            print(f"Filling {remaining_slots} remaining slots...")
            
            # Calculate how many more of each type we need
            needed_counts = {
                meal_type: max(0, target_per_type - count)
                for meal_type, count in meal_type_counts.items()
            }
            
            # Sort remaining patterns by meal type priority and glucose rise
            remaining_patterns = [p for p in patterns if p not in selected]
            
            # Prioritize meal types that need more representation
            for meal_type in sorted(needed_counts, key=lambda x: needed_counts[x], reverse=True):
                if needed_counts[meal_type] <= 0:
                    continue
                    
                meal_type_patterns = [p for p in remaining_patterns if p['meal_type'] == meal_type]
                meal_type_patterns.sort(key=lambda x: x['glucose_rise'], reverse=True)
                
                for pattern in meal_type_patterns:
                    if len(selected) >= target_count:
                        break
                        
                    meal_datetime = pattern['datetime']
                    date_key = pattern['date']
                    
                    # Skip if date has too many meals
                    if date_key in date_meal_types and len(date_meal_types[date_key]) >= 2:
                        continue
                    
                    # Check time proximity (same as above)
                    too_close = False
                    for selected_point in selected:
                        selected_datetime = selected_point['datetime']
                        time_diff = abs((meal_datetime - selected_datetime).total_seconds() / 3600)
                        
                        if time_diff < self.min_time_between_meals:
                            too_close = True
                            break
                            
                        if meal_type == selected_point['meal_type'] and time_diff < self.min_time_between_same_type:
                            too_close = True
                            break
                    
                    if too_close:
                        continue
                    
                    # Add this meal point
                    selected.append(pattern)
                    meal_type_counts[meal_type] += 1
                    
                    if date_key not in date_meal_types:
                        date_meal_types[date_key] = set()
                    date_meal_types[date_key].add(meal_type)
                    
                    print(f"Added {meal_type} on {date_key} at {pattern['time']} (glucose rise: {pattern['glucose_rise']:.1f})")
                    
                    if meal_type_counts[meal_type] >= target_per_type:
                        break
        
        # Final check - make sure we have enough points
        if len(selected) < target_count:
            print(f"Warning: Could only find {len(selected)} optimal meal points with proper spacing")
            
            # If we're really short, relax the constraints a bit for the remaining slots
            if len(selected) < target_count * 0.8:  # Less than 80% filled
                print("Relaxing constraints to fill more slots...")
                
                for pattern in patterns:
                    if pattern in selected or len(selected) >= target_count:
                        continue
                        
                    meal_datetime = pattern['datetime']
                    meal_type = pattern['meal_type']
                    date_key = pattern['date']
                    
                    # Check minimum 90 minutes between any meals
                    too_close = False
                    for selected_point in selected:
                        selected_datetime = selected_point['datetime']
                        time_diff = abs((meal_datetime - selected_datetime).total_seconds() / 60)  # minutes
                        
                        if time_diff < 90:  # Relaxed constraint: 90 minutes instead of 2 hours
                            too_close = True
                            break
                    
                    if too_close:
                        continue
                    
                    selected.append(pattern)
                    print(f"Added with relaxed constraints: {meal_type} on {date_key} at {pattern['time']}")
                    
                    if len(selected) >= target_count:
                        break
        
        # Sort selected by date and time
        selected.sort(key=lambda x: x['datetime'])
        
        # Final stats
        print(f"\nFinal meal point distribution:")
        for meal_type, count in meal_type_counts.items():
            print(f"  {meal_type}: {count}")
            
        dates_used = len(date_meal_types)
        print(f"  Total dates used: {dates_used}")
        print(f"  Average meals per date: {len(selected)/dates_used:.1f}")
        
        return selected[:target_count]

class BatchProcessor:
    """Processes multiple meal images with CGM insights"""
    
    def __init__(self, user_id: str = "100022075"):
        self.user_id = user_id
        self.cgm_path = f"{user_id}/CGM.csv"
        self.food_path = f"{user_id}/Food.csv"
        self.results_csv = "batch_results.csv"
        self.images_folder = "images/"
        
        # Initialize components
        self.analyzer = MealTimingAnalyzer()
        self.metrics_calc = CGMMetricsCalculator(verbose=False)  # No console spam
        self.gemini_api = GeminiAPI()
        
        # Load optimal meal points
        self.meal_points = self._get_optimal_meal_points()
        
    def _get_optimal_meal_points(self) -> List[Dict]:
        """Get 30 realistic human meal timing points within CGM data range (April 16-23, 2025)"""
        print("🕐 Creating realistic human meal schedule within CGM data range...")
        
        # Define 30 realistic meal points across April 16-23, 2025 ONLY
        # 8 days × 3.75 meals per day = 30 meals
        # Some days have 4 meals (including snacks), some have 3
        realistic_meals = [
            # April 16, 2025 (4 meals - breakfast, lunch, snack, dinner)
            {'datetime': datetime(2025, 4, 16, 8, 0), 'date': 'April 16, 2025', 'time': '08:00', 'meal_type': 'BREAKFAST'},
            {'datetime': datetime(2025, 4, 16, 13, 0), 'date': 'April 16, 2025', 'time': '13:00', 'meal_type': 'LUNCH'}, 
            {'datetime': datetime(2025, 4, 16, 16, 30), 'date': 'April 16, 2025', 'time': '16:30', 'meal_type': 'LUNCH'},  # Afternoon snack as lunch
            {'datetime': datetime(2025, 4, 16, 19, 30), 'date': 'April 16, 2025', 'time': '19:30', 'meal_type': 'DINNER'},
            
            # April 17, 2025 (4 meals)
            {'datetime': datetime(2025, 4, 17, 7, 30), 'date': 'April 17, 2025', 'time': '07:30', 'meal_type': 'BREAKFAST'},
            {'datetime': datetime(2025, 4, 17, 12, 30), 'date': 'April 17, 2025', 'time': '12:30', 'meal_type': 'LUNCH'},
            {'datetime': datetime(2025, 4, 17, 17, 0), 'date': 'April 17, 2025', 'time': '17:00', 'meal_type': 'DINNER'},  # Early dinner
            {'datetime': datetime(2025, 4, 17, 20, 30), 'date': 'April 17, 2025', 'time': '20:30', 'meal_type': 'DINNER'},  # Late dinner/snack
            
            # April 18, 2025 (4 meals)
            {'datetime': datetime(2025, 4, 18, 8, 30), 'date': 'April 18, 2025', 'time': '08:30', 'meal_type': 'BREAKFAST'},
            {'datetime': datetime(2025, 4, 18, 11, 0), 'date': 'April 18, 2025', 'time': '11:00', 'meal_type': 'BREAKFAST'},  # Brunch
            {'datetime': datetime(2025, 4, 18, 14, 0), 'date': 'April 18, 2025', 'time': '14:00', 'meal_type': 'LUNCH'},
            {'datetime': datetime(2025, 4, 18, 19, 0), 'date': 'April 18, 2025', 'time': '19:00', 'meal_type': 'DINNER'},
            
            # April 19, 2025 (3 meals - standard)
            {'datetime': datetime(2025, 4, 19, 7, 0), 'date': 'April 19, 2025', 'time': '07:00', 'meal_type': 'BREAKFAST'},
            {'datetime': datetime(2025, 4, 19, 13, 30), 'date': 'April 19, 2025', 'time': '13:30', 'meal_type': 'LUNCH'},
            {'datetime': datetime(2025, 4, 19, 18, 0), 'date': 'April 19, 2025', 'time': '18:00', 'meal_type': 'DINNER'},
            
            # April 20, 2025 (4 meals)
            {'datetime': datetime(2025, 4, 20, 9, 0), 'date': 'April 20, 2025', 'time': '09:00', 'meal_type': 'BREAKFAST'},
            {'datetime': datetime(2025, 4, 20, 12, 0), 'date': 'April 20, 2025', 'time': '12:00', 'meal_type': 'LUNCH'},
            {'datetime': datetime(2025, 4, 20, 14, 30), 'date': 'April 20, 2025', 'time': '14:30', 'meal_type': 'LUNCH'},  # Late lunch
            {'datetime': datetime(2025, 4, 20, 18, 30), 'date': 'April 20, 2025', 'time': '18:30', 'meal_type': 'DINNER'},
            
            # April 21, 2025 (4 meals)
            {'datetime': datetime(2025, 4, 21, 8, 0), 'date': 'April 21, 2025', 'time': '08:00', 'meal_type': 'BREAKFAST'},
            {'datetime': datetime(2025, 4, 21, 10, 30), 'date': 'April 21, 2025', 'time': '10:30', 'meal_type': 'BREAKFAST'},  # Late breakfast/snack
            {'datetime': datetime(2025, 4, 21, 13, 0), 'date': 'April 21, 2025', 'time': '13:00', 'meal_type': 'LUNCH'},
            {'datetime': datetime(2025, 4, 21, 19, 30), 'date': 'April 21, 2025', 'time': '19:30', 'meal_type': 'DINNER'},
            
            # April 22, 2025 (3 meals - standard)
            {'datetime': datetime(2025, 4, 22, 7, 30), 'date': 'April 22, 2025', 'time': '07:30', 'meal_type': 'BREAKFAST'},
            {'datetime': datetime(2025, 4, 22, 14, 0), 'date': 'April 22, 2025', 'time': '14:00', 'meal_type': 'LUNCH'},
            {'datetime': datetime(2025, 4, 22, 20, 0), 'date': 'April 22, 2025', 'time': '20:00', 'meal_type': 'DINNER'},
            
            # April 23, 2025 (4 meals - final day)
            {'datetime': datetime(2025, 4, 23, 9, 30), 'date': 'April 23, 2025', 'time': '09:30', 'meal_type': 'BREAKFAST'},
            {'datetime': datetime(2025, 4, 23, 12, 30), 'date': 'April 23, 2025', 'time': '12:30', 'meal_type': 'LUNCH'},
            {'datetime': datetime(2025, 4, 23, 15, 30), 'date': 'April 23, 2025', 'time': '15:30', 'meal_type': 'LUNCH'},  # Afternoon snack
            {'datetime': datetime(2025, 4, 23, 19, 0), 'date': 'April 23, 2025', 'time': '19:00', 'meal_type': 'DINNER'},
        ]
        
        # Sort by datetime to ensure chronological order
        realistic_meals.sort(key=lambda x: x['datetime'])
        
        # Print realistic meal schedule
        print(f"\n🍽️  Realistic Human Meal Schedule ({len(realistic_meals)} meals within CGM data range):")
        meal_counts = {'BREAKFAST': 0, 'LUNCH': 0, 'DINNER': 0}
        current_date = None
        
        for meal in realistic_meals:
            meal_counts[meal['meal_type']] += 1
            if meal['date'] != current_date:
                print(f"\n{meal['date']}:")
                current_date = meal['date']
            print(f"  {meal['time']} - {meal['meal_type']}")
        
        print(f"\n📊 Meal Distribution:")
        for meal_type, count in meal_counts.items():
            print(f"  {meal_type}: {count} meals")
        
        print(f"\n✅ CGM Data Coverage:")
        print(f"  Date Range: April 16-23, 2025 (8 days)")
        print(f"  All {len(realistic_meals)} meals within CGM data range")
        
        return realistic_meals
    
    def _initialize_results_csv(self):
        """Initialize results CSV with headers including markdown data"""
        if not os.path.exists(self.results_csv):
            headers = [
                'image_name', 'processing_date', 'meal_date', 'meal_time', 'meal_type', 
                'raw_metrics_json', 'insights_text', 'raw_data_passed'
            ]
            with open(self.results_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def _get_image_files(self) -> List[str]:
        """Get list of image files to process"""
        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)
            
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.images_folder, ext)))
            image_files.extend(glob.glob(os.path.join(self.images_folder, ext.upper())))
        
        return sorted(image_files)
    
    def _get_reference_meals_from_slot(self, meal_datetime: datetime, meal_type: str) -> List[Dict]:
        """Get reference meals from same slot using actual food data or detected patterns"""
        reference_meals = []
        
        try:
            # Try to load actual food data first
            if os.path.exists(self.food_path):
                food_df = pd.read_csv(self.food_path)
                if not food_df.empty:
                    food_df['datetime'] = pd.to_datetime(food_df['date'] + ' ' + food_df['time'])
                    
                    # Filter for same slot in last 7 days
                    seven_days_ago = meal_datetime - timedelta(days=7)
                    same_slot_meals = food_df[
                        (food_df['slot'] == meal_type) &
                        (food_df['datetime'] >= seven_days_ago) &
                        (food_df['datetime'] < meal_datetime)
                    ]
                    
                    # Convert to list of dicts
                    for _, meal in same_slot_meals.iterrows():
                        reference_meals.append({
                            'date': meal['date'],
                            'time': meal['time'],
                            'slot': meal['slot'],
                            'meal name': meal['meal name']
                        })
            
            # If no actual meals found, use detected patterns
            if not reference_meals:
                df = self.analyzer.load_cgm_data(self.cgm_path)
                patterns = self.analyzer.detect_glucose_rise_patterns(df, min_rise=15)
                
                seven_days_ago = meal_datetime - timedelta(days=7)
                for pattern in patterns:
                    pattern_datetime = pattern['datetime']
                    
                    if (pattern['meal_type'] == meal_type and 
                        seven_days_ago <= pattern_datetime < meal_datetime):
                        reference_meals.append({
                            'date': pattern['date'],
                            'time': pattern['time'],
                            'slot': pattern['meal_type'],
                            'meal name': f'Detected {meal_type.title()} Meal'
                        })
            
            # Limit to 5 most recent
            if reference_meals:
                reference_meals.sort(key=lambda x: x['date'] + ' ' + x['time'], reverse=True)
                reference_meals = reference_meals[:5]
                
        except Exception as e:
            print(f"❌ Error getting reference meals: {e}")
        
        return reference_meals
    
    def _load_cgm_data_as_list(self) -> List[Dict]:
        """Load CGM data as list of dicts for metrics calculator"""
        df = pd.read_csv(self.cgm_path)
        return df.to_dict('records')
    
    async def process_single_image(self, image_path: str, meal_point: Dict, model_choice: str = "flash") -> Dict:
        """Process a single image with CGM insights using MealImpactAnalysis model"""
        try:
            # Load image
            image = Image.open(image_path)
            image_name = os.path.basename(image_path)
            
            print(f"🔄 Processing {image_name} for {meal_point['meal_type']} at {meal_point['date']} {meal_point['time']}")
            
            # Create mock meal time
            meal_datetime = datetime.strptime(f"{meal_point['date']} {meal_point['time']}", "%B %d, %Y %H:%M")
            
            # Load CGM data
            cgm_data = self._load_cgm_data_as_list()
            
            # ✅ Create current meal data structure
            current_meal_data = [{
                'date': meal_point['date'],
                'time': meal_point['time'],
                'slot': meal_point['meal_type'],
                'meal name': f'Live Upload - {meal_point["meal_type"].title()} Meal'
            }]
            
            # ✅ Get reference meals from same slot
            reference_meals_data = self._get_reference_meals_from_slot(meal_datetime, meal_point['meal_type'])
            
            # ✅ USE METRICS.PY TO CREATE EXACT MODEL STRUCTURE
            meal_impact_analysis: MealImpactAnalysis = self.metrics_calc.create_meal_impact_analysis(
                cgm_data, current_meal_data, reference_meals_data
            )
            
            # ✅ Convert to dict for API and storage
            meal_analysis_dict = asdict(meal_impact_analysis)
            
            # ✅ Prepare data for Gemini API
            current_meal_details = {
                'meal_time': meal_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'meal_type': meal_point['meal_type'],
                'meal_date': meal_point['date'],
                'meal_time_str': meal_point['time']
            }
            
            reference_meal_details = [
                {
                    'meal_name': ref_meal.get('meal name', 'Unknown'),
                    'date': ref_meal.get('date', ''),
                    'time': ref_meal.get('time', ''),
                    'slot': ref_meal.get('slot', '')
                } for ref_meal in reference_meals_data
            ]
            
            # ✅ Generate markdown formatted data for LLM and frontend
            markdown_data = self.gemini_api.get_formatted_data_for_frontend(
                meal_analysis_dict,
                meal_analysis_dict["reference_meals"],
                current_meal_details,
                reference_meal_details
            )
            
            # ✅ Generate insights using structured data (SIMPLE VERSION)
            insights_text = ""
            try:
                insights_chunks = []
                async for chunk in self.gemini_api.get_insights_stream(
                    choice=model_choice,
                    current_meal_metrics=meal_analysis_dict,
                    reference_meals_metrics=meal_analysis_dict["reference_meals"],
                    meal_image=image,
                    current_meal_details=current_meal_details,
                    reference_meal_details=reference_meal_details
                ):
                    insights_chunks.append(chunk)
                insights_text = ''.join(insights_chunks)
            except Exception as e:
                insights_text = f"Error generating insights: {str(e)}"
                print(f"❌ Insight generation error: {e}")
            
            # ✅ Store structured result with markdown data  
            result = {
                'image_name': image_name,
                'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'meal_date': meal_point['date'],
                'meal_time': meal_point['time'],
                'meal_type': meal_point['meal_type'],
                'raw_metrics_json': json.dumps(meal_analysis_dict, default=str),
                'insights_text': insights_text,
                'raw_data_passed': markdown_data
            }
            
            print(f"✅ Completed {image_name} - Found {meal_impact_analysis.reference_meal_count} reference meals")
            return result
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_result_to_csv(self, result: Dict):
        """Save single result to CSV"""
        if result is None:
            return
            
        try:
            # Check if result already exists
            existing_results = []
            if os.path.exists(self.results_csv):
                with open(self.results_csv, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    existing_results = list(reader)
            
            # Check for duplicate
            for existing in existing_results:
                if existing['image_name'] == result['image_name']:
                    print(f"⚠️ Result for {result['image_name']} already exists, skipping...")
                    return
            
            # Append new result
            with open(self.results_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())
                if os.path.getsize(self.results_csv) == 0:  # Empty file
                    writer.writeheader()
                writer.writerow(result)
                
            print(f"💾 Saved result for {result['image_name']}")
            
        except Exception as e:
            print(f"❌ Error saving result: {e}")
    
    async def process_all_images(self, model_choice: str = "flash", max_concurrent: int = 2):
        """Process all images in the images folder"""
        print("🚀 Starting batch processing...")
        
        # Initialize CSV
        self._initialize_results_csv()
        
        # Get image files
        image_files = self._get_image_files()
        
        if not image_files:
            print("❌ No images found in images/ folder")
            return []
        
        if len(image_files) > len(self.meal_points):
            print(f"⚠️ Found {len(image_files)} images but only {len(self.meal_points)} meal points")
            print("Using first 30 images...")
            image_files = image_files[:30]
        
        print(f"📊 Processing {len(image_files)} images with {len(self.meal_points)} meal points")
        
        # Process images with semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(image_path, meal_point):
            async with semaphore:
                result = await self.process_single_image(image_path, meal_point, model_choice)
                if result:
                    self._save_result_to_csv(result)
                return result
        
        # Create tasks
        tasks = []
        for i, image_path in enumerate(image_files):
            if i < len(self.meal_points):
                meal_point = self.meal_points[i]
                task = process_with_semaphore(image_path, meal_point)
                tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Summary
        successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        failed = len(results) - successful
        
        print(f"\n🎉 Batch processing complete!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📄 Results saved to: {self.results_csv}")
        
        return [r for r in results if r is not None and not isinstance(r, Exception)]

# Convenience function for app.py integration
async def run_batch_processing(model_choice: str = "flash"):
    """Run batch processing - called from app.py"""
    processor = BatchProcessor()
    return await processor.process_all_images(model_choice)

if __name__ == "__main__":
    # Test run
    async def main():
        processor = BatchProcessor()
        await processor.process_all_images("flash")
    
    asyncio.run(main())