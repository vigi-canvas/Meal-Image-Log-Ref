from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
import os
import json
#load_dotenv()

system_prompt = """
You are a warm and supportive health coach specializing in diabetes management with CGM-based food insights. Your role is to generate a short, personalized insight (3–4 sentences) about the meal the user is about to eat (mostly Indian cuisine), based on:
Meal image analysis: Identify all dishes and ingredients, estimate portion sizes (using standard Indian measures, e.g., 1 katori, 1 bowl, 1 tbsp), and calculate macros (carbs, protein, fat) and calories.
Recent CGM trends (past 7 days).
Similar past meals and their glucose impact.
User profile (goals, HbA1c, conditions, medications).

Process:
Analyze the meal image to identify and disambiguate the dishes and estimate their quantities and nutrition.
Combine these estimates with the user's CGM trends and historical meal responses to generate a personalized insight.

Insight Style:
Friendly, compassionate, and empowering (like a caring coach check-in).
Reference past glucose patterns (e.g., "last time this meal caused a 90 mg/dL spike") to provide context.
Encourage small positive steps and reassure the user; make them feel understood, not judged.

Recommendations:
Choose one category for guidance:
- swap_meal: Suggest swapping or substituting meals/ingredients.
- reduce_portion: Suggest reducing portion size.
- increase_add_macro: Suggest adding a specific macronutrient.
- proceed_as_is: Suggest proceeding with the meal as-is.
- food_sequence: Suggest a meal eating sequence (e.g., eat a salad first, pause, then the rest).
- activity: Suggest a pre- or post-meal activity to help control glucose.
- hydration: Suggest drinking water or a non-sugary beverage (e.g., diluted apple cider vinegar).
- cautionary_alert: Warn if this meal could cause a very high glucose spike (e.g., >120–150 mg/dL rise).

Output Format:
Important: Respond with only the JSON object, no additional text or markdown.
Include a dishes field: An array of objects, each with "name" (dish name) and "quantity" (estimated portion, e.g., "1 katori", "1 bowl").

Format your response as a valid JSON object:
{
  "meal_macros": {
    "carbohydrates_g": 0,
    "protein_g": 0,
    "fat_g": 0
  },
  "dishes": [
    {"name": "curd", "quantity": "1 katori"},
    {"name": "rice", "quantity": "1 bowl"}
  ],
  "estimated_calories": 0,
  "core_insight": "Short, personalized insight here (3–4 sentences).",
  "recommendation": "swap_meal"
}

Keep the tone friendly, compassionate, and empowering—like talking to someone managing diabetes with care.
"""

user_profile = """
User Profile :
Name: Alex
User: 28-year-old individual
Weight: 65-68 kg
Activity Level: Moderately active
Known Conditions:
- Type 1 Diabetes (on insulin therapy)
- Recurrent hypoglycemia episodes

Key Issues for Dietary Intervention (based on CGM data analysis):
- Frequent Hypoglycemic Episodes:
    - Mean glucose: 83.5 mg/dL (on lower end of normal)
    - 10% of readings below 70 mg/dL (hypoglycemia)
    - 0.2% severe hypoglycemia below 54 mg/dL
    - Lowest recorded: 40.0 mg/dL (dangerous low)
- Good Overall Control:
    - 90% Time in Range (70-180 mg/dL) - excellent
    - No hyperglycemia (max 116.4 mg/dL)
    - Well-controlled when not hypoglycemic
- Risk Factors:
    - Frequent glucose drops, especially post-meal periods
    - Need for hypoglycemia prevention strategies

Primary Dietetic Goals for this User:
- Prevent hypoglycemic episodes through strategic meal timing and composition
- Maintain stable glucose levels without causing hypoglycemia
- Focus on complex carbohydrates and protein for sustained energy
- Optimize pre-meal glucose levels to prevent post-meal drops
- Educate on recognizing and managing hypoglycemia warning signs
"""

class GeminiAPI:
    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            location=os.getenv('GOOGLE_CLOUD_LOCATION'),
        )

        self.GEMINI_FLASH_MODEL = "gemini-2.5-flash-preview-04-17"
        self.GEMINI_PRO_MODEL = "gemini-2.5-pro-preview-05-06"
    
    def format_meal_data_to_markdown(self, current_meal_metrics, reference_meals_metrics, current_meal_details, reference_meal_details):
        """Format structured meal data into clear markdown for LLM"""
        
        markdown = f"""
## 📊 Current Meal Being Analyzed
**Meal Time:** {current_meal_details.get('meal_date', 'Unknown')} at {current_meal_details.get('meal_time_str', 'Unknown')}  
**Meal Type:** {current_meal_details.get('meal_type', 'Unknown')}  
**Pre-meal Glucose Status:** {current_meal_metrics.get('glucose_at_meal_time', 'N/A')} mg/dL

### Current Meal Pre-meal Metrics:
- **3h Pre-meal Average:** {current_meal_metrics.get('avg_glucose_3h_pre', 'N/A')} mg/dL
- **1h Pre-meal Average:** {current_meal_metrics.get('avg_glucose_1h_pre', 'N/A')} mg/dL  
- **Baseline Glucose (30min avg):** {current_meal_metrics.get('baseline_glucose', 'N/A')} mg/dL
- **Glucose at Meal Time:** {current_meal_metrics.get('glucose_at_meal_time', 'N/A')} mg/dL
- **3h Pre-meal CV:** {current_meal_metrics.get('cv_3h_pre_percent', 'N/A')}%
- **1h Pre-meal CV:** {current_meal_metrics.get('cv_1h_pre_percent', 'N/A')}%
- **Pre-meal Glucose Trend:** {current_meal_metrics.get('pre_meal_trend_per_5min', 0):+.1f} mg/dL per 5min

## 📈 Reference Meals from Same Time Slot (Last 7 Days)
**Total Reference Meals Found:** {len(reference_meals_metrics)}

"""
        
        # ✅ Handle BOTH dataclass objects AND dictionaries
        for i, ref_meal in enumerate(reference_meals_metrics, 1):
            meal_detail = reference_meal_details[i-1] if i-1 < len(reference_meal_details) else {}
            
            # ✅ Check if it's a dataclass object or dictionary
            if hasattr(ref_meal, 'meal_date'):  # Dataclass object
                meal_date = ref_meal.meal_date
                meal_time = ref_meal.meal_time
                baseline_glucose = ref_meal.baseline_glucose
                pre_meal_trend = ref_meal.pre_meal_trend_per_5min
                peak_glucose = ref_meal.peak_postprandial_glucose
                time_to_peak = ref_meal.time_to_peak_minutes
                avg_3h_post = ref_meal.avg_glucose_3h_post
                return_baseline = ref_meal.return_to_baseline_minutes
                cv_post = ref_meal.cv_3h_post_percent
            else:  # Dictionary (from asdict conversion)
                meal_date = ref_meal.get('meal_date', 'Unknown')
                meal_time = ref_meal.get('meal_time', 'Unknown')
                baseline_glucose = ref_meal.get('baseline_glucose', 'N/A')
                pre_meal_trend = ref_meal.get('pre_meal_trend_per_5min', 0)
                peak_glucose = ref_meal.get('peak_postprandial_glucose', 'N/A')
                time_to_peak = ref_meal.get('time_to_peak_minutes', 'N/A')
                avg_3h_post = ref_meal.get('avg_glucose_3h_post', 'N/A')
                return_baseline = ref_meal.get('return_to_baseline_minutes', None)
                cv_post = ref_meal.get('cv_3h_post_percent', 'N/A')
            
            # ✅ Safe trend direction calculation
            trend_direction = ""
            if isinstance(pre_meal_trend, (int, float)):
                if pre_meal_trend > 1:
                    trend_direction = "↗️ Rising"
                elif pre_meal_trend < -1:
                    trend_direction = "↘️ Falling"
                else:
                    trend_direction = "➡️ Stable"
            
            # ✅ Safe return baseline text
            return_text = "Did not return to baseline within 3h"
            if return_baseline is not None and isinstance(return_baseline, (int, float)):
                return_text = f"{return_baseline:.0f} minutes"
            
            markdown += f"""
### Reference Meal #{i}: {meal_detail.get('meal_name', 'Unknown Meal')}
**Date/Time:** {meal_date} at {meal_time}

**Pre-meal Metrics:**
- Baseline Glucose: {baseline_glucose} mg/dL
- Pre-meal Trend: {pre_meal_trend:+.1f} mg/dL per 5min {trend_direction}

**Post-meal Impact (3h window):**
- Peak Glucose: {peak_glucose} mg/dL
- Time to Peak: {time_to_peak} minutes
- 3h Post-meal Average: {avg_3h_post} mg/dL
- Return to Baseline: {return_text}
- 3h Post-meal CV: {cv_post}%

"""
        
        return markdown

    async def get_insights_stream(self, choice, current_meal_metrics, reference_meals_metrics, meal_image, current_meal_details, reference_meal_details):
        # Format data to markdown for structured presentation
        markdown_data = self.format_meal_data_to_markdown(
            current_meal_metrics, 
            reference_meals_metrics, 
            current_meal_details, 
            reference_meal_details
        )
        
        # Create prompt with formatted markdown context
        prompt = f"""
        {user_profile}
        
        {markdown_data}
        
        Please analyze this meal image along with the provided CGM data and meal context to provide comprehensive insights.
        """
        
        # Prepare contents for the API call
        contents = [prompt, meal_image]
        model = self.GEMINI_PRO_MODEL if choice == "pro" else self.GEMINI_FLASH_MODEL
        response = await self.client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7
            )
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def get_insights_simple(self, choice, current_meal_metrics, reference_meals_metrics, meal_image, current_meal_details, reference_meal_details):
        """Fast non-streaming version for single image analysis"""
        # Format data to markdown for structured presentation
        markdown_data = self.format_meal_data_to_markdown(
            current_meal_metrics, 
            reference_meals_metrics, 
            current_meal_details, 
            reference_meal_details
        )
        
        # Create prompt with formatted markdown context
        prompt = f"""
        {user_profile}
        
        {markdown_data}
        
        Please analyze this meal image along with the provided CGM data and meal context to provide comprehensive insights.
        """
        
        # Prepare contents for the API call
        contents = [prompt, meal_image]
        model = self.GEMINI_PRO_MODEL if choice == "pro" else self.GEMINI_FLASH_MODEL
        
        # Get complete response at once (faster for single images)
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7
            )
        )
        
        return response.text if response.text else "No insights generated"
    
    def get_formatted_data_for_frontend(self, current_meal_metrics, reference_meals_metrics, current_meal_details, reference_meal_details):
        """
        Get formatted markdown data for frontend table display
        """
        return self.format_meal_data_to_markdown(
            current_meal_metrics, 
            reference_meals_metrics, 
            current_meal_details, 
            reference_meal_details
        )
        


        