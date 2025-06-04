from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
import os
import json
load_dotenv()

system_prompt = """
You are a warm, supportive health coach specializing in diabetes and CGM-based food insights.
Your role is to generate a short, personalized insight (max 3–4 sentences) for the current meal the user is about to eat, mostly Indian cuisine, based on:
Their meal image analysis (carb/protein/fat & calories estimates)

Recent CGM trends (past 7 days)

Similar meals in the past and their glucose impact

User profile context (goals, HbA1c, conditions, medications)

Process :
For food image analysis, identify and disambiguate the dishes and ingredients, estimate calories and macro, and use the same with CGM data and historical data to provide a personalized insight.

Your output should feel like a friendly check-in from a coach. The insight should gently guide the user on what to do as per following categories:
- Swap/Substitute Meals/Ingredients
- Reduce Portion
- Increase/Add specific macro
- Proceed as-is
- Food Sequencing (Suggest a meal eating sequence that will be best for their glucose. Eg:- "Try eating salad first, then take a small break and have the rest")
- Activity (Some pre-meal or post-meal activity to control glucose levels)
- Hydration or Beverage Choices (Opting for extra hydration or non-sugary beverages or diluted ACV ~5ml in water)
- Cautionary Alert (If food choice has chances of causing really significant glucose rise such as more than 120-150 mg/dL rise or chances of peak going beyond 250 mg/dL or so)

Reference past patterns (e.g., "last time this caused a 90 mg/dL spike"), and connect that meaningfully to this meal. Encourage even small positive behaviors and help the user feel understood, not judged.

Format your response as JSON:
{
  "meal_macros": {
    "carbohydrates_g": 0,
    "protein_g": 0,
    "fat_g": 0
  },
  "estimated_calories" : 0,
  "core_insight": "Short, personalized insight here (max 3–4 sentences).",
  "recommendation": "swap" | "reduce_portion" | "proceed_as_is"
}

Keep it friendly, compassionate, and empowering — like talking to someone managing diabetes with care.
"""

user_profile = """
User Profile :
Name: Neera
User: 25-year-old female
Weight: 75-78 kg
Activity Level: Sedentary
Known Conditions:
- Diabetes (on Ryzodeg 30/70 insulin)
- Hypothyroidism (on Thyronorm 75mcg - TSH 0.571, controlled)

Key Issues for Dietary Intervention (based on latest data):
- Severe Glycemic Imbalance:
    - HbA1c: 8.8% (despite insulin)
    - Fasting Glucose: 191 mg/dL
    - Random Glucose: 205.86 mg/dL
- Nutritional Concerns:
    - Mild Anemia: Hb 12 g/dL (MCHC 31.8, MCH 26.1)
    - Elevated ESR: 34 mm/hr (inflammation marker)
- Supplements: On A TO Z Gold multivitamin.

Primary Dietetic Goals for this User:
- Improve blood sugar control significantly.
- Address anemia through diet and support inflammation reduction.
- Support weight management efforts.
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

## 📈 Reference Meals from Same Time Slot (Last 7 Days)
**Total Reference Meals Found:** {current_meal_metrics.get('reference_meal_count', 0)}

"""
        
        if reference_meals_metrics and len(reference_meals_metrics) > 0:
            for i, ref_meal in enumerate(reference_meals_metrics, 1):
                ref_detail = reference_meal_details[i-1] if i-1 < len(reference_meal_details) else {}
                
                return_status = f"{ref_meal.get('return_to_baseline_minutes', 'Did not return')} minutes" if ref_meal.get('return_to_baseline_minutes') else "Did not return to baseline within 3h"
                
                markdown += f"""
### Reference Meal #{i}: {ref_detail.get('meal_name', 'Unknown')}
**Date/Time:** {ref_meal.get('meal_date', 'Unknown')} at {ref_meal.get('meal_time', 'Unknown')}

**Pre-meal Metrics:**
- Baseline Glucose: {ref_meal.get('baseline_glucose', 'N/A')} mg/dL

**Post-meal Impact (3h window):**  
- Peak Glucose: {ref_meal.get('peak_postprandial_glucose', 'N/A')} mg/dL
- Time to Peak: {ref_meal.get('time_to_peak_minutes', 'N/A')} minutes
- 3h Post-meal Average: {ref_meal.get('avg_glucose_3h_post', 'N/A')} mg/dL
- Return to Baseline: {return_status}
- **3h Post-meal CV:** {ref_meal.get('cv_3h_post_percent', 'N/A')}%

"""
        else:
            markdown += "\n*No reference meals found for this time slot in the last 7 days.*\n"
        
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
                temperature=0.4,
            )
        )

        async for chunk in response:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text
    
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
        


        