import pandas as pd
import json
import re
from collections import defaultdict

def analyze_insights():
    # Load the CSV
    df = pd.read_csv('batch_results.csv')
    print(f'Total insights: {len(df)}')
    
    # Analyze insights for different categories
    insights_data = []
    for idx, row in df.iterrows():
        try:
            # Parse the insights JSON
            insights_json = json.loads(row['insights_text'])
            
            # Extract key data
            insight_data = {
                'index': idx,
                'image_name': row['image_name'],
                'meal_type': row['meal_type'],
                'meal_time': row['meal_time'],
                'meal_date': row['meal_date'],
                'recommendation': insights_json.get('recommendation', 'unknown'),
                'calories': insights_json.get('estimated_calories', 0),
                'carbs': insights_json.get('meal_macros', {}).get('carbohydrates_g', 0),
                'protein': insights_json.get('meal_macros', {}).get('protein_g', 0),
                'fat': insights_json.get('meal_macros', {}).get('fat_g', 0),
                'core_insight': insights_json.get('core_insight', ''),
                'dishes': insights_json.get('dishes', []),
                'num_dishes': len(insights_json.get('dishes', [])),
                'raw_data': row['raw_data_passed']
            }
            
            # Calculate interest scores
            insight_data['calorie_variety_score'] = min(abs(insight_data['calories'] - 200) / 50, 3)
            insight_data['macro_balance_score'] = (insight_data['protein'] > 5) + (insight_data['fat'] > 3) + (insight_data['carbs'] > 10)
            insight_data['dish_complexity_score'] = min(insight_data['num_dishes'], 3)
            insight_data['insight_length_score'] = min(len(insight_data['core_insight']) / 100, 3)
            
            insights_data.append(insight_data)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    # Analyze recommendations
    recommendations = defaultdict(int)
    for item in insights_data:
        recommendations[item['recommendation']] += 1
    
    print('\nRecommendation categories:')
    for rec, count in recommendations.items():
        print(f'{rec}: {count}')
    
    # Score insights for selection
    for item in insights_data:
        # Base score factors
        calorie_diversity = 3 if 100 < item['calories'] < 400 else 1
        macro_balance = item['macro_balance_score']
        dish_variety = item['dish_complexity_score']
        insight_quality = item['insight_length_score']
        
        # Meal type diversity bonus
        meal_type_bonus = {'BREAKFAST': 1.2, 'LUNCH': 1.1, 'DINNER': 1.0}[item['meal_type']]
        
        # Recommendation category bonus
        rec_bonus = {
            'increase_add_macro': 1.3,
            'timing_adjustment': 1.4,
            'portion_control': 1.2,
            'meal_composition': 1.3,
            'hypoglycemia_prevention': 1.5,
            'unknown': 0.8
        }.get(item['recommendation'], 1.0)
        
        # Calculate total score
        item['total_score'] = (calorie_diversity + macro_balance + dish_variety + insight_quality) * meal_type_bonus * rec_bonus
    
    # Sort by score and select top insights
    insights_data.sort(key=lambda x: x['total_score'], reverse=True)
    
    print(f'\nTop 15 insights by score:')
    print('='*80)
    
    selected_insights = []
    used_recommendations = set()
    meal_type_counts = defaultdict(int)
    
    # First pass: Ensure diversity
    for item in insights_data:
        if len(selected_insights) >= 15:
            break
            
        # Ensure we have variety in recommendations and meal types
        rec = item['recommendation']
        meal_type = item['meal_type']
        
        # Skip if we already have 3 of this recommendation type or 6 of this meal type
        if (used_recommendations.count(rec) < 3 and 
            meal_type_counts[meal_type] < 6):
            
            selected_insights.append(item)
            used_recommendations.add(rec)
            meal_type_counts[meal_type] += 1
            
            print(f"{len(selected_insights):2d}. {item['image_name'][:30]:<30} | {item['meal_type']:<9} | {item['recommendation']:<20} | Score: {item['total_score']:.2f}")
            print(f"    Calories: {item['calories']}, Dishes: {item['num_dishes']}, Insight: {item['core_insight'][:100]}...")
            print()
    
    return selected_insights

if __name__ == "__main__":
    best_insights = analyze_insights() 