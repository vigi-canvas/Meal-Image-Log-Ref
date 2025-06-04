from typing import List, Union
from dataclasses import dataclass

@dataclass
class ReferenceMealAnalysis:
    meal_date: str
    meal_time: str
    baseline_glucose: float
    peak_postprandial_glucose: float
    avg_glucose_3h_post: float
    time_to_peak_minutes: float
    glucose_variability_post: float
    cv_3h_post_percent: float
    return_to_baseline_minutes: Union[float, None] = None  # ✅ Default value moved to end

@dataclass
class MealImpactAnalysis:
    meal_time: str
    meal_name: str
    slot: str
    avg_glucose_3h_pre: float
    glucose_3h_back: float
    cv_3h_pre_percent: float
    avg_glucose_1h_pre: float
    cv_1h_pre_percent: float
    baseline_glucose: float
    glucose_at_meal_time: float
    reference_meal_count: int = 0  # ✅ Default value moved to end
    reference_meals: Union[List[ReferenceMealAnalysis], None] = None  # ✅ Default value at end
    
    def __post_init__(self):
        if self.reference_meals is not None:
            self.reference_meal_count = len(self.reference_meals)
        else:
            self.reference_meal_count = 0