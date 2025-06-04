import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class CGMMetricsCalculator:
    """
    Calculates CGM metrics for a given time frame and meal events.
    Accepts CGM data as a list of dicts (from CSV), converts to DataFrame,
    and computes metrics for each meal time.
    """

    def __init__(self, baseline_window_minutes=30, avg_3h_window=180, avg_1h_window=60, verbose=True):
        self.baseline_window = baseline_window_minutes
        self.avg_3h_window = avg_3h_window
        self.avg_1h_window = avg_1h_window
        self.verbose = verbose

    def _print_metrics(self, title, metrics_dict):
        """Print metrics to console for debugging and verification."""
        if not self.verbose:
            return
            
        print(f"\n{'='*60}")
        print(f"📊 {title}")
        print(f"{'='*60}")
        
        for key, value in metrics_dict.items():
            if isinstance(value, datetime):
                formatted_value = value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(value, float) and value is not None:
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value) if value is not None else "N/A"
            
            # Format key for better readability
            formatted_key = key.replace('_', ' ').title()
            print(f"  {formatted_key:<35}: {formatted_value}")
        
        print(f"{'='*60}\n")

    def cgm_csv_to_df(self, cgm_data):
        """
        Accepts CGM data as a list of dicts (from CSV).
        Expects keys: 'user_id', 'date', 'time', 'value'
        Returns DataFrame with 'timestamp' and 'glucose' columns.
        """
        df = pd.DataFrame(cgm_data)
        # Parse timestamp
        try:
            # Try to parse as "Month Day, Year" format
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        except:
            # If that fails, try standard format
            df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
        
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp')
        df['glucose'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['glucose'])
        
        if self.verbose:
            print(f"\n🔄 CGM Data Processing:")
            print(f"  Total CGM records processed: {len(df)}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Glucose range: {df['glucose'].min():.1f} - {df['glucose'].max():.1f} mg/dL")
        
        return df[['timestamp', 'glucose']]

    def meal_csv_to_df(self, meal_data):
        """
        Accepts meal data as a list of dicts (from CSV).
        Expects keys: 'date', 'time', 'slot', 'meal name'
        Returns DataFrame with 'timestamp', 'slot', 'meal_name' columns.
        """
        df = pd.DataFrame(meal_data)
        df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp')
        df['slot'] = df['slot'].astype(str)
        df['meal_name'] = df['meal name'].astype(str)
        
        if self.verbose:
            print(f"\n🍽️  Meal Data Processing:")
            print(f"  Total meal records processed: {len(df)}")
            if len(df) > 0:
                print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df[['timestamp', 'slot', 'meal_name']]

    def _get_window(self, df, center_time, window_minutes, before=True):
        """
        Returns a DataFrame window before or after center_time.
        Fixed to use proper time boundaries (inclusive start, exclusive end for before=True).
        """
        if before:
            start = center_time - timedelta(minutes=window_minutes)
            end = center_time
            window_df = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]
        else:
            start = center_time
            end = center_time + timedelta(minutes=window_minutes)
            window_df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
            
        if self.verbose and len(window_df) > 0:
            direction = "before" if before else "after"
            print(f"    📊 {window_minutes}min {direction} window: {len(window_df)} readings")
            print(f"        Time range: {start.strftime('%H:%M')} to {end.strftime('%H:%M')}")
            if len(window_df) > 0:
                print(f"        Glucose range: {window_df['glucose'].min():.1f} - {window_df['glucose'].max():.1f} mg/dL")
        
        return window_df

    def _get_baseline_glucose(self, df, meal_time):
        """
        Calculate baseline glucose as average of 30-minute window before meal.
        This is more accurate than using a single closest point.
        """
        baseline_window = self._get_window(df, meal_time, self.baseline_window, before=True)
        if baseline_window.empty:
            if self.verbose:
                print(f"    ⚠️  No CGM data in {self.baseline_window}min baseline window")
            return None
        
        baseline_avg = baseline_window['glucose'].mean()
        if self.verbose:
            print(f"    📊 Baseline calculation:")
            print(f"        Window: {self.baseline_window} minutes before meal")
            print(f"        Readings: {len(baseline_window)}")
            print(f"        Average glucose: {baseline_avg:.2f} mg/dL")
            print(f"        Range: {baseline_window['glucose'].min():.1f} - {baseline_window['glucose'].max():.1f} mg/dL")
        
        return baseline_avg

    def _get_value_at_time(self, df, target_time):
        """
        Returns the glucose value closest to target_time.
        Enhanced with better interpolation logic.
        """
        if df.empty:
            return None
        
        # Find closest timestamp
        time_diffs = (df['timestamp'] - target_time).abs()
        idx = time_diffs.idxmin()
        
        # If closest reading is within 15 minutes, use it
        closest_time_diff = time_diffs.loc[idx].total_seconds() / 60
        if closest_time_diff <= 15:
            glucose_value = df.loc[idx, 'glucose']
            if self.verbose:
                actual_time = df.loc[idx, 'timestamp']
                print(f"    🎯 Closest glucose to {target_time.strftime('%H:%M')}: {glucose_value:.2f} mg/dL")
                print(f"        Actual time: {actual_time.strftime('%H:%M')} ({closest_time_diff:.1f}min difference)")
            return glucose_value
        
        if self.verbose:
            print(f"    ⚠️  No CGM reading within 15min of {target_time.strftime('%H:%M')} (closest: {closest_time_diff:.1f}min)")
        return None

    def _calculate_cv(self, df):
        """
        Calculate Coefficient of Variation (CV) = (std/mean) * 100
        """
        if df.empty or len(df) < 2:
            return None
        mean_glucose = df['glucose'].mean()
        std_glucose = df['glucose'].std()
        if mean_glucose > 0:
            cv = (std_glucose / mean_glucose) * 100
            if self.verbose:
                print(f"        CV: {cv:.2f}% (std: {std_glucose:.2f}, mean: {mean_glucose:.2f})")
            return cv
        return None

    def calculate_meal_metrics(
        self,
        cgm_data,
        meal_data,
        reference_mode=False,
        postprandial_window_minutes=180
    ):
        """
        Calculates metrics for each meal event.
        If reference_mode=True, also calculates postprandial metrics.
        Returns a list of dicts, one per meal.
        """
        if self.verbose:
            print(f"\n🚀 Starting CGM Metrics Calculation")
            print(f"Reference mode: {'ON' if reference_mode else 'OFF'}")
            print(f"Postprandial window: {postprandial_window_minutes} minutes")
        
        cgm_df = self.cgm_csv_to_df(cgm_data)
        meal_df = self.meal_csv_to_df(meal_data)
        results = []

        for meal_idx, (_, meal) in enumerate(meal_df.iterrows(), 1):
            meal_time = meal['timestamp']
            slot = meal['slot']
            meal_name = meal['meal_name']

            if self.verbose:
                print(f"\n🍽️  Processing Meal {meal_idx} of {len(meal_df)}")
                print(f"  Meal: {meal_name}")
                print(f"  Slot: {slot}")
                print(f"  Time: {meal_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # 3h pre-meal window
            if self.verbose:
                print(f"\n  📊 Calculating 3-hour pre-meal metrics:")
            pre3h_df = self._get_window(cgm_df, meal_time, self.avg_3h_window, before=True)
            avg_glucose_3h_pre = pre3h_df['glucose'].mean() if not pre3h_df.empty else None
            glucose_3h_back = self._get_value_at_time(cgm_df, meal_time - timedelta(minutes=self.avg_3h_window))
            cv_3h_pre = self._calculate_cv(pre3h_df)

            # 1h pre-meal window
            if self.verbose:
                print(f"\n  📊 Calculating 1-hour pre-meal metrics:")
            pre1h_df = self._get_window(cgm_df, meal_time, self.avg_1h_window, before=True)
            avg_glucose_1h_pre = pre1h_df['glucose'].mean() if not pre1h_df.empty else None
            cv_1h_pre = self._calculate_cv(pre1h_df)

            # FIXED: Baseline glucose using average of 30-min window instead of single point
            if self.verbose:
                print(f"\n  📊 Calculating baseline glucose (30-min average):")
            baseline_glucose_30min = self._get_baseline_glucose(cgm_df, meal_time)

            # Glucose at meal time (closest reading within 15 minutes)
            if self.verbose:
                print(f"\n  📊 Finding glucose at meal time:")
            glucose_at_meal = self._get_value_at_time(cgm_df, meal_time)

            metrics = {
                'meal_time': meal_time,
                'meal_name': meal_name,
                'slot': slot,
                'avg_glucose_3h_pre': round(avg_glucose_3h_pre, 2) if avg_glucose_3h_pre is not None else None,
                'glucose_3h_back': round(glucose_3h_back, 2) if glucose_3h_back is not None else None,
                'cv_3h_pre_percent': round(cv_3h_pre, 2) if cv_3h_pre is not None else None,
                'avg_glucose_1h_pre': round(avg_glucose_1h_pre, 2) if avg_glucose_1h_pre is not None else None,
                'cv_1h_pre_percent': round(cv_1h_pre, 2) if cv_1h_pre is not None else None,
                'baseline_glucose_30min': round(baseline_glucose_30min, 2) if baseline_glucose_30min is not None else None,
                'glucose_at_meal_time': round(glucose_at_meal, 2) if glucose_at_meal is not None else None,
            }

            if reference_mode:
                if self.verbose:
                    print(f"\n  📊 Calculating postprandial metrics ({postprandial_window_minutes}min):")
                
                # Postprandial metrics (3h after meal)
                post3h_df = self._get_window(cgm_df, meal_time, postprandial_window_minutes, before=False)
                avg_glucose_3h_post = post3h_df['glucose'].mean() if not post3h_df.empty else None
                glucose_3h_after = self._get_value_at_time(cgm_df, meal_time + timedelta(minutes=postprandial_window_minutes))

                # Peak glucose in postprandial window
                peak_glucose = post3h_df['glucose'].max() if not post3h_df.empty else None
                peak_time = None
                time_to_peak = None
                
                if not post3h_df.empty and peak_glucose is not None:
                    peak_idx = post3h_df['glucose'].idxmax()
                    peak_time = post3h_df.loc[peak_idx, 'timestamp']
                    time_to_peak = (peak_time - meal_time).total_seconds() / 60
                    
                    if self.verbose:
                        print(f"    🔺 Peak glucose: {peak_glucose:.2f} mg/dL")
                        print(f"        Peak time: {peak_time.strftime('%H:%M')}")
                        print(f"        Time to peak: {time_to_peak:.1f} minutes")

                # FIXED: Use baseline average for glucose rise calculation
                glucose_rise = None
                if peak_glucose and baseline_glucose_30min:
                    glucose_rise = max(0, peak_glucose - baseline_glucose_30min)  # Ensure non-negative
                    if self.verbose:
                        print(f"    📈 Glucose rise: {glucose_rise:.2f} mg/dL")
                        print(f"        (Peak {peak_glucose:.2f} - Baseline {baseline_glucose_30min:.2f})")
                
                # Time to return to baseline (within 15% tolerance, more realistic than 10%)
                baseline_threshold = baseline_glucose_30min * 1.15 if baseline_glucose_30min else None
                return_to_baseline_time = None
                
                if baseline_threshold and not post3h_df.empty and peak_time is not None:
                    # Look for return to baseline only after peak
                    post_peak_data = post3h_df[post3h_df['timestamp'] > peak_time]
                    baseline_return = post_peak_data[post_peak_data['glucose'] <= baseline_threshold]
                    if not baseline_return.empty:
                        return_time = baseline_return.iloc[0]['timestamp']
                        return_to_baseline_time = (return_time - meal_time).total_seconds() / 60
                        if self.verbose:
                            print(f"    🔄 Return to baseline: {return_to_baseline_time:.1f} minutes")
                            print(f"        Threshold: ≤{baseline_threshold:.2f} mg/dL (115% of baseline)")
                    else:
                        # NEW: Estimate return time when not in window
                        # If peak is near end of window, estimate return time based on rate of decline
                        if time_to_peak and time_to_peak > (postprandial_window_minutes * 0.7):  # If peak is in last 30% of window
                            # Get last 2+ readings to calculate rate of decline
                            if len(post_peak_data) >= 2:
                                # Sort by timestamp to ensure correct order
                                post_peak_data = post_peak_data.sort_values('timestamp')
                                
                                # Get last glucose reading and its time
                                last_glucose = post_peak_data.iloc[-1]['glucose']
                                last_time = post_peak_data.iloc[-1]['timestamp']
                                
                                # Get second-to-last reading for rate calculation
                                prev_glucose = post_peak_data.iloc[-2]['glucose']
                                prev_time = post_peak_data.iloc[-2]['timestamp']
                                
                                # Calculate rate of decline (mg/dL per minute)
                                time_diff_minutes = (last_time - prev_time).total_seconds() / 60
                                if time_diff_minutes > 0:
                                    glucose_diff = prev_glucose - last_glucose  # Change in glucose
                                    decline_rate = glucose_diff / time_diff_minutes  # mg/dL per minute
                                    
                                    # Only estimate if glucose is actually declining
                                    if decline_rate > 0:
                                        # Calculate how much more glucose needs to drop
                                        remaining_drop = last_glucose - baseline_threshold
                                        
                                        # Estimate additional time needed
                                        if decline_rate > 0:
                                            additional_minutes = remaining_drop / decline_rate
                                            
                                            # Only use reasonable estimates (no more than 2 hours additional)
                                            if additional_minutes > 0 and additional_minutes < 120:
                                                # Total return time = time to last reading + estimated additional time
                                                return_to_baseline_time = ((last_time - meal_time).total_seconds() / 60) + additional_minutes
                                                
                                                if self.verbose:
                                                    print(f"    🔄 Estimated return to baseline: {return_to_baseline_time:.1f} minutes")
                                                    print(f"        Based on decline rate: {decline_rate:.2f} mg/dL per minute")
                
                if return_to_baseline_time is None and self.verbose:
                    print(f"    ⚠️  No return to baseline within {postprandial_window_minutes}min window")
                
                # Glucose variability (standard deviation)
                glucose_variability_post = post3h_df['glucose'].std() if not post3h_df.empty else None
                
                # Minimum glucose in post-meal period
                min_glucose_post = post3h_df['glucose'].min() if not post3h_df.empty else None
                
                # Rate of glucose rise (mg/dL per minute) - using baseline
                rate_of_rise = None
                if glucose_rise and time_to_peak and time_to_peak > 0:
                    rate_of_rise = glucose_rise / time_to_peak
                    if self.verbose:
                        print(f"    📊 Rate of rise: {rate_of_rise:.3f} mg/dL per minute")

                # Post-meal CV
                cv_3h_post = self._calculate_cv(post3h_df)

                metrics.update({
                    'avg_glucose_3h_post': round(avg_glucose_3h_post, 2) if avg_glucose_3h_post is not None else None,
                    'glucose_3h_after': round(glucose_3h_after, 2) if glucose_3h_after is not None else None,
                    'peak_postprandial_glucose': round(peak_glucose, 2) if peak_glucose is not None else None,
                    'time_to_peak_minutes': round(time_to_peak, 2) if time_to_peak is not None else None,
                    'glucose_rise_mg_dl': round(glucose_rise, 2) if glucose_rise is not None else None,
                    'return_to_baseline_minutes': round(return_to_baseline_time, 2) if return_to_baseline_time is not None else None,
                    'glucose_variability_post': round(glucose_variability_post, 2) if glucose_variability_post is not None else None,
                    'min_glucose_post': round(min_glucose_post, 2) if min_glucose_post is not None else None,
                    'rate_of_rise_mg_dl_per_min': round(rate_of_rise, 2) if rate_of_rise is not None else None,
                    'cv_3h_post_percent': round(cv_3h_post, 2) if cv_3h_post is not None else None,
                })

            # Print the complete metrics for this meal
            meal_title = f"MEAL {meal_idx}: {meal_name} ({slot}) - {meal_time.strftime('%Y-%m-%d %H:%M')}"
            self._print_metrics(meal_title, metrics)

            results.append(metrics)

        if self.verbose:
            print(f"\n✅ Completed metrics calculation for {len(results)} meals")
            
        return results

    def create_meal_impact_analysis(self, cgm_data, current_meal_data, reference_meal_data_list):
        """
        Creates MealImpactAnalysis structure using the exact model from models.py
        
        Args:
            cgm_data: List of CGM records
            current_meal_data: List with single current meal dict
            reference_meal_data_list: List of reference meal dicts
        
        Returns:
            MealImpactAnalysis with properly calculated metrics
        """
        from models import MealImpactAnalysis, ReferenceMealAnalysis
        
        if self.verbose:
            print(f"\n🏗️  Creating MealImpactAnalysis structure")
            print(f"   Current meal: {current_meal_data[0]['meal name'] if current_meal_data else 'None'}")
            print(f"   Reference meals: {len(reference_meal_data_list)}")
        
        # ✅ Calculate current meal metrics (reference_mode=False - pre-meal only)
        current_meal_metrics = self.calculate_meal_metrics(
            cgm_data, current_meal_data, reference_mode=False
        )
        
        if not current_meal_metrics:
            raise ValueError("Could not calculate current meal metrics")
        
        current_metrics = current_meal_metrics[0]
        
        # ✅ Calculate reference meals metrics (reference_mode=True - complete data)
        reference_meal_analyses = []
        for ref_meal_data in reference_meal_data_list:
            ref_metrics = self.calculate_meal_metrics(
                cgm_data, [ref_meal_data], reference_mode=True
            )
            
            if ref_metrics:
                ref_metric = ref_metrics[0]
                meal_time = ref_metric['meal_time']
                
                # Create ReferenceMealAnalysis using exact model structure
                ref_analysis = ReferenceMealAnalysis(
                    meal_date=meal_time.strftime('%Y-%m-%d'),
                    meal_time=meal_time.strftime('%H:%M'),
                    baseline_glucose=ref_metric.get('baseline_glucose_30min') or 0.0,
                    peak_postprandial_glucose=ref_metric.get('peak_postprandial_glucose') or 0.0,
                    avg_glucose_3h_post=ref_metric.get('avg_glucose_3h_post') or 0.0,
                    time_to_peak_minutes=ref_metric.get('time_to_peak_minutes') or 0.0,
                    glucose_variability_post=ref_metric.get('glucose_variability_post') or 0.0,
                    cv_3h_post_percent=ref_metric.get('cv_3h_post_percent') or 0.0,
                    return_to_baseline_minutes=ref_metric.get('return_to_baseline_minutes')  # Can be None
                )
                reference_meal_analyses.append(ref_analysis)
        
        # Create MealImpactAnalysis using exact model structure
        meal_time = current_metrics['meal_time']
        meal_impact = MealImpactAnalysis(
            meal_time=meal_time.strftime('%Y-%m-%d %H:%M:%S'),
            meal_name=current_metrics.get('meal_name') or '',
            slot=current_metrics.get('slot') or '',
            avg_glucose_3h_pre=current_metrics.get('avg_glucose_3h_pre') or 0.0,
            glucose_3h_back=current_metrics.get('glucose_3h_back') or 0.0,
            cv_3h_pre_percent=current_metrics.get('cv_3h_pre_percent') or 0.0,
            avg_glucose_1h_pre=current_metrics.get('avg_glucose_1h_pre') or 0.0,
            cv_1h_pre_percent=current_metrics.get('cv_1h_pre_percent') or 0.0,
            baseline_glucose=current_metrics.get('baseline_glucose_30min') or 0.0,
            glucose_at_meal_time=current_metrics.get('glucose_at_meal_time') or 0.0,
            reference_meals=reference_meal_analyses  # List of ReferenceMealAnalysis objects
        )
        
        if self.verbose:
            print(f"✅ Created MealImpactAnalysis with {meal_impact.reference_meal_count} reference meals")
        
        return meal_impact