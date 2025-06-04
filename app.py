import streamlit as st
import pandas as pd
import csv
import os
# import asyncio  # ✅ Comment out - not needed for results viewing
# import threading  # ✅ Comment out - not needed for results viewing
# import queue  # ✅ Comment out - not needed for results viewing
from datetime import datetime, timedelta
# from metrics import CGMMetricsCalculator  # ✅ Comment out - not needed for results viewing
# from gemini_api import GeminiAPI  # ✅ Comment out - not needed for results viewing
from PIL import Image
import time
import base64
import json
# from dataclasses import asdict  # ✅ Comment out - not needed for results viewing

# Import batch processor and models - ✅ COMMENT OUT FOR STREAMLIT DEPLOY
# from batch_processor import run_batch_processing
# from models import MealImpactAnalysis, ReferenceMealAnalysis

# Page config
st.set_page_config(
    page_title="Meal Log Insights",
    page_icon="🍽️",
    layout="wide"
)

# Title
st.title("🍽️ Meal Log Insights")
st.markdown("Upload a meal image and get personalized CGM insights")

# Create tabs - ✅ ONLY SHOW RESULTS TAB FOR DEPLOY
# tab1, tab2, tab3 = st.tabs(["📸 Single Image", "🔄 Batch Processing", "📊 Past Results"])
tab3 = st.tabs(["📊 Past Results"])[0]  # ✅ Only results tab

# ✅ COMMENT OUT TAB1 AND TAB2 ENTIRELY
"""
with tab1:
    # Single image processing code commented out for deploy
    st.info("Single image processing not available in this deployment. Use local version for full functionality.")

with tab2:
    # Batch processing code commented out for deploy  
    st.info("Batch processing not available in this deployment. Use local version for full functionality.")
"""

with tab3:
    st.header("📊 Past Results")
    st.markdown("View previously processed meal insights")
    
    results_csv = "batch_results.csv"
    if os.path.exists(results_csv):
        results_df = pd.read_csv(results_csv)
        
        if not results_df.empty:
            st.success(f"📄 Found {len(results_df)} processed meals")
            
            # ✅ Create dropdown for image selection
            image_options = []
            for idx, row in results_df.iterrows():
                option_label = f"{row['image_name']} - {row['meal_type']} on {row['meal_date']} at {row['meal_time']}"
                image_options.append(option_label)
            
            selected_image = st.selectbox(
                "🍽️ Select a processed meal to view details:",
                options=image_options,
                index=0 if image_options else None
            )
            
            if selected_image:
                # Get the selected row
                selected_idx = image_options.index(selected_image)
                selected_row = results_df.iloc[selected_idx]
                
                # ✅ Display the meal image
                image_path = f"images/{selected_row['image_name']}"
                if os.path.exists(image_path):
                    st.image(image_path, caption=f"📷 {selected_row['image_name']}", width=400)
                else:
                    st.warning(f"⚠️ Image file not found: {image_path}")
                
                # ✅ Show insights first
                st.header("🤖 Generated Insights")
                st.markdown(selected_row['insights_text'])
                
                # ✅ Show raw data passed to LLM (this has everything!)
                if 'raw_data_passed' in selected_row and pd.notna(selected_row['raw_data_passed']):
                    st.header("📊 Raw Data Passed to LLM")
                    st.markdown(selected_row['raw_data_passed'])
                else:
                    st.info("Raw data not available for this meal")
            
        else:
            st.info("No results found. Upload batch_results.csv and images folder to view results.")
    else:
        st.info("No results file found. Upload batch_results.csv to view processed meal insights.")
        st.markdown("""
        **To view results:**
        1. Upload your `batch_results.csv` file to the app directory
        2. Upload your `images/` folder with meal photos
        3. Refresh the app to see processed meal insights
        """)
