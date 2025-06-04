import streamlit as st
import pandas as pd
import csv
import os
import asyncio
import threading
import queue
from datetime import datetime, timedelta
from metrics import CGMMetricsCalculator
from gemini_api import GeminiAPI
from PIL import Image
import time
import base64
import json
from dataclasses import asdict

# Import batch processor and models
from batch_processor import run_batch_processing
from models import MealImpactAnalysis, ReferenceMealAnalysis

# Page config
st.set_page_config(
    page_title="Meal Log Insights",
    page_icon="🍽️",
    layout="wide"
)

# Title
st.title("🍽️ Meal Log Insights")
st.markdown("Upload a meal image and get personalized CGM insights")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📸 Single Image", "🔄 Batch Processing", "📊 Past Results"])

with tab1:
    st.header("📸 Single Meal Analysis")
    
    # ✅ Clean UI Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        # Model selection as buttons with icons
        st.markdown("**🤖 AI Model:**")
        model_col1, model_col2 = st.columns(2)
        with model_col1:
            flash_selected = st.button("⚡ Flash", use_container_width=True, 
                                    help="Faster analysis")
        with model_col2:
            pro_selected = st.button("🚀 Pro", use_container_width=True,
                                   help="More detailed analysis")
        
        # Set model choice based on button clicks
        if 'model_choice' not in st.session_state:
            st.session_state.model_choice = "flash"  # Default
        
        if flash_selected:
            st.session_state.model_choice = "flash"
        elif pro_selected:
            st.session_state.model_choice = "pro"
        
        # Show selected model
        st.info(f"Selected: Gemini 2.5 {'⚡ Flash' if st.session_state.model_choice == 'flash' else '🚀 Pro'}")
        
        # File upload
        uploaded_file = st.file_uploader(
        "Choose a meal image...", 
        type=['png', 'jpg', 'jpeg'],
            help="Upload an image of your meal for analysis"
        )
        
        # Image preview
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded meal image", use_column_width=True)
    
    with col2:
        # ✅ FIXED: Date/Time/Slot Selection with session state
        st.markdown("**Meal Details:**")
        
        # Initialize session state for time input
        if 'meal_time_input' not in st.session_state:
            st.session_state.meal_time_input = datetime.now().strftime("%H:%M")
        
        meal_date = st.date_input(
            "📅 Meal Date", 
            value=datetime.now().date(),
            help="Select the date when you had this meal"
        )
        
        # ✅ FIXED: Time input with session state to prevent reversion
        meal_time_str = st.text_input(
            "🕐 Meal Time (24h format)", 
            value=st.session_state.meal_time_input,
            help="Enter time in 24-hour format (e.g., 14:20, 08:30, 19:45)",
            key="meal_time_key"
        )
        
        # Update session state when time changes
        if meal_time_str != st.session_state.meal_time_input:
            st.session_state.meal_time_input = meal_time_str
        
        meal_slot = st.selectbox(
            "🍽️ Meal Type",
            ["BREAKFAST", "LUNCH", "DINNER"],
            index=1,  # Default to LUNCH
            help="Select the type of meal"
        )

    # Store in session state
    st.session_state.selected_model = st.session_state.model_choice

    # ✅ Fixed User ID (hardcoded, removed input)
    user_id = "100022075"  # Fixed user ID
    
    # Cache for data loading
    @st.cache_data
    def load_cgm_data(user_id):
        """Load and cache CGM data"""
        csv_path = f"{user_id}/CGM.csv"
        if not os.path.exists(csv_path):
            st.error(f"CGM data file not found: {csv_path}")
            return None
        
        cgm_data = []
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            cgm_data = list(reader)
        return cgm_data

    @st.cache_data  
    def load_food_data(user_id):
        """Load and cache food data"""
        csv_path = f"{user_id}/Food.csv"
        if not os.path.exists(csv_path):
            return []
        
        food_data = []
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            food_data = list(reader)
        return food_data

    def optimized_stream_insights(gemini_api, model_choice, current_meal_metrics, reference_meals_metrics, 
                                meal_image, current_meal_details, reference_meal_details):
        """Fixed streaming function without event loop conflicts"""
        try:
            import asyncio
            
            async def get_insights():
                insights = ""
                async for chunk in gemini_api.get_insights_stream(
                        choice=model_choice,
                        current_meal_metrics=current_meal_metrics,
                        reference_meals_metrics=reference_meals_metrics,
                        meal_image=meal_image,
                    current_meal_details=current_meal_details,
                    reference_meal_details=reference_meal_details
                ):
                    insights += chunk
                    yield chunk
            
            # Use asyncio.run() properly without manual loop management
            async def run_generator():
                all_chunks = []
                async for chunk in get_insights():
                    all_chunks.append(chunk)
                    yield chunk
            
            # Run the async generator in a clean way
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            async def stream_all():
                result = []
                async for chunk in run_generator():
                    result.append(chunk)
                    yield chunk
            
            # Convert to sync generator
            gen = stream_all()
            while True:
                try:
                    chunk = loop.run_until_complete(gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
                
        except Exception as e:
            yield f"Error: {str(e)}"

    # ✅ Processing section
    if uploaded_file:
        # Load data
        cgm_data = load_cgm_data(user_id)
        food_data = load_food_data(user_id)
        
        if cgm_data:
            # ✅ Validate time input
            try:
                meal_time = datetime.strptime(meal_time_str, "%H:%M").time()
                selected_datetime = datetime.combine(meal_date, meal_time)
                
                # Show current selection
                st.success(f"📊 Ready to analyze: {meal_slot} on {selected_datetime.strftime('%B %d, %Y at %H:%M')}")
                
            except ValueError:
                st.error("⚠️ Please enter time in 24-hour format (HH:MM). Examples: 08:30, 14:20, 19:45")
                st.stop()
            
            # Initialize components  
            metrics_calc = CGMMetricsCalculator(verbose=False)
            gemini_api = GeminiAPI()
            
            # Convert uploaded file to PIL Image
            meal_image = Image.open(uploaded_file)
            
            if st.button("🔍 Generate Insights", type="primary", use_container_width=True):
                with st.spinner("⚡ Processing CGM data and generating insights..."):
                    start_time = time.time()
                    
                    # ✅ USE SELECTED DATE/TIME AND SLOT
                    current_meal_data = [{
                        'date': selected_datetime.strftime('%B %d, %Y'),
                        'time': selected_datetime.strftime('%H:%M'),
                        'slot': meal_slot,
                        'meal name': f'Live Upload - {meal_slot.title()} Meal'
                    }]
                    
                    # ✅ Get reference meals from same slot in last 7 days
                    seven_days_ago = selected_datetime - timedelta(days=7)
                    reference_meals_data = []
                    
                    for meal in food_data:
                        try:
                            meal_datetime = datetime.strptime(f"{meal['date']} {meal['time']}", '%Y-%m-%d %H:%M')
                            # Same slot, within last 7 days, before selected meal time
                            if (meal['slot'] == meal_slot and 
                                        seven_days_ago <= meal_datetime < selected_datetime):
                                    reference_meals_data.append(meal)
                        except:
                            continue
                    
                    # Limit to 5 most recent reference meals
                    reference_meals_data.sort(key=lambda x: f"{x['date']} {x['time']}", reverse=True)
                    reference_meals_data = reference_meals_data[:5]
                    
                    # ✅ USE METRICS.PY TO CREATE EXACT MODEL STRUCTURE
                    meal_impact_analysis: MealImpactAnalysis = metrics_calc.create_meal_impact_analysis(
                        cgm_data, current_meal_data, reference_meals_data
                    )
                    
                    # Convert to dict for display and API
                    meal_analysis_dict = asdict(meal_impact_analysis)
                    
                    processing_time = time.time() - start_time
                    st.success(f"⚡ Data processed in {processing_time:.2f} seconds")
                    
                    # Prepare API call data
                    current_meal_details = {
                        'meal_time': selected_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                        'meal_type': meal_slot,
                        'meal_date': selected_datetime.strftime('%B %d, %Y'),
                        'meal_time_str': selected_datetime.strftime('%H:%M')
                    }
                    
                    reference_meal_details = [
                        {
                            'meal_name': meal.get('meal name', 'Unknown'),
                            'date': meal.get('date', ''),
                            'time': meal.get('time', ''),
                            'slot': meal.get('slot', '')
                        } for meal in reference_meals_data
                    ]
                    
                    # Get selected model from session state
                    selected_model = st.session_state.selected_model
                    
                    # Stream insights
                    st.header("🤖 Generated Insights")
                    insights_placeholder = st.empty()
                    insights_text = ""
                    
                    try:
                        for chunk in optimized_stream_insights(
                            gemini_api, selected_model, meal_analysis_dict,
                            meal_analysis_dict["reference_meals"],
                            meal_image, current_meal_details, reference_meal_details
                        ):
                            insights_text += chunk
                            insights_placeholder.markdown(insights_text)
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
                    
                    # ✅ Generate markdown data for LLM  
                    markdown_data = gemini_api.get_formatted_data_for_frontend(
                        meal_analysis_dict,
                        meal_analysis_dict["reference_meals"], 
                        current_meal_details,
                        reference_meal_details
                    )
                    
                    # ✅ Display raw data passed to LLM
                    st.header("📊 Raw Data Passed to LLM")
                    st.markdown(markdown_data)
                    
                    # Show summary info
                    st.info(f"📅 **Meal Time:** {selected_datetime.strftime('%B %d, %Y at %H:%M')} ({meal_slot})")
                    st.info(f"🔄 **Reference Meals:** Found {meal_impact_analysis.reference_meal_count} {meal_slot} meals from last 7 days")
                    
                    # Debug info
                    with st.expander("🔧 Data Structure Verification"):
                        st.markdown("""
                        **Verification Details:**
                        - ✅ Current meal calculated with `reference_mode=False` (pre-meal only)
                        - ✅ Reference meals calculated with `reference_mode=True` (complete metrics)
                        - ✅ Data passed to LLM: MealImpactAnalysis model structure
                        - ✅ User ID: 100022075 (fixed)
                        """)
                        st.write(f"**Total reference meals found:** {meal_impact_analysis.reference_meal_count}")
        else:
            st.error("Could not load CGM data. Please check if the data files exist.")
    else:
        st.info("👆 Please upload a meal image to get started")

with tab2:
    st.header("🔄 Batch Processing")
    st.markdown("Process all images in the `images/` folder with CGM insights")
    
    # Model selection for batch processing as buttons with icons
    st.markdown("**🤖 Select AI Model for Batch Processing:**")
    batch_model_col1, batch_model_col2 = st.columns(2)
    with batch_model_col1:
        batch_flash_selected = st.button("⚡ Flash", use_container_width=True, 
                                help="Faster for batch processing", key="batch_flash")
    with batch_model_col2:
        batch_pro_selected = st.button("🚀 Pro", use_container_width=True,
                               help="More detailed analysis", key="batch_pro")

    # Set batch model choice based on button clicks
    if 'batch_model_choice' not in st.session_state:
        st.session_state.batch_model_choice = "flash"  # Default

    if batch_flash_selected:
        st.session_state.batch_model_choice = "flash"
    elif batch_pro_selected:
        st.session_state.batch_model_choice = "pro"

    # Show selected model
    st.info(f"Selected for batch: Gemini 2.5 {'⚡ Flash' if st.session_state.batch_model_choice == 'flash' else '🚀 Pro'}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Check images folder
        images_folder = "images/"
        if os.path.exists(images_folder):
            image_files = [f for f in os.listdir(images_folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            st.info(f"📁 Found {len(image_files)} images in {images_folder}")
            
            if image_files and len(image_files) > 5:
                with st.expander("🖼️ Preview Images"):
                    for i, img_file in enumerate(image_files[:5]):  # Show first 5
                        img_path = os.path.join(images_folder, img_file)
                        img = Image.open(img_path)
                        st.image(img, caption=img_file, width=150)
                    if len(image_files) > 5:
                        st.markdown(f"... and {len(image_files) - 5} more images")
        else:
            st.warning(f"📁 Images folder `{images_folder}` not found")
    
    with col2:
        # Check results file
        results_csv = "batch_results.csv"
        if os.path.exists(results_csv):
            try:
                results_df = pd.read_csv(results_csv)
                st.info(f"📊 Found {len(results_df)} existing results in {results_csv}")
            except:
                st.warning("⚠️ Results file exists but couldn't be read")
        else:
            st.info("📊 No existing results file found")
    
    # Batch processing button
    if st.button("🚀 Process All Images", type="primary"):
        if not os.path.exists(images_folder):
            st.error(f"Images folder `{images_folder}` not found!")
        else:
            with st.spinner("🔄 Running batch processing... This may take several minutes."):
                try:
                    # Run batch processing
                    results = asyncio.run(run_batch_processing(st.session_state.batch_model_choice))
                    
                    if results:
                        st.success(f"✅ Successfully processed {len(results)} images!")
                        st.balloons()
                except Exception as e:
                    st.error(f"❌ Batch processing failed: {str(e)}")

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
            st.info("No results found. Run batch processing first.")
    else:
        st.info("No results file found. Run batch processing first.")
