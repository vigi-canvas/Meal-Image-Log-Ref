import streamlit as st
import pandas as pd
import csv
import os
# import asyncio
# import threading
# import queue
from datetime import datetime, timedelta
# from metrics import CGMMetricsCalculator
# from gemini_api import GeminiAPI
# from PIL import Image
# import time
# import base64
# import json
# from dataclasses import asdict

# Import batch processor and models
# from batch_processor import run_batch_processing
# from models import MealImpactAnalysis, ReferenceMealAnalysis

# Page config
st.set_page_config(
    page_title="Meal Log Insights",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Meal Log Insights with CGM Data")
st.markdown("Upload meal images and get personalized insights based on your CGM patterns")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📸 Single Image", "🔄 Batch Processing", "📊 Past Results"])

with tab1:
    st.header("📸 Single Image Analysis")
    st.info("🚀 This feature is functional in local development but disabled for Streamlit deployment.")
    st.markdown("**Features available locally:**")
    st.markdown("- Real-time CGM analysis")
    st.markdown("- AI-powered meal insights")
    st.markdown("- Pre-meal glucose trend analysis")
    st.markdown("- Reference meal comparisons")
    
    # st.markdown("Upload a meal image and get real-time CGM insights")
    # 
    # # Model selection as buttons with icons
    # st.markdown("**🤖 Select AI Model:**")
    # 
    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("⚡ Gemini 2.5 Flash", key="flash", use_container_width=True):
    #         st.session_state.selected_model = "flash"
    # 
    # with col2:
    #     if st.button("🧠 Gemini 2.5 Pro", key="pro", use_container_width=True):
    #         st.session_state.selected_model = "pro"
    # 
    # # Initialize default model
    # if 'selected_model' not in st.session_state:
    #     st.session_state.selected_model = "flash"
    # 
    # # Display selected model
    # model_name = "Gemini 2.5 Flash ⚡" if st.session_state.selected_model == "flash" else "Gemini 2.5 Pro 🧠"
    # st.info(f"🤖 Selected Model: {model_name}")
    # 
    # # File upload
    # uploaded_file = st.file_uploader(
    #     "Choose a meal image...",
    #     type=['png', 'jpg', 'jpeg'],
    #     help="Upload an image of your meal to get CGM-based insights"
    # )
    # 
    # # Date and time input section
    # st.markdown("### 📅 Meal Information")
    # col1, col2, col3 = st.columns([2, 2, 2])
    # 
    # with col1:
    #     meal_date = st.date_input(
    #         "📅 Meal Date",
    #         value=datetime.now().date(),
    #         help="Select date when you had/will have this meal"
    #     )
    #
    # with col2:
    #     meal_time_str = st.text_input(
    #         "🕐 Meal Time (24h format)", 
    #         value="14:20",
    #         placeholder="HH:MM (e.g., 08:30, 14:20, 19:45)",
    #         help="Enter time in 24-hour format: HH:MM"
    #     )
    #
    # with col3:
    #     meal_slot = st.selectbox(
    #         "🍽️ Meal Type",
    #         options=["BREAKFAST", "LUNCH", "DINNER"],
    #         index=1,  # Default to LUNCH
    #         help="Select the type of meal for appropriate CGM comparison"
    #     )
    #
    # user_id = "101679328"  # Updated user ID
    #
    # @st.cache_data  
    # def load_cgm_data(user_id):
    #     """Load and cache CGM data"""
    #     csv_path = f"{user_id}/CGM.csv"
    #     if not os.path.exists(csv_path):
    #         return []
    #     
    #     cgm_data = []
    #     with open(csv_path, 'r') as file:
    #         reader = csv.DictReader(file)
    #         cgm_data = list(reader)
    #     return cgm_data
    #
    # @st.cache_data  
    # def load_food_data(user_id):
    #     """Load and cache food data"""
    #     csv_path = f"{user_id}/Food.csv"
    #     if not os.path.exists(csv_path):
    #         return []
    #     
    #     food_data = []
    #     with open(csv_path, 'r') as file:
    #         reader = csv.DictReader(file)
    #         food_data = list(reader)
    #     return food_data
    #
    # def optimized_stream_insights(gemini_api, model_choice, current_meal_metrics, reference_meals_metrics, 
    #                             meal_image, current_meal_details, reference_meal_details):
    #     """Simplified streaming function for faster insights"""
    #     try:
    #         # Simple direct call to get insights without complex async management
    #         loop = asyncio.new_event_loop()
    #         asyncio.set_event_loop(loop)
    #         
    #         async def get_insights_simple():
    #             insights_text = ""
    #             async for chunk in gemini_api.get_insights_stream(
    #                 choice=model_choice,
    #                 current_meal_metrics=current_meal_metrics,
    #                 reference_meals_metrics=reference_meals_metrics,
    #                 meal_image=meal_image,
    #                 current_meal_details=current_meal_details,
    #                 reference_meal_details=reference_meal_details
    #             ):
    #                 insights_text += chunk
    #                 yield chunk
    #             
    #         # Run the simplified async generator
    #         async_gen = get_insights_simple()
    #         
    #         while True:
    #             try:
    #                 chunk = loop.run_until_complete(async_gen.__anext__())
    #                 yield chunk
    #             except StopAsyncIteration:
    #                 break
    #                 
    #     except Exception as e:
    #         yield f"Error: {str(e)}"
    #
    # # ✅ Processing section
    # if uploaded_file:
    #     # Load data
    #     cgm_data = load_cgm_data(user_id)
    #     food_data = load_food_data(user_id)
    #     
    #     if cgm_data:
    #         # ✅ Validate time input
    #         try:
    #             meal_time = datetime.strptime(meal_time_str, "%H:%M").time()
    #             selected_datetime = datetime.combine(meal_date, meal_time)
    #             
    #             # Show current selection
    #             st.success(f"📊 Ready to analyze: {meal_slot} on {selected_datetime.strftime('%B %d, %Y at %H:%M')}")
    #             
    #         except ValueError:
    #             st.error("⚠️ Please enter time in 24-hour format (HH:MM). Examples: 08:30, 14:20, 19:45")
    #             st.stop()
    #         
    #         # Initialize components  
    #         metrics_calc = CGMMetricsCalculator(verbose=False)
    #         gemini_api = GeminiAPI()
    #         
    #         # Convert uploaded file to PIL Image
    #         meal_image = Image.open(uploaded_file)
    #         
    #         if st.button("🔍 Generate Insights", type="primary", use_container_width=True):
    #             with st.spinner("⚡ Processing CGM data and generating insights..."):
    #                 start_time = time.time()
    #                 
    #                 # ✅ USE SELECTED DATE/TIME AND SLOT
    #                 current_meal_data = [{
    #                     'date': selected_datetime.strftime('%B %d, %Y'),
    #                     'time': selected_datetime.strftime('%H:%M'),
    #                     'slot': meal_slot,
    #                     'meal name': f'Live Upload - {meal_slot.title()} Meal'
    #                 }]
    #                 
    #                 # ✅ Get reference meals from same slot in last 7 days BUT NOT CURRENT DATE
    #                 seven_days_ago = selected_datetime - timedelta(days=7)
    #                 current_date = selected_datetime.date()
    #                 reference_meals_data = []
    #                 
    #                 for meal in food_data:
    #                     try:
    #                         meal_datetime = datetime.strptime(f"{meal['date']} {meal['time']}", '%Y-%m-%d %H:%M')
    #                         meal_date_only = meal_datetime.date()
    #                         
    #                         # ✅ FIXED: Same slot, within last 7 days, NOT on current date
    #                         if (meal['slot'] == meal_slot and 
    #                             seven_days_ago <= meal_datetime < selected_datetime and
    #                             meal_date_only != current_date):  # ✅ EXCLUDE CURRENT DATE
    #                                 reference_meals_data.append(meal)
    #                     except:
    #                         continue
    #                 
    #                 # Limit to 5 most recent reference meals
    #                 reference_meals_data.sort(key=lambda x: f"{x['date']} {x['time']}", reverse=True)
    #                 reference_meals_data = reference_meals_data[:5]
    #                 
    #                 # ✅ USE METRICS.PY TO CREATE EXACT MODEL STRUCTURE
    #                 meal_impact_analysis: MealImpactAnalysis = metrics_calc.create_meal_impact_analysis(
    #                     cgm_data, current_meal_data, reference_meals_data
    #                 )
    #                 
    #                 # Convert to dict for display and API
    #                 meal_analysis_dict = asdict(meal_impact_analysis)
    #                 
    #                 processing_time = time.time() - start_time
    #                 st.success(f"⚡ Data processed in {processing_time:.2f} seconds")
    #                 
    #                 # Prepare API call data
    #                 current_meal_details = {
    #                     'meal_time': selected_datetime.strftime('%Y-%m-%d %H:%M:%S'),
    #                     'meal_type': meal_slot,
    #                     'meal_date': selected_datetime.strftime('%B %d, %Y'),
    #                     'meal_time_str': selected_datetime.strftime('%H:%M')
    #                 }
    #                 
    #                 reference_meal_details = [
    #                     {
    #                         'meal_name': meal.get('meal name', 'Unknown'),
    #                         'date': meal.get('date', ''),
    #                         'time': meal.get('time', ''),
    #                         'slot': meal.get('slot', '')
    #                     } for meal in reference_meals_data
    #                 ]
    #                 
    #                 # Get selected model from session state
    #                 selected_model = st.session_state.selected_model
    #                 
    #                 # Stream insights
    #                 st.header("🤖 Generated Insights")
    #                 insights_placeholder = st.empty()
    #                 insights_text = ""
    #                 
    #                 try:
    #                     # Use faster non-streaming approach for single images
    #                     with st.spinner("🤖 Generating AI insights..."):
    #                         loop = asyncio.new_event_loop()
    #                         asyncio.set_event_loop(loop)
    #                         
    #                         # Get insights all at once (faster than streaming for single images)
    #                         insights_text = loop.run_until_complete(
    #                             gemini_api.get_insights_simple(
    #                                 choice=selected_model,
    #                                 current_meal_metrics=meal_analysis_dict,
    #                                 reference_meals_metrics=meal_analysis_dict["reference_meals"],
    #                                 meal_image=meal_image,
    #                                 current_meal_details=current_meal_details,
    #                                 reference_meal_details=reference_meal_details
    #                             )
    #                         )
    #                         
    #                     # Display the complete insights at once
    #                     insights_placeholder.markdown(insights_text)
    #                     
    #                 except AttributeError:
    #                     # Fallback to streaming if get_insights_simple doesn't exist
    #                     for chunk in optimized_stream_insights(
    #                         gemini_api, selected_model, meal_analysis_dict,
    #                         meal_analysis_dict["reference_meals"],
    #                         meal_image, current_meal_details, reference_meal_details
    #                     ):
    #                         insights_text += chunk
    #                         insights_placeholder.markdown(insights_text)
    #                 except Exception as e:
    #                     st.error(f"Error generating insights: {str(e)}")
    #                 
    #                 # ✅ Get markdown data for LLM  
    #                 markdown_data = gemini_api.get_formatted_data_for_frontend(
    #                     meal_analysis_dict,
    #                     meal_analysis_dict["reference_meals"], 
    #                     current_meal_details,
    #                     reference_meal_details
    #                 )
    #
    #                 # ✅ Display raw data structure below insights
    #                 st.header("📊 Raw Data Passed to LLM")
    #                 st.markdown(markdown_data)
    #     else:
    #         st.error("Could not load CGM data. Please check if the data files exist.")
    # else:
    #     st.info("👆 Please upload a meal image to get started")

with tab2:
    st.header("🔄 Batch Processing")
    st.info("🚀 This feature is functional in local development but disabled for Streamlit deployment.")
    st.markdown("**Features available locally:**")
    st.markdown("- Process multiple meal images automatically")
    st.markdown("- AI model selection (Gemini 2.5 Flash/Pro)")
    st.markdown("- Concurrent processing with progress tracking")
    st.markdown("- Automatic CSV result generation")
    
    # st.markdown("Process all images in the `images/` folder with CGM insights")
    # 
    # # Model selection for batch processing as buttons with icons
    # st.markdown("**🤖 Select AI Model for Batch Processing:**")
    # 
    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("⚡ Gemini 2.5 Flash", key="batch_flash", use_container_width=True):
    #         st.session_state.batch_model_choice = "flash"
    # 
    # with col2:
    #     if st.button("🧠 Gemini 2.5 Pro", key="batch_pro", use_container_width=True):
    #         st.session_state.batch_model_choice = "pro"
    # 
    # # Initialize default batch model
    # if 'batch_model_choice' not in st.session_state:
    #     st.session_state.batch_model_choice = "flash"
    # 
    # # Display selected batch model
    # batch_model_name = "Gemini 2.5 Flash ⚡" if st.session_state.batch_model_choice == "flash" else "Gemini 2.5 Pro 🧠"
    # st.info(f"🤖 Selected Batch Model: {batch_model_name}")
    # 
    # # Info about the process
    # images_folder = "images"
    # if os.path.exists(images_folder):
    #     image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    #     st.info(f"📁 Found {len(image_files)} images in `{images_folder}/` folder")
    #     
    #     if image_files:
    #         # Show first few image names as preview
    #         preview_count = min(5, len(image_files))
    #         st.markdown("**📸 Image Preview:**")
    #         for i in range(preview_count):
    #             st.markdown(f"- {image_files[i]}")
    #         if len(image_files) > preview_count:
    #             st.markdown(f"- ... and {len(image_files) - preview_count} more images")
    # else:
    #     st.warning(f"📁 `{images_folder}/` folder not found. Please create it and add meal images.")
    # 
    # # Batch processing button
    # if st.button("🚀 Process All Images", type="primary"):
    #     if not os.path.exists(images_folder):
    #         st.error(f"Images folder `{images_folder}` not found!")
    #     else:
    #         with st.spinner("🔄 Running batch processing... This may take several minutes."):
    #             try:
    #                 # Run batch processing
    #                 results = asyncio.run(run_batch_processing(st.session_state.batch_model_choice))
    #                 
    #                 if results:
    #                     st.success(f"✅ Successfully processed {len(results)} images!")
    #                     st.balloons()
    #             except Exception as e:
    #                 st.error(f"❌ Batch processing failed: {str(e)}")

with tab3:
    st.header("📊 Past Results")
    st.markdown("View previously processed meal insights with detailed analysis")
    
    results_csv = "batch_results.csv"
    
    # ✅ Initialize session state for meal selection FIRST
    if 'selected_meal_index' not in st.session_state:
        st.session_state.selected_meal_index = 0
    if 'meal_selection_made' not in st.session_state:
        st.session_state.meal_selection_made = False
    
    if os.path.exists(results_csv):
        try:
            results_df = pd.read_csv(results_csv)
            
            if not results_df.empty and len(results_df) > 0:
                st.success(f"📄 Found {len(results_df)} processed meals")
                
                # ✅ Create dropdown for image selection
                image_options = ["Select a meal to view details..."]  # Default option
                for idx, row in results_df.iterrows():
                    try:
                        option_label = f"{row['image_name']} - {row['meal_type']} on {row['meal_date']} at {row['meal_time']}"
                        image_options.append(option_label)
                    except:
                        continue
                
                # Only proceed if we have valid options
                if len(image_options) > 1:
                    selected_image = st.selectbox(
                        "🍽️ Select a processed meal to view details:",
                        options=image_options,
                        index=st.session_state.selected_meal_index,
                        key="meal_selector"
                    )
                    
                    # Update session state when selection changes
                    current_index = image_options.index(selected_image)
                    if current_index != st.session_state.selected_meal_index:
                        st.session_state.selected_meal_index = current_index
                        st.session_state.meal_selection_made = (current_index > 0)
                    
                    # ✅ STRICT CONDITIONS: Only show content if EXPLICITLY selected AND valid
                    if (selected_image != "Select a meal to view details..." and 
                        st.session_state.selected_meal_index > 0 and 
                        st.session_state.meal_selection_made and
                        st.session_state.selected_meal_index < len(results_df) + 1):
                        
                        try:
                            # Get the selected row
                            selected_idx = st.session_state.selected_meal_index - 1  # Subtract 1 for default option
                            selected_row = results_df.iloc[selected_idx]
                            
                            # ✅ Display the meal image only after explicit selection
                            image_path = f"images/{selected_row['image_name']}"
                            if os.path.exists(image_path):
                                st.image(image_path, caption=f"📷 {selected_row['image_name']}", width=400)
                            else:
                                st.warning(f"⚠️ Image file not found: {image_path}")
                            
                            # ✅ Show insights
                            st.header("🤖 Generated Insights")
                            if pd.notna(selected_row['insights_text']) and selected_row['insights_text'].strip():
                                st.markdown(selected_row['insights_text'])
                            else:
                                st.info("No insights available for this meal")
                            
                            # ✅ Show raw data passed to LLM
                            if 'raw_data_passed' in selected_row and pd.notna(selected_row['raw_data_passed']):
                                st.header("📊 Raw Data Passed to LLM") 
                                st.markdown(selected_row['raw_data_passed'])
                            else:
                                st.info("Raw data not available for this meal")
                                
                        except Exception as e:
                            st.error("Error loading meal details. Please try selecting another meal.")
                            st.session_state.selected_meal_index = 0
                            st.session_state.meal_selection_made = False
                    else:
                        # Show instruction when nothing is selected or default option is selected
                        st.info("👆 Please select a meal from the dropdown above to view its insights and analysis")
                else:
                    st.warning("No valid meal data found in results file.")
            else:
                st.info("No results found. Run batch processing first.")
                
        except Exception as e:
            st.error("Error reading results file. Please upload a valid batch_results.csv file.")
            st.session_state.selected_meal_index = 0
            st.session_state.meal_selection_made = False
    else:
        st.info("No results file found. Run batch processing first.")
        
        # ✅ Add file upload option for CSV
        st.markdown("### 📁 Upload Results File")
        uploaded_csv = st.file_uploader(
            "Upload batch_results.csv file",
            type=['csv'],
            help="Upload your batch processing results to view meal insights"
        )
        
        if uploaded_csv is not None:
            try:
                # Save uploaded file
                with open("batch_results.csv", "wb") as f:
                    f.write(uploaded_csv.getbuffer())
                st.success("✅ File uploaded successfully! Please refresh the page to view results.")
                st.rerun()
            except Exception as e:
                st.error("Error uploading file. Please try again.")
