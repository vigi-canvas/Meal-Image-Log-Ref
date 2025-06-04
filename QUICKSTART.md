# Quick Start Guide - Meal Log Insights

Follow these steps to get the application running quickly:

## 1. Install Dependencies

First, install all required Python packages:

```bash
pip install -r requirements.txt
```

## 2. Get a Gemini API Key

1. Visit https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

## 3. Prepare Your Data

The application expects two CSV files:

### CGM Data CSV
Your CGM data should have these columns:
- `user_id`: User identifier
- `date`: Date in format like "April 16, 2025"
- `time`: Time in HH:MM format
- `value`: Glucose value in mg/dL

### Food Logs CSV
Your food logs should have these columns:
- `user_id`: User identifier
- `date`: Date in YYYY-MM-DD format
- `time`: Time in HH:MM format
- `slot`: BREAKFAST, LUNCH, DINNER, or SNACK
- `meal name`: Name of the meal
- `calories`, `protein`, `carbohydrates`, `fat`, `fibre`: Nutritional values

Sample files are included: `sample_cgm_data.csv` and `sample_food_logs.csv`

## 4. Run the Application

```bash
streamlit run app.py
```

## 5. Using the Application

1. **In the sidebar:**
   - Enter your Gemini API key
   - Upload your CGM data CSV
   - Upload your food logs CSV
   - Enter your User ID (default: 100067072)

2. **In the main area:**
   - Select the date and time of your meal
   - Choose the meal slot
   - Upload a photo of your meal
   - Click "Analyze Meal"

3. **Review results:**
   - Check the identified dishes and nutrition
   - View glucose metrics
   - Read AI-generated insights

4. **Edit if needed:**
   - Modify the dishes or nutritional values
   - Click "Update & Re-analyze" for updated insights

## Troubleshooting

### "Module not found" errors
Run: `pip install -r requirements.txt`

### "No CGM data available"
- Check that your CSV files are properly formatted
- Ensure the user_id matches
- Verify dates are in the correct format

### API errors
- Verify your Gemini API key is correct
- Check your internet connection
- Ensure you have API quota remaining

## Test Your Setup

Run the test script to verify everything is installed correctly:

```bash
python test_setup.py
```

All modules should show ✅ if properly installed. 