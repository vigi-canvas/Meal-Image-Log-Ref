# Meal Log Insights - CGM Analysis PoC

A Streamlit-based proof of concept that analyzes meal images and their impact on glucose levels using Continuous Glucose Monitor (CGM) data.

## 🚀 Features

- **Meal Image Analysis**: Upload meal photos to automatically identify dishes and estimate nutritional content using Google's Gemini AI
- **CGM Data Integration**: Analyze glucose response patterns before and after meals
- **Meal Impact Analysis**: Calculate key metrics like glucose rise, time to peak, and return to baseline
- **Reference Comparisons**: Compare current meal impacts with similar past meals
- **Editable Nutrition Data**: Manually adjust identified dishes and nutritional estimates
- **AI-Powered Insights**: Get personalized recommendations based on your glucose response patterns

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- CGM data in CSV format
- Food log data in CSV format

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Meal-Log-Insights
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data Format

### CGM Data CSV Format
```csv
user_id,date,time,value
100067072,"April 16, 2025",01:05,77.14
```

### Food Logs CSV Format
```csv
user_id,date,time,slot,meal name,calories,protein,carbohydrates,fat,fibre
100067072,2025-04-17,08:57,BREAKFAST,"Masala Omelette, Wheat Roti",429.54,18.69,41.88,21.8,6.87
```

## 🚀 Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

## 📖 Usage

1. **Configuration (Sidebar)**:
   - Enter your Gemini API key
   - Upload your CGM data CSV file
   - Upload your food logs CSV file
   - (Optional) Add a custom system prompt for AI insights
   - Enter your User ID

2. **Meal Analysis**:
   - Select the date and time of your meal
   - Choose the meal slot (Breakfast, Lunch, Dinner, Snack)
   - Upload a photo of your meal
   - Click "Analyze Meal"

3. **Review Results**:
   - View identified dishes and estimated nutrition
   - Check glucose metrics (pre-meal, peak, rise)
   - Read AI-generated insights and recommendations

4. **Edit and Re-analyze** (Optional):
   - Modify identified dishes or nutritional values
   - Click "Update & Re-analyze" for updated insights

## 🏗️ Project Structure

```
Meal-Log-Insights/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── models.py                       # Pydantic models for data structures
├── data_processing/
│   ├── __init__.py
│   ├── cgm_data_processor.py      # CGM data loading and preprocessing
│   ├── metrics_processor.py        # Glucose metrics calculations
│   └── meal_impact_analysis.py    # Meal impact analysis
└── utils/
    └── llm_handler.py             # Gemini AI integration
```

## 🔧 Key Components

### Metrics Calculated
- **Pre-meal glucose**: Average glucose 30 minutes before meal
- **Post-meal peak**: Maximum glucose within 3 hours after meal
- **Time to peak**: Minutes from meal to peak glucose
- **Glucose rise**: Difference between peak and pre-meal glucose
- **Return to baseline**: Whether glucose returned to pre-meal levels

### Time Windows
- Pre-meal window: 30 minutes before meal
- Post-meal window: 3 hours after meal (2 hours for snacks)

## ⚠️ Important Notes

- This is a proof of concept and should not be used for medical decisions
- Always consult healthcare professionals for diabetes management
- Ensure your CSV files follow the exact format specified
- The system requires sufficient CGM data around meal times for accurate analysis

## 🤝 Contributing

Feel free to submit issues or pull requests to improve the functionality.

## 📄 License

This project is for demonstration purposes only. 