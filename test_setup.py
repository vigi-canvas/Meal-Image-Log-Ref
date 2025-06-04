#!/usr/bin/env python3
"""
Test script to verify the Meal Log Insights setup
"""

import sys
import importlib

def test_imports():
    """Test if all required modules can be imported"""
    modules_to_test = [
        ('streamlit', 'Streamlit'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('pydantic', 'Pydantic'),
        ('google.generativeai', 'Google GenerativeAI'),
        ('PIL', 'Pillow'),
    ]
    
    print("Testing imports...")
    all_good = True
    
    for module_name, display_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"✅ {display_name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {display_name}: {e}")
            all_good = False
    
    return all_good

def test_local_modules():
    """Test if local modules can be imported"""
    print("\nTesting local modules...")
    local_modules = [
        'models',
        'data_processing.cgm_data_processor',
        'data_processing.metrics_processor',
        'data_processing.meal_impact_analysis',
        'utils.llm_handler',
    ]
    
    all_good = True
    
    for module in local_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            all_good = False
    
    return all_good

def main():
    print("=" * 50)
    print("Meal Log Insights - Setup Test")
    print("=" * 50)
    
    # Test external imports
    external_ok = test_imports()
    
    # Test local modules
    local_ok = test_local_modules()
    
    print("\n" + "=" * 50)
    
    if external_ok and local_ok:
        print("✅ All tests passed! Your setup is ready.")
        print("\nYou can now run the app with:")
        print("  streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nMake sure to run:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main() 