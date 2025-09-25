
# test_fixes.py - Test script to verify fixes
import pandas as pd
import sys
import os

# Test basic imports
print("Testing imports...")
try:
    from scripts.data_preprocessor import ReviewPreprocessor
    print("✓ Data preprocessor import successful")
except Exception as e:
    print(f"✗ Data preprocessor import failed: {e}")
    sys.exit(1)

# Test preprocessing
print("\nTesting preprocessing...")
try:
    preprocessor = ReviewPreprocessor()
    
    # Test data
    test_data = pd.DataFrame({
        'content': [
            'Aplikasi sangat bagus!',
            'App jelek banget',
            'Biasa aja'
        ],
        'score': [5, 1, 3],
        'app_name': ['tokopedia', 'shopee', 'tokopedia']
    })
    
    processed = preprocessor.preprocess_dataframe(test_data)
    print(f"✓ Preprocessing successful! Processed {len(processed)} reviews")
    print(f"  Sentiment distribution: {dict(processed['sentiment'].value_counts())}")
    
except Exception as e:
    print(f"✗ Preprocessing test failed: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests completed!")
