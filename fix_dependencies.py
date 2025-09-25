# fix_dependencies.py - Run this script to fix the immediate issues
import os
import sys
import subprocess

def install_and_download_nltk():
    """Install NLTK and download required data"""
    print("Installing and setting up NLTK...")
    
    try:
        import nltk
        print("NLTK already installed")
    except ImportError:
        print("Installing NLTK...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
        import nltk
    
    # Download required NLTK data
    required_data = [
        'punkt',
        'punkt_tab', 
        'stopwords'
    ]
    
    for data_name in required_data:
        try:
            print(f"Downloading NLTK {data_name}...")
            nltk.download(data_name, quiet=False)
            print(f"Successfully downloaded {data_name}")
        except Exception as e:
            print(f"Error downloading {data_name}: {e}")
            # Try alternative download
            try:
                nltk.download(data_name, download_dir=os.path.expanduser('~/nltk_data'))
                print(f"Downloaded {data_name} to user directory")
            except Exception as e2:
                print(f"Failed to download {data_name}: {e2}")

def test_nltk_setup():
    """Test if NLTK is properly set up"""
    print("\nTesting NLTK setup...")
    
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        # Test tokenization
        test_text = "This is a test sentence."
        tokens = word_tokenize(test_text)
        print(f"Tokenization test: '{test_text}' -> {tokens}")
        
        # Test stopwords
        english_stopwords = stopwords.words('english')
        print(f"English stopwords loaded: {len(english_stopwords)} words")
        
        try:
            indonesian_stopwords = stopwords.words('indonesian')
            print(f"Indonesian stopwords loaded: {len(indonesian_stopwords)} words")
        except Exception as e:
            print(f"Indonesian stopwords not available: {e}")
            print("Will use manual Indonesian stopwords")
        
        print("NLTK setup successful!")
        return True
        
    except Exception as e:
        print(f"NLTK test failed: {e}")
        return False

def fix_encoding_issues():
    """Fix Windows encoding issues"""
    print("\nFixing Windows encoding issues...")
    
    # Set environment variables for better Unicode support
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Try to set console to UTF-8 on Windows
    if os.name == 'nt':  # Windows
        try:
            os.system('chcp 65001')  # Set console to UTF-8
            print("Console encoding set to UTF-8")
        except Exception as e:
            print(f"Could not set console encoding: {e}")
    
    print("Encoding fixes applied")

def create_test_script():
    """Create a simple test script to verify the fixes"""
    test_script = """
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
print("\\nTesting preprocessing...")
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

print("\\nAll tests completed!")
"""
    
    with open('test_fixes.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("Created test_fixes.py - run this to verify the fixes work")

def main():
    print("=" * 60)
    print("FIXING SENTIMENT ANALYSIS PROJECT DEPENDENCIES")
    print("=" * 60)
    
    # Fix 1: Install and setup NLTK
    install_and_download_nltk()
    
    # Fix 2: Test NLTK setup
    nltk_success = test_nltk_setup()
    
    # Fix 3: Fix encoding issues
    fix_encoding_issues()
    
    # Fix 4: Create test script
    create_test_script()
    
    print("\n" + "=" * 60)
    if nltk_success:
        print("FIXES APPLIED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Replace your main.py with the fixed version (no emojis)")
        print("2. Replace your data_preprocessor.py with the fixed version")
        print("3. Replace your logging_config.py with the fixed version")
        print("4. Run: python test_fixes.py  (to verify fixes)")
        print("5. Run: python main.py --steps preprocess baseline")
    else:
        print("SOME FIXES FAILED!")
        print("Manual steps needed:")
        print("1. Open Python and run: import nltk; nltk.download('all')")
        print("2. Check your internet connection")
        print("3. Try running the test script: python test_fixes.py")
    
    print("=" * 60)

if __name__ == "__main__":
    main()