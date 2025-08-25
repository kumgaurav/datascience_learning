#!/usr/bin/env python3
"""
Helper script to set up the .env file for Google API key configuration.
"""

import os

def create_env_file():
    """Create a .env file with Google API key configuration."""
    
    env_content = """# Google Gemini API Configuration
# Replace YOUR_ACTUAL_API_KEY_HERE with your real Google API key
GOOGLE_API_KEY=YOUR_ACTUAL_API_KEY_HERE

# Other environment variables can be added here
# CURL_CA_BUNDLE=
# REQUESTS_CA_BUNDLE=
"""
    
    # Check if .env file already exists
    if os.path.exists('.env'):
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("❌ Setup cancelled.")
            return
    
    # Create the .env file
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully!")
        print("\n📝 Next steps:")
        print("1. Open the .env file in your text editor")
        print("2. Replace 'YOUR_ACTUAL_API_KEY_HERE' with your actual Google API key")
        print("3. Save the file")
        print("4. Restart your Streamlit app")
        print("\n🔗 Get your Google API key from: https://makersuite.google.com/app/apikey")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def check_env_setup():
    """Check if the .env file is properly configured."""
    
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Run: python setup_env.py")
        return False
    
    # Load the .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file!")
        return False
    
    if api_key == 'YOUR_ACTUAL_API_KEY_HERE':
        print("⚠️  GOOGLE_API_KEY still has placeholder value!")
        print("Please replace 'YOUR_ACTUAL_API_KEY_HERE' with your actual API key")
        return False
    
    print("✅ .env file is properly configured!")
    print(f"API Key: {api_key[:10]}...")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_env_setup()
    else:
        create_env_file()
