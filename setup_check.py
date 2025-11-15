"""
Quick setup verification script
Run this to check if all dependencies and components are ready
"""

import sys
import os


def check_imports():
    """Check if all required packages are installed"""
    print("Checking Python packages...")

    packages = [
        ('gymnasium', 'Gymnasium'),
        ('stable_baselines3', 'Stable-Baselines3'),
        ('torch', 'PyTorch'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('requests', 'Requests'),
        ('streamlit', 'Streamlit'),
        ('plotly', 'Plotly'),
    ]

    all_good = True

    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} - NOT INSTALLED")
            all_good = False

    return all_good


def check_directories():
    """Check if all required directories exist"""
    print("\nChecking directory structure...")

    dirs = [
        'data/raw',
        'data/processed',
        'src',
        'models',
        'dashboard',
    ]

    all_good = True

    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ - NOT FOUND")
            all_good = False

    return all_good


def check_files():
    """Check if all required source files exist"""
    print("\nChecking source files...")

    files = [
        'src/environment.py',
        'src/agent.py',
        'src/llm_integration.py',
        'src/data_loader.py',
        'src/utils.py',
        'dashboard/app.py',
        'requirements.txt',
        'README.md',
    ]

    all_good = True

    for file_path in files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - NOT FOUND")
            all_good = False

    return all_good


def check_ollama():
    """Check if Ollama is running"""
    print("\nChecking Ollama (LLM)...")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("  ✅ Ollama is running")
            return True
        else:
            print("  ⚠️  Ollama is not responding correctly")
            return False
    except Exception as e:
        print(f"  ⚠️  Ollama is not running (optional)")
        print(f"     Start with: ollama serve")
        return False


def check_src_imports():
    """Check if src modules can be imported"""
    print("\nChecking src module imports...")

    modules = [
        'src.environment',
        'src.agent',
        'src.llm_integration',
        'src.data_loader',
        'src.utils',
    ]

    all_good = True

    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module} - ERROR: {str(e)[:50]}")
            all_good = False

    return all_good


def main():
    print("="*60)
    print("  Diggyy Fresh - Setup Verification")
    print("="*60)

    results = []

    results.append(("Python Packages", check_imports()))
    results.append(("Directory Structure", check_directories()))
    results.append(("Source Files", check_files()))
    results.append(("Module Imports", check_src_imports()))
    results.append(("Ollama (Optional)", check_ollama()))

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)

    for name, status in results:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")

    all_good = all([r[1] for r in results[:4]])  # Exclude optional Ollama check

    if all_good:
        print("\n🎉 All checks passed! You're ready to go!")
        print("\nNext steps:")
        print("  1. Train agent:    python src/agent.py")
        print("  2. Launch dashboard: streamlit run dashboard/app.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
