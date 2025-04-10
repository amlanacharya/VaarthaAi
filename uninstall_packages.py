import subprocess
import sys

def uninstall_packages():
    """Uninstall all packages listed in requirements.txt"""
    try:
        with open('requirements.txt', 'r') as f:
            packages = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(packages)} packages to uninstall:")
        for pkg in packages:
            print(f"  - {pkg}")
        
        confirm = input("\nAre you sure you want to uninstall these packages? (y/n): ")
        if confirm.lower() != 'y':
            print("Uninstall cancelled.")
            return
        
        print("\nUninstalling packages...")
        for pkg in packages:
            print(f"Uninstalling {pkg}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", pkg])
                print(f"Successfully uninstalled {pkg}")
            except subprocess.CalledProcessError:
                print(f"Failed to uninstall {pkg}")
        
        print("\nAll packages have been uninstalled.")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    uninstall_packages()
