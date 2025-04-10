import os
import subprocess
import sys

def run_command(command):
    """Run a shell command and print the output."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(f"Error: {result.stderr}")
    
    return result.returncode == 0

def git_commit():
    """Commit changes to Git."""
    print("Preparing to commit changes to Git...")
    
    # Check if Git is installed
    if not run_command("git --version"):
        print("Error: Git is not installed or not in the PATH.")
        return False
    
    # Check if we're in a Git repository
    if not run_command("git rev-parse --is-inside-work-tree"):
        print("Error: Not in a Git repository.")
        return False
    
    # Show status
    run_command("git status")
    
    # Ask for confirmation
    confirm = input("\nDo you want to add and commit these changes? (y/n): ")
    if confirm.lower() != 'y':
        print("Commit cancelled.")
        return False
    
    # Add files
    files_to_add = [
        "models/smart_classifier.py",
        "utils/database.py",
        "app/app.py",
        "run_smart_app.py",
        "run_smart_classifier.py",
        "scripts/",
        "batch/",
        "README.md"
    ]
    
    for file in files_to_add:
        run_command(f"git add {file}")
    
    # Commit
    commit_message = "Add smart classifier and fix thread safety issues"
    run_command(f'git commit -m "{commit_message}"')
    
    # Ask about pushing
    push = input("\nDo you want to push these changes to the remote repository? (y/n): ")
    if push.lower() == 'y':
        run_command("git push")
    
    print("\nGit operations completed!")
    return True

if __name__ == "__main__":
    git_commit()
