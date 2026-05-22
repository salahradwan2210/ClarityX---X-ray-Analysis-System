
# Script to update diagnose_image.py threshold
import os
import re

# Optimal threshold from training
NEW_THRESHOLD = 0.45

def update_diagnose_script():
    if not os.path.exists('diagnose_image.py'):
        print("diagnose_image.py not found")
        return False
    with open('diagnose_image.py', 'r') as file:
        content = file.read()
    updated_content = re.sub(
        r'THRESHOLD\s*=\s*0\.\d+',
        f'THRESHOLD = 0.45',
        content
    )
    with open('diagnose_image.py', 'w') as file:
        file.write(updated_content)
    print(f"Updated diagnose_image.py threshold to 0.45")
    return True

if __name__ == "__main__":
    update_diagnose_script()
