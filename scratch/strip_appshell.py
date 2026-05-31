import os
import glob
import re

files = glob.glob('/Users/ebelthomasseiko/clippedai/frontend/src/app/(dashboard)/**/*.tsx', recursive=True)
for f in files:
    with open(f, 'r') as file:
        content = file.read()
    
    if 'import AppShell' in content:
        # Remove import
        content = re.sub(r'import AppShell from "~/components/app-shell";\n?', '', content)
        # Replace <AppShell> with Fragment to preserve indentation
        content = re.sub(r'<AppShell>', '<>', content)
        content = re.sub(r'</AppShell>', '</>', content)
        
        with open(f, 'w') as file:
            file.write(content)
        print(f"Stripped from {f}")

