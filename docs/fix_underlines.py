#!/usr/bin/env python3
"""Fix RST title underline warnings in Sphinx documentation."""

import os
import re
from pathlib import Path

def fix_underlines(file_path):
    """Fix title underlines in an RST file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    i = 0
    while i < len(lines) - 1:
        # Check if current line is a title (has text) and next line is an underline
        current = lines[i].rstrip()
        if i + 1 < len(lines):
            next_line = lines[i + 1].rstrip()
            
            # Check if next line consists only of underline characters
            if next_line and all(c in '=-~^"+*' for c in next_line):
                # Make sure underline is at least as long as the title
                if len(next_line) < len(current):
                    char = next_line[0]
                    lines[i + 1] = char * len(current) + '\n'
                    modified = True
                    print(f"Fixed underline in {file_path} at line {i + 2}")
        i += 1
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    return modified

def main():
    """Fix all RST files in the docs directory."""
    docs_dir = Path(__file__).parent
    fixed_count = 0
    
    for rst_file in docs_dir.rglob('*.rst'):
        if '_build' not in str(rst_file):
            if fix_underlines(rst_file):
                fixed_count += 1
    
    print(f"\nFixed underlines in {fixed_count} files")

if __name__ == '__main__':
    main()