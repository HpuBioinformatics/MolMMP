# -*- coding: utf-8 -*-
"""
Auto-detect and batch-convert all .py files to UTF-8
Author: ChatGPT
Date: 2025-08
"""

import os
import sys
import chardet

def detect_encoding(file_path, nbytes=4096):
    """Detect file encoding"""
    with open(file_path, "rb") as f:
        raw = f.read(nbytes)
    result = chardet.detect(raw)
    enc = result["encoding"]
    return enc

def convert_file(file_path):
    """Attempt to convert a single file to UTF-8"""
    try:
        enc = detect_encoding(file_path)
        if enc is None:
            print(f"[SKIP] {file_path} (encoding undetectable)")
            return

        if enc.lower() == "utf-8":
            # Already UTF-8
            return

        # Read original content
        with open(file_path, "r", encoding=enc, errors="ignore") as f:
            content = f.read()

        # Write back as UTF-8
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[CONVERTED] {file_path} ({enc} -> utf-8)")
    except Exception as e:
        print(f"[FAILED] {file_path}: {e}")

def convert_all_py(root_dir):
    """Walk through all .py files and convert them"""
    for subdir, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(subdir, fname)
                convert_file(fpath)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = "."

    print(f"Start scanning and converting directory: {os.path.abspath(root)}")
    convert_all_py(root)
    print("All done ?")