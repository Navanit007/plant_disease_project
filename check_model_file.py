#!/usr/bin/env python3
"""
check_and_load_model.py

Usage:
  python check_and_load_model.py "C:\path\to\plant_disease_recog_model_pwp.keras"

What it does:
- Validates file header (HTML preview, HDF5, ZIP (.keras), or SavedModel directory).
- If ZIP (.keras): lists contents and scans for HTML inside files.
- Attempts to load model using tensorflow.keras.models.load_model.
- If load fails for a ZIP, optionally extracts to a temp folder and retries.
- Prints clear guidance for next steps.

Dependencies:
  pip install tensorflow gdown
  (gdown only needed if you want to re-download from Google Drive)
"""
import sys
import os
import io
import zipfile
import tempfile
import shutil
import traceback
from pathlib import Path

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ZIP_MAGIC = b"PK\x03\x04"

def read_head(path, n=1024):
    with open(path, "rb") as f:
        return f.read(n)

def looks_like_html(head: bytes) -> bool:
    if not head:
        return False
    low = head.lstrip().lower()
    return low.startswith(b"<") and (b"<html" in low or b"<!doctype" in low or b"<!doctype html" in low)

def is_hdf5(head: bytes) -> bool:
    return head.startswith(HDF5_MAGIC)

def is_zip(head: bytes) -> bool:
    return head.startswith(ZIP_MAGIC)

def inspect_zip(path: Path):
    print(f"\nListing ZIP contents for: {path}")
    try:
        with zipfile.ZipFile(path, "r") as z:
            infos = z.infolist()
            for info in infos:
                print(f" - {info.filename}  ({info.file_size} bytes)")
            # scan first few files for HTML
            print("\nScanning first files for embedded HTML preview content...")
            for name in z.namelist()[:20]:
                try:
                    with z.open(name) as f:
                        head = f.read(1024).lower()
                        if b"<html" in head or b"<!doctype" in head:
                            print("  !! HTML-like content found in", name)
                except RuntimeError:
                    # encrypted or not readable as text
                    pass
    except zipfile.BadZipFile:
        print("ERROR: Not a valid ZIP archive (BadZipFile).")

def try_load_model(path_str: str):
    print("\nAttempting to load model with tensorflow.keras.models.load_model...")
    try:
        # Import inside function so script can run even if TF missing for initial checks
        from tensorflow import keras
    except Exception as e:
        print("TensorFlow import failed:", type(e).__name__, e)
        print("Install tensorflow (pip install tensorflow) to attempt loading the model.")
        return False

    try:
        model = keras.models.load_model(path_str)
        print("Model loaded successfully.")
        try:
            model.summary()
        except Exception:
            # Some models print a lot; still okay
            print("Model loaded but printing summary failed (model.summary()).")
        return True
    except Exception as e:
        print("Failed to load model:", type(e).__name__, e)
        traceback.print_exc()
        return False

def extract_zip_and_try(path: Path):
    tmpdir = Path(tempfile.mkdtemp(prefix="model_extract_"))
    print(f"\nExtracting ZIP to temporary directory: {tmpdir}")
    try:
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmpdir)
        # Try common saved_model paths inside extracted content
        candidates = [tmpdir]
        # add top-level directories inside tmpdir
        for p in tmpdir.iterdir():
            if p.is_dir():
                candidates.append(p)
        for cand in candidates:
            print("Trying extracted path:", cand)
            if try_load_model(str(cand)):
                print("Loaded model from extracted path:", cand)
                return True
        print("Could not load model from any extracted candidate path.")
        return False
    finally:
        # keep the tmpdir by default for debugging — remove only if you want
        print(f"Temporary extract folder kept at: {tmpdir}")
        print("Remove it manually when you no longer need it, or re-run with manual cleanup.")
        # If you want automatic cleanup, uncomment the next line:
        # shutil.rmtree(tmpdir)

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_and_load_model.py path/to/model.keras")
        sys.exit(1)
    p = Path(sys.argv[1]).expanduser().resolve()
    if not p.exists():
        print("ERROR: Path does not exist:", p)
        sys.exit(1)

    print("File:", p)
    size = p.stat().st_size
    print("Size (bytes):", size)

    # If path is a directory, assume SavedModel format and try to load directly
    if p.is_dir():
        print("Path is a directory — treating as SavedModel directory")
        loaded = try_load_model(str(p))
        if not loaded:
            print("Failed to load SavedModel directory. See traceback above.")
        return

    head = read_head(p, 2048)
    if looks_like_html(head):
        print("Result: FILE CONTAINS HTML (Google Drive preview or similar).")
        print("Action: Re-download the model and make sure the Drive file is shared 'Anyone with the link', or use gdown with the file id.")
        return

    if is_hdf5(head):
        print("Result: Looks like HDF5 (.h5) model (HDF5 magic header detected).")
        loaded = try_load_model(str(p))
        if not loaded:
            print("If loading failed, consider re-downloading the file or checking TF version compatibility.")
        return

    if is_zip(head):
        print("Result: ZIP archive detected (likely a .keras Keras archive).")
        inspect_zip(p)
        loaded = try_load_model(str(p))
        if loaded:
            return
        # If load failed when passing the zip file path, try extract & load
        print("\nLoad from ZIP failed — will extract and attempt to load from extracted content.")
        extract_zip_and_try(p)
        return

    print("Unknown file header; file may be corrupted or in an unexpected format.")
    print("First 512 bytes (utf-8, errors replaced):")
    try:
        print(head.decode("utf-8", errors="replace")[:1000])
    except Exception:
        print(head[:200])

if __name__ == "__main__":
    main()