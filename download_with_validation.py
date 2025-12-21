# download_with_validation.py
import os
import requests
import gdown

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

def looks_like_html(bytes_chunk: bytes) -> bool:
    b = bytes_chunk.lstrip()
    return b.startswith(b"<") and (b"<html" in b or b"<!doctype" in b)

def is_hdf5_bytes(bytes_chunk: bytes) -> bool:
    return bytes_chunk.startswith(HDF5_MAGIC)

def download_via_requests(url: str, out_path: str) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for i, chunk in enumerate(r.iter_content(chunk_size=8192)):
                if not chunk:
                    continue
                f.write(chunk)
                if i == 0:  # inspect the first chunk
                    head = chunk[:512].lower()
                    if looks_like_html(head):
                        raise ValueError("Downloaded content looks like HTML (likely Drive page).")
    # optional final header check
    with open(out_path, "rb") as f:
        head = f.read(8)
    if not is_hdf5_bytes(head):
        raise ValueError("Downloaded file doesn't look like HDF5 (.h5) model.")

def download_with_gdown(file_id_or_url: str, out_path: str) -> None:
    # gdown accepts file id or uc? url; it handles large-file confirm tokens
    gdown.download(f"https://drive.google.com/uc?id={file_id_or_url}", out_path, quiet=False)

def download_model(primary_url: str, drive_file_id: str, out_path: str) -> None:
    # Try direct HTTP first (if you have a direct link)
    try:
        download_via_requests(primary_url, out_path)
        print("Downloaded model via requests and validated HDF5 header.")
        return
    except Exception as e:
        print("Primary download failed or returned HTML:", e)

    # Fallback to gdown (requires file to be shared as 'Anyone with the link')
    try:
        download_with_gdown(drive_file_id, out_path)
        with open(out_path, "rb") as f:
            head = f.read(8)
        if is_hdf5_bytes(head):
            print("Downloaded model via gdown and validated HDF5 header.")
            return
        raise ValueError("gdown download did not yield a valid HDF5 model.")
    except Exception as e:
        # cleanup possibly invalid file
        if os.path.exists(out_path):
            os.remove(out_path)
        raise RuntimeError("Model download failed (requests + gdown). Make sure Drive sharing is public.") from ep