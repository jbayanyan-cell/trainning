"""
External dataset ZIP support for Railway (Option 2).

Set Railway env:
  DATASET_ZIP_URL=https://...direct-or-drive-link-to-zip...

Preferred ZIP layout (any of these work after normalize):
  dataset_organized/100.v1i.folder/train/<class>/*.jpg
  dataset_organized/classification/train/<class>/*.jpg
  Dataset20260715folder/train/<class>/*.jpg
  100.v1i.folder/train/<class>/*.jpg
  train/<class>/*.jpg
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse


# Default starter dataset (Google Drive folder). Override with Railway Variable DATASET_ZIP_URL.
DEFAULT_DATASET_ZIP_URL = (
    "https://drive.google.com/drive/folders/1H_TDkyyCZus54yH92vgFVwsjsSPK5Z1Y?usp=sharing"
)


def get_dataset_zip_url() -> str:
    """Resolve dataset URL from env (several aliases) or built-in default."""
    for key in (
        "DATASET_ZIP_URL",
        "DATASET_URL",
        "DATASET_DRIVE_URL",
        "GOOGLE_DRIVE_DATASET_URL",
    ):
        val = (os.getenv(key) or "").strip().strip('"').strip("'")
        if val:
            return val
    return DEFAULT_DATASET_ZIP_URL



def resolve_dataset_zip_url(url: str) -> str:
    """Turn common Google Drive share links into a direct download URL."""
    url = (url or "").strip()
    if not url:
        return url

    # https://drive.google.com/file/d/FILE_ID/view?...
    m = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"

    # https://drive.google.com/open?id=FILE_ID
    parsed = urlparse(url)
    if "drive.google.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "id" in qs and qs["id"]:
            return f"https://drive.google.com/uc?export=download&id={qs['id'][0]}"

    return url


def _download_url_to_file(url: str, dest: Path, timeout: int = 600) -> None:
    import requests

    url = resolve_dataset_zip_url(url)
    session = requests.Session()
    resp = session.get(url, stream=True, timeout=timeout, allow_redirects=True)

    # Google Drive large-file confirm page
    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" in content_type and "drive.google.com" in url:
        text = resp.text
        confirm = re.search(r"confirm=([0-9A-Za-z_]+)", text)
        file_id = re.search(r"id=([^&\"']+)", url)
        if confirm and file_id:
            confirm_url = (
                f"https://drive.google.com/uc?export=download"
                f"&confirm={confirm.group(1)}&id={file_id.group(1)}"
            )
            resp = session.get(confirm_url, stream=True, timeout=timeout)

    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)


def dataset_ready(script_dir: Path) -> bool:
    organized = script_dir / "training_data" / "dataset_organized"
    return organized.exists() and (
        (organized / "data.yaml").exists()
        or (organized / "classification" / "train").exists()
        or (organized / "100.v1i.folder" / "train").exists()
    )


def _copy_class_tree(src_split: Path, dst_split: Path, normalize_names: bool) -> int:
    count = 0
    if not src_split.is_dir():
        return 0
    dst_split.mkdir(parents=True, exist_ok=True)
    for class_dir in src_split.iterdir():
        if not class_dir.is_dir():
            continue
        name = class_dir.name
        if normalize_names:
            name = CLASS_NAME_MAP.get(name, CLASS_NAME_MAP.get(name.lower(), name))
            name = name.replace(" ", "_").replace("-", "_")
        target = dst_split / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(class_dir, target)
        count += sum(1 for _ in target.rglob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return count


def normalize_dataset_layout(script_dir: Path) -> bool:
    """
    Ensure training_data/dataset_organized/{100.v1i.folder|classification} exists.
    """
    training_data = script_dir / "training_data"
    organized = training_data / "dataset_organized"
    organized.mkdir(parents=True, exist_ok=True)

    if dataset_ready(script_dir):
        return True

    candidates = [
        training_data / "Dataset20260715folder",
        script_dir / "Dataset20260715folder",
        training_data / "100.v1i.folder",
        organized / "Dataset20260715folder",
        training_data / "dataset_organized",  # maybe train/ is directly here
    ]

    # Also search one level for any folder containing train/<classes>
    for root in [training_data, script_dir, organized]:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and (child / "train").is_dir():
                candidates.append(child)

    source = None
    for cand in candidates:
        if cand.is_dir() and (cand / "train").is_dir():
            # Prefer folder-style class datasets (not YOLO train/images)
            sample = next((cand / "train").iterdir(), None)
            if sample and sample.is_dir() and sample.name != "images":
                source = cand
                break

    if source is None:
        return False

    print(f"[INFO] Normalizing dataset from: {source}", flush=True)

    # 100.v1i.folder (keep original folder class names for Roboflow path)
    robo = organized / "100.v1i.folder"
    if robo.exists():
        shutil.rmtree(robo)
    shutil.copytree(source, robo)

    # classification with normalized class codes + val from valid
    class_root = organized / "classification"
    if class_root.exists():
        shutil.rmtree(class_root)
    n_train = _copy_class_tree(source / "train", class_root / "train", True)
    valid_src = source / "valid" if (source / "valid").exists() else source / "val"
    n_val = _copy_class_tree(valid_src, class_root / "val", True)

    yaml_path = organized / "data.yaml"
    classes = sorted([p.name for p in (class_root / "train").iterdir() if p.is_dir()])
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("names:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {c}\n")
        f.write(f"nc: {len(classes)}\n")

    print(f"[OK] Normalized dataset: train_images~{n_train}, val_images~{n_val}, classes={classes}", flush=True)
    return dataset_ready(script_dir)


def is_google_drive_folder(url: str) -> bool:
    return bool(re.search(r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)", url or ""))


def extract_drive_folder_id(url: str) -> Optional[str]:
    m = re.search(r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)", url or "")
    return m.group(1) if m else None


def download_google_drive_folder(url: str, dest_dir: Path) -> Path:
    """Download a shared Drive folder using gdown. Returns the folder path with train/."""
    import gdown

    dest_dir.mkdir(parents=True, exist_ok=True)
    folder_id = extract_drive_folder_id(url)
    if not folder_id:
        raise ValueError(f"Not a Google Drive folder URL: {url}")

    # gdown creates dest_dir/<folder_name>/...
    print(f"[INFO] Downloading Google Drive folder id={folder_id} via gdown...", flush=True)
    gdown.download_folder(
        id=folder_id,
        output=str(dest_dir),
        quiet=False,
        use_cookies=False,
    )

    # Find a directory that contains train/
    for child in sorted(dest_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if child.is_dir() and (child / "train").is_dir():
            return child
    if (dest_dir / "train").is_dir():
        return dest_dir
    raise FileNotFoundError(
        f"Drive folder downloaded but no train/ found under {dest_dir}"
    )


def download_external_dataset_zip(script_dir: Path, logger=None) -> bool:
    """
    Download DATASET_ZIP_URL (ZIP file OR Google Drive folder) and normalize layout.
    Returns True if a usable dataset is ready afterward.
    Returns False only when DATASET_ZIP_URL is unset (caller may try PHP).
    Raises RuntimeError when URL is set but download/normalize fails.
    """
    url = get_dataset_zip_url()
    if not url:
        print("[WARN] DATASET_ZIP_URL is not set — will try PHP download_dataset.php", flush=True)
        return False

    env_set = bool(
        (os.getenv("DATASET_ZIP_URL") or "").strip()
        or (os.getenv("DATASET_URL") or "").strip()
        or (os.getenv("DATASET_DRIVE_URL") or "").strip()
    )
    if not env_set:
        print(
            "[INFO] Using built-in default Drive folder URL "
            "(set Railway Variable DATASET_ZIP_URL to override)",
            flush=True,
        )
    if dataset_ready(script_dir):
        print("[INFO] Dataset already present; skipping DATASET_ZIP_URL download", flush=True)
        return True

    training_data = script_dir / "training_data"
    training_data.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Using external dataset source (DATASET_ZIP_URL)", flush=True)
    if logger:
        try:
            logger.info("Downloading external dataset from DATASET_ZIP_URL")
        except Exception:
            pass

    try:
        # --- Google Drive folder ---
        if is_google_drive_folder(url):
            print(f"[INFO] DATASET_ZIP_URL is a Google Drive folder", flush=True)
            print(f"  {url[:100]}", flush=True)
            downloaded = download_google_drive_folder(url, training_data / "_drive_download")
            target = training_data / "Dataset20260715folder"
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(downloaded), str(target))
            drive_root = training_data / "_drive_download"
            if drive_root.exists():
                shutil.rmtree(drive_root, ignore_errors=True)
            ok = normalize_dataset_layout(script_dir)
            if not ok:
                raise RuntimeError("Could not normalize Drive folder into dataset_organized")
            print("[OK] External Drive folder dataset ready", flush=True)
            return True

        # --- ZIP file URL ---
        print(f"[INFO] Downloading external dataset ZIP...", flush=True)
        print(f"  URL: {resolve_dataset_zip_url(url)[:120]}...", flush=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp_path = Path(tmp.name)

        try:
            _download_url_to_file(url, tmp_path)
            size_mb = tmp_path.stat().st_size / (1024 * 1024)
            print(f"[OK] ZIP downloaded ({size_mb:.1f} MB)", flush=True)
            if tmp_path.stat().st_size < 1024:
                raise RuntimeError("ZIP too small — check DATASET_ZIP_URL is a direct download link")

            with open(tmp_path, "rb") as f:
                magic = f.read(4)
            if magic[:2] != b"PK":
                raise RuntimeError(
                    "Downloaded file is not a ZIP (bad Drive link?). "
                    "Use a file share link or a Drive folder URL."
                )

            with zipfile.ZipFile(tmp_path, "r") as zf:
                names = zf.namelist()
                images = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))]
                print(f"[INFO] ZIP files={len(names)} images={len(images)}", flush=True)
                if not images:
                    raise RuntimeError("ZIP has no images")
                zf.extractall(training_data)

            ok = normalize_dataset_layout(script_dir)
            if not ok:
                raise RuntimeError("Could not normalize extracted ZIP into dataset_organized")
            print("[OK] External ZIP dataset ready", flush=True)
            return True
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    except Exception as exc:
        print(f"[ERROR] External dataset download failed: {exc}", flush=True)
        if logger:
            try:
                logger.error(f"External dataset download failed: {exc}")
            except Exception:
                pass
        # Do NOT silently fall back to PHP when URL was configured
        raise RuntimeError(f"DATASET_ZIP_URL failed: {exc}") from exc
