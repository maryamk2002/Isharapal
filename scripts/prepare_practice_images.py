#!/usr/bin/env python3
"""
Prepare Sign Reference Images for Frontend

This script copies and optimizes reference images from the backend dataset
to the frontend assets folder for use in the Practice Mode and Disambiguation features.

Features:
- Copies one representative image per sign to frontend
- Resizes and optimizes images for web (512x512 WebP)
- Generates a manifest JSON for the frontend

Usage:
    python prepare_practice_images.py

Output:
    - frontend/assets/signs/{SignName}.webp
    - frontend/assets/signs/manifest.json
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Try to import PIL for image optimization
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARNING] Pillow not installed. Images will be copied without optimization.")
    print("         Install with: pip install Pillow")

# Try to import OpenCV for video frame extraction
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARNING] OpenCV not installed. Cannot extract frames from videos.")
    print("         Install with: pip install opencv-python")

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DATA = PROJECT_ROOT / "backend" / "data" / "Pakistan Sign Language Urdu Alphabets"
FRONTEND_ASSETS = PROJECT_ROOT / "frontend" / "assets" / "signs"

# Image settings
TARGET_SIZE = (512, 512)
JPEG_QUALITY = 85
WEBP_QUALITY = 80

# Urdu mapping for labels
URDU_MAPPING = {
    "1-Hay": "ہ",
    "2-Hay": "ھ",
    "Ain": "ع",
    "Alif": "ا",
    "Alifmad": "آ",
    "Aray": "ڑ",
    "Bariyay": "ے",  # Bari Yay (same as Byeh)
    "Bay": "ب",
    "Byeh": "ے",
    "Chay": "چ",
    "Chotiyay": "ی",  # Choti Yay (same as Cyeh)
    "Cyeh": "ی",
    "Daal": "ڈ",
    "Dal": "د",
    "Dochahay": "ح",
    "Fay": "ف",
    "Gaaf": "گ",
    "Ghain": "غ",
    "Hamza": "ء",
    "Jeem": "ج",
    "Kaf": "ک",
    "Khay": "خ",
    "Kiaf": "ق",
    "Lam": "ل",
    "Meem": "م",
    "Nuun": "ن",
    "Nuungh": "ں",
    "Pay": "پ",
    "Ray": "ر",
    "Say": "ث",
    "Seen": "س",
    "Sheen": "ش",
    "Suad": "ص",
    "Taay": "ط",
    "Tay": "ت",
    "Tuey": "ٹ",
    "Wao": "و",
    "Zaal": "ذ",
    "Zaey": "ض",
    "Zay": "ز",
    "Zuad": "ظ",
    "Zuey": "ژ"
}


def find_best_image(sign_dir: Path) -> tuple[Path | None, bool]:
    """
    Find the best representative image for a sign.
    Returns (path, is_video) tuple.
    """
    # Look for JPG/JPEG files (prefer uppercase extensions as they're often higher quality)
    jpg_files = list(sign_dir.glob("*.JPG")) + list(sign_dir.glob("*.jpg")) + list(sign_dir.glob("*.jpeg"))
    
    if not jpg_files:
        # Try PNG as fallback
        jpg_files = list(sign_dir.glob("*.png")) + list(sign_dir.glob("*.PNG"))
    
    if jpg_files:
        # Sort by file size (larger often means higher quality) and pick a middle-sized one
        jpg_files.sort(key=lambda f: f.stat().st_size, reverse=True)
        # Pick one from the top 20% (good quality but not outlier)
        top_idx = max(0, len(jpg_files) // 5)
        return jpg_files[top_idx], False
    
    # No images found - try to extract from video
    if HAS_CV2:
        mp4_files = list(sign_dir.glob("*.mp4")) + list(sign_dir.glob("*.MP4"))
        if mp4_files:
            # Pick the first video (or one with a descriptive name)
            for video in mp4_files:
                # Prefer videos with the sign name in filename
                if sign_dir.name.lower() in video.name.lower():
                    return video, True
            # Otherwise just use the first one
            return mp4_files[0], True
    
    return None, False


def extract_frame_from_video(video_path: Path, output_path: Path) -> bool:
    """Extract a representative frame from video."""
    if not HAS_CV2:
        return False
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open video: {video_path.name}")
            return False
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"  [ERROR] Video has no frames: {video_path.name}")
            cap.release()
            return False
        
        # Seek to middle of video (likely shows the sign gesture)
        target_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print(f"  [ERROR] Cannot read frame from video: {video_path.name}")
            return False
        
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save as temporary image for processing
        temp_path = output_path.with_suffix('.temp.jpg')
        Image.fromarray(frame_rgb).save(temp_path, 'JPEG', quality=95)
        
        return temp_path
        
    except Exception as e:
        print(f"  [ERROR] Video extraction failed: {e}")
        return False


def optimize_image(src_path: Path, dst_path: Path) -> bool:
    """Optimize and resize image for web."""
    if not HAS_PIL:
        # Just copy the file
        shutil.copy2(src_path, dst_path.with_suffix(src_path.suffix))
        return True
    
    try:
        with Image.open(src_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Calculate aspect-preserving resize
            img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # Create a square canvas
            canvas = Image.new('RGB', TARGET_SIZE, (255, 255, 255))
            
            # Center the image
            x = (TARGET_SIZE[0] - img.width) // 2
            y = (TARGET_SIZE[1] - img.height) // 2
            canvas.paste(img, (x, y))
            
            # Save as WebP for best size/quality ratio
            webp_path = dst_path.with_suffix('.webp')
            canvas.save(webp_path, 'WEBP', quality=WEBP_QUALITY, method=6)
            
            # Also save as JPG for fallback (older browsers)
            jpg_path = dst_path.with_suffix('.jpg')
            canvas.save(jpg_path, 'JPEG', quality=JPEG_QUALITY, optimize=True)
            
            return True
            
    except Exception as e:
        print(f"  [ERROR] Failed to optimize {src_path.name}: {e}")
        return False


def main():
    print("=" * 60)
    print("PSL Practice Images Preparation Script")
    print("=" * 60)
    
    # Verify source directory exists
    if not BACKEND_DATA.exists():
        print(f"[ERROR] Source directory not found: {BACKEND_DATA}")
        print("Please ensure the sign language images are in backend/data/Pakistan Sign Language Urdu Alphabets/")
        sys.exit(1)
    
    # Create output directory
    FRONTEND_ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Source: {BACKEND_DATA}")
    print(f"[INFO] Output: {FRONTEND_ASSETS}")
    
    # Process each sign directory
    manifest = {
        "version": "1.0",
        "generated": "",
        "signs": {}
    }
    
    # Get all sign directories
    sign_dirs = [d for d in BACKEND_DATA.iterdir() if d.is_dir()]
    print(f"\n[INFO] Found {len(sign_dirs)} sign directories\n")
    
    success_count = 0
    skip_count = 0
    
    for sign_dir in sorted(sign_dirs):
        sign_name = sign_dir.name
        
        # Skip non-sign directories
        if sign_name.lower() in ('test', 'train', 'val', 'processed', 'raw'):
            continue
        
        # Find best image or video
        best_source, is_video = find_best_image(sign_dir)
        
        if best_source is None:
            print(f"  [SKIP] {sign_name}: No suitable image or video found")
            skip_count += 1
            continue
        
        # Process image
        dst_path = FRONTEND_ASSETS / sign_name
        source_type = "image"
        temp_file = None
        
        if is_video:
            # Extract frame from video
            temp_file = extract_frame_from_video(best_source, dst_path)
            if not temp_file:
                print(f"  [SKIP] {sign_name}: Could not extract frame from video")
                skip_count += 1
                continue
            best_source_for_optimize = temp_file
            source_type = "video_frame"
        else:
            best_source_for_optimize = best_source
        
        if optimize_image(best_source_for_optimize, dst_path):
            urdu_char = URDU_MAPPING.get(sign_name, "?")
            
            manifest["signs"][sign_name] = {
                "romanized": sign_name,
                "urdu": urdu_char,
                "webp": f"{sign_name}.webp",
                "jpg": f"{sign_name}.jpg",
                "source": best_source.name,
                "type": source_type
            }
            
            print(f"  [OK] {sign_name} ({urdu_char}) [{source_type}]")
            success_count += 1
        else:
            skip_count += 1
        
        # Clean up temp file
        if temp_file and Path(temp_file).exists():
            try:
                Path(temp_file).unlink()
            except:
                pass
    
    # Generate timestamp
    from datetime import datetime
    manifest["generated"] = datetime.now().isoformat()
    manifest["total_signs"] = success_count
    
    # Write manifest
    manifest_path = FRONTEND_ASSETS / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"[DONE] Processed {success_count} signs, skipped {skip_count}")
    print(f"[DONE] Manifest saved to {manifest_path}")
    print("=" * 60)
    
    # Print usage instructions
    print("\n[NEXT STEPS]")
    print("1. The images are now available in frontend/assets/signs/")
    print("2. Use manifest.json to load sign information in JavaScript")
    print("3. Reference images in Practice Mode and Disambiguation UI")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


