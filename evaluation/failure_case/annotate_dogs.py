#!/usr/bin/env python3
"""
Interactive Per-Video Annotation Tool for Best- vs. Worst-Performing Identities
==============================================================================
Annotates each video clip (84 total: 53 query, 31 gallery) across the 20 target dogs
(10 best-performing with 100% mAP and 10 worst-performing) across the visual attributes:
  1. Distinctive visual features (Yes / No)
  2. Occlusion (Yes / No)
  3. Blurriness / low resolution (Yes / No)
  4. Lighting (Good / Poor)
  5. Viewpoint variation (Low / High)
  6. Free-form Notes (optional text)

All video-level annotations are saved directly to:
  evaluation/failure_case/best_vs_worst_video_annotations.csv

Usage:
  python evaluation/failure_case/annotate_dogs.py --port 8050
"""

import argparse
import base64
import http.server
import io
import json
import os
import re
import socketserver
import sys
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from PIL import Image

# Setup root path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[2]
VIDEO_ANNOT_CSV = SCRIPT_DIR / "best_vs_worst_video_annotations.csv"
DOG_ANNOT_CSV = SCRIPT_DIR / "best_vs_worst_annotations.csv"
SPLITS_PATH = ROOT_DIR / "splits.csv"
BBOX_PATH = ROOT_DIR / "bounding_boxes.csv"
IMAGES_DIR = ROOT_DIR / "Images"
VIDEOS_DIR = ROOT_DIR / "Videos"

ATTRIBUTE_DEFS = [
    {
        "id": "Distinctive visual features",
        "name": "Distinctive visual features",
        "options": ["Yes", "No"],
        "desc": "Distinctive coat patterns, facial markings, unusual ears, tail, etc."
    },
    {
        "id": "Occlusion",
        "name": "Occlusion",
        "options": ["Yes", "No"],
        "desc": "Dog is partially or significantly obscured in this video."
    },
    {
        "id": "Blurriness / low resolution",
        "name": "Blurriness / low resolution",
        "options": ["Yes", "No"],
        "desc": "Dog appearance is noticeably blurry, motion-blurred, or low res."
    },
    {
        "id": "Lighting",
        "name": "Lighting",
        "options": ["Good", "Poor"],
        "desc": "Sufficient illumination vs underexposed / overexposed / back-lit."
    },
    {
        "id": "Viewpoint variation",
        "name": "Viewpoint variation",
        "options": ["Low", "High"],
        "desc": "Static / steady angle vs dynamic changing perspectives (front, profile, rear)."
    }
]

EXPECTED_COLUMNS = [
    "Group", "Short ID", "DOG_ID", "VIDEO_ID", "Split", "mAP (%)", "Rank-1 (%)",
    "Distinctive visual features", "Occlusion", "Blurriness / low resolution",
    "Lighting", "Viewpoint variation", "Notes"
]


def load_bounding_boxes() -> Dict[tuple, tuple]:
    """Load ground truth bounding boxes."""
    if not BBOX_PATH.exists():
        return {}
    df = pd.read_csv(BBOX_PATH)
    return {
        (str(r["DOG_ID"]), str(r["VIDEO_ID"])): (
            int(r["x_top_left"]), int(r["y_top_left"]), int(r["width"]), int(r["height"])
        )
        for _, r in df.iterrows()
    }


def load_annotations_df() -> pd.DataFrame:
    """Load or initialize per-video annotations CSV."""
    if VIDEO_ANNOT_CSV.exists():
        df = pd.read_csv(VIDEO_ANNOT_CSV)
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[EXPECTED_COLUMNS]

    # Initialize from DOG_ANNOT_CSV and splits.csv
    splits = pd.read_csv(SPLITS_PATH)
    if DOG_ANNOT_CSV.exists():
        df_dogs = pd.read_csv(DOG_ANNOT_CSV)
    else:
        raise FileNotFoundError(f"{DOG_ANNOT_CSV} not found to initialize video annotations.")

    rows = []
    for _, r in df_dogs.iterrows():
        d_id = str(r["Full DOG_ID"])
        sp = splits[splits["DOG_ID"] == d_id]
        for _, srow in sp.iterrows():
            v_id = str(srow["VIDEO_ID"])
            rows.append({
                "Group": str(r["Group"]),
                "Short ID": str(r["Short ID"]),
                "DOG_ID": d_id,
                "VIDEO_ID": v_id,
                "Split": str(srow["SPLIT_CLOSED_SET"]),
                "mAP (%)": float(r["mAP (%)"]),
                "Rank-1 (%)": float(r["Rank-1 (%)"]),
                "Distinctive visual features": "",
                "Occlusion": "",
                "Blurriness / low resolution": "",
                "Lighting": "",
                "Viewpoint variation": "",
                "Notes": ""
            })
    df = pd.DataFrame(rows)
    save_annotations_df(df)
    return df


def save_annotations_df(df: pd.DataFrame) -> None:
    """Save dataframe atomically to CSV."""
    df = df[EXPECTED_COLUMNS]
    tmp_path = VIDEO_ANNOT_CSV.with_suffix(".tmp.csv")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(VIDEO_ANNOT_CSV)


def get_image_base64(dog_id: str, video_id: str, bbox_map: dict, max_size: int = 360) -> str:
    """Load, crop by bbox, resize, and convert to base64 jpeg string."""
    img_path = IMAGES_DIR / dog_id / f"{dog_id}-{video_id}.jpg"
    if not img_path.exists():
        return ""
    try:
        im = Image.open(img_path).convert("RGB")
        box = bbox_map.get((str(dog_id), str(video_id)))
        if box:
            x, y, w, h = box
            pad = 0.08
            x1 = max(0, int(x - w * pad))
            y1 = max(0, int(y - h * pad))
            x2 = min(im.width, int(x + w * (1.0 + pad)))
            y2 = min(im.height, int(y + h * (1.0 + pad)))
            if x2 > x1 and y2 > y1:
                im = im.crop((x1, y1, x2, y2))
        im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return ""


def prepare_dataset_payload() -> List[Dict[str, Any]]:
    """Assemble all dog metadata, video clips info, annotations and base64 crops."""
    df_video_annot = load_annotations_df()
    bbox_map = load_bounding_boxes()

    # Index annotations by (DOG_ID, VIDEO_ID)
    annot_lookup = {}
    for _, row in df_video_annot.iterrows():
        key = (str(row["DOG_ID"]), str(row["VIDEO_ID"]))
        annot_lookup[key] = {
            attr["id"]: ("" if pd.isna(row[attr["id"]]) else str(row[attr["id"]]))
            for attr in ATTRIBUTE_DEFS
        }
        annot_lookup[key]["Notes"] = "" if pd.isna(row["Notes"]) else str(row["Notes"])

    # Group by dog_id preserving order
    unique_dogs = []
    seen = set()
    for _, row in df_video_annot.iterrows():
        d_id = str(row["DOG_ID"])
        if d_id not in seen:
            seen.add(d_id)
            unique_dogs.append({
                "dog_id": d_id,
                "short_id": str(row["Short ID"]),
                "group": str(row["Group"]),
                "map_pct": float(row["mAP (%)"]),
                "rank1_pct": float(row["Rank-1 (%)"]),
            })

    dogs_data = []
    print(f"Loading visual data for {len(unique_dogs)} target identities ({len(df_video_annot)} videos total)...")
    for idx, d_info in enumerate(unique_dogs):
        d_id = d_info["dog_id"]
        dog_rows = df_video_annot[df_video_annot["DOG_ID"] == d_id]

        queries = []
        gallery = []

        for _, vrow in dog_rows.iterrows():
            v_id = str(vrow["VIDEO_ID"])
            split = str(vrow["Split"])
            b64 = get_image_base64(d_id, v_id, bbox_map)
            has_vid = (VIDEOS_DIR / d_id / f"{d_id}-{v_id}.mp4").exists()
            annots = annot_lookup.get((d_id, v_id), {a["id"]: "" for a in ATTRIBUTE_DEFS})
            if "Notes" not in annots: annots["Notes"] = ""

            clip_item = {
                "video_id": v_id,
                "short_vid": v_id[:8] + "...",
                "split": split,
                "image_b64": b64,
                "has_video": has_vid,
                "annotations": annots
            }

            if split == "query":
                queries.append(clip_item)
            else:
                gallery.append(clip_item)

        dogs_data.append({
            "index": idx,
            "dog_id": d_id,
            "short_id": d_info["short_id"],
            "group": d_info["group"],
            "map_pct": d_info["map_pct"],
            "rank1_pct": d_info["rank1_pct"],
            "n_q": len(queries),
            "n_g": len(gallery),
            "queries": queries,
            "gallery": gallery
        })

    print(f"Successfully assembled payload for {len(dogs_data)} identities.")
    return dogs_data


def generate_html(dogs_data: List[Dict[str, Any]], standalone: bool = False) -> str:
    """Generate the modern interactive Per-Video HTML application."""
    data_json = json.dumps(dogs_data)
    attrs_json = json.dumps(ATTRIBUTE_DEFS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DogReID Per-Video Attribute Annotator</title>
  <style>
    :root {{
      --bg-primary: #0b1120;
      --bg-surface: #1e293b;
      --bg-card: #334155;
      --bg-active: #1e3a8a;
      --border-color: #475569;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --accent-green: #22c55e;
      --accent-orange: #f97316;
      --accent-red: #ef4444;
      --accent-purple: #a855f7;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    body {{
      background: var(--bg-primary);
      color: var(--text-main);
      padding: 14px 20px;
      min-height: 100vh;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border-color);
    }}
    .header-left h1 {{
      font-size: 1.25rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .header-left p {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 2px;
    }}
    .header-actions {{
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    .btn {{
      background: #0284c7;
      color: #fff;
      border: none;
      padding: 7px 14px;
      border-radius: 6px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s;
    }}
    .btn:hover {{ background: #0369a1; }}
    .btn-secondary {{ background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-main); }}
    .btn-secondary:hover {{ background: #475569; }}
    .btn-success {{ background: #16a34a; }}
    .btn-success:hover {{ background: #15803d; }}

    .badge-progress {{
      background: #0f766e;
      color: #ccfbf1;
      padding: 5px 10px;
      border-radius: 12px;
      font-size: 0.8rem;
      font-weight: 700;
    }}

    /* Dog Navigation bar */
    .nav-bar {{
      display: flex;
      gap: 6px;
      overflow-x: auto;
      padding: 4px 0 10px 0;
      margin-bottom: 14px;
    }}
    .nav-pill {{
      flex: 0 0 auto;
      padding: 6px 12px;
      border-radius: 8px;
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      cursor: pointer;
      font-size: 0.8rem;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s;
    }}
    .nav-pill:hover {{ border-color: var(--accent-blue); }}
    .nav-pill.active {{
      background: #0284c7;
      border-color: #38bdf8;
      color: #fff;
      font-weight: 700;
    }}
    .nav-pill.all-done {{
      border-color: #22c55e !important;
      background: rgba(34, 197, 94, 0.15);
    }}
    .nav-pill.all-done.active {{
      background: #15803d !important;
      border-color: #4ade80 !important;
    }}
    .nav-pill.partial-done {{
      border-left: 4px solid var(--accent-orange);
    }}
    .nav-pill.best-group {{ color: #7dd3fc; }}
    .nav-pill.worst-group {{ color: #fdba74; }}
    .nav-pill.active.best-group {{ color: #fff; }}
    .nav-pill.active.worst-group {{ color: #fff; }}

    /* Main Grid Layout */
    .main-grid {{
      display: grid;
      grid-template-columns: 1fr 420px;
      gap: 16px;
    }}
    @media (max-width: 1200px) {{
      .main-grid {{ grid-template-columns: 1fr; }}
    }}

    .panel {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 16px;
    }}

    /* Dog info header */
    .dog-info-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border-color);
    }}
    .group-tag {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
    }}
    .tag-best {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #0284c7; }}
    .tag-worst {{ background: rgba(249, 115, 22, 0.2); color: #f97316; border: 1px solid #c2410c; }}

    .clips-section-title {{
      font-size: 0.92rem;
      font-weight: 700;
      color: var(--accent-blue);
      margin: 12px 0 8px 0;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .clips-section-title.gallery {{
      color: var(--accent-green);
    }}

    /* Video cards grid */
    .clips-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .clip-card {{
      background: var(--bg-card);
      border: 2px solid var(--border-color);
      border-radius: 10px;
      overflow: hidden;
      cursor: pointer;
      transition: all 0.15s ease;
      position: relative;
      display: flex;
      flex-direction: column;
    }}
    .clip-card:hover {{
      border-color: var(--accent-blue);
      transform: translateY(-2px);
    }}
    .clip-card.selected {{
      border-color: #38bdf8;
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.35);
      background: #1e3a5f;
    }}
    .clip-card.annotated {{
      border-top: 4px solid var(--accent-green);
    }}
    .clip-thumb {{
      position: relative;
      width: 100%;
      height: 130px;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .clip-thumb img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
    }}
    .play-overlay {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 38px;
      height: 38px;
      background: rgba(15, 23, 42, 0.75);
      border: 1.5px solid #fff;
      border-radius: 50%;
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      opacity: 0.85;
      transition: all 0.2s;
    }}
    .clip-card:hover .play-overlay {{
      opacity: 1;
      transform: translate(-50%, -50%) scale(1.1);
      background: #0284c7;
    }}
    .clip-badge {{
      position: absolute;
      top: 6px;
      left: 6px;
      font-size: 0.7rem;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(15, 23, 42, 0.85);
      color: #fff;
    }}
    .clip-status {{
      position: absolute;
      top: 6px;
      right: 6px;
      font-size: 0.7rem;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(34, 197, 94, 0.85);
      color: #fff;
    }}
    .clip-info {{
      padding: 8px 10px;
      font-size: 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}
    .clip-name {{
      font-weight: 700;
      color: #fff;
      font-family: monospace;
    }}
    .clip-annot-summary {{
      color: var(--text-muted);
      font-size: 0.7rem;
      line-height: 1.2;
    }}

    /* Active Video Player & Form Panel (Right Column) */
    .inspector-panel {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .player-card {{
      background: #000;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 220px;
      position: relative;
    }}
    .player-card video {{
      width: 100%;
      max-height: 280px;
      outline: none;
      background: #000;
    }}
    .player-placeholder {{
      color: var(--text-muted);
      font-size: 0.85rem;
      padding: 40px 20px;
      text-align: center;
    }}

    .active-video-details {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      font-size: 0.85rem;
      padding: 4px 2px;
    }}

    /* Form Fields */
    .form-group {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 8px;
    }}
    .form-group.filled {{
      border-left: 4px solid var(--accent-green);
    }}
    .form-title {{
      font-size: 0.82rem;
      font-weight: 700;
      margin-bottom: 2px;
      display: flex;
      justify-content: space-between;
    }}
    .form-desc {{
      font-size: 0.72rem;
      color: var(--text-muted);
      margin-bottom: 8px;
    }}
    .btn-row {{
      display: flex;
      gap: 8px;
    }}
    .btn-opt {{
      flex: 1;
      padding: 6px 10px;
      border-radius: 6px;
      background: var(--bg-surface);
      color: var(--text-main);
      border: 1.5px solid var(--border-color);
      font-size: 0.82rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .btn-opt:hover {{ border-color: var(--accent-blue); }}
    .btn-opt.sel-yes, .btn-opt.sel-good, .btn-opt.sel-low {{
      background: #0284c7;
      border-color: #38bdf8;
      color: #fff;
    }}
    .btn-opt.sel-no, .btn-opt.sel-poor, .btn-opt.sel-high {{
      background: #b45309;
      border-color: #f59e0b;
      color: #fff;
    }}

    .notes-box {{
      width: 100%;
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 6px 10px;
      color: #fff;
      font-size: 0.8rem;
      resize: vertical;
      min-height: 50px;
    }}

    /* Bottom summary table */
    .summary-card {{
      margin-top: 18px;
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
    }}
    th, td {{
      padding: 6px 8px;
      text-align: left;
      border-bottom: 1px solid var(--border-color);
    }}
    th {{
      background: var(--bg-card);
      color: #cbd5e1;
    }}
    tr:hover {{ background: rgba(255, 255, 255, 0.03); cursor: pointer; }}
    .cell-yes {{ color: var(--accent-green); font-weight: 600; }}
    .cell-no {{ color: var(--accent-orange); font-weight: 600; }}
  </style>
</head>
<body>

  <header>
    <div class="header-left">
      <h1>🎬 DogReID Per-Video Attribute Annotator</h1>
      <p>Annotating all 84 video clips (53 Query, 31 Gallery) across 20 target identities &bull; Auto-saves to <code>evaluation/failure_case/best_vs_worst_video_annotations.csv</code></p>
    </div>
    <div class="header-actions">
      <span class="badge-progress" id="progress-badge">Annotated: 0 / 84 clips (0%)</span>
      <button class="btn btn-secondary" onclick="jumpToNextUnannotated()">Next Unannotated [U]</button>
      <button class="btn btn-success" onclick="saveData(true)">💾 Save CSV [Ctrl+S]</button>
      <button class="btn btn-secondary" onclick="downloadCSV()">⬇ Export CSV</button>
    </div>
  </header>

  <!-- Identity Selector Bar -->
  <div class="nav-bar" id="dog-nav-bar"></div>

  <!-- Main View -->
  <div class="main-grid">
    
    <!-- Left Column: Video Clips for Selected Dog -->
    <div class="panel">
      <div class="dog-info-header">
        <div>
          <h2 style="font-size:1.15rem; display:flex; align-items:center; gap:8px;">
            <span id="current-dog-title">#1: a6c4d7b5...</span>
            <span id="current-dog-group" class="group-tag tag-best">Best 10</span>
          </h2>
          <div style="font-size:0.75rem; color:var(--text-muted); font-family:monospace; margin-top:3px;" id="current-dog-fullid"></div>
        </div>
        <div style="font-size:0.82rem; text-align:right;">
          mAP: <b id="dog-map">100.0%</b> | Rank-1: <b id="dog-rank1">100.0%</b><br>
          <span style="color:var(--text-muted)">Query clips: <b id="dog-nq">1</b> | Gallery clips: <b id="dog-ng">4</b></span>
        </div>
      </div>

      <!-- Query Clips -->
      <div class="clips-section-title">
        🔍 Query Video Clips (<span id="query-section-count">0</span>)
        <span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">&bull; Click any clip to play & annotate</span>
      </div>
      <div class="clips-grid" id="query-clips-grid"></div>

      <!-- Gallery Clips -->
      <div class="clips-section-title gallery">
        🏛 Gallery Video Clips (<span id="gallery-section-count">0</span>)
        <span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">&bull; Click any clip to play & annotate</span>
      </div>
      <div class="clips-grid" id="gallery-clips-grid"></div>
    </div>

    <!-- Right Column: Video Player & Attribute Form -->
    <div class="panel inspector-panel">
      
      <!-- Video Player -->
      <div class="player-card" id="player-container">
        <div class="player-placeholder" id="player-placeholder">Select a video clip to play and annotate</div>
      </div>

      <div class="active-video-details">
        <div>
          <b id="active-video-name" style="font-family:monospace; color:var(--accent-blue);">No video selected</b>
          <span id="active-video-badge" style="margin-left:6px; font-size:0.75rem;"></span>
        </div>
        <div id="save-status" style="font-size:0.78rem; color:var(--accent-green); font-weight:700;">✓ Ready</div>
      </div>

      <!-- Form Inputs -->
      <div id="form-container">
        <!-- Injected dynamically based on active video -->
      </div>

      <!-- Video Notes -->
      <div class="form-group" style="margin-bottom:0;">
        <div class="form-title">
          <span>Notes & Observations (Optional)</span>
        </div>
        <div class="form-desc">Any extra details: distinct collar, wet fur, camera shake, severe occlusion...</div>
        <textarea id="video-notes-input" class="notes-box" placeholder="Write any specific clip observations here..." oninput="onNotesInput(this.value)"></textarea>
      </div>

      <!-- Navigation buttons between clips -->
      <div style="display:flex; justify-content:space-between; margin-top:8px;">
        <button class="btn btn-secondary" style="flex:1; margin-right:6px;" onclick="prevClip()">◀ Prev Clip [←]</button>
        <button class="btn" style="flex:1; margin-left:6px;" onclick="nextClip()">Next Clip [→] ▶</button>
      </div>

    </div>

  </div>

  <!-- Summary Table (All 84 Videos) -->
  <div class="summary-card">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
      <h3 style="font-size:0.95rem;">All Video Clips & Annotation Status (<span id="table-total-count">84</span> total)</h3>
      <span style="font-size:0.75rem; color:var(--text-muted);">Click any row to jump directly to that video</span>
    </div>
    <div style="overflow-x:auto; max-height:360px;">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Dog Group</th>
            <th>Dog ID</th>
            <th>Split</th>
            <th>Video ID</th>
            <th>Distinctive Features</th>
            <th>Occlusion</th>
            <th>Blurriness</th>
            <th>Lighting</th>
            <th>Viewpoint</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody id="summary-tbody"></tbody>
      </table>
    </div>
  </div>

  <script>
    const dogs = {data_json};
    const attributes = {attrs_json};
    const isStandalone = {str(standalone).lower()};

    // Flatten all clips into a unified list for fast index navigation
    let allClips = [];
    dogs.forEach((d, dIdx) => {{
      d.queries.forEach((q, qIdx) => {{
        allClips.push({{ dogIdx: dIdx, dog: d, clip: q, split: "query", clipIdx: qIdx }});
      }});
      d.gallery.forEach((g, gIdx) => {{
        allClips.push({{ dogIdx: dIdx, dog: d, clip: g, split: "gallery", clipIdx: gIdx }});
      }});
    }});

    // Find first unannotated clip on startup to resume smoothly
    let firstUnannotated = allClips.find(c => !isClipAnnotated(c.clip));
    let startItem = firstUnannotated || (allClips.length > 0 ? allClips[0] : null);

    let currentDogIndex = startItem ? startItem.dogIdx : 0;
    let activeClipKey = startItem ? (startItem.dog.dog_id + "_" + startItem.clip.video_id) : "";

    function isClipAnnotated(clip) {{
      return attributes.every(attr => clip.annotations[attr.id] && clip.annotations[attr.id].trim() !== "");
    }}

    function isDogFullyAnnotated(d) {{
      const clips = [...d.queries, ...d.gallery];
      return clips.every(isClipAnnotated);
    }}

    function updateProgress() {{
      const total = allClips.length;
      const done = allClips.filter(c => isClipAnnotated(c.clip)).length;
      const pct = total === 0 ? 0 : Math.round((done / total) * 100);
      document.getElementById("progress-badge").innerText = `Annotated: ${{done}} / ${{total}} clips (${{pct}}%)`;
      document.getElementById("table-total-count").innerText = total;
    }}

    function renderDogNavBar() {{
      const nav = document.getElementById("dog-nav-bar");
      nav.innerHTML = "";
      dogs.forEach((d, i) => {{
        const pill = document.createElement("div");
        const allDone = isDogFullyAnnotated(d);
        const doneClips = [...d.queries, ...d.gallery].filter(isClipAnnotated).length;
        const totalClips = d.queries.length + d.gallery.length;
        const partialDone = doneClips > 0 && !allDone;

        pill.className = "nav-pill" + 
          (i === currentDogIndex ? " active" : "") +
          (allDone ? " all-done" : (partialDone ? " partial-done" : "")) +
          (d.group.includes("Best") ? " best-group" : " worst-group");

        let statusBadge = "";
        if (allDone) {{
          statusBadge = `<span style="background:#22c55e; color:#fff; padding:1px 5px; border-radius:10px; font-size:0.68rem; font-weight:bold;">✓ Done</span>`;
        }} else if (doneClips > 0) {{
          statusBadge = `<span style="background:#ea580c; color:#fff; padding:1px 5px; border-radius:10px; font-size:0.68rem;">${{doneClips}}/${{totalClips}}</span>`;
        }} else {{
          statusBadge = `<span style="opacity:0.6; font-size:0.68rem;">0/${{totalClips}}</span>`;
        }}

        pill.innerHTML = `<span>#${{i+1}} ${{d.short_id}}</span> ${{statusBadge}}`;
        pill.onclick = () => selectDog(i);
        nav.appendChild(pill);
      }});
    }}

    function selectDog(index) {{
      currentDogIndex = index;
      const d = dogs[currentDogIndex];
      // Select first clip of this dog if active clip is not in this dog
      const dogClips = [...d.queries, ...d.gallery];
      const hasActive = dogClips.some(c => (d.dog_id + "_" + c.video_id) === activeClipKey);
      if (!hasActive && dogClips.length > 0) {{
        activeClipKey = d.dog_id + "_" + dogClips[0].video_id;
      }}
      renderAll();
    }}

    function selectClip(dogIdx, videoId) {{
      currentDogIndex = dogIdx;
      activeClipKey = dogs[dogIdx].dog_id + "_" + videoId;
      renderAll();
    }}

    function renderAll() {{
      renderDogNavBar();
      renderCurrentDogPanel();
      renderInspector();
      renderSummaryTable();
      updateProgress();
    }}

    function renderCurrentDogPanel() {{
      const d = dogs[currentDogIndex];
      document.getElementById("current-dog-title").innerText = `#${{currentDogIndex+1}}: ${{d.short_id}}`;
      document.getElementById("current-dog-fullid").innerText = d.dog_id;
      
      const grpTag = document.getElementById("current-dog-group");
      grpTag.innerText = d.group;
      grpTag.className = "group-tag " + (d.group.includes("Best") ? "tag-best" : "tag-worst");

      document.getElementById("dog-map").innerText = d.map_pct.toFixed(1) + "%";
      document.getElementById("dog-rank1").innerText = d.rank1_pct.toFixed(1) + "%";
      document.getElementById("dog-nq").innerText = d.n_q;
      document.getElementById("dog-ng").innerText = d.n_g;

      // Render Queries
      const qGrid = document.getElementById("query-clips-grid");
      document.getElementById("query-section-count").innerText = d.queries.length;
      qGrid.innerHTML = "";
      d.queries.forEach((q, idx) => {{
        qGrid.appendChild(createClipCard(d, q, "Query", idx));
      }});

      // Render Gallery
      const gGrid = document.getElementById("gallery-clips-grid");
      document.getElementById("gallery-section-count").innerText = d.gallery.length;
      gGrid.innerHTML = "";
      d.gallery.forEach((g, idx) => {{
        gGrid.appendChild(createClipCard(d, g, "Gallery", idx));
      }});
    }}

    function createClipCard(dog, clip, splitLabel, idx) {{
      const key = dog.dog_id + "_" + clip.video_id;
      const isSelected = key === activeClipKey;
      const done = isClipAnnotated(clip);

      const card = document.createElement("div");
      card.className = "clip-card" + (isSelected ? " selected" : "") + (done ? " annotated" : "");
      card.onclick = () => selectClip(dog.index, clip.video_id);

      // Brief summary of annotations
      let summaryText = "";
      if (done) {{
        summaryText = `<b style="color:var(--accent-green)">✓ Annotated</b> &bull; Feat:${{clip.annotations["Distinctive visual features"]}} | Occl:${{clip.annotations["Occlusion"]}} | Light:${{clip.annotations["Lighting"]}}`;
      }} else {{
        summaryText = `<span style="color:#f59e0b;">⏳ Pending</span> &bull; Click to watch & annotate`;
      }}

      card.innerHTML = `
        <div class="clip-thumb">
          <img src="data:image/jpeg;base64,${{clip.image_b64}}" alt="Thumbnail">
          <div class="play-overlay">▶</div>
          <div class="clip-badge">${{splitLabel}} #${{idx+1}}</div>
          ${{done ? '<div class="clip-status" style="background:#16a34a; font-weight:bold;">✓ Done</div>' : '<div class="clip-status" style="background:#475569; opacity:0.8;">Unset</div>'}}
        </div>
        <div class="clip-info">
          <div class="clip-name">${{clip.short_vid}}</div>
          <div class="clip-annot-summary">${{summaryText}}</div>
        </div>
      `;
      return card;
    }}

    function getActiveClipObject() {{
      for (const item of allClips) {{
        if ((item.dog.dog_id + "_" + item.clip.video_id) === activeClipKey) {{
          return item;
        }}
      }}
      return null;
    }}

    function renderInspector() {{
      const activeItem = getActiveClipObject();
      const container = document.getElementById("form-container");
      const playerBox = document.getElementById("player-container");
      const notesBox = document.getElementById("video-notes-input");

      if (!activeItem) {{
        playerBox.innerHTML = '<div class="player-placeholder">No video selected.</div>';
        container.innerHTML = "";
        return;
      }}

      const {{ dog, clip, split, clipIdx }} = activeItem;

      // Update Header
      document.getElementById("active-video-name").innerText = `${{split.toUpperCase()}} #${{clipIdx+1}}: ${{clip.video_id}}`;
      document.getElementById("active-video-badge").innerText = `(Dog #${{dog.index+1}} ${{dog.short_id}})`;

      // Update Video Player
      if (clip.has_video && !isStandalone) {{
        const vidUrl = `/api/video?dog_id=${{encodeURIComponent(dog.dog_id)}}&video_id=${{encodeURIComponent(clip.video_id)}}`;
        playerBox.innerHTML = `
          <video controls autoplay loop muted playsinline style="width:100%; max-height:280px;">
            <source src="${{vidUrl}}" type="video/mp4">
            Your browser does not support HTML5 video.
          </video>
        `;
      }} else {{
        playerBox.innerHTML = `
          <img src="data:image/jpeg;base64,${{clip.image_b64}}" style="width:100%; max-height:280px; object-fit:contain; background:#000;">
        `;
      }}

      // Update Notes
      notesBox.value = clip.annotations["Notes"] || "";

      // Render Attribute Form
      container.innerHTML = "";
      attributes.forEach(attr => {{
        const val = clip.annotations[attr.id] || "";
        const grp = document.createElement("div");
        grp.className = "form-group" + (val ? " filled" : "");

        let btnsHtml = "";
        attr.options.forEach(opt => {{
          const isSel = val === opt;
          const optLower = opt.toLowerCase();
          const selClass = isSel ? ("sel-" + optLower) : "";
          btnsHtml += `
            <button class="btn-opt ${{selClass}}" onclick="setClipAnnotation('${{attr.id}}', '${{opt}}')">
              ${{opt}}
            </button>
          `;
        }});

        grp.innerHTML = `
          <div class="form-title">
            <span>${{attr.name}}</span>
            ${{val ? '<span style="color:var(--accent-green); font-size:0.75rem;">✓ ' + val + '</span>' : ''}}
          </div>
          <div class="form-desc">${{attr.desc}}</div>
          <div class="btn-row">${{btnsHtml}}</div>
        `;
        container.appendChild(grp);
      }});
    }}

    function setClipAnnotation(attrId, value) {{
      const activeItem = getActiveClipObject();
      if (!activeItem) return;

      activeItem.clip.annotations[attrId] = value;
      renderInspector();
      renderCurrentDogPanel();
      renderDogNavBar();
      renderSummaryTable();
      updateProgress();
      saveData(false);
    }}

    function onNotesInput(val) {{
      const activeItem = getActiveClipObject();
      if (!activeItem) return;
      activeItem.clip.annotations["Notes"] = val;
      saveData(false);
    }}

    function prevClip() {{
      const idx = allClips.findIndex(c => (c.dog.dog_id + "_" + c.clip.video_id) === activeClipKey);
      if (idx > 0) {{
        const prev = allClips[idx - 1];
        selectClip(prev.dogIdx, prev.clip.video_id);
      }}
    }}

    function nextClip() {{
      const idx = allClips.findIndex(c => (c.dog.dog_id + "_" + c.clip.video_id) === activeClipKey);
      if (idx >= 0 && idx < allClips.length - 1) {{
        const next = allClips[idx + 1];
        selectClip(next.dogIdx, next.clip.video_id);
      }}
    }}

    function jumpToNextUnannotated() {{
      const curIdx = allClips.findIndex(c => (c.dog.dog_id + "_" + c.clip.video_id) === activeClipKey);
      for (let i = 1; i <= allClips.length; i++) {{
        const testIdx = (curIdx + i) % allClips.length;
        if (!isClipAnnotated(allClips[testIdx].clip)) {{
          const target = allClips[testIdx];
          selectClip(target.dogIdx, target.clip.video_id);
          return;
        }}
      }}
      alert("All 84 videos across all 20 identities have been annotated!");
    }}

    function renderSummaryTable() {{
      const tbody = document.getElementById("summary-tbody");
      tbody.innerHTML = "";
      allClips.forEach((item, i) => {{
        const {{ dog, clip, split }} = item;
        const key = dog.dog_id + "_" + clip.video_id;
        const isCurrent = key === activeClipKey;

        const tr = document.createElement("tr");
        if (isCurrent) tr.style.background = "rgba(56, 189, 248, 0.18)";

        const cell = (v) => {{
          if (v === "Yes" || v === "Good" || v === "Low") return `<span class="cell-yes">${{v}}</span>`;
          if (v === "No" || v === "Poor" || v === "High") return `<span class="cell-no">${{v}}</span>`;
          return `<span style="color:var(--text-muted)">-</span>`;
        }};

        tr.innerHTML = `
          <td>${{i+1}}</td>
          <td><span class="${{dog.group.includes("Best") ? "tag-best" : "tag-worst"}}" style="padding:1px 5px; border-radius:3px;">${{dog.group.includes("Best") ? "Best" : "Worst"}}</span></td>
          <td><code>${{dog.short_id}}</code></td>
          <td><b style="color:${{split === 'query' ? 'var(--accent-blue)' : 'var(--accent-green)'}}">${{split.toUpperCase()}}</b></td>
          <td><code>${{clip.short_vid}}</code></td>
          <td>${{cell(clip.annotations["Distinctive visual features"])}}</td>
          <td>${{cell(clip.annotations["Occlusion"])}}</td>
          <td>${{cell(clip.annotations["Blurriness / low resolution"])}}</td>
          <td>${{cell(clip.annotations["Lighting"])}}</td>
          <td>${{cell(clip.annotations["Viewpoint variation"])}}</td>
          <td style="max-width:140px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${{clip.annotations["Notes"] || ""}}</td>
        `;
        tr.onclick = () => selectClip(dog.index, clip.video_id);
        tbody.appendChild(tr);
      }});
    }}

    function buildCSV() {{
      const headers = [
        "Group", "Short ID", "DOG_ID", "VIDEO_ID", "Split", "mAP (%)", "Rank-1 (%)",
        "Distinctive visual features", "Occlusion", "Blurriness / low resolution",
        "Lighting", "Viewpoint variation", "Notes"
      ];
      let csv = headers.join(",") + "\\n";
      allClips.forEach(item => {{
        const {{ dog, clip, split }} = item;
        const row = [
          `"${{dog.group}}"`,
          `"${{dog.short_id}}"`,
          `"${{dog.dog_id}}"`,
          `"${{clip.video_id}}"`,
          `"${{split}}"`,
          dog.map_pct.toFixed(1),
          dog.rank1_pct.toFixed(1),
          `"${{clip.annotations["Distinctive visual features"] || ""}}"`,
          `"${{clip.annotations["Occlusion"] || ""}}"`,
          `"${{clip.annotations["Blurriness / low resolution"] || ""}}"`,
          `"${{clip.annotations["Lighting"] || ""}}"`,
          `"${{clip.annotations["Viewpoint variation"] || ""}}"`,
          `"${{(clip.annotations["Notes"] || "").replace(/"/g, '""')}}"`
        ];
        csv += row.join(",") + "\\n";
      }});
      return csv;
    }}

    function saveData(showNotice = false) {{
      const csvContent = buildCSV();
      const statusLbl = document.getElementById("save-status");

      if (isStandalone) {{
        localStorage.setItem("dogreid_video_annotations_csv", csvContent);
        statusLbl.innerText = "✓ Saved LocalStorage";
        if (showNotice) alert("Saved to local browser storage! Use 'Export CSV' to download.");
        return;
      }}

      fetch("/api/save", {{
        method: "POST",
        headers: {{ "Content-Type": "text/csv" }},
        body: csvContent
      }})
      .then(res => res.json())
      .then(data => {{
        statusLbl.innerText = "✓ Saved to CSV";
        if (showNotice) alert("Successfully saved annotations to evaluation/failure_case/best_vs_worst_video_annotations.csv!");
      }})
      .catch(err => {{
        console.error("Save error:", err);
        statusLbl.innerText = "✗ Save failed!";
        statusLbl.style.color = "var(--accent-red)";
      }});
    }}

    function downloadCSV() {{
      const csv = buildCSV();
      const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", "best_vs_worst_video_annotations.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }}

    // Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {{
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === "ArrowLeft") prevClip();
      if (e.key === "ArrowRight") nextClip();
      if (e.key === "u" || e.key === "U") jumpToNextUnannotated();
      if (e.key === "s" && (e.ctrlKey || e.metaKey)) {{
        e.preventDefault();
        saveData(true);
      }}
    }});

    // Initialize
    renderAll();
  </script>
</body>
</html>
"""


class AnnotationServerHandler(http.server.BaseHTTPRequestHandler):
    """Custom HTTP handler serving the per-video annotation GUI and REST endpoints."""

    dogs_data = []

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            # Re-read CSV fresh on page load so external or latest edits are instantly reflected
            self.dogs_data = prepare_dataset_payload()
            html_content = generate_html(self.dogs_data, standalone=False)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))
        elif self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.dogs_data).encode("utf-8"))
        elif self.path == "/api/export":
            if VIDEO_ANNOT_CSV.exists():
                with open(VIDEO_ANNOT_CSV, "rb") as f:
                    csv_data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", "attachment; filename=best_vs_worst_video_annotations.csv")
                self.end_headers()
                self.wfile.write(csv_data)
            else:
                self.send_error(404, "File not found")
        elif self.path.startswith("/api/video"):
            self.handle_video_stream()
        else:
            self.send_error(404, "Not Found")

    def handle_video_stream(self):
        """Stream MP4 video file with HTTP 206 Partial Content support."""
        query_str = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query_str)
        dog_id = params.get("dog_id", [""])[0]
        video_id = params.get("video_id", [""])[0]

        video_path = VIDEOS_DIR / dog_id / f"{dog_id}-{video_id}.mp4"
        if not video_path.exists():
            self.send_error(404, f"Video not found: {video_path.name}")
            return

        file_size = os.path.getsize(video_path)
        range_header = self.headers.get("Range")

        start = 0
        end = file_size - 1

        if range_header:
            range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if range_match:
                start = int(range_match.group(1))
                if range_match.group(2):
                    end = int(range_match.group(2))

        length = end - start + 1
        with open(video_path, "rb") as f:
            f.seek(start)
            data = f.read(length)

        if range_header:
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        else:
            self.send_response(200)

        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path == "/api/save":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8")
            try:
                tmp_path = VIDEO_ANNOT_CSV.with_suffix(".tmp.csv")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(post_data)
                tmp_path.replace(VIDEO_ANNOT_CSV)
                print(f"-> [AUTO-SAVED] Successfully wrote updated annotations to {VIDEO_ANNOT_CSV.name}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "file": str(VIDEO_ANNOT_CSV)}).encode("utf-8"))
            except Exception as e:
                print(f"[ERROR] Failed to save CSV: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

    def log_message(self, format, *args):
        sys.stderr.write(f"[Server] {args[0]} -> {args[1]}\n")


def run_server(port: int = 8050, open_browser: bool = False):
    """Run lightweight HTTP server with auto port finding."""
    dogs_data = prepare_dataset_payload()
    AnnotationServerHandler.dogs_data = dogs_data

    current_port = port
    while current_port < port + 50:
        try:
            with socketserver.TCPServer(("", current_port), AnnotationServerHandler) as httpd:
                url = f"http://localhost:{current_port}"
                print("\n" + "=" * 65)
                print(f"🎬 DogReID Per-Video Attribute Annotation Tool Live!")
                print(f"   URL: {url}")
                print(f"   Target CSV: {VIDEO_ANNOT_CSV}")
                print(f"   Total Videos: {sum(d['n_q'] + d['n_g'] for d in dogs_data)} across {len(dogs_data)} dogs")
                print(f"   Press Ctrl+C to stop server")
                print("=" * 65 + "\n")

                if open_browser:
                    webbrowser.open(url)
                httpd.serve_forever()
                break
        except OSError as e:
            if "Address already in use" in str(e) or e.errno == 98 or e.errno == 10048:
                current_port += 1
            else:
                raise e


def export_standalone_html(output_path: Path):
    """Export self-contained static HTML file with embedded base64 images."""
    dogs_data = prepare_dataset_payload()
    html_content = generate_html(dogs_data, standalone=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"-> [SUCCESS] Exported standalone offline annotator to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DogReID Per-Video Attribute Annotation Tool")
    parser.add_argument("--port", type=int, default=8050, help="Local server port (default: 8050)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--export-html", type=str, default="", help="Export standalone static HTML file to path")

    args = parser.parse_args()

    if args.export_html:
        export_standalone_html(Path(args.export_html))
    else:
        run_server(port=args.port, open_browser=not args.no_browser)
