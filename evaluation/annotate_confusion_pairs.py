#!/usr/bin/env python3
"""
Interactive Inter-Identity Similarity & Confusion Annotator (Top 20 Pairs)
==========================================================================
Analyzes the Top 20 most frequently confused dog pairs:
  - Pairs #1 to #16: All 16 Mutual Confusion (Bidirectional Top-1) pairs
  - Pairs #17 to #20: Top 4 highest error pairs
For each confused pair (Dog A <-> Dog B):
  - Displays side-by-side native HTML5 VIDEO PLAYERS for Dog A (Left) and Dog B (Right)
  - Seamlessly loops and plays clips side-by-side with synchronized controls and thumbnail selectors
  - Quick annotation buttons for the 5 morphological comparisons:
      1. Breed match (Same breed / Different breed)
      2. Coat colour (Similar / Different)
      3. Coat pattern (Similar / Different)
      4. Body shape & size (Similar / Different)
      5. Facial appearance (Similar / Different)
      6. Key discriminating feature missed / Notes (text)

Saves all annotations to:
  evaluation/confusion_pairs_annotations.csv

Usage:
  python evaluation/annotate_confusion_pairs.py --port 8055
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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
CONF_CSV_PATH = ROOT_DIR / "evaluation" / "confusion_pairs_annotations.csv"
DIST_MATRIX_CSV = ROOT_DIR / "evaluation" / "csvs_closed_results" / "baseline_vid2vid" / "closed_dist_matrix.csv"
SPLITS_PATH = ROOT_DIR / "splits.csv"
BBOX_PATH = ROOT_DIR / "bounding_boxes.csv"
IMAGES_DIR = ROOT_DIR / "Images"
VIDEOS_DIR = ROOT_DIR / "Videos"

COMPARISON_ATTRS = [
    {
        "id": "Breed match",
        "name": "Breed match",
        "options": ["Same breed", "Different breed"],
        "desc": "Do Dog A and Dog B appear to be the same breed / dog type?"
    },
    {
        "id": "Coat colour",
        "name": "Coat colour",
        "options": ["Similar", "Different"],
        "desc": "Are the primary coat colors/shades closely matching or distinct?"
    },
    {
        "id": "Coat pattern",
        "name": "Coat pattern",
        "options": ["Similar", "Different"],
        "desc": "Do they share similar markings, patches, bi-color/tri-color distribution?"
    },
    {
        "id": "Body shape",
        "name": "Body shape & size",
        "options": ["Similar", "Different"],
        "desc": "Similar overall body proportions, build, height, tail carriage?"
    },
    {
        "id": "Facial appearance",
        "name": "Facial appearance",
        "options": ["Similar", "Different"],
        "desc": "Similar muzzle length, ear carriage (pricked/floppy), eye mask?"
    }
]

EXPECTED_COLUMNS = [
    "Pair Index", "Confusion Type", "Dog A (ID)", "Dog B (ID)", "Short A", "Short B",
    "Mutual Top-1 Errors", "Mutual Top-5 Errors", "A_to_B Top-1", "B_to_A Top-1",
    "Breed match", "Coat colour", "Coat pattern", "Body shape", "Facial appearance",
    "Key discriminating feature missed / Notes"
]


def load_bounding_boxes() -> Dict[tuple, tuple]:
    if not BBOX_PATH.exists():
        return {}
    df = pd.read_csv(BBOX_PATH)
    return {
        (str(r["DOG_ID"]), str(r["VIDEO_ID"])): (
            int(r["x_top_left"]), int(r["y_top_left"]), int(r["width"]), int(r["height"])
        )
        for _, r in df.iterrows()
    }


def get_image_base64(dog_id: str, video_id: str, bbox_map: dict, max_size: int = 400) -> str:
    img_path = IMAGES_DIR / dog_id / f"{dog_id}-{video_id}.jpg"
    if not img_path.exists():
        dog_dir = IMAGES_DIR / dog_id
        if not dog_dir.exists(): return ""
        files = sorted(list(dog_dir.glob("*.jpg")))
        if not files: return ""
        img_path = files[0]
        video_id = img_path.stem[37:]

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
        return ""


def find_top_confusion_pairs(n_total: int = 20) -> List[Dict[str, Any]]:
    """Compute Top 20 confusion pairs: 16 strictly mutual Top-1 + 4 top unidirectional/top-5 mutual."""
    if not DIST_MATRIX_CSV.exists():
        raise FileNotFoundError(f"Distance matrix missing at {DIST_MATRIX_CSV}")

    df = pd.read_csv(DIST_MATRIX_CSV)
    get_id = lambda x: str(x).rsplit('_', 1)[0]

    q_ids = df['queryId'].values
    g_ids = df.columns[1:].values
    q_labels = [get_id(x) for x in q_ids]
    g_labels = [get_id(x) for x in g_ids]
    dist_mat = df.iloc[:, 1:].values
    sort_idx = dist_mat.argsort(axis=1)

    pair_stats = defaultdict(lambda: {
        'A_to_B_top1': 0, 'B_to_A_top1': 0,
        'A_to_B_top5': 0, 'B_to_A_top5': 0
    })

    for i, q_lab in enumerate(q_labels):
        ranked_g_labels = [g_labels[idx] for idx in sort_idx[i]]
        top1_lab = ranked_g_labels[0]

        if top1_lab != q_lab:
            pair = tuple(sorted([q_lab, top1_lab]))
            if q_lab == pair[0]:
                pair_stats[pair]['A_to_B_top1'] += 1
            else:
                pair_stats[pair]['B_to_A_top1'] += 1

        for r in range(min(5, len(ranked_g_labels))):
            g_lab = ranked_g_labels[r]
            if g_lab != q_lab:
                pair = tuple(sorted([q_lab, g_lab]))
                if q_lab == pair[0]: pair_stats[pair]['A_to_B_top5'] += 1
                else: pair_stats[pair]['B_to_A_top5'] += 1

    # 1. Strictly Mutual Top-1 pairs
    mutual_pairs = []
    for (d_a, d_b), s in pair_stats.items():
        if s['A_to_B_top1'] > 0 and s['B_to_A_top1'] > 0:
            mutual_pairs.append({
                'dog_a': d_a,
                'dog_b': d_b,
                'confusion_type': 'Mutual (Bidirectional)',
                'short_a': d_a[:8] + '...',
                'short_b': d_b[:8] + '...',
                'tot_top1': s['A_to_B_top1'] + s['B_to_A_top1'],
                'tot_top5': s['A_to_B_top5'] + s['B_to_A_top5'],
                'a_to_b_top1': s['A_to_B_top1'],
                'b_to_a_top1': s['B_to_A_top1']
            })
    mutual_pairs.sort(key=lambda x: (x['tot_top1'], x['tot_top5']), reverse=True)

    # 2. Next highest error pairs up to n_total
    seen = set((m['dog_a'], m['dog_b']) for m in mutual_pairs)
    additional = []
    for (d_a, d_b), s in pair_stats.items():
        if (d_a, d_b) not in seen:
            tot_top1 = s['A_to_B_top1'] + s['B_to_A_top1']
            tot_top5 = s['A_to_B_top5'] + s['B_to_A_top5']
            if tot_top1 > 0:
                c_type = 'Bidirectional (Top-5)' if (s['A_to_B_top5'] > 0 and s['B_to_A_top5'] > 0) else 'Unidirectional'
                additional.append({
                    'dog_a': d_a,
                    'dog_b': d_b,
                    'confusion_type': c_type,
                    'short_a': d_a[:8] + '...',
                    'short_b': d_b[:8] + '...',
                    'tot_top1': tot_top1,
                    'tot_top5': tot_top5,
                    'a_to_b_top1': s['A_to_B_top1'],
                    'b_to_a_top1': s['B_to_A_top1']
                })
    additional.sort(key=lambda x: (x['tot_top1'], x['tot_top5']), reverse=True)

    top_pairs = mutual_pairs + additional[:max(0, n_total - len(mutual_pairs))]
    return top_pairs


def load_or_init_csv(top_pairs: List[Dict[str, Any]]) -> pd.DataFrame:
    existing_annotations = {}
    if CONF_CSV_PATH.exists():
        df_old = pd.read_csv(CONF_CSV_PATH)
        for _, r in df_old.iterrows():
            key = (str(r["Dog A (ID)"]), str(r["Dog B (ID)"]))
            existing_annotations[key] = {
                a["id"]: ("" if pd.isna(r.get(a["id"])) else str(r[a["id"]]))
                for a in COMPARISON_ATTRS
            }
            existing_annotations[key]["Notes"] = "" if pd.isna(r.get("Key discriminating feature missed / Notes")) else str(r["Key discriminating feature missed / Notes"])

    rows = []
    for idx, p in enumerate(top_pairs):
        key = (p["dog_a"], p["dog_b"])
        prev = existing_annotations.get(key, {})
        rows.append({
            "Pair Index": idx + 1,
            "Confusion Type": p["confusion_type"],
            "Dog A (ID)": p["dog_a"],
            "Dog B (ID)": p["dog_b"],
            "Short A": p["short_a"],
            "Short B": p["short_b"],
            "Mutual Top-1 Errors": p["tot_top1"],
            "Mutual Top-5 Errors": p["tot_top5"],
            "A_to_B Top-1": p["a_to_b_top1"],
            "B_to_A Top-1": p["b_to_a_top1"],
            "Breed match": prev.get("Breed match", ""),
            "Coat colour": prev.get("Coat colour", ""),
            "Coat pattern": prev.get("Coat pattern", ""),
            "Body shape": prev.get("Body shape", ""),
            "Facial appearance": prev.get("Facial appearance", ""),
            "Key discriminating feature missed / Notes": prev.get("Notes", "")
        })
    df = pd.DataFrame(rows)
    df.to_csv(CONF_CSV_PATH, index=False)
    return df


def prepare_pairs_payload() -> List[Dict[str, Any]]:
    top_pairs = find_top_confusion_pairs(n_total=20)
    df_annot = load_or_init_csv(top_pairs)
    bbox_map = load_bounding_boxes()
    splits_df = pd.read_csv(SPLITS_PATH)

    annot_dict = {}
    for _, r in df_annot.iterrows():
        key = (str(r["Dog A (ID)"]), str(r["Dog B (ID)"]))
        annot_dict[key] = {
            a["id"]: ("" if pd.isna(r[a["id"]]) else str(r[a["id"]]))
            for a in COMPARISON_ATTRS
        }
        annot_dict[key]["Notes"] = "" if pd.isna(r["Key discriminating feature missed / Notes"]) else str(r["Key discriminating feature missed / Notes"])

    payload = []
    print(f"Loading video data for {len(top_pairs)} top confusion pairs...")
    for idx, p in enumerate(top_pairs):
        d_a = p["dog_a"]
        d_b = p["dog_b"]

        sp_a = splits_df[splits_df["DOG_ID"] == d_a]
        clips_a = []
        for _, s in sp_a.iterrows():
            v_id = str(s["VIDEO_ID"])
            b64 = get_image_base64(d_a, v_id, bbox_map, max_size=400)
            has_vid = (VIDEOS_DIR / d_a / f"{d_a}-{v_id}.mp4").exists()
            clips_a.append({
                "video_id": v_id,
                "short_vid": v_id[:8] + "...",
                "split": str(s["SPLIT_CLOSED_SET"]),
                "image_b64": b64,
                "has_video": has_vid
            })

        sp_b = splits_df[splits_df["DOG_ID"] == d_b]
        clips_b = []
        for _, s in sp_b.iterrows():
            v_id = str(s["VIDEO_ID"])
            b64 = get_image_base64(d_b, v_id, bbox_map, max_size=400)
            has_vid = (VIDEOS_DIR / d_b / f"{d_b}-{v_id}.mp4").exists()
            clips_b.append({
                "video_id": v_id,
                "short_vid": v_id[:8] + "...",
                "split": str(s["SPLIT_CLOSED_SET"]),
                "image_b64": b64,
                "has_video": has_vid
            })

        key = (d_a, d_b)
        ann = annot_dict.get(key, {a["id"]: "" for a in COMPARISON_ATTRS})
        if "Notes" not in ann: ann["Notes"] = ""

        payload.append({
            "index": idx,
            "dog_a": d_a,
            "dog_b": d_b,
            "confusion_type": p["confusion_type"],
            "short_a": p["short_a"],
            "short_b": p["short_b"],
            "tot_top1": p["tot_top1"],
            "tot_top5": p["tot_top5"],
            "a_to_b_top1": p["a_to_b_top1"],
            "b_to_a_top1": p["b_to_a_top1"],
            "clips_a": clips_a,
            "clips_b": clips_b,
            "annotations": ann
        })

    return payload


def generate_html(pairs_data: List[Dict[str, Any]]) -> str:
    data_json = json.dumps(pairs_data)
    attrs_json = json.dumps(COMPARISON_ATTRS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DogReID Inter-Identity Similarity & Confusion Tool (Top 20)</title>
  <style>
    :root {{
      --bg-primary: #0a0f1d;
      --bg-surface: #1e293b;
      --bg-card: #2d3748;
      --border-color: #475569;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --accent-orange: #f97316;
      --accent-green: #22c55e;
      --accent-red: #ef4444;
      --accent-purple: #c084fc;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    body {{ background: var(--bg-primary); color: var(--text-main); padding: 14px 22px; }}
    header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 12px; margin-bottom: 14px; }}
    .btn {{ background: #0284c7; color: #fff; border: none; padding: 7px 14px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; cursor: pointer; }}
    .btn:hover {{ background: #0369a1; }}
    .btn-secondary {{ background: var(--bg-card); border: 1px solid var(--border-color); color: #fff; }}
    .btn-success {{ background: #16a34a; }}

    /* Top Pair Selector */
    .nav-bar {{ display: flex; gap: 6px; overflow-x: auto; padding-bottom: 10px; margin-bottom: 14px; }}
    .pair-pill {{ flex: 0 0 auto; padding: 6px 12px; border-radius: 8px; background: var(--bg-surface); border: 1px solid var(--border-color); cursor: pointer; font-size: 0.8rem; }}
    .pair-pill:hover {{ border-color: var(--accent-blue); }}
    .pair-pill.active {{ background: #0284c7; border-color: #38bdf8; color: #fff; font-weight: 700; }}
    .pair-pill.done {{
      border-color: #22c55e !important;
      background: rgba(34, 197, 94, 0.15);
    }}
    .pair-pill.done.active {{
      background: #15803d !important;
      border-color: #4ade80 !important;
    }}

    /* Side-by-side comparison grid */
    .compare-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 16px; }}
    .dog-box {{ background: var(--bg-surface); border: 2px solid var(--border-color); border-radius: 12px; padding: 16px; }}
    .dog-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; margin-bottom: 12px; }}
    .dog-box.dog-a {{ border-top: 4px solid var(--accent-blue); }}
    .dog-box.dog-a .dog-header h2 {{ color: var(--accent-blue); font-size: 1.15rem; }}
    .dog-box.dog-b {{ border-top: 4px solid var(--accent-orange); }}
    .dog-box.dog-b .dog-header h2 {{ color: var(--accent-orange); font-size: 1.15rem; }}

    /* Video Player Box */
    .video-player-container {{
      width: 100%;
      height: 330px;
      background: #000;
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--border-color);
      margin-bottom: 10px;
      position: relative;
    }}
    .video-player-container video {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #000;
    }}
    .video-player-container img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #000;
    }}

    /* Clip selection thumbnails under each video player */
    .clips-strip {{
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding: 4px 0 8px 0;
    }}
    .strip-thumb {{
      flex: 0 0 95px;
      height: 70px;
      border-radius: 6px;
      overflow: hidden;
      border: 2px solid var(--border-color);
      cursor: pointer;
      position: relative;
      background: #000;
      transition: all 0.15s;
    }}
    .strip-thumb:hover {{
      border-color: var(--accent-blue);
      transform: scale(1.05);
    }}
    .strip-thumb.active {{
      border-color: #38bdf8;
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.4);
    }}
    .strip-thumb img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
    }}
    .strip-thumb .thumb-tag {{
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(15, 23, 42, 0.85);
      font-size: 0.65rem;
      text-align: center;
      padding: 2px;
      font-family: monospace;
      color: #fff;
    }}

    /* Form Panel */
    .annotation-panel {{ background: var(--bg-surface); border: 1px solid var(--border-color); border-radius: 12px; padding: 18px; }}
    .form-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 14px; }}
    .attr-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; }}
    .attr-card.filled {{ border-left: 4px solid var(--accent-green); }}
    .attr-title {{ font-size: 0.85rem; font-weight: 700; display: flex; justify-content: space-between; margin-bottom: 2px; }}
    .attr-desc {{ font-size: 0.72rem; color: var(--text-muted); margin-bottom: 8px; line-height: 1.25; }}
    .btn-row {{ display: flex; gap: 8px; }}
    .btn-choice {{
      flex: 1;
      padding: 8px;
      border-radius: 6px;
      background: var(--bg-surface);
      color: #fff;
      border: 1.5px solid var(--border-color);
      font-size: 0.82rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .btn-choice:hover {{ border-color: var(--accent-blue); }}
    .btn-choice.sel-same-breed, .btn-choice.sel-similar {{
      background: #0284c7;
      border-color: #38bdf8;
      color: #fff;
    }}
    .btn-choice.sel-different-breed, .btn-choice.sel-different {{
      background: #b45309;
      border-color: #f59e0b;
      color: #fff;
    }}

    .notes-box {{ width: 100%; min-height: 60px; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 6px; color: #fff; padding: 8px 12px; font-size: 0.85rem; }}

    /* Summary Table */
    table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-top: 14px; }}
    th, td {{ padding: 7px 9px; text-align: left; border-bottom: 1px solid var(--border-color); }}
    th {{ background: var(--bg-card); }}
    tr:hover {{ background: rgba(255, 255, 255, 0.04); cursor: pointer; }}
  </style>
</head>
<body>

  <header>
    <div>
      <h1 style="font-size: 1.3rem;">🎬 Inter-Identity Similarity & Confusion Video Annotator (Top 20 Pairs)</h1>
      <p style="font-size: 0.82rem; color: var(--text-muted); margin-top: 2px;">
        Side-by-Side Synchronized Video Playback &bull; Auto-saves to <code>evaluation/confusion_pairs_annotations.csv</code>
      </p>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
      <span id="progress-badge" style="background:#0f766e; color:#ccfbf1; padding:5px 10px; border-radius:10px; font-size:0.82rem; font-weight:bold;">0 / 20 Done</span>
      <button class="btn btn-secondary" onclick="jumpNextUnannotated()">Next Unannotated [U]</button>
      <button class="btn btn-success" onclick="saveData(true)">💾 Save CSV [Ctrl+S]</button>
      <button class="btn btn-secondary" onclick="downloadCSV()">⬇ Export CSV</button>
    </div>
  </header>

  <!-- Navigation bar for all 20 pairs -->
  <div class="nav-bar" id="pair-nav-bar"></div>

  <!-- Comparison Header & Metrics -->
  <div style="background:var(--bg-surface); border:1px solid var(--border-color); border-radius:10px; padding:12px 16px; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center;">
    <div>
      <div style="display:flex; align-items:center; gap:8px;">
        <h2 id="pair-title" style="font-size:1.15rem; font-weight:700;">Pair #1: Dog A vs. Dog B</h2>
        <span id="pair-type-badge" style="padding:2px 8px; border-radius:5px; font-size:0.75rem; font-weight:bold; background:#1e3a8a; color:#bfdbfe;"></span>
      </div>
      <div id="pair-subtitle" style="font-size:0.75rem; color:var(--text-muted); font-family:monospace; margin-top:2px;"></div>
    </div>
    <div style="text-align:right; font-size:0.85rem;">
      <span style="color:var(--accent-red); font-weight:bold; font-size:0.95rem;" id="pair-errors">Top-1 Errors: 4 | Top-5 Errors: 8</span><br>
      <span style="color:var(--text-muted);" id="pair-breakdown">A &rarr; B: 3 Top-1 | B &rarr; A: 1 Top-1</span>
    </div>
  </div>

  <!-- Side-by-Side Videos -->
  <div class="compare-container">
    
    <!-- Dog A Video Player -->
    <div class="dog-box dog-a">
      <div class="dog-header">
        <h2>Dog A (<span id="name-a"></span>)</h2>
        <span style="font-size:0.8rem; color:var(--accent-blue); font-family:monospace;" id="active-clip-label-a"></span>
      </div>
      <div class="video-player-container" id="player-box-a">
        <!-- Replaced dynamically with <video> -->
      </div>
      <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:4px;">Available clips (<span id="count-a"></span>): Click thumbnail to switch video</div>
      <div class="clips-strip" id="strip-a"></div>
    </div>

    <!-- Dog B Video Player -->
    <div class="dog-box dog-b">
      <div class="dog-header">
        <h2>Dog B (<span id="name-b"></span>)</h2>
        <span style="font-size:0.8rem; color:var(--accent-orange); font-family:monospace;" id="active-clip-label-b"></span>
      </div>
      <div class="video-player-container" id="player-box-b">
        <!-- Replaced dynamically with <video> -->
      </div>
      <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:4px;">Available clips (<span id="count-b"></span>): Click thumbnail to switch video</div>
      <div class="clips-strip" id="strip-b"></div>
    </div>

  </div>

  <!-- Attribute Comparison Panel -->
  <div class="annotation-panel">
    <h3 style="font-size: 1.05rem; margin-bottom: 12px; display:flex; align-items:center; gap:8px;">
      <span>Morphological Comparison Attributes</span>
      <span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">(Click choices below to annotate)</span>
    </h3>
    <div class="form-grid" id="form-grid"></div>

    <div style="margin-top: 10px;">
      <div style="font-size: 0.85rem; font-weight: 700; margin-bottom: 4px;">Key Discriminating Feature Missed / Notes</div>
      <textarea id="notes-input" class="notes-box" placeholder="e.g. Same breed, identical coat color, but Dog A has a wider white chest patch and floppy ears..." oninput="onNotesChange(this.value)"></textarea>
    </div>

    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:14px;">
      <button class="btn btn-secondary" onclick="prevPair()">◀ Prev Pair [←]</button>
      <span id="save-status" style="color:var(--accent-green); font-size:0.85rem; font-weight:700;">✓ Ready</span>
      <button class="btn" onclick="nextPair()">Next Pair [→] ▶</button>
    </div>
  </div>

  <!-- Summary Table -->
  <div style="background:var(--bg-surface); border:1px solid var(--border-color); border-radius:12px; padding:14px; margin-top:18px;">
    <h3 style="font-size:0.95rem; margin-bottom:8px;">All 20 Confusion Pairs Overview</h3>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Type</th>
            <th>Dog A</th>
            <th>Dog B</th>
            <th>Top-1 Errors</th>
            <th>Top-5 Errors</th>
            <th>Breed Match</th>
            <th>Coat Colour</th>
            <th>Coat Pattern</th>
            <th>Body Shape</th>
            <th>Facial Appearance</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody id="table-tbody"></tbody>
      </table>
    </div>
  </div>

  <script>
    const pairs = {data_json};
    const attributes = {attrs_json};
    let currentPairIdx = 0;
    let activeClipIdxA = 0;
    let activeClipIdxB = 0;

    function isPairAnnotated(p) {{
      return attributes.every(a => p.annotations[a.id] && p.annotations[a.id].trim() !== "");
    }}

    function updateProgress() {{
      const done = pairs.filter(isPairAnnotated).length;
      document.getElementById("progress-badge").innerText = `${{done}} / ${{pairs.length}} Done`;
    }}

    function renderNavBar() {{
      const nav = document.getElementById("pair-nav-bar");
      nav.innerHTML = "";
      pairs.forEach((p, idx) => {{
        const pill = document.createElement("div");
        const done = isPairAnnotated(p);
        pill.className = "pair-pill" + (idx === currentPairIdx ? " active" : "") + (done ? " done" : "");
        pill.innerHTML = `#${{idx+1}} ${{p.short_a}} &harr; ${{p.short_b}} ${{done ? "✓" : ""}}`;
        pill.onclick = () => selectPair(idx);
        nav.appendChild(pill);
      }});
    }}

    function selectPair(idx) {{
      currentPairIdx = idx;
      activeClipIdxA = 0;
      activeClipIdxB = 0;
      renderAll();
    }}

    function renderAll() {{
      renderNavBar();
      renderCurrentPair();
      renderForm();
      renderTable();
      updateProgress();
    }}

    function renderCurrentPair() {{
      const p = pairs[currentPairIdx];
      document.getElementById("pair-title").innerText = `Pair #${{currentPairIdx+1}}: ${{p.short_a}} &harr; ${{p.short_b}}`;
      document.getElementById("pair-subtitle").innerText = `Dog A: ${{p.dog_a}} | Dog B: ${{p.dog_b}}`;
      document.getElementById("pair-type-badge").innerText = p.confusion_type;
      document.getElementById("pair-errors").innerText = `Top-1 Errors: ${{p.tot_top1}} | Top-5 Errors: ${{p.tot_top5}}`;
      document.getElementById("pair-breakdown").innerText = `A &rarr; B: ${{p.a_to_b_top1}} Top-1 | B &rarr; A: ${{p.b_to_a_top1}} Top-1`;

      document.getElementById("name-a").innerText = p.short_a;
      document.getElementById("name-b").innerText = p.short_b;
      document.getElementById("count-a").innerText = p.clips_a.length;
      document.getElementById("count-b").innerText = p.clips_b.length;

      renderPlayerA();
      renderPlayerB();

      document.getElementById("notes-input").value = p.annotations["Notes"] || "";
    }}

    function renderPlayerA() {{
      const p = pairs[currentPairIdx];
      const clip = p.clips_a[activeClipIdxA];
      const box = document.getElementById("player-box-a");
      const label = document.getElementById("active-clip-label-a");
      label.innerText = `${{clip.split.toUpperCase()}} #${{activeClipIdxA+1}} (${{clip.short_vid}})`;

      if (clip.has_video) {{
        const vidUrl = `/api/video?dog_id=${{encodeURIComponent(p.dog_a)}}&video_id=${{encodeURIComponent(clip.video_id)}}`;
        box.innerHTML = `
          <video controls autoplay loop muted playsinline style="width:100%; height:100%;">
            <source src="${{vidUrl}}" type="video/mp4">
          </video>
        `;
      }} else {{
        box.innerHTML = `<img src="data:image/jpeg;base64,${{clip.image_b64}}">`;
      }}

      // Strip
      const strip = document.getElementById("strip-a");
      strip.innerHTML = "";
      p.clips_a.forEach((c, i) => {{
        const thumb = document.createElement("div");
        thumb.className = "strip-thumb" + (i === activeClipIdxA ? " active" : "");
        thumb.onclick = () => {{ activeClipIdxA = i; renderPlayerA(); }};
        thumb.innerHTML = `
          <img src="data:image/jpeg;base64,${{c.image_b64}}">
          <div class="thumb-tag">${{c.split[0].toUpperCase()}}#${{i+1}}</div>
        `;
        strip.appendChild(thumb);
      }});
    }}

    function renderPlayerB() {{
      const p = pairs[currentPairIdx];
      const clip = p.clips_b[activeClipIdxB];
      const box = document.getElementById("player-box-b");
      const label = document.getElementById("active-clip-label-b");
      label.innerText = `${{clip.split.toUpperCase()}} #${{activeClipIdxB+1}} (${{clip.short_vid}})`;

      if (clip.has_video) {{
        const vidUrl = `/api/video?dog_id=${{encodeURIComponent(p.dog_b)}}&video_id=${{encodeURIComponent(clip.video_id)}}`;
        box.innerHTML = `
          <video controls autoplay loop muted playsinline style="width:100%; height:100%;">
            <source src="${{vidUrl}}" type="video/mp4">
          </video>
        `;
      }} else {{
        box.innerHTML = `<img src="data:image/jpeg;base64,${{clip.image_b64}}">`;
      }}

      // Strip
      const strip = document.getElementById("strip-b");
      strip.innerHTML = "";
      p.clips_b.forEach((c, i) => {{
        const thumb = document.createElement("div");
        thumb.className = "strip-thumb" + (i === activeClipIdxB ? " active" : "");
        thumb.onclick = () => {{ activeClipIdxB = i; renderPlayerB(); }};
        thumb.innerHTML = `
          <img src="data:image/jpeg;base64,${{c.image_b64}}">
          <div class="thumb-tag">${{c.split[0].toUpperCase()}}#${{i+1}}</div>
        `;
        strip.appendChild(thumb);
      }});
    }}

    function renderForm() {{
      const p = pairs[currentPairIdx];
      const grid = document.getElementById("form-grid");
      grid.innerHTML = "";
      attributes.forEach(attr => {{
        const val = p.annotations[attr.id] || "";
        const card = document.createElement("div");
        card.className = "attr-card" + (val ? " filled" : "");

        let btns = "";
        attr.options.forEach(opt => {{
          const isSel = val === opt;
          const optLower = opt.toLowerCase().replace(/\\s+/g, '-');
          const selClass = isSel ? ("sel-" + optLower) : "";
          btns += `<button class="btn-choice ${{selClass}}" onclick="setAttr('${{attr.id}}', '${{opt}}')">${{opt}}</button>`;
        }});

        card.innerHTML = `
          <div class="attr-title">
            <span>${{attr.name}}</span>
            ${{val ? '<span style="color:var(--accent-green)">✓ ' + val + '</span>' : ''}}
          </div>
          <div class="attr-desc">${{attr.desc}}</div>
          <div class="btn-row">${{btns}}</div>
        `;
        grid.appendChild(card);
      }});
    }}

    function setAttr(attrId, val) {{
      const p = pairs[currentPairIdx];
      p.annotations[attrId] = val;
      renderAll();
      saveData(false);
    }}

    function onNotesChange(val) {{
      pairs[currentPairIdx].annotations["Notes"] = val;
      saveData(false);
    }}

    function prevPair() {{
      if (currentPairIdx > 0) selectPair(currentPairIdx - 1);
    }}
    function nextPair() {{
      if (currentPairIdx < pairs.length - 1) selectPair(currentPairIdx + 1);
    }}
    function jumpNextUnannotated() {{
      for (let i = 1; i <= pairs.length; i++) {{
        const tIdx = (currentPairIdx + i) % pairs.length;
        if (!isPairAnnotated(pairs[tIdx])) {{
          selectPair(tIdx);
          return;
        }}
      }}
      alert("All 20 confusion pairs are fully annotated!");
    }}

    function renderTable() {{
      const tbody = document.getElementById("table-tbody");
      tbody.innerHTML = "";
      pairs.forEach((p, idx) => {{
        const tr = document.createElement("tr");
        if (idx === currentPairIdx) tr.style.background = "rgba(56, 189, 248, 0.18)";
        tr.onclick = () => selectPair(idx);

        const cell = (v) => {{
          if (v === "Similar" || v === "Same breed") return `<span style="color:var(--accent-green); font-weight:bold;">${{v}}</span>`;
          if (v === "Different" || v === "Different breed") return `<span style="color:var(--accent-orange); font-weight:bold;">${{v}}</span>`;
          return `<span style="color:var(--text-muted)">-</span>`;
        }};

        tr.innerHTML = `
          <td>${{idx+1}}</td>
          <td><span style="font-size:0.7rem; color:var(--text-muted);">${{p.confusion_type}}</span></td>
          <td><code>${{p.short_a}}</code></td>
          <td><code>${{p.short_b}}</code></td>
          <td style="color:var(--accent-red); font-weight:bold;">${{p.tot_top1}}</td>
          <td>${{p.tot_top5}}</td>
          <td>${{cell(p.annotations["Breed match"])}}</td>
          <td>${{cell(p.annotations["Coat colour"])}}</td>
          <td>${{cell(p.annotations["Coat pattern"])}}</td>
          <td>${{cell(p.annotations["Body shape"])}}</td>
          <td>${{cell(p.annotations["Facial appearance"])}}</td>
          <td style="max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${{p.annotations["Notes"] || ""}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function buildCSV() {{
      const headers = [
        "Pair Index", "Confusion Type", "Dog A (ID)", "Dog B (ID)", "Short A", "Short B",
        "Mutual Top-1 Errors", "Mutual Top-5 Errors", "A_to_B Top-1", "B_to_A Top-1",
        "Breed match", "Coat colour", "Coat pattern", "Body shape", "Facial appearance",
        "Key discriminating feature missed / Notes"
      ];
      let csv = headers.join(",") + "\\n";
      pairs.forEach((p, idx) => {{
        const row = [
          idx + 1,
          `"${{p.confusion_type}}"`,
          `"${{p.dog_a}}"`,
          `"${{p.dog_b}}"`,
          `"${{p.short_a}}"`,
          `"${{p.short_b}}"`,
          p.tot_top1,
          p.tot_top5,
          p.a_to_b_top1,
          p.b_to_a_top1,
          `"${{p.annotations["Breed match"] || ""}}"`,
          `"${{p.annotations["Coat colour"] || ""}}"`,
          `"${{p.annotations["Coat pattern"] || ""}}"`,
          `"${{p.annotations["Body shape"] || ""}}"`,
          `"${{p.annotations["Facial appearance"] || ""}}"`,
          `"${{(p.annotations["Notes"] || "").replace(/"/g, '""')}}"`
        ];
        csv += row.join(",") + "\\n";
      }});
      return csv;
    }}

    function saveData(showNotice = false) {{
      const csv = buildCSV();
      const statusLbl = document.getElementById("save-status");
      fetch("/api/save", {{
        method: "POST",
        headers: {{ "Content-Type": "text/csv" }},
        body: csv
      }})
      .then(res => res.json())
      .then(d => {{
        statusLbl.innerText = "✓ Saved to CSV";
        if (showNotice) alert("Successfully saved annotations to evaluation/confusion_pairs_annotations.csv!");
      }})
      .catch(e => {{
        statusLbl.innerText = "✗ Save failed!";
        statusLbl.style.color = "var(--accent-red)";
      }});
    }}

    function downloadCSV() {{
      const csv = buildCSV();
      const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "confusion_pairs_annotations.csv";
      link.click();
    }}

    window.addEventListener("keydown", (e) => {{
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === "ArrowLeft") prevPair();
      if (e.key === "ArrowRight") nextPair();
      if (e.key === "u" || e.key === "U") jumpNextUnannotated();
      if (e.key === "s" && (e.ctrlKey || e.metaKey)) {{
        e.preventDefault();
        saveData(true);
      }}
    }});

    // Init: start on first unannotated
    const firstUn = pairs.findIndex(p => !isPairAnnotated(p));
    currentPairIdx = firstUn >= 0 ? firstUn : 0;
    renderAll();
  </script>
</body>
</html>
"""


class ConfusionServerHandler(http.server.BaseHTTPRequestHandler):
    pairs_data = []

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.pairs_data = prepare_pairs_payload()
            html = generate_html(self.pairs_data)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.pairs_data).encode("utf-8"))
        elif self.path.startswith("/api/video"):
            self.handle_video_stream()
        elif self.path == "/api/export":
            if CONF_CSV_PATH.exists():
                with open(CONF_CSV_PATH, "rb") as f:
                    csv_data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", "attachment; filename=confusion_pairs_annotations.csv")
                self.end_headers()
                self.wfile.write(csv_data)
            else:
                self.send_error(404, "File not found")
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
                tmp_path = CONF_CSV_PATH.with_suffix(".tmp.csv")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(post_data)
                tmp_path.replace(CONF_CSV_PATH)
                print(f"-> [AUTO-SAVED] Successfully wrote updated annotations to {CONF_CSV_PATH.name}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "file": str(CONF_CSV_PATH)}).encode("utf-8"))
            except Exception as e:
                print(f"[ERROR] Failed to save CSV: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

    def log_message(self, format, *args):
        sys.stderr.write(f"[ConfusionServer] {args[0]} -> {args[1]}\n")


def run_server(port: int = 8055, open_browser: bool = False):
    pairs_data = prepare_pairs_payload()
    ConfusionServerHandler.pairs_data = pairs_data

    current_port = port
    while current_port < port + 50:
        try:
            with socketserver.TCPServer(("", current_port), ConfusionServerHandler) as httpd:
                url = f"http://localhost:{current_port}"
                print("\n" + "=" * 65)
                print(f"🎬 DogReID Inter-Identity Confusion Video Annotator Live!")
                print(f"   URL: {url}")
                print(f"   Target CSV: {CONF_CSV_PATH}")
                print(f"   Total Confusion Pairs: {len(pairs_data)}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DogReID Inter-Identity Confusion Annotator")
    parser.add_argument("--port", type=int, default=8055, help="Server port (default: 8055)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()
    run_server(port=args.port, open_browser=not args.no_browser)
