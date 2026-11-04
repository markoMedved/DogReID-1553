#!/usr/bin/env python3
"""
Interactive Annotation Tool for Best- vs. Worst-Performing Identities
====================================================================
Annotates the 20 selected dogs (10 best-performing with 100% mAP and 10 worst-performing)
across the 5 visual attributes defined in the study:
  1. Distinctive visual features (Yes / No)
  2. Occlusion (Yes / No)
  3. Blurriness / low resolution (Yes / No)
  4. Lighting (Good / Poor)
  5. Viewpoint variation (Low / High)

All annotations are saved to:
  evaluation/best_vs_worst_annotations.csv

Usage:
  # Launch local web server on port 8050:
  python evaluation/annotate_dogs.py --port 8050

  # Export standalone static HTML file (works offline anywhere):
  python evaluation/annotate_dogs.py --export-html evaluation/best_vs_worst_annotator.html
"""

import argparse
import base64
import http.server
import io
import json
import os
import socketserver
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from PIL import Image

# Setup root path
ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "evaluation" / "best_vs_worst_annotations.csv"
TEMPLATE_PATH = ROOT_DIR / "evaluation" / "best_vs_worst_annotation_template.csv"
SPLITS_PATH = ROOT_DIR / "splits.csv"
BBOX_PATH = ROOT_DIR / "bounding_boxes.csv"
IMAGES_DIR = ROOT_DIR / "Images"

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
        "desc": "Dog is partially obscured in one or more relevant videos."
    },
    {
        "id": "Blurriness / low resolution",
        "name": "Blurriness / low resolution",
        "options": ["Yes", "No"],
        "desc": "Dog's appearance is noticeably blurred or lacks sufficient detail."
    },
    {
        "id": "Lighting",
        "name": "Lighting",
        "options": ["Good", "Poor"],
        "desc": "Sufficient lighting versus underexposure, overexposure or difficult lighting."
    },
    {
        "id": "Viewpoint variation",
        "name": "Viewpoint variation",
        "options": ["Low", "High"],
        "desc": "Limited viewpoint changes versus substantial changes (front, side, rear)."
    }
]

EXPECTED_COLUMNS = [
    "Group", "Short ID", "mAP (%)", "Rank-1 (%)", "N_q", "N_g", "Full DOG_ID",
    "Distinctive visual features", "Occlusion", "Blurriness / low resolution",
    "Lighting", "Viewpoint variation"
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
    """Load or initialize annotations CSV dataframe."""
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        # Verify required columns exist
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[EXPECTED_COLUMNS]

    if TEMPLATE_PATH.exists():
        template = pd.read_csv(TEMPLATE_PATH)
        df = pd.DataFrame({
            "Group": template["Group"],
            "Short ID": template["Short ID"],
            "mAP (%)": template["mAP (%)"],
            "Rank-1 (%)": template["Rank-1 (%)"],
            "N_q": template["N_q"],
            "N_g": template["N_g"],
            "Full DOG_ID": template["DOG_ID"],
            "Distinctive visual features": "",
            "Occlusion": "",
            "Blurriness / low resolution": "",
            "Lighting": "",
            "Viewpoint variation": ""
        })
        save_annotations_df(df)
        return df

    raise FileNotFoundError(f"Neither {CSV_PATH} nor {TEMPLATE_PATH} exists.")


def save_annotations_df(df: pd.DataFrame) -> None:
    """Save dataframe atomically to CSV."""
    df = df[EXPECTED_COLUMNS]
    tmp_path = CSV_PATH.with_suffix(".tmp.csv")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(CSV_PATH)


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
    """Assemble all dog metadata, query/gallery video info, and base64 crops."""
    df_annot = load_annotations_df()
    splits_df = pd.read_csv(SPLITS_PATH)
    bbox_map = load_bounding_boxes()

    dogs_data = []
    print(f"Loading visual crops for {len(df_annot)} identities...")
    for idx, row in df_annot.iterrows():
        dog_id = str(row["Full DOG_ID"])
        sp = splits_df[splits_df["DOG_ID"] == dog_id]
        
        q_vids = sp[sp["SPLIT_CLOSED_SET"] == "query"]["VIDEO_ID"].tolist()
        g_vids = sp[sp["SPLIT_CLOSED_SET"] == "gallery"]["VIDEO_ID"].tolist()

        queries = []
        for v in q_vids:
            b64 = get_image_base64(dog_id, v, bbox_map)
            queries.append({"video_id": v, "short_vid": v[:8] + "...", "image_b64": b64})

        gallery = []
        for v in g_vids:
            b64 = get_image_base64(dog_id, v, bbox_map)
            gallery.append({"video_id": v, "short_vid": v[:8] + "...", "image_b64": b64})

        annotations = {
            attr["id"]: ("" if pd.isna(row[attr["id"]]) else str(row[attr["id"]]))
            for attr in ATTRIBUTE_DEFS
        }

        dogs_data.append({
            "index": idx,
            "dog_id": dog_id,
            "short_id": str(row["Short ID"]),
            "group": str(row["Group"]),
            "map_pct": float(row["mAP (%)"]),
            "rank1_pct": float(row["Rank-1 (%)"]),
            "n_q": int(row["N_q"]),
            "n_g": int(row["N_g"]),
            "queries": queries,
            "gallery": gallery,
            "annotations": annotations
        })

    print(f"Loaded visual data for {len(dogs_data)} identities.")
    return dogs_data


def generate_html(dogs_data: List[Dict[str, Any]], standalone: bool = False) -> str:
    """Generate the modern interactive HTML application."""
    data_json = json.dumps(dogs_data)
    attrs_json = json.dumps(ATTRIBUTE_DEFS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DogReID Attribute Annotator: Best vs. Worst Identities</title>
  <style>
    :root {{
      --bg-primary: #0f172a;
      --bg-surface: #1e293b;
      --bg-card: #334155;
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
      padding: 16px 24px;
      min-height: 100vh;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border-color);
      margin-bottom: 18px;
    }}
    .header-left h1 {{
      font-size: 1.4rem;
      font-weight: 700;
      color: #fff;
    }}
    .header-left p {{
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 4px;
    }}
    .header-stats {{
      display: flex;
      gap: 16px;
      align-items: center;
    }}
    .stat-badge {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 0.85rem;
    }}
    .btn {{
      background: #2563eb;
      color: #fff;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }}
    .btn:hover {{ background: #1d4ed8; }}
    .btn-secondary {{ background: var(--bg-card); color: #fff; }}
    .btn-secondary:hover {{ background: #475569; }}
    .btn-success {{ background: #16a34a; }}
    .btn-success:hover {{ background: #15803d; }}
    
    /* Navigation Bar */
    .nav-bar {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-bottom: 20px;
      background: var(--bg-surface);
      padding: 10px;
      border-radius: 10px;
      border: 1px solid var(--border-color);
    }}
    .nav-pill {{
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 600;
      background: var(--bg-card);
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      border: 1px solid transparent;
      transition: all 0.1s;
    }}
    .nav-pill:hover {{ color: #fff; border-color: var(--accent-blue); }}
    .nav-pill.active {{
      background: #0284c7;
      color: #fff;
      border-color: #38bdf8;
    }}
    .nav-pill.annotated {{
      border-left: 4px solid var(--accent-green);
    }}
    .nav-pill.best-group {{
      border-top: 2px solid #38bdf8;
    }}
    .nav-pill.worst-group {{
      border-top: 2px solid #f97316;
    }}

    /* Main Layout */
    .container {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 20px;
    }}
    @media (max-width: 1080px) {{
      .container {{ grid-template-columns: 1fr; }}
    }}

    /* Visual Panel */
    .panel {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 18px;
    }}
    .dog-info-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border-color);
    }}
    .dog-title {{
      font-size: 1.25rem;
      font-weight: 700;
    }}
    .dog-subtitle {{
      font-size: 0.8rem;
      color: var(--text-muted);
      font-family: monospace;
      margin-top: 2px;
    }}
    .group-tag {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 700;
    }}
    .tag-best {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #0284c7; }}
    .tag-worst {{ background: rgba(249, 115, 22, 0.2); color: #f97316; border: 1px solid #c2410c; }}

    .metrics-row {{
      display: flex;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .metric-box {{
      background: var(--bg-card);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 0.82rem;
    }}
    .metric-box strong {{
      display: block;
      font-size: 1.1rem;
      color: #fff;
    }}

    .section-title {{
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--accent-blue);
      margin: 14px 0 8px 0;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .section-title.gallery-title {{
      color: var(--accent-green);
    }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(135px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .image-card {{
      background: var(--bg-card);
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border-color);
      position: relative;
      cursor: zoom-in;
      transition: transform 0.15s;
    }}
    .image-card:hover {{
      transform: scale(1.02);
      border-color: var(--accent-blue);
    }}
    .image-card img {{
      width: 100%;
      height: 135px;
      object-fit: contain;
      background: #000;
      display: block;
    }}
    .image-card .caption {{
      font-size: 0.72rem;
      padding: 4px 6px;
      color: #cbd5e1;
      font-family: monospace;
      text-align: center;
      background: rgba(15, 23, 42, 0.85);
    }}

    /* Form Panel */
    .form-group {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 14px;
      transition: border-color 0.15s;
    }}
    .form-group.filled {{
      border-left: 4px solid var(--accent-green);
    }}
    .form-label {{
      font-size: 0.95rem;
      font-weight: 700;
      margin-bottom: 4px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .form-desc {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-bottom: 10px;
      line-height: 1.3;
    }}
    .options-row {{
      display: flex;
      gap: 10px;
    }}
    .option-btn {{
      flex: 1;
      padding: 10px 14px;
      border-radius: 8px;
      background: var(--bg-surface);
      color: var(--text-main);
      border: 2px solid var(--border-color);
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      text-align: center;
      transition: all 0.15s;
    }}
    .option-btn:hover {{
      border-color: #38bdf8;
    }}
    .option-btn.selected-yes, .option-btn.selected-good, .option-btn.selected-low {{
      background: #0369a1;
      border-color: #38bdf8;
      color: #fff;
    }}
    .option-btn.selected-no, .option-btn.selected-poor, .option-btn.selected-high {{
      background: #9a3412;
      border-color: #f97316;
      color: #fff;
    }}

    .bottom-nav {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 18px;
      padding-top: 14px;
      border-top: 1px solid var(--border-color);
    }}
    .save-indicator {{
      font-size: 0.85rem;
      color: var(--accent-green);
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 4px;
    }}

    /* Lightbox Modal */
    .modal {{
      display: none;
      position: fixed;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0, 0, 0, 0.85);
      z-index: 1000;
      justify-content: center;
      align-items: center;
      cursor: zoom-out;
    }}
    .modal img {{
      max-width: 90%;
      max-height: 90%;
      border-radius: 8px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.8);
    }}

    /* Live Summary Table */
    .summary-card {{
      margin-top: 24px;
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 16px;
    }}
    .summary-card h2 {{
      font-size: 1.05rem;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }}
    th, td {{
      padding: 7px 10px;
      text-align: left;
      border-bottom: 1px solid var(--border-color);
    }}
    th {{
      background: var(--bg-card);
      color: #cbd5e1;
      font-weight: 600;
    }}
    tr:hover {{ background: rgba(255, 255, 255, 0.03); }}
    .cell-yes {{ color: var(--accent-green); font-weight: 600; }}
    .cell-no {{ color: var(--accent-orange); font-weight: 600; }}
  </style>
</head>
<body>

  <header>
    <div class="header-left">
      <h1>DogReID Attribute Annotator: Best vs. Worst Identities</h1>
      <p>Annotate 10 best-performing (100% mAP) and 10 worst-performing identities across 5 recording and visual traits.</p>
    </div>
    <div class="header-stats">
      <div class="stat-badge" id="progress-badge">Annotated: 0 / 20 (0%)</div>
      <button class="btn btn-secondary" onclick="jumpToNextUnannotated()">Next Unannotated</button>
      <button class="btn btn-success" onclick="saveData(true)">💾 Save to CSV</button>
      <button class="btn btn-secondary" onclick="downloadCSV()">⬇ Export CSV</button>
    </div>
  </header>

  <!-- Navigation Bar -->
  <div class="nav-bar" id="nav-bar"></div>

  <!-- Main Container -->
  <div class="container">
    <!-- Left: Dog Info & Crops -->
    <div class="panel">
      <div class="dog-info-header">
        <div>
          <div class="dog-title" id="dog-title">Identity #1</div>
          <div class="dog-subtitle" id="dog-subtitle">Full UUID</div>
        </div>
        <div id="dog-group-tag" class="group-tag tag-best">Best 10</div>
      </div>

      <div class="metrics-row">
        <div class="metric-box"><span>mAP:</span> <strong id="metric-map">100.0%</strong></div>
        <div class="metric-box"><span>Rank-1:</span> <strong id="metric-rank1">100.0%</strong></div>
        <div class="metric-box"><span>N_query:</span> <strong id="metric-nq">1</strong></div>
        <div class="metric-box"><span>N_gallery:</span> <strong id="metric-ng">4</strong></div>
      </div>

      <div class="section-title">🔍 Query Videos (<span id="query-count">0</span> clips)</div>
      <div class="image-grid" id="query-grid"></div>

      <div class="section-title gallery-title">🏛 Gallery Videos (<span id="gallery-count">0</span> clips)</div>
      <div class="image-grid" id="gallery-grid"></div>
    </div>

    <!-- Right: Annotation Form -->
    <div class="panel">
      <h2 style="font-size: 1.15rem; font-weight: 700; margin-bottom: 14px;">Attribute Annotations</h2>
      <div id="form-container"></div>

      <div class="bottom-nav">
        <button class="btn btn-secondary" onclick="prevDog()">⬅ Previous [←]</button>
        <div class="save-indicator" id="save-indicator">Saved to CSV ✓</div>
        <button class="btn" onclick="nextDog()">Next [→] ➡</button>
      </div>
    </div>
  </div>

  <!-- Summary Table -->
  <div class="summary-card">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
      <h2>Dataset Annotation Summary (20 Target Dogs)</h2>
      <span style="font-size:0.8rem; color:var(--text-muted)">Outputs written directly to: <code>evaluation/best_vs_worst_annotations.csv</code></span>
    </div>
    <div style="overflow-x: auto;">
      <table id="summary-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Group</th>
            <th>Short ID</th>
            <th>mAP (%)</th>
            <th>Rank-1 (%)</th>
            <th>N_q</th>
            <th>N_g</th>
            <th>Distinctive features</th>
            <th>Occlusion</th>
            <th>Blurriness</th>
            <th>Lighting</th>
            <th>Viewpoint var.</th>
          </tr>
        </thead>
        <tbody id="summary-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- Lightbox Modal -->
  <div class="modal" id="modal" onclick="closeModal()">
    <img id="modal-img" src="" alt="Full Crop">
  </div>

  <script>
    const dogs = {data_json};
    const attributes = {attrs_json};
    const isStandalone = {str(standalone).lower()};
    let currentIndex = 0;

    function renderNav() {{
      const nav = document.getElementById("nav-bar");
      nav.innerHTML = "";
      dogs.forEach((d, i) => {{
        const pill = document.createElement("div");
        const isAnnotated = isDogFullyAnnotated(d);
        pill.className = "nav-pill" + 
          (i === currentIndex ? " active" : "") +
          (isAnnotated ? " annotated" : "") +
          (d.group.includes("Best") ? " best-group" : " worst-group");
        
        pill.innerHTML = `<span>#${{i + 1}} ${{d.short_id}}</span>${{isAnnotated ? " ✓" : ""}}`;
        pill.onclick = () => selectDog(i);
        nav.appendChild(pill);
      }});
      updateProgressBadge();
    }}

    function isDogFullyAnnotated(dog) {{
      return attributes.every(attr => dog.annotations[attr.id] && dog.annotations[attr.id].trim() !== "");
    }}

    function updateProgressBadge() {{
      const count = dogs.filter(isDogFullyAnnotated).length;
      const pct = Math.round((count / dogs.length) * 100);
      document.getElementById("progress-badge").innerText = `Annotated: ${{count}} / ${{dogs.length}} (${{pct}}%)`;
    }}

    function selectDog(index) {{
      currentIndex = index;
      renderNav();
      renderCurrentDog();
      renderSummaryTable();
    }}

    function prevDog() {{
      if (currentIndex > 0) selectDog(currentIndex - 1);
    }}

    function nextDog() {{
      if (currentIndex < dogs.length - 1) selectDog(currentIndex + 1);
    }}

    function jumpToNextUnannotated() {{
      for (let i = 0; i < dogs.length; i++) {{
        const idx = (currentIndex + 1 + i) % dogs.length;
        if (!isDogFullyAnnotated(dogs[idx])) {{
          selectDog(idx);
          return;
        }}
      }}
      alert("All 20 dogs are fully annotated!");
    }}

    function renderCurrentDog() {{
      const d = dogs[currentIndex];
      document.getElementById("dog-title").innerText = `#${{currentIndex + 1}}: ${{d.short_id}} (${{d.group}})`;
      document.getElementById("dog-subtitle").innerText = d.dog_id;
      
      const tag = document.getElementById("dog-group-tag");
      tag.innerText = d.group;
      tag.className = "group-tag " + (d.group.includes("Best") ? "tag-best" : "tag-worst");

      document.getElementById("metric-map").innerText = d.map_pct.toFixed(1) + "%";
      document.getElementById("metric-rank1").innerText = d.rank1_pct.toFixed(1) + "%";
      document.getElementById("metric-nq").innerText = d.n_q;
      document.getElementById("metric-ng").innerText = d.n_g;

      // Query Grid
      const qGrid = document.getElementById("query-grid");
      document.getElementById("query-count").innerText = d.queries.length;
      qGrid.innerHTML = "";
      d.queries.forEach((q, idx) => {{
        const card = document.createElement("div");
        card.className = "image-card";
        card.onclick = () => openModal("data:image/jpeg;base64," + q.image_b64);
        card.innerHTML = `
          <img src="data:image/jpeg;base64,${{q.image_b64}}" alt="Query ${{idx+1}}">
          <div class="caption">Query #${{idx+1}} (${{q.short_vid}})</div>
        `;
        qGrid.appendChild(card);
      }});

      // Gallery Grid
      const gGrid = document.getElementById("gallery-grid");
      document.getElementById("gallery-count").innerText = d.gallery.length;
      gGrid.innerHTML = "";
      d.gallery.forEach((g, idx) => {{
        const card = document.createElement("div");
        card.className = "image-card";
        card.onclick = () => openModal("data:image/jpeg;base64," + g.image_b64);
        card.innerHTML = `
          <img src="data:image/jpeg;base64,${{g.image_b64}}" alt="Gallery ${{idx+1}}">
          <div class="caption">Gallery #${{idx+1}} (${{g.short_vid}})</div>
        `;
        gGrid.appendChild(card);
      }});

      // Form
      const form = document.getElementById("form-container");
      form.innerHTML = "";
      attributes.forEach(attr => {{
        const val = d.annotations[attr.id] || "";
        const grp = document.createElement("div");
        grp.className = "form-group" + (val ? " filled" : "");
        
        let optionsHtml = "";
        attr.options.forEach(opt => {{
          const isSelected = val === opt;
          const optClass = isSelected ? ("selected-" + opt.toLowerCase()) : "";
          optionsHtml += `
            <button class="option-btn ${{optClass}}" onclick="setAnnotation('${{attr.id}}', '${{opt}}')">
              ${{opt}}
            </button>
          `;
        }});

        grp.innerHTML = `
          <div class="form-label">
            <span>${{attr.name}}</span>
            ${{val ? '<span style="font-size:0.75rem; color:var(--accent-green)">✓ ' + val + '</span>' : ''}}
          </div>
          <div class="form-desc">${{attr.desc}}</div>
          <div class="options-row">${{optionsHtml}}</div>
        `;
        form.appendChild(grp);
      }});
    }}

    function setAnnotation(attrId, value) {{
      const d = dogs[currentIndex];
      // Toggle or set
      d.annotations[attrId] = value;
      renderCurrentDog();
      renderNav();
      renderSummaryTable();
      saveData(false);
    }}

    function openModal(src) {{
      document.getElementById("modal-img").src = src;
      document.getElementById("modal").style.display = "flex";
    }}

    function closeModal() {{
      document.getElementById("modal").style.display = "none";
    }}

    function renderSummaryTable() {{
      const tbody = document.getElementById("summary-tbody");
      tbody.innerHTML = "";
      dogs.forEach((d, i) => {{
        const tr = document.createElement("tr");
        if (i === currentIndex) tr.style.background = "rgba(56, 189, 248, 0.15)";
        
        const cell = (v) => {{
          if (v === "Yes" || v === "Good" || v === "Low") return `<span class="cell-yes">${{v}}</span>`;
          if (v === "No" || v === "Poor" || v === "High") return `<span class="cell-no">${{v}}</span>`;
          return `<span style="color:var(--text-muted)">-</span>`;
        }};

        tr.innerHTML = `
          <td>${{i + 1}}</td>
          <td><span class="${{d.group.includes("Best") ? "tag-best" : "tag-worst"}}" style="padding:2px 6px; border-radius:4px; font-size:0.75rem;">${{d.group.includes("Best") ? "Best" : "Worst"}}</span></td>
          <td><code>${{d.short_id}}</code></td>
          <td>${{d.map_pct.toFixed(1)}}</td>
          <td>${{d.rank1_pct.toFixed(1)}}</td>
          <td>${{d.n_q}}</td>
          <td>${{d.n_g}}</td>
          <td>${{cell(d.annotations["Distinctive visual features"])}}</td>
          <td>${{cell(d.annotations["Occlusion"])}}</td>
          <td>${{cell(d.annotations["Blurriness / low resolution"])}}</td>
          <td>${{cell(d.annotations["Lighting"])}}</td>
          <td>${{cell(d.annotations["Viewpoint variation"])}}</td>
        `;
        tr.onclick = () => selectDog(i);
        tr.style.cursor = "pointer";
        tbody.appendChild(tr);
      }});
    }}

    function buildCSV() {{
      const headers = ["Group", "Short ID", "mAP (%)", "Rank-1 (%)", "N_q", "N_g", "Full DOG_ID",
                       "Distinctive visual features", "Occlusion", "Blurriness / low resolution",
                       "Lighting", "Viewpoint variation"];
      let csv = headers.join(",") + "\\n";
      dogs.forEach(d => {{
        const row = [
          `"${{d.group}}"`,
          `"${{d.short_id}}"`,
          d.map_pct.toFixed(1),
          d.rank1_pct.toFixed(1),
          d.n_q,
          d.n_g,
          `"${{d.dog_id}}"`,
          `"${{d.annotations["Distinctive visual features"] || ""}}"`,
          `"${{d.annotations["Occlusion"] || ""}}"`,
          `"${{d.annotations["Blurriness / low resolution"] || ""}}"`,
          `"${{d.annotations["Lighting"] || ""}}"`,
          `"${{d.annotations["Viewpoint variation"] || ""}}"`
        ];
        csv += row.join(",") + "\\n";
      }});
      return csv;
    }}

    function saveData(showNotice = false) {{
      const csvContent = buildCSV();
      const indicator = document.getElementById("save-indicator");

      if (isStandalone) {{
        // Running as static offline HTML: persist to localStorage
        localStorage.setItem("dogreid_annotations_csv", csvContent);
        if (showNotice) alert("Annotations saved to browser storage! Click 'Export CSV' to download the CSV file.");
        indicator.innerText = "Saved to LocalStorage ✓";
        setTimeout(() => {{ indicator.innerText = "Saved ✓"; }}, 2000);
        return;
      }}

      // Running against python HTTP server
      fetch("/api/save", {{
        method: "POST",
        headers: {{ "Content-Type": "text/csv" }},
        body: csvContent
      }})
      .then(res => res.json())
      .then(data => {{
        indicator.innerText = "Saved to CSV ✓";
        if (showNotice) alert("Successfully saved annotations to evaluation/best_vs_worst_annotations.csv!");
      }})
      .catch(err => {{
        console.error("Save error:", err);
        indicator.innerText = "Save failed!";
        indicator.style.color = "var(--accent-red)";
      }});
    }}

    function downloadCSV() {{
      const csv = buildCSV();
      const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", "best_vs_worst_annotations.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }}

    // Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {{
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === "ArrowLeft" || e.key === "p") prevDog();
      if (e.key === "ArrowRight" || e.key === "n") nextDog();
      if (e.key === "u") jumpToNextUnannotated();
      if (e.key === "s" && (e.ctrlKey || e.metaKey)) {{
        e.preventDefault();
        saveData(true);
      }}
      if (e.key === "Escape") closeModal();
    }});

    // Initialize
    renderNav();
    selectDog(0);
  </script>
</body>
</html>
"""


class AnnotationServerHandler(http.server.BaseHTTPRequestHandler):
    """Custom HTTP handler serving the annotation GUI and REST endpoints."""

    dogs_data = []

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
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
            if CSV_PATH.exists():
                with open(CSV_PATH, "rb") as f:
                    csv_data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", "attachment; filename=best_vs_worst_annotations.csv")
                self.end_headers()
                self.wfile.write(csv_data)
            else:
                self.send_error(404, "File not found")
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/save":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8")
            try:
                # Write directly to best_vs_worst_annotations.csv
                tmp_path = CSV_PATH.with_suffix(".tmp.csv")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(post_data)
                tmp_path.replace(CSV_PATH)
                print(f"-> [AUTO-SAVED] Successfully wrote updated annotations to {CSV_PATH.name}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "file": str(CSV_PATH)}).encode("utf-8"))
            except Exception as e:
                print(f"[ERROR] Failed to save CSV: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[Server] {args[0]} -> {args[1]}\n")


def run_server(port: int = 8050, open_browser: bool = False):
    """Run lightweight HTTP server with auto port finding."""
    dogs_data = prepare_dataset_payload()
    AnnotationServerHandler.dogs_data = dogs_data

    # Find free port
    current_port = port
    while current_port < port + 50:
        try:
            with socketserver.TCPServer(("", current_port), AnnotationServerHandler) as httpd:
                url = f"http://localhost:{current_port}"
                print("\n" + "=" * 65)
                print(f"🐾 DogReID Attribute Annotation Tool Live!")
                print(f"   URL: {url}")
                print(f"   Target CSV: {CSV_PATH}")
                print(f"   Press Ctrl+C to stop server")
                print("=" * 65 + "\n")
                if open_browser:
                    webbrowser.open(url)
                httpd.serve_forever()
                break
        except OSError as e:
            if e.errno == 98 or "address already in use" in str(e).lower():
                current_port += 1
            else:
                raise e


def export_standalone_html(output_html_path: Path):
    """Generate and write standalone HTML file."""
    dogs_data = prepare_dataset_payload()
    html_content = generate_html(dogs_data, standalone=True)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"-> [SUCCESS] Exported standalone offline annotator to: {output_html_path}")
    print(f"   File size: {output_html_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"   You can open this file in any web browser directly (no server needed).")


def main():
    parser = argparse.ArgumentParser(description="DogReID Best vs. Worst Attribute Annotation Tool")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the local server on (default: 8050)")
    parser.add_argument("--open", action="store_true", help="Automatically open browser window")
    parser.add_argument("--export-html", type=str, default="", help="Path to export a self-contained offline HTML annotator")
    args = parser.parse_args()

    if args.export_html:
        export_standalone_html(Path(args.export_html))
    else:
        run_server(port=args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
