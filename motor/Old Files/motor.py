# ============================================================
# Rexair Motor & Magnet Analysis – FINAL SCRIPT v8
# Fix: Magnet PDFs may be cover-only, data-only, or cover+data.
# We ONLY analyze the page that contains the "3.7±0.5°" timing table.
# ============================================================

import os
import re
import platform
import subprocess
from datetime import datetime
import threading
import queue
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
import tkinter as tk
from tkinter import ttk
import fitz

try:
    from PIL import Image, ImageEnhance, ImageOps
    import pytesseract
    _HAS_TESSERACT = True
except:
    _HAS_TESSERACT = False

REPORT_FOLDER_MOTOR = "Motor Reports"
REPORT_FOLDER_MAGNET = "Magnet Reports"

MAGNET_TARGET = 3.7
MAGNET_TOL = 0.5
LSL = MAGNET_TARGET - MAGNET_TOL
USL = MAGNET_TARGET + MAGNET_TOL

os.makedirs(REPORT_FOLDER_MOTOR, exist_ok=True)
os.makedirs(REPORT_FOLDER_MAGNET, exist_ok=True)

print("RUNNING FINAL SCRIPT v8")
print(f"[INFO] Working directory: {os.getcwd()}")

# ============================================================
# UTILITIES
# ============================================================

def open_file(filepath):
    try:
        if platform.system() == "Windows":
            os.startfile(filepath)
        elif platform.system() == "Darwin":
            subprocess.run(["open", filepath])
        else:
            subprocess.run(["xdg-open", filepath])
    except:
        pass

def versioned_filename(folder, base_name):
    path = os.path.join(folder, base_name)
    if not os.path.exists(path):
        return path
    name, ext = os.path.splitext(base_name)
    i = 1
    while True:
        new_name = f"{name}_v{i}{ext}"
        new_path = os.path.join(folder, new_name)
        if not os.path.exists(new_path):
            return new_path
        i += 1

def safe_numeric_array(values):
    s = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s.to_numpy(dtype=float)

def safe_mean(values):
    arr = safe_numeric_array(values)
    return float(arr.mean()) if arr.size else 0.0

def safe_median(values):
    arr = safe_numeric_array(values)
    return float(np.median(arr)) if arr.size else 0.0

def safe_stdev(values):
    arr = safe_numeric_array(values)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))

def compute_cp_cpk(mean, std):
    if std <= 0:
        return 0.0, 0.0
    cp = (USL - LSL) / (6.0 * std)
    cpk = min((mean - LSL) / (3.0 * std), (USL - mean) / (3.0 * std))
    return cp, cpk

# ============================================================
# MOTOR SECTION (UNCHANGED – WORKING)
# ============================================================

def extract_date_from_filename(fname):
    m = re.search(r"C6521(\d{6})", fname)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%y%m%d")
    except:
        return None

def find_input_power_column(df):
    for i in range(min(15, len(df))):
        row = df.iloc[i].astype(str).tolist()
        for j, cell in enumerate(row):
            cl = cell.lower()
            if "high speed(open)" in cl and "input power" in cl:
                return j
    for i in range(min(15, len(df))):
        row = df.iloc[i].astype(str).tolist()
        for j, cell in enumerate(row):
            if "input power" in cell.lower():
                return j
    return None

def read_input_power(filepath):
    try:
        df = pd.read_excel(filepath, header=None, engine=None)
    except Exception as e:
        print(f"[MOTOR] ERROR reading {os.path.basename(filepath)} -> {e}")
        return []

    col_idx = find_input_power_column(df)
    if col_idx is None:
        print(f"[MOTOR] SKIP (no Input Power column): {os.path.basename(filepath)}")
        return []

    values = []
    for v in df.iloc[:, col_idx].tolist():
        try:
            fv = float(v)
            if np.isfinite(fv):
                values.append(fv)
        except:
            continue

    if not values:
        print(f"[MOTOR] SKIP (0 numeric values): {os.path.basename(filepath)}")
        return []

    print(f"[MOTOR] OK: {os.path.basename(filepath)} -> {len(values)} values")
    return values

def create_motor_pdf(output_path, months, month_values):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rexair Motor Test Trend Analysis", ln=True)

    month_labels = [m.strftime("%b %Y") for m in months]

    plt.figure(figsize=(8.5, 4.2))
    plt.boxplot(month_values, tick_labels=month_labels, showfliers=False)
    plt.title("Input Power (W): Monthly Distribution")
    plt.ylabel("Watts")
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_box = "motor_boxplot_temp.png"
    plt.savefig(img_box, dpi=300)
    plt.close()
    pdf.image(img_box, x=10, y=30, w=190)

    monthly_stds = [safe_stdev(v) for v in month_values]
    plt.figure(figsize=(8.5, 3.0))
    plt.plot(month_labels, monthly_stds, marker="o")
    plt.title("Standard Deviation Over Time")
    plt.ylabel("Std Dev (W)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_std = "motor_std_temp.png"
    plt.savefig(img_std, dpi=300)
    plt.close()
    pdf.image(img_std, x=10, y=125, w=190)

    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Monthly Summary", ln=True)
    pdf.set_font("Arial", "", 11)

    for m, vals in zip(months, month_values):
        pdf.cell(
            0, 8,
            f"{m.strftime('%Y-%m')}: Mean={safe_mean(vals):.2f} W, "
            f"Median={safe_median(vals):.2f} W, StdDev={safe_stdev(vals):.4f}",
            ln=True
        )

    pdf.output(output_path)

    for tmp in (img_box, img_std):
        if os.path.exists(tmp):
            os.remove(tmp)

def run_motor_analysis(q=None, cancel_flag=None):
    base = os.getcwd()

    files = [
        f for f in os.listdir(base)
        if f.lower().endswith((".xlsx", ".xls"))
        and "motor test data" in f.lower()
    ]

    dated = []
    for f in files:
        d = extract_date_from_filename(f)
        if d:
            dated.append((d, f))

    dated.sort(key=lambda x: x[0])

    monthly_data = {}

    for d, f in dated:
        vals = read_input_power(os.path.join(base, f))
        if vals:
            monthly_data.setdefault(d.strftime("%Y-%m"), []).extend(vals)

    if not monthly_data:
        print("❌ No motor Input Power data found.")
        return

    print("[MOTOR] Months detected:", sorted(monthly_data.keys()))

    all_months = sorted(datetime.strptime(m, "%Y-%m") for m in monthly_data.keys())
    recent_months = all_months[-12:]
    month_values = [monthly_data[m.strftime("%Y-%m")] for m in recent_months]

    filename = f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}.pdf"
    outpath = versioned_filename(REPORT_FOLDER_MOTOR, filename)

    create_motor_pdf(outpath, recent_months, month_values)
    print(f"[INFO] ✅ Motor report created: {outpath}")
    open_file(outpath)

# ============================================================
# MAGNET SECTION (NEW: DETECT DATA PAGE THEN OCR TABLE)
# ============================================================

def create_magnet_pdf(output_path, rows):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Report", ln=True)

    labels = [r["label"] for r in rows]
    cpks = [r["cpk"] for r in rows]

    plt.figure(figsize=(8.5, 4))
    plt.plot(labels, cpks, marker="o")
    plt.axhline(1.33, linestyle="--")
    plt.ylabel("CpK")
    plt.title("Magnet Timing CpK Over Time")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    img = "magnet_cpk.png"
    plt.savefig(img, dpi=300)
    plt.close()

    pdf.image(img, x=10, y=30, w=190)

    pdf.add_page()
    pdf.set_font("Arial", "", 11)
    for r in rows:
        pdf.cell(
            0, 8,
            f"{r['label']} Mean={r['mean']:.4f} Std={r['std']:.4f} Cp={r['cp']:.2f} CpK={r['cpk']:.2f}",
            ln=True
        )

    pdf.output(output_path)

    if os.path.exists(img):
        os.remove(img)

    print(f"[MAGNET] REPORT CREATED: {output_path}")
    open_file(output_path)

def extract_month_from_pdf_filename(fname, full_path):
    m = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", fname)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).replace(day=1)
    return datetime.fromtimestamp(os.path.getmtime(full_path)).replace(day=1)

def _normalize_ocr_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("°", " ")
    t = t.replace(",", ".").replace("·", ".").replace(":", ".")
    t = t.replace('O', '0').replace('o', '0')
    t = t.replace('S', '5')
    t = t.replace('I', '1').replace('l', '1')
    # collapse whitespace BETWEEN digits: "3 7 2" -> "372"
    t = re.sub(r'(?<=\d)\s+(?=\d)', '', t)
    return t

def _extract_3p7_column_values(ocr_text: str, max_count=40):
    """
    Pull values that look like 3.xx / 4.xx (timing values).
    """
    t = _normalize_ocr_text(ocr_text)
    vals = []

    # 3.72 style
    for m in re.finditer(r"\b([34]\.[0-9]{1,3})\b", t):
        try:
            v = float(m.group(1))
            if 3.0 <= v <= 4.5:
                vals.append(v)
        except:
            pass

    # 372 style -> 3.72
    for m in re.finditer(r"\b([34][0-9]{2})\b", t):
        try:
            v = float(m.group(1)) / 100.0
            if 3.0 <= v <= 4.5:
                vals.append(v)
        except:
            pass

    # de-dupe
    out = []
    seen = set()
    for v in vals:
        k = round(v, 3)
        if k not in seen:
            seen.add(k)
            out.append(v)
        if len(out) >= max_count:
            break
    return out

def _render_page_image(doc, page_index, dpi=300):
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples).convert('L')
    return img

def _rotate_upright(img_gray):
    """
    Many of your pages are rotated sideways. Try 0/90/180/270 and choose
    the orientation that produces the most keyword hits for the data page.
    """
    candidates = [
        ("r0", img_gray),
        ("r90", img_gray.rotate(90, expand=True)),
        ("r180", img_gray.rotate(180, expand=True)),
        ("r270", img_gray.rotate(270, expand=True)),
    ]
    best = candidates[0]
    best_score = -1

    for name, im in candidates:
        test = ImageOps.autocontrast(im)
        test = ImageEnhance.Contrast(test).enhance(2.0)
        # quick OCR, low cost
        txt = pytesseract.image_to_string(test, config="--psm 6") if _HAS_TESSERACT else ""
        nt = _normalize_ocr_text(txt)
        score = 0
        if "角度数据" in nt:
            score += 5
        if "3.7" in nt:
            score += 2
        if "0.5" in nt or "±0.5" in nt:
            score += 2
        if score > best_score:
            best_score = score
            best = (name, im)

    return best[0], best[1], best_score

def find_data_page_index(pdf_path):
    """
    Return (page_index, upright_rotation_name) for the page containing the timing data table.
    If no such page exists -> return (None, None)
    """
    if not _HAS_TESSERACT:
        return None, None

    doc = fitz.open(pdf_path)
    try:
        for i in range(doc.page_count):
            img = _render_page_image(doc, i, dpi=220)
            rot_name, upright, score = _rotate_upright(img)
            # Strong keyword presence -> data page
            if score >= 4:
                print(f"[MAGNET DEBUG] {os.path.basename(pdf_path)} data_page_detect p{i+1} {rot_name} score={score}")
                return i, rot_name
        return None, None
    finally:
        doc.close()

def extract_timing_values_from_data_page(pdf_path, page_index):
    """
    Once we know the data page, crop the table and OCR the rightmost timing column (3.7±0.5).
    """
    if not _HAS_TESSERACT:
        return []

    doc = fitz.open(pdf_path)
    try:
        img = _render_page_image(doc, page_index, dpi=450)

        # Rotate upright (best effort)
        rot_name, upright, _ = _rotate_upright(img)

        # Crop to the table region (works well for your provided samples)
        w, h = upright.size

        # The data page is basically the table; crop margins slightly
        table = upright.crop((int(w*0.08), int(h*0.05), int(w*0.95), int(h*0.97)))

        # Enhance for OCR
        table = ImageOps.autocontrast(table)
        table = ImageEnhance.Contrast(table).enhance(2.6)

        # OCR entire table then extract 3.7 column values
        txt = pytesseract.image_to_string(table, config="--oem 1 --psm 6")
        vals = _extract_3p7_column_values(txt, max_count=60)

        print(f"[MAGNET DEBUG] {os.path.basename(pdf_path)} p{page_index+1} {rot_name} -> {len(vals)} values (table OCR)")
        return vals[:30]
    finally:
        doc.close()

def extract_magnet_timing_values(pdf_path):
    """
    Key rule you gave:
      - If PDF does NOT contain the data page (3.7 table), disregard the entire PDF.
      - If it contains it (either as page 2 of 2 or page 1 of 1), extract from that page only.
    """
    page_idx, _ = find_data_page_index(pdf_path)
    if page_idx is None:
        return []  # skip entire PDF
    return extract_timing_values_from_data_page(pdf_path, page_idx)

def run_magnet_analysis(q=None, cancel_flag=None):
    base = os.getcwd()

    pdf_files = sorted([
        f for f in os.listdir(base)
        if f.lower().endswith(".pdf")
        and "magnet ring" in f.lower()
    ])

    rows = []

    for f in pdf_files:
        full = os.path.join(base, f)

        vals = extract_magnet_timing_values(full)

        if len(vals) < 20:
            print(f"[MAGNET] SKIP (no data page or insufficient values): {f}")
            continue

        arr = np.array(vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1))
        cp, cpk = compute_cp_cpk(mean, std)
        month_dt = extract_month_from_pdf_filename(f, full)

        rows.append({
            "month_dt": month_dt,
            "mean": mean,
            "std": std,
            "cp": cp,
            "cpk": cpk,
            "source_file": f
        })

        print(f"[MAGNET] OK: {f} -> {month_dt.strftime('%Y-%m')} CpK={cpk:.2f}")

    if not rows:
        print("❌ No magnet data found")
        return

    rows.sort(key=lambda r: (r["month_dt"], r["source_file"]))

    from collections import defaultdict
    month_counters = defaultdict(int)
    for r in rows:
        month_counters[r["month_dt"]] += 1
        r["label"] = f"{r['month_dt'].strftime('%B')} {month_counters[r['month_dt']]}"

    filename = f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    outpath = versioned_filename(REPORT_FOLDER_MAGNET, filename)

    create_magnet_pdf(outpath, rows)

# ============================================================
# GUI
# ============================================================

def start_threaded(funcs):
    root = tk.Tk()
    root.title("Running...")
    root.geometry("420x120")

    tk.Label(root, text="Processing...", font=("Segoe UI", 12)).pack(pady=10)
    bar = ttk.Progressbar(root, mode="indeterminate")
    bar.pack(pady=6, fill="x", padx=20)
    bar.start()

    q = queue.Queue()
    cancel_flag = {"stop": False}

    def worker():
        for func in funcs:
            try:
                func(q, cancel_flag)
            except Exception as e:
                print(f"[ERROR] Worker exception: {e}")
        q.put("done")

    def check_done():
        try:
            msg = q.get_nowait()
            if msg == "done":
                bar.stop()
                root.destroy()
                return
        except queue.Empty:
            pass
        root.after(200, check_done)

    root.after(200, check_done)
    threading.Thread(target=worker, daemon=True).start()
    root.mainloop()

def launch_gui():
    root = tk.Tk()
    root.title("Motor & Magnet Analysis")
    root.geometry("520x300")

    tk.Label(root, text="Motor & Magnet Analysis", font=("Segoe UI", 14, "bold")).pack(pady=12)

    tk.Button(root, text="Run Motor Analysis", width=45,
              command=lambda: start_threaded([run_motor_analysis])).pack(pady=8)

    tk.Button(root, text="Run Magnet Analysis", width=45,
              command=lambda: start_threaded([run_magnet_analysis])).pack(pady=8)

    tk.Button(root, text="Run Both", width=45,
              command=lambda: start_threaded([run_motor_analysis, run_magnet_analysis])).pack(pady=8)

    root.mainloop()

if __name__ == "__main__":
    launch_gui()