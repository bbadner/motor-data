# ============================================================
# Rexair Magnet Timing Analysis (Standalone)
# Focused magnet-only script for supplier PDF reports
# ============================================================

import os
import re
import platform
import subprocess
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
import tkinter as tk
from tkinter import ttk, messagebox
import fitz  # PyMuPDF

# Optional OCR support
try:
    from PIL import Image, ImageEnhance
    import pytesseract
    HAS_OCR = True
except Exception:
    Image = None
    ImageEnhance = None
    pytesseract = None
    HAS_OCR = False

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

REPORT_FOLDER = "Magnet Reports"
DEBUG_FOLDER = "Magnet Debug"
LOGO_FILE = "rexair_logo.png"

TARGET_COUNT = 30
MAGNET_TARGET = 3.7
MAGNET_TOL = 0.5
LSL = MAGNET_TARGET - MAGNET_TOL
USL = MAGNET_TARGET + MAGNET_TOL

MIN_VALUES_REQUIRED = 15      # accept complete reports even if OCR misses a few rows
TARGET_VALUES = 30            # preferred count from supplier page 2
DUPLICATE_ROUND_DECIMALS = 4  # duplicate comparison stability

os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# GENERAL UTILITIES
# ------------------------------------------------------------

def open_file(filepath):
    try:
        if platform.system() == "Windows":
            os.startfile(filepath)
        elif platform.system() == "Darwin":
            subprocess.run(["open", filepath], check=False)
        else:
            subprocess.run(["xdg-open", filepath], check=False)
    except Exception as exc:
        print(f"[WARN] Could not open file: {exc}")


def compute_cp_cpk(mean, std):
    if std is None or std <= 0:
        return 0.0, 0.0
    cp = (USL - LSL) / (6.0 * std)
    cpk = min((mean - LSL) / (3.0 * std), (USL - mean) / (3.0 * std))
    return cp, cpk


def duplicate_key(values):
    return tuple(round(float(v), DUPLICATE_ROUND_DECIMALS) for v in values)


def dump_debug_text(pdf_name, page1_text, page2_text, note=""):
    base = os.path.splitext(os.path.basename(pdf_name))[0]
    out_txt = os.path.join(DEBUG_FOLDER, f"{base}_debug.txt")
    try:
        with open(out_txt, "w", encoding="utf-8", errors="ignore") as f:
            if note:
                f.write(f"NOTE: {note}\n\n")
            f.write("--- PAGE 1 ---\n")
            f.write(page1_text or "")
            f.write("\n\n--- PAGE 2 ---\n")
            f.write(page2_text or "")
        print(f"[DEBUG] Wrote debug file: {out_txt}")
    except Exception as exc:
        print(f"[WARN] Could not write debug dump for {pdf_name}: {exc}")


# ------------------------------------------------------------
# DATE EXTRACTION
# ------------------------------------------------------------

def extract_month_from_filename(filename):
    """
    Fallback date extraction from filenames like:
      magnet ring-2026-01-16 57-58.pdf
      Magnet Checking Data（19850PCS) 20250414.pdf
    Returns datetime(month first day) or None.
    """
    patterns = [
        r"(20\d{2})[-_](\d{2})[-_](\d{2})",
        r"(20\d{2})(\d{2})(\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, filename)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), 1)
            except Exception:
                pass
    return None


def extract_inspection_month(page1_text, filename):
    """
    Preferred method: read inspection date from page 1.
    Fallback: date parsed from filename.
    Returns datetime set to first day of month, or None.
    """
    if page1_text:
        patterns = [
            r"INSPECTION\s*[:：]?\s*([A-Za-z]{3}/\d{1,2}/\d{4})",
            r"INSPECTION\s*DATE\s*[:：]?\s*([A-Za-z]{3}/\d{1,2}/\d{4})",
            r"Inspection\s*Date\s*[:：]?\s*([A-Za-z]{3}/\d{1,2}/\d{4})",
            r"Inspection\s*[:：]?\s*([A-Za-z]{3}/\d{1,2}/\d{4})",
        ]
        for pat in patterns:
            m = re.search(pat, page1_text, re.IGNORECASE)
            if m:
                raw = m.group(1).strip()
                try:
                    return datetime.strptime(raw, "%b/%d/%Y").replace(day=1)
                except Exception:
                    pass

    return extract_month_from_filename(filename)


# ------------------------------------------------------------
# PDF / OCR EXTRACTION
# ------------------------------------------------------------

def extract_page_texts(pdf_path):
    """
    Return a list of page texts.
    Uses text layer first. If a page is sparse and OCR is available, tries OCR.
    """
    texts = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        return [], f"read_error:{exc}"

    try:
        for page in doc:
            txt = page.get_text("text") or ""
            if len(txt.strip()) < 30 and HAS_OCR:
                try:
                    tp = page.get_textpage_ocr(language="eng", dpi=300, full=True)
                    ocr_txt = page.get_text("text", textpage=tp) or ""
                    if len(ocr_txt.strip()) > len(txt.strip()):
                        txt = ocr_txt
                except Exception:
                    pass
            texts.append(txt)
    finally:
        doc.close()

    return texts, "ok"


def normalize_numeric_text(text):
    if not text:
        return ""
    t = text
    t = t.replace("\u00a0", " ")
    t = t.replace("，", ",").replace("：", ":").replace("；", ";")
    t = t.replace("·", ".").replace("º", "°")
    t = t.replace("O", "0").replace("o", "0")
    t = t.replace("I", "1").replace("l", "1")
    t = t.replace("S", "5")
    return t


def extract_37_values_from_page2_text(page2_text, target_count=TARGET_VALUES):
    """
    Preferred parser for supplier page 2.
    It looks for row lines and takes the RIGHT-MOST 3.x value on each line,
    which corresponds to the 3.7 ± 0.5° column.
    """
    if not page2_text:
        return []

    t = normalize_numeric_text(page2_text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    values = []
    for line in lines:
        # Row starts with 1..30 in most reports
        if not re.match(r"^\d{1,2}\b", line):
            continue

        # Get decimal values on the row; use the right-most 3.x value
        nums = re.findall(r"\b([34]\.\d{1,2})\b", line)
        if nums:
            try:
                v = float(nums[-1])
                if 3.0 <= v <= 4.5:
                    values.append(v)
            except Exception:
                pass

    if len(values) >= MIN_VALUES_REQUIRED:
        return values[:target_count]

    # Fallback 1: only scan text after the 3.7 header
    header_match = re.search(r"3\s*\.?\s*7\s*(?:±|\+/?-)\s*0\s*\.?\s*5", t)
    scan_text = t[header_match.start():] if header_match else t

    values = []
    for m in re.finditer(r"\b([34]\.\d{1,2})\b", scan_text):
        try:
            v = float(m.group(1))
        except Exception:
            continue
        if 3.0 <= v <= 4.5:
            values.append(v)
        if len(values) >= target_count:
            return values[:target_count]

    # Fallback 2: OCR sometimes loses the decimal point: 380 -> 3.80
    for m in re.finditer(r"\b([34]\d{2})\b", scan_text):
        try:
            v = float(m.group(1)) / 100.0
        except Exception:
            continue
        if 3.0 <= v <= 4.5:
            values.append(v)
        if len(values) >= target_count:
            return values[:target_count]

    return values[:target_count]


def extract_37_values_from_page2_image(pdf_path, target_page_index=1, dpi=450):
    """
    OCR fallback for page 2 only, cropping likely right-side table region.
    """
    if not HAS_OCR:
        return []

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []

    try:
        if doc.page_count <= target_page_index:
            return []
        page = doc.load_page(target_page_index)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("L")
        w, h = img.size

        crop_boxes = [
            (0.50, 0.08, 0.96, 0.96),
            (0.56, 0.08, 0.98, 0.96),
            (0.60, 0.10, 0.98, 0.98),
        ]
        configs = [
            '--psm 6 -c tessedit_char_whitelist=0123456789.°/-±',
            '--psm 4 -c tessedit_char_whitelist=0123456789.°/-±',
            '--psm 11 -c tessedit_char_whitelist=0123456789.°/-±',
        ]

        for lx, ty, rx, by in crop_boxes:
            crop = img.crop((int(w * lx), int(h * ty), int(w * rx), int(h * by)))
            crop = ImageEnhance.Contrast(crop).enhance(3.0)
            crop = crop.point(lambda p: 255 if p > 180 else 0)

            for cfg in configs:
                try:
                    ocr_text = pytesseract.image_to_string(crop, config=cfg)
                except Exception:
                    continue
                #vals = extract_37_values_from_page2_text(ocr_text, target_count=target_count)
                vals = extract_37_values_from_page2_text(ocr_text, target_count=TARGET_VALUES)
		if len(vals) >= MIN_VALUES_REQUIRED:
                    return vals[:TARGET_VALUES]
                    #return vals[:target_count]
        return []
    finally:
        doc.close()


# ------------------------------------------------------------
# REPORT GENERATION
# ------------------------------------------------------------

def add_red_note(pdf, text, bold=True):
    pdf.set_text_color(255, 0, 0)
    pdf.set_font("Arial", "B" if bold else "", 11)
    pdf.multi_cell(190, 6, text)
    pdf.set_text_color(0, 0, 0)


def create_magnet_pdf(output_path, rows, incomplete_files, duplicate_log):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    if os.path.exists(LOGO_FILE):
        try:
            pdf.image(LOGO_FILE, x=10, y=8, w=28)
        except Exception:
            pass

    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Report", ln=True)

    labels = [r["label"] for r in rows]
    cpks = [r["cpk"] for r in rows]

    plt.figure(figsize=(8.5, 4.0))
    plt.plot(labels, cpks, marker="o")
    plt.axhline(1.33, color="black", linestyle="--", linewidth=1.3)
    plt.title("Magnet Timing CpK Over Time (Per File)")
    plt.xlabel("File within Month")
    plt.ylabel("CpK")
    ymax = max(4.0, (max(cpks) + 0.25) if cpks else 4.0)
    plt.ylim(0, ymax)
    plt.grid(True)

    ax = plt.gca()
    latest_cpk = cpks[-1] if cpks else None
    if latest_cpk is not None:
        if latest_cpk >= 1.33:
            ax.text(0.5, 0.5, "CpK is Good", transform=ax.transAxes,
                    fontsize=18, fontweight="bold", color="green",
                    ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "CpK is BAD", transform=ax.transAxes,
                    fontsize=18, fontweight="bold", color="red",
                    ha="center", va="center")

    plt.xticks(rotation=45)
    plt.tight_layout()
    chart_png = "magnet_cpk_temp.png"
    plt.savefig(chart_png, dpi=300)
    plt.close()

    pdf.image(chart_png, x=10, y=30, w=190)

    pdf.set_xy(10, 125)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Per-File Summary Table", ln=True)
    pdf.set_font("Arial", "", 11)

    for row in rows:
        line = (
            f"{row['label']} "
            f"Mean: {row['mean']:.4f}  "
            f"Std: {row['std']:.4f}  "
            f"Cp: {row['cp']:.2f}  "
            f"CpK: {row['cpk']:.2f}"
        )
        pdf.cell(0, 8, line, ln=True)

    pdf.ln(4)
    if incomplete_files:
        add_red_note(pdf, "INCOMPLETE REPORT RECEIVED")
        pdf.set_text_color(255, 0, 0)
        pdf.set_font("Arial", "", 10)
        for f in incomplete_files:
            pdf.multi_cell(190, 5, f" - {f}")
        pdf.set_text_color(0, 0, 0)

    if duplicate_log:
        pdf.ln(2)
        add_red_note(pdf, "DUPLICATE REPORTS INCLUDED")
        pdf.set_text_color(255, 0, 0)
        pdf.set_font("Arial", "", 10)
        for dup_file, kept_file in duplicate_log:
            pdf.multi_cell(190, 5, f" - {dup_file} duplicates {kept_file}")
        pdf.set_text_color(0, 0, 0)

    pdf.output(output_path)

    try:
        if os.path.exists(chart_png):
            os.remove(chart_png)
    except Exception:
        pass


# ------------------------------------------------------------
# CORE ANALYSIS
# ------------------------------------------------------------

def analyze_single_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    page_texts, mode = extract_page_texts(pdf_path)

    if mode.startswith("read_error") or not page_texts:
        return {
            "status": "incomplete",
            "file": filename,
            "reason": f"Could not read PDF: {mode}",
        }

    if len(page_texts) < 2:
        return {
            "status": "incomplete",
            "file": filename,
            "reason": "PDF does not contain two pages",
        }

    page1_text = page_texts[0] or ""
    page2_text = page_texts[1] or ""

    month_dt = extract_inspection_month(page1_text, filename)
    if month_dt is None:
        dump_debug_text(filename, page1_text, page2_text, note="Inspection date not found")
        return {
            "status": "incomplete",
            "file": filename,
            "reason": "Inspection date not found on page 1 and no valid filename fallback",
        }

    values = extract_37_values_from_page2_text(page2_text, target_count=TARGET_VALUES)

    if len(values) < MIN_VALUES_REQUIRED:
        img_vals = extract_37_values_from_page2_image(pdf_path, target_page_index=1)
        if len(img_vals) >= len(values):
            values = img_vals

    if len(values) < MIN_VALUES_REQUIRED:
        dump_debug_text(filename, page1_text, page2_text, note="Too few 3.7 column values found")
        return {
            "status": "incomplete",
            "file": filename,
            "reason": f"Only found {len(values)} timing values on page 2",
            "month_dt": month_dt,
        }

    values = values[:TARGET_VALUES]
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    cp, cpk = compute_cp_cpk(mean, std)

    return {
        "status": "ok",
        "file": filename,
        "month_dt": month_dt,
        "values": values,
        "mean": mean,
        "std": std,
        "cp": cp,
        "cpk": cpk,
    }


def run_magnet_analysis(progress_callback=None):
    base = os.getcwd()
    report_dir = os.path.join(base, REPORT_FOLDER)
    os.makedirs(report_dir, exist_ok=True)

    pdf_files = sorted([f for f in os.listdir(base) if f.lower().endswith(".pdf")])
    if not pdf_files:
        raise RuntimeError("No PDF files found in the script folder.")

    rows = []
    incomplete_files = []
    duplicate_log = []
    month_registry = defaultdict(dict)  # month -> duplicate_key -> kept filename

    total = len(pdf_files)
    for idx, filename in enumerate(pdf_files, start=1):
        if progress_callback:
            progress_callback(f"Scanning {idx}/{total}: {filename}")

        result = analyze_single_pdf(os.path.join(base, filename))

        if result["status"] != "ok":
            print(f"[MAGNET] INCOMPLETE: {filename} -> {result.get('reason', 'Unknown reason')}")
            incomplete_files.append(filename)
            continue

        month_key = result["month_dt"].strftime("%Y-%m")
        key = duplicate_key(result["values"])
        if key in month_registry[month_key]:
            kept_file = month_registry[month_key][key]
            duplicate_log.append((filename, kept_file))
            print(f"[MAGNET] DUPLICATE: {filename} duplicates {kept_file}")
            continue

        month_registry[month_key][key] = filename
        rows.append(result)
        print(
            f"[MAGNET] OK: {filename} | {result['month_dt'].strftime('%b %Y')} | "
            f"n={len(result['values'])} mean={result['mean']:.4f} std={result['std']:.4f} "
            f"cp={result['cp']:.2f} cpk={result['cpk']:.2f}"
        )

    if not rows:
        raise RuntimeError("No valid magnet data found. Check the Magnet Debug folder.")

    rows.sort(key=lambda r: (r["month_dt"], r["file"].lower()))
    month_counts = defaultdict(int)
    for row in rows:
        month_counts[row["month_dt"]] += 1
        row["label"] = f"{row['month_dt'].strftime('%B')} {month_counts[row['month_dt']]}"

    output_path = os.path.join(report_dir, f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf")
    create_magnet_pdf(output_path, rows, incomplete_files, duplicate_log)

    if progress_callback:
        progress_callback("Done")

    return {
        "output_path": output_path,
        "rows": rows,
        "incomplete_files": incomplete_files,
        "duplicate_log": duplicate_log,
    }


# ------------------------------------------------------------
# SIMPLE GUI
# ------------------------------------------------------------

class MagnetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rexair Magnet Analysis")
        self.root.geometry("520x240")

        tk.Label(root, text="Rexair Magnet Analysis", font=("Segoe UI", 14, "bold")).pack(pady=14)
        tk.Label(
            root,
            text="Place this script in the same folder as the supplier magnet PDFs.",
            font=("Segoe UI", 10),
        ).pack(pady=4)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(root, textvariable=self.status_var, font=("Segoe UI", 10)).pack(pady=8)

        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(fill="x", padx=30, pady=8)

        tk.Button(
            root,
            text="Run Magnet Analysis",
            width=32,
            command=self.run_analysis,
        ).pack(pady=10)

        tk.Button(
            root,
            text="Open Magnet Reports Folder",
            width=32,
            command=self.open_reports_folder,
        ).pack(pady=4)

    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def open_reports_folder(self):
        folder = os.path.join(os.getcwd(), REPORT_FOLDER)
        os.makedirs(folder, exist_ok=True)
        open_file(folder)

    def run_analysis(self):
        self.progress.start()
        self.set_status("Running...")
        self.root.update()
        try:
            result = run_magnet_analysis(progress_callback=self.set_status)
            self.progress.stop()
            self.set_status("Report created")
            msg = [f"Report created:\n{result['output_path']}"]
            if result["incomplete_files"]:
                msg.append(f"\nIncomplete reports: {len(result['incomplete_files'])}")
            if result["duplicate_log"]:
                msg.append(f"\nDuplicate reports skipped: {len(result['duplicate_log'])}")
            messagebox.showinfo("Magnet Analysis", "".join(msg))
            open_file(result["output_path"])
        except Exception as exc:
            self.progress.stop()
            self.set_status("Failed")
            messagebox.showerror("Magnet Analysis", str(exc))


def main():
    root = tk.Tk()
    app = MagnetApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
