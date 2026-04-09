# ============================================================
# PART 1 — IMPORTS & GLOBAL CONFIG
# ============================================================

import os
import re
import threading
import traceback
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance

import pytesseract
import cv2

# ------------------------------------------------------------
# Tesseract configuration (HARD-WIRED, NO PATH ISSUES)
# ------------------------------------------------------------
TESSERACT_PATH = r"C:\Users\bbadner\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

_HAS_TESSERACT = False
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    _HAS_TESSERACT = True
    print("[INFO] Using Tesseract at:", TESSERACT_PATH)
else:
    print("[ERROR] Tesseract not found:", TESSERACT_PATH)

# ------------------------------------------------------------
# Global parameters
# ------------------------------------------------------------
MIN_OCR_VALUES = 10   # minimum numeric values to accept OCR page

# ============================================================
# PART 2 — MOTOR ANALYSIS (SAFE STUB)
# ============================================================

def run_motor_analysis():
    print("[INFO] Motor analysis placeholder (already working in your version)")

# ============================================================
# PART 2 — MOTOR ANALYSIS (SAFE STUB)
# ============================================================

def run_motor_analysis():
    print("[INFO] Motor analysis placeholder (already working in your version)")
# ============================================================
# PART 3 — STATISTICS & METADATA UTILITIES
# ============================================================

def compute_cp_cpk(mean, std, lsl=-1.0, usl=1.0):
    if std is None or std <= 0:
        return None, None

    cp = (usl - lsl) / (6 * std)
    cpk = min(
        (usl - mean) / (3 * std),
        (mean - lsl) / (3 * std)
    )
    return cp, cpk


def extract_mean_std_cp_cpk(text):
    """
    Extract mean and std dev from summary-style PDFs (April / May).
    """
    result = {"mean": None, "std": None}

    if not text:
        return result

    mean_match = re.search(r"Mean\s*[:=]\s*([-+]?\d*\.\d+|\d+)", text, re.I)
    std_match = re.search(
        r"(Std\s*Dev|Standard\s*Deviation)\s*[:=]\s*([-+]?\d*\.\d+|\d+)",
        text,
        re.I
    )

    try:
        if mean_match:
            result["mean"] = float(mean_match.group(1))
        if std_match:
            result["std"] = float(std_match.group(2))
    except Exception:
        pass

    return result


def extract_month_from_pdf_filename(filename):
    match = re.search(r"(20\d{2})(\d{2})\d{2}", filename)
    if match:
        return datetime(int(match.group(1)), int(match.group(2)), 1)

    match = re.search(r"(20\d{2})[-_](\d{2})", filename)
    if match:
        return datetime(int(match.group(1)), int(match.group(2)), 1)

    return datetime.now().replace(day=1)

# ============================================================
# PART 4 — PDF & OCR EXTRACTION (CROPPED NUMERIC COLUMN)
# ============================================================

def extract_pdf_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception:
        return ""


def extract_timing_values_from_pdf_images(pdf_path):
    """
    OCR for scanned magnet timing PDFs.

    - Crops to numeric timing column (right side)
    - Saves debug images to ./ocr_debug/
    """
    if not _HAS_TESSERACT:
        print("[MAGNET][OCR] Tesseract unavailable")
        return []

    values = []

    debug_dir = os.path.join(os.getcwd(), "ocr_debug")
    os.makedirs(debug_dir, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        for page_index, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=500)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Convert to grayscale
            img = img.convert("L")

            # Convert to numpy
            img_np = np.array(img)
            h, w = img_np.shape

            # ------------------------------------------------
            # CROP TO RIGHT NUMERIC COLUMN (KEY FIX)
            # ------------------------------------------------
            x_start = int(w * 0.60)
            x_end   = w
            y_start = int(h * 0.10)
            y_end   = int(h * 0.95)

            img_np = img_np[y_start:y_end, x_start:x_end]

            # Enlarge
            img_np = cv2.resize(
                img_np,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            # Increase contrast
            img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX)

            # Threshold
            _, img_np = cv2.threshold(
                img_np,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Slight dilation
            kernel = np.ones((2, 2), np.uint8)
            img_np = cv2.dilate(img_np, kernel, iterations=1)

            # ------------------------------------------------
            # SAVE DEBUG IMAGE
            # ------------------------------------------------
            debug_path = os.path.join(
                debug_dir,
                f"{pdf_name}_page_{page_index}_CROPPED.png"
            )
            cv2.imwrite(debug_path, img_np)
            print(f"[MAGNET][OCR][DEBUG] Saved: {debug_path}")

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------
            config = (
                "--oem 3 "
                "--psm 6 "
                "-c tessedit_char_whitelist=0123456789."
            )

            text = pytesseract.image_to_string(img_np, config=config)

            for token in re.findall(r"\d+\.\d+", text):
                try:
                    val = float(token)
                    if 0.1 < val < 10.0:  # timing sanity range
                        values.append(val)
                except Exception:
                    pass

        doc.close()

    except Exception as e:
        print("[MAGNET][OCR] Exception:", e)

    if values:
        print(f"[MAGNET][OCR] Extracted {len(values)} timing values")
    else:
        print(f"[MAGNET][OCR] No timing values found")

    return values


# ============================================================
# PART 5 — MAGNET ANALYSIS (OCR-DRIVEN, ROBUST)
# ============================================================

def run_magnet_analysis():
    print("[INFO] Starting magnet analysis")

    folder = filedialog.askdirectory(title="Select Magnet Data Folder")
    if not folder:
        print("[MAGNET] No folder selected")
        return

    records = []

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".pdf"):
            continue

        full_path = os.path.join(folder, fname)
        print(f"[MAGNET] Processing: {fname}")

        # ----------------------------------------------------
        # Attempt text-based extraction first (summary PDFs)
        # ----------------------------------------------------
        text = extract_pdf_text(full_path)
        stats = extract_mean_std_cp_cpk(text)

        if stats.get("mean") is not None and stats.get("std") is not None:
            mean = float(stats["mean"])
            std = float(stats["std"])
            source = "summary"

            print(f"[MAGNET] Using summary stats from {fname}")

        else:
            # ------------------------------------------------
            # OCR path (scanned PDFs)
            # ------------------------------------------------
            values = extract_timing_values_from_pdf_images(full_path)

            if not values:
                print("[MAGNET] SKIP (no OCR values)")
                continue

            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            source = "ocr"

            print(f"[MAGNET] Using {len(values)} OCR values from {fname}")

        # ----------------------------------------------------
        # Compute Cp / Cpk
        # ----------------------------------------------------
        cp, cpk = compute_cp_cpk(mean, std)

        # ----------------------------------------------------
        # Determine month
        # ----------------------------------------------------
        month = extract_month_from_pdf_filename(fname)

        records.append({
            "Month": month.strftime("%Y-%m"),
            "Mean": mean,
            "StdDev": std,
            "Cp": cp,
            "Cpk": cpk,
            "Source": source,
            "File": fname
        })

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------
    if not records:
        messagebox.showerror(
            "Magnet Report",
            "No magnet data found.\n\n"
            "OCR ran, but no valid datasets passed filtering."
        )
        return

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------
    df = pd.DataFrame(records)
    df = df.sort_values("Month")

    # --------------------------------------------------------
    # Output report
    # --------------------------------------------------------
    out_path = os.path.join(
        folder,
        "Rexair_Magnet_Timing_CpK_Report.csv"
    )

    df.to_csv(out_path, index=False)

    messagebox.showinfo(
        "Magnet Report",
        f"Magnet report created successfully:\n\n{out_path}"
    )

    print("[MAGNET] Report created:", out_path)


# ============================================================
# PART 6 — GUI & THREADING
# ============================================================

def start_threaded(funcs):
    def runner():
        try:
            for f in funcs:
                f()
        except Exception:
            traceback.print_exc()
            messagebox.showerror("Error", "An error occurred.\nSee console.")

    threading.Thread(target=runner, daemon=True).start()


def launch_gui():
    root = tk.Tk()
    root.title("Motor & Magnet Analysis")
    root.geometry("520x220")

    tk.Label(
        root,
        text="Motor & Magnet Analysis",
        font=("Segoe UI", 14, "bold")
    ).pack(pady=12)

    tk.Button(
        root,
        text="Run Magnet Analysis",
        width=45,
        command=lambda: start_threaded([run_magnet_analysis])
    ).pack(pady=12)

    root.mainloop()


if __name__ == "__main__":
    launch_gui()
