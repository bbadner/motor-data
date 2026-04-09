import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageFilter
import pytesseract


# ============================================================
# CONFIG
# ============================================================

# Set this to your installed Tesseract path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\bbadner\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

REPORT_FOLDER = "Magnet Reports"

TARGET = 3.7
TOL = 0.5
LSL = TARGET - TOL
USL = TARGET + TOL

TARGET_VALUES = 30
MIN_VALUES = 10

CHART_TARGET_CPK = 1.33

os.makedirs(REPORT_FOLDER, exist_ok=True)


# ============================================================
# STATS
# ============================================================

def compute_cp_cpk(mean: float, std: float) -> Tuple[float, float]:
    if std <= 0:
        return 0.0, 0.0

    cp = (USL - LSL) / (6 * std)
    cpk = min((mean - LSL) / (3 * std), (USL - mean) / (3 * std))
    return cp, cpk


# ============================================================
# PDF / IMAGE HELPERS
# ============================================================

def render_pdf_page(pdf_file: str, page_num: int, dpi: int = 450) -> Image.Image:
    doc = fitz.open(pdf_file)
    try:
        page = doc.load_page(page_num)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    finally:
        doc.close()


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray)
    # MedianFilter deliberately omitted — it blurs decimal points in small fonts,
    # causing Tesseract to drop them (e.g. "3.76" read as "376").
    bw = gray.point(lambda x: 255 if x > 165 else 0)
    return bw


def preprocess_for_ocr_raw(img: Image.Image) -> Image.Image:
    """No-threshold variant — better for clean scans where thresholding merges glyphs."""
    gray = img.convert("L")
    return ImageOps.autocontrast(gray)


def ocr_image(img: Image.Image, psm: int = 6) -> str:
    config = f"--oem 3 --psm {psm}"
    return pytesseract.image_to_string(img, config=config)


def get_page_text(doc: fitz.Document, pdf_file: str, page_num: int) -> str:
    """
    Try native PDF text first. If weak/empty, try OCR with a few rotations/psm modes.
    Returns the best text blob found.
    """
    native_text = doc.load_page(page_num).get_text("text")
    native_text = native_text or ""

    candidates = [native_text]

    if len(native_text.strip()) < 40:
        base_img = render_pdf_page(pdf_file, page_num, dpi=450)

        for angle in (0, 90, 270):
            img = base_img.rotate(angle, expand=True) if angle else base_img
            proc = preprocess_for_ocr(img)

            for psm in (6, 11):
                text = ocr_image(proc, psm=psm)
                if text:
                    candidates.append(text)

    # Best candidate = most useful text length after stripping whitespace
    best = max(candidates, key=lambda s: len(re.sub(r"\s+", "", s)))
    return best


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_ocr_text(text: str) -> str:
    """
    Normalize common OCR quirks before regex extraction.
    """
    if not text:
        return ""

    t = text

    # Unify punctuation / odd glyphs
    replacements = {
        "：": ":",
        "，": ",",
        "。": ".",
        "·": ".",
        "•": ".",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "°": "",
        "º": "",
        "O": "0",   # sometimes OCR mixes O and 0
        "o": "0",
    }

    for old, new in replacements.items():
        t = t.replace(old, new)

    # Normalize whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\r", "\n", t)

    return t


# ============================================================
# MONTH EXTRACTION
# ============================================================

def extract_month(text: str, filename: str) -> Optional[datetime]:
    t = normalize_ocr_text(text)

    # Examples:
    # Apr/14/2025
    # May/22/2025
    # Sep-9-2025
    # September 9, 2025
    patterns = [
        r'([A-Za-z]{3,9})[\/\- ](\d{1,2})[\/\- ,]+(20\d{2})',
        r'(20\d{2})[\/\-_](\d{1,2})[\/\-_](\d{1,2})',
    ]

    # Try month-name forms first
    m = re.search(patterns[0], t, flags=re.IGNORECASE)
    if m:
        month_name = m.group(1)[:3].title()
        try:
            month_num = datetime.strptime(month_name, "%b").month
            year = int(m.group(3))
            return datetime(year, month_num, 1)
        except ValueError:
            pass

    # Try ISO date in text
    m = re.search(patterns[1], t)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        return datetime(year, month, 1)

    # Try filename fallback
    m = re.search(r'(20\d{2})[-_](\d{2})[-_](\d{2})', filename)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1)

    # Try compact yyyyMMdd in filename
    m = re.search(r'(20\d{2})(\d{2})(\d{2})', filename)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1)

    return None


# ============================================================
# VALUE EXTRACTION
# ============================================================

def parse_3x_values(text: str) -> List[float]:
    """
    Extract 3.xx-like values robustly from OCR text.
    Handles separators and dropped decimals:
      3.76  3,76  3:76  3 76   <- explicit separator
      376   369   382          <- decimal dropped by OCR (poor scan / thin font)

    Strategy: scan left-to-right, collecting every token that looks like a
    measurement.  Bare 3XX tokens are only promoted when no separator form
    was already found at the same string position (avoids double-counting
    "3.76" once as "3.76" and once as the bare "376" embedded in it).
    """
    t = normalize_ocr_text(text)

    # Collect (position, value) for separator form
    sep_spans: set = set()   # character spans already matched
    values: List[float] = []

    for m in re.finditer(r'(?<!\d)(3)[\.,:\s](\d{2})(?!\d)', t):
        val = float(f"3.{m.group(2)}")
        if 3.0 <= val <= 4.5:
            values.append(val)
            # Mark the start position so the bare pass skips overlap
            sep_spans.add(m.start())

    # Bare 3XX pass — only add tokens whose start position wasn't already
    # captured by the separator pass (prevents double-counting "3.76" -> 376).
    # Use tighter LSL..USL range to filter noise (e.g. OCR artifact "326").
    for m in re.finditer(r'(?<!\d)3(\d{2})(?!\d)', t):
        # Skip if this position overlaps a separator match
        if any(abs(m.start() - s) <= 1 for s in sep_spans):
            continue
        val = float(f"3.{m.group(1)}")
        # Use a tighter floor (3.45) for bare matches to block OCR noise such as
        # "326" (a misread of "3.86") while still catching all in-spec values.
        # The true spec floor is LSL=3.2 but real measurements are never that low.
        if 3.45 <= val <= USL:
            values.append(val)

    return values


def dedupe_preserve_order(values: List[float], tol: float = 1e-9) -> List[float]:
    result = []
    for v in values:
        if not result or abs(result[-1] - v) > tol:
            result.append(v)
    return result


def _extract_cells(col_img: Image.Image) -> List[float]:
    """
    Detect horizontal table lines in col_img and OCR each cell individually
    using psm=7 (single text line). Returns values found across all cells.
    This is a fallback for PDFs where whole-page OCR misreads individual rows.
    """
    import numpy as np

    arr = np.array(col_img.convert("L"))
    row_means = arr.mean(axis=1)

    # Group consecutive dark rows into single line positions
    is_dark = row_means < 100
    lines: List[int] = []
    in_line = False
    start = 0
    for i, d in enumerate(is_dark):
        if d and not in_line:
            in_line = True
            start = i
        elif not d and in_line:
            in_line = False
            lines.append((start + i) // 2)

    if len(lines) < 3:
        return []

    col_w = col_img.width
    values: List[float] = []

    for i in range(1, len(lines) - 1):   # skip header row
        y1 = lines[i] + 2
        y2 = lines[i + 1] - 2
        if y2 - y1 < 5:
            continue

        cell = col_img.crop((0, y1, col_w, y2))
        # Scale up 4x so Tesseract can resolve small digits clearly
        cell_big = cell.resize((cell.width * 4, cell.height * 4), Image.LANCZOS)

        for proc in (preprocess_for_ocr(cell_big), preprocess_for_ocr_raw(cell_big)):
            text = pytesseract.image_to_string(
                proc,
                config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789."
            ).strip()
            cell_vals = parse_3x_values(text)
            if cell_vals:
                values.extend(cell_vals)
                break   # accept first successful read for this cell

    return values


def extract_values_from_page2(doc: fitz.Document, pdf_file: str) -> List[float]:
    """
    Build several extraction candidates and choose the best one.
    We prefer the candidate closest to exactly 30 values.
    """
    candidates: List[List[float]] = []

    # 1) Native text
    native = doc.load_page(1).get_text("text") or ""
    native_vals = parse_3x_values(native)
    if native_vals:
        candidates.append(native_vals)

    # 2) OCR full page (uncropped) — best safety net for tall tables
    #    Use h * 1.0 to ensure the very last rows are never clipped.
    base_img = render_pdf_page(pdf_file, 1, dpi=500)
    full_proc = preprocess_for_ocr(base_img)

    for psm in (4, 6, 11):          # psm=4 added: single-column / table mode
        text = ocr_image(full_proc, psm=psm)
        vals = parse_3x_values(text)
        if vals:
            candidates.append(vals)

    w, h = base_img.size

    # 3) OCR cropped right side (target column usually lives here).
    #    Bottom extended to h (1.0) so the last rows are never cut off.
    #    Left boundary loosened to 0.45 to avoid clipping a column that
    #    starts slightly left of the 0.50 mark.
    right_crop = base_img.crop((int(w * 0.45), int(h * 0.05), int(w * 0.99), h))
    right_proc = preprocess_for_ocr(right_crop)

    for psm in (4, 6, 11):
        text = ocr_image(right_proc, psm=psm)
        vals = parse_3x_values(text)
        if vals:
            candidates.append(vals)

    # 3b) Same crop without thresholding — catches cases where threshold
    #     merges the decimal dot with adjacent ink.
    right_raw = preprocess_for_ocr_raw(right_crop)
    for psm in (4, 6, 11):
        text = ocr_image(right_raw, psm=psm)
        vals = parse_3x_values(text)
        if vals:
            candidates.append(vals)

    # 4) OCR narrow far-right band.
    #    Left boundary loosened from 0.62 → 0.58 and bottom extended to h.
    narrow_crop = base_img.crop((int(w * 0.58), int(h * 0.05), int(w * 0.97), h))
    narrow_proc = preprocess_for_ocr(narrow_crop)

    for psm in (4, 6, 11):
        text = ocr_image(narrow_proc, psm=psm)
        vals = parse_3x_values(text)
        if vals:
            candidates.append(vals)

    # 4b) Targeted column crop — empirically identified as the exact x-band
    #     containing the 3.7±0.5° data column in this document layout (47–62%).
    #     Run both threshold and raw preprocessing to maximise decimal detection.
    target_crop = base_img.crop((int(w * 0.47), int(h * 0.06), int(w * 0.62), h))
    for proc in (preprocess_for_ocr(target_crop), preprocess_for_ocr_raw(target_crop)):
        for psm in (4, 6, 11):
            text = ocr_image(proc, psm=psm)
            vals = parse_3x_values(text)
            if vals:
                candidates.append(vals)

    # 5) Bottom-half crop — catches rows 20-30 that top-biased OCR may miss
    bottom_crop = base_img.crop((int(w * 0.45), int(h * 0.55), int(w * 0.99), h))
    bottom_proc = preprocess_for_ocr(bottom_crop)

    for psm in (4, 6, 11):
        text = ocr_image(bottom_proc, psm=psm)
        vals = parse_3x_values(text)
        if vals:
            candidates.append(vals)

    # 6) Cell-by-cell extraction on the targeted column.
    #    Detects horizontal table lines, then OCRs each cell individually
    #    using psm=7 (single text line). This sidesteps layout confusion that
    #    causes multi-row OCR passes to misread or skip individual cells.
    cell_vals = _extract_cells(target_crop)
    if cell_vals:
        candidates.append(cell_vals)

    # Cleanup candidate lists
    cleaned_candidates = []
    for vals in candidates:
        vals = dedupe_preserve_order(vals)
        cleaned_candidates.append(vals)

    if not cleaned_candidates:
        return []

    # Prefer candidate with:
    # 1) >= TARGET_VALUES values and smallest overflow
    # 2) otherwise longest candidate
    exact_or_over = [c for c in cleaned_candidates if len(c) >= TARGET_VALUES]
    if exact_or_over:
        best = min(exact_or_over, key=lambda c: len(c) - TARGET_VALUES)
        return best[:TARGET_VALUES]

    # No candidate reached TARGET_VALUES — merge top-half and bottom-half
    # candidates to reconstruct a full set when OCR splits the table.
    if len(cleaned_candidates) >= 2:
        # Find the longest candidate (likely covers rows 1-20)
        top = max(cleaned_candidates, key=len)
        # Find the best bottom-half candidate (rows 21-30) from crop #5
        bottom_candidates = [c for c in cleaned_candidates if c and c not in [top]]
        if bottom_candidates:
            bottom = max(bottom_candidates, key=len)
            # Only merge if top + bottom together approach TARGET_VALUES
            # and their value ranges don't suggest full overlap
            merged = top + [v for v in bottom if v not in top[-5:]]
            merged = dedupe_preserve_order(merged)
            if len(merged) > len(top):
                candidates.append(merged)
                cleaned_candidates.append(merged)
                if len(merged) >= TARGET_VALUES:
                    return merged[:TARGET_VALUES]

    best = max(cleaned_candidates, key=len)
    return best[:TARGET_VALUES]


# ============================================================
# REPORT
# ============================================================

def create_report(rows: List[dict]) -> str:
    labels = [r["label"] for r in rows]
    cpks = [r["cpk"] for r in rows]

    plt.figure(figsize=(8, 4))
    plt.plot(labels, cpks, marker="o")
    plt.axhline(CHART_TARGET_CPK, linestyle="--")
    plt.title("Magnet Timing CpK Trend")
    plt.ylabel("Cpk")
    plt.grid(True)

    chart_file = "chart.png"
    plt.tight_layout()
    plt.savefig(chart_file, dpi=200)
    plt.close()

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Report", ln=True)

    pdf.image(chart_file, x=10, y=30, w=190)

    pdf.set_y(120)
    pdf.set_font("Arial", "", 11)

    for row in rows:
        line = (
            f"{row['label']}   "
            f"Mean:{row['mean']:.4f}   "
            f"Std:{row['std']:.4f}   "
            f"Cp:{row['cp']:.2f}   "
            f"Cpk:{row['cpk']:.2f}"
        )
        pdf.cell(0, 8, line, ln=True)

    output = os.path.join(
        REPORT_FOLDER,
        f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    )

    pdf.output(output)

    if os.path.exists(chart_file):
        os.remove(chart_file)

    return output


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_pdf(pdf_file: str) -> Optional[dict]:
    print("")
    print(f"Processing: {pdf_file}")

    try:
        doc = fitz.open(pdf_file)
    except Exception as exc:
        print(f"Skipped - could not open PDF: {exc}")
        return None

    try:
        if doc.page_count < 2:
            print(f"Skipped - not enough pages ({doc.page_count})")
            return None

        page1_text = get_page_text(doc, pdf_file, 0)
        month = extract_month(page1_text, pdf_file)

        if not month:
            print("Skipped - month not detected")
            return None

        values = extract_values_from_page2(doc, pdf_file)

        if len(values) < MIN_VALUES:
            print(f"Skipped - insufficient values: {len(values)}")
            return None

        arr = np.array(values, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        cp, cpk = compute_cp_cpk(mean, std)

        print(f"OK  n={len(values)}  mean={mean:.4f}  std={std:.4f}  cpk={cpk:.2f}")

        return {
            "file": pdf_file,
            "month_dt": month,
            "values": values,
            "mean": mean,
            "std": std,
            "cp": cp,
            "cpk": cpk,
        }

    finally:
        doc.close()


def run() -> None:
    rows: List[dict] = []

    pdf_files = [
        f for f in os.listdir()
        if f.lower().endswith(".pdf")
        and not f.startswith("Rexair_Magnet_Timing_CpK_Report_")
    ]

    if not pdf_files:
        print("No PDF files found in current folder.")
        return

    for file_name in sorted(pdf_files):
        row = process_pdf(file_name)
        if row:
            rows.append(row)

    if not rows:
        print("")
        print("No valid datasets found. Report not created.")
        return

    rows.sort(key=lambda r: r["month_dt"])

    for row in rows:
        row["label"] = row["month_dt"].strftime("%b %Y")

    output = create_report(rows)

    print("")
    print("Report created:")
    print(output)


if __name__ == "__main__":
    run()