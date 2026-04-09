import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import statistics

# === CONFIGURATION ===
LOGO_FILE = "rexair_logo.png"
REPORT_FOLDER = "Motor Reports"
YMIN, YMAX = 700, 1100
OUTLIER_THRESHOLD = 0.10  # ±10%

# === POPUP ===
def show_processing_popup():
    popup = tk.Tk()
    popup.title("Rexair Motor Test Trend Analysis")
    popup.geometry("320x100")
    popup.eval('tk::PlaceWindow . center')
    label = ttk.Label(popup, text="Processing motor test data...", font=("Arial", 11))
    label.pack(expand=True)
    popup.update()
    return popup

# === FIND INPUT POWER COLUMN ===
def find_input_power_column(df):
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).str.lower()
        for idx, val in enumerate(row):
            if "input" in val and "power" in val:
                return idx
    return None

# === READ EXCEL FILE ===
def read_input_power(filepath):
    try:
        excel = pd.ExcelFile(filepath)
        for sheet in excel.sheet_names:
            df = excel.parse(sheet, header=None)
            col = find_input_power_column(df)
            if col is not None:
                values = pd.to_numeric(df.iloc[10:, col], errors="coerce").dropna()
                if len(values) > 0:
                    return values.tolist()
        return []
    except Exception:
        return []

# === PARSE DATE FROM FILENAME ===
def extract_date_from_filename(fname):
    m = re.search(r"IN(\d{6})", fname)
    if m:
        try:
            return datetime.strptime(m.group(1), "%y%m%d")
        except Exception:
            return None
    return None

# === DETECT OUTLIERS ===
def detect_outliers(months, means):
    outliers = []
    for i in range(1, len(means)):
        if means[i - 1] != 0:
            diff = (means[i] - means[i - 1]) / means[i - 1]
            if abs(diff) >= OUTLIER_THRESHOLD:
                outliers.append((months[i], diff))
    return outliers

# === CREATE PDF REPORT ===
def create_pdf_report(output_path, logo_path, month_labels, month_values, month_means, month_stds, outliers):
    pdf = FPDF()
    pdf.add_page()

    # HEADER
    if os.path.exists(logo_path):
        pdf.image(logo_path, 10, 8, 33)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(80)
    pdf.cell(30, 10, "Rexair Motor Test Trend Analysis", ln=1, align="C")

    pdf.set_font("Arial", "", 11)
    start, end = month_labels[0].strftime("%b %Y"), month_labels[-1].strftime("%b %Y")
    pdf.cell(0, 10, f"Period: {start} - {end}", ln=1, align="C")

    # CHARTS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    plt.subplots_adjust(hspace=0.4)

    positions = list(range(1, len(month_labels) + 1))

    # BOX & WHISKER
    ax1.boxplot(month_values, positions=positions, patch_artist=True)
    ax1.set_title("Monthly Input Power Distribution (W)")
    ax1.set_ylabel("Watts")
    ax1.set_ylim(YMIN, YMAX)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.plot(positions, month_means, color="blue", marker="o", label="Mean (W)")

    for (m, diff) in outliers:
        idx = month_labels.index(m)
        val = month_means[idx]
        ax1.plot(idx + 1, val, "ro")
        ax1.text(idx + 1, val + 10, f"{'+' if diff > 0 else ''}{diff*100:.0f}%", color="red",
                 ha="center", fontsize=8, fontweight="bold")

    ax1.legend()

    # STD DEV LINE
    ax2.plot(positions, month_stds, color="orange", marker="s", label="Std Dev (W)")
    ax2.set_title("Monthly Standard Deviation (W)")
    ax2.set_ylabel("Std Dev (W)")
    ax2.set_xlabel("Month")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    ax2.set_xticks(positions)
    ax2.set_xticklabels([d.strftime("%b %Y") for d in month_labels], rotation=45, ha="right")

    chart_path = "chart_temp.png"
    plt.tight_layout()
    plt.savefig(chart_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    pdf.image(chart_path, x=10, y=50, w=190)
    os.remove(chart_path)

    # TABLE
    pdf.set_y(180)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Monthly Summary", ln=1, align="C")

    col_widths = [60, 60, 60]
    headers = ["Month", "Mean (W)", "Std Dev (W)"]
    pdf.set_font("Arial", "B", 10)

    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    fill = False

    for i, month in enumerate(month_labels):
        mean = month_means[i]
        std = month_stds[i]
        if i > 0 and month_means[i - 1] != 0:
            diff = (mean - month_means[i - 1]) / month_means[i - 1]
        else:
            diff = 0

        if fill:
            pdf.set_fill_color(240, 240, 240)
        else:
            pdf.set_fill_color(255, 255, 255)
        fill = not fill

        if abs(diff) >= OUTLIER_THRESHOLD:
            pdf.set_text_color(255, 0, 0)
        else:
            pdf.set_text_color(0, 0, 0)

        pdf.cell(col_widths[0], 8, month.strftime("%b %Y"), border=1, align="C", fill=True)
        pdf.cell(col_widths[1], 8, f"{mean:.1f}", border=1, align="C", fill=True)
        pdf.cell(col_widths[2], 8, f"{std:.1f}", border=1, align="C", fill=True)
        pdf.ln()

    pdf.set_text_color(0, 0, 0)
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}", 0, 0, "C")

    pdf.output(output_path, "F")

# === MAIN ===
def main():
    base_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    report_dir = os.path.join(base_dir, REPORT_FOLDER)
    os.makedirs(report_dir, exist_ok=True)

    popup = show_processing_popup()
    result = {"success": False, "report": None, "error": None}

    def run_analysis():
        try:
            excel_files = [f for f in os.listdir(base_dir) if f.lower().endswith(".xlsx")]
            if not excel_files:
                raise ValueError("No Excel (.xlsx) files found in this folder.")

            # --- NEW LOGIC: pick the last 12 unique months found ---
            dated_files = []
            for fname in excel_files:
                fdate = extract_date_from_filename(fname)
                if fdate:
                    dated_files.append((fdate, fname))
            if not dated_files:
                raise ValueError("No valid dates found in filenames.")

            dated_files.sort(key=lambda x: x[0])
            all_months = sorted({d.strftime("%Y-%m") for d, _ in dated_files})
            selected_months = all_months[-12:] if len(all_months) > 12 else all_months

            monthly_data = {}
            for fdate, fname in dated_files:
                month = fdate.strftime("%Y-%m")
                if month not in selected_months:
                    continue
                values = read_input_power(os.path.join(base_dir, fname))
                if len(values) == 0:
                    continue
                monthly_data.setdefault(month, []).extend(values)

            if not monthly_data:
                raise ValueError("No valid 'Input Power' data found in the last 12 months of files.")

            sorted_months = sorted(monthly_data.keys())
            month_labels = [datetime.strptime(m, "%Y-%m") for m in sorted_months]
            month_values = [monthly_data[m] for m in sorted_months]
            month_means = [statistics.mean(v) if len(v) > 0 else 0 for v in month_values]
            month_stds = [statistics.stdev(v) if len(v) > 1 else 0 for v in month_values]
            outliers = detect_outliers(month_labels, month_means)

            report_name = f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}.pdf"
            report_path = os.path.join(report_dir, report_name)
            create_pdf_report(report_path, os.path.join(base_dir, LOGO_FILE),
                              month_labels, month_values, month_means, month_stds, outliers)
            result["success"] = True
            result["report"] = report_path
        except Exception as e:
            result["error"] = str(e)
        finally:
            popup.quit()

    thread = threading.Thread(target=run_analysis)
    thread.start()
    popup.mainloop()
    popup.destroy()

    msg_root = tk.Tk()
    msg_root.withdraw()
    if result["success"]:
        messagebox.showinfo("Rexair Motor Test Trend Analysis", f"✅ Report created:\n\n{result['report']}")
        os.startfile(result["report"])
    else:
        messagebox.showerror("Rexair Motor Test Trend Analysis", f"❌ Error:\n{result['error']}")
    msg_root.destroy()


if __name__ == "__main__":
    main()
