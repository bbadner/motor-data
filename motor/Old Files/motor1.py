# ============================================================
# Rexair Motor & Magnet Analysis – FINAL SCRIPT v11
# Restores full reports while keeping improved OCR extraction
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

print("RUNNING FINAL SCRIPT v11")

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


def compute_cp_cpk(mean, std):
    if std <= 0:
        return 0, 0
    cp = (USL - LSL) / (6 * std)
    cpk = min((mean - LSL) / (3 * std), (USL - mean) / (3 * std))
    return cp, cpk


# ============================================================
# MOTOR SECTION
# ============================================================

def extract_date_from_filename(fname):
    m = re.search(r"C6521(\d{6})", fname)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%y%m%d")


def find_input_power_column(df):
    for i in range(min(15, len(df))):
        row = df.iloc[i].astype(str).tolist()
        for j, cell in enumerate(row):
            if "input power" in cell.lower():
                return j
    return None


def read_input_power(filepath):

    try:
        df = pd.read_excel(filepath, header=None)
    except:
        return []

    col_idx = find_input_power_column(df)
    if col_idx is None:
        return []

    values = []
    for v in df.iloc[:, col_idx]:
        try:
            values.append(float(v))
        except:
            pass

    return values


def run_motor_analysis(q=None, cancel_flag=None):

    base = os.getcwd()

    files = [
        f for f in os.listdir(base)
        if f.lower().endswith((".xlsx", ".xls"))
        and "motor test data" in f.lower()
    ]

    dated = []

    for f in files:
        try:
            dated.append((extract_date_from_filename(f), f))
        except:
            pass

    dated.sort(key=lambda x: x[0])

    monthly = {}

    for d, f in dated:

        vals = read_input_power(os.path.join(base, f))

        if vals:
            monthly.setdefault(d.strftime("%Y-%m"), []).extend(vals)

    if not monthly:
        print("❌ No motor data found.")
        return

    months = sorted(monthly.keys())
    data = [monthly[m] for m in months]

    # ----------------
    # Create plots
    # ----------------

    plt.figure(figsize=(8,4))
    plt.boxplot(data)
    plt.xticks(range(1,len(months)+1),months,rotation=45)
    plt.title("Monthly Motor Input Power Distribution")
    plt.tight_layout()
    boxplot="motor_box.png"
    plt.savefig(boxplot,dpi=300)
    plt.close()

    all_values = np.concatenate(data)

    plt.figure(figsize=(6,4))
    plt.hist(all_values,bins=30)
    plt.title("Motor Input Power Histogram")
    plt.tight_layout()
    hist="motor_hist.png"
    plt.savefig(hist,dpi=300)
    plt.close()

    # ----------------
    # PDF
    # ----------------

    name=f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}.pdf"
    out=versioned_filename(REPORT_FOLDER_MOTOR,name)

    pdf=FPDF()

    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"Rexair Motor Test Trend Analysis",ln=True)

    pdf.image(boxplot,x=10,y=30,w=190)

    pdf.add_page()
    pdf.image(hist,x=20,y=40,w=160)

    pdf.output(out)

    os.remove(boxplot)
    os.remove(hist)

    print("Motor report created:",out)
    open_file(out)

# ============================================================
# MAGNET OCR
# ============================================================

def render_page(doc,index,dpi=450):

    page=doc.load_page(index)
    pix=page.get_pixmap(dpi=dpi)

    return Image.frombytes("RGB",[pix.width,pix.height],pix.samples).convert("L")


def extract_3p7_values(text):

    text=text.replace(",",".")
    vals=[]

    for m in re.finditer(r"\b([34]\.[0-9]{1,3})\b",text):
        try:
            v=float(m.group(1))
            if 3.0<=v<=4.5:
                vals.append(v)
        except:
            pass

    return vals

def extract_magnet_values(pdf_path):

    if not _HAS_TESSERACT:
        return []

    doc = fitz.open(pdf_path)

    try:
        for i in range(doc.page_count):

            img = render_page(doc, i)

            best_vals = []

            for angle in [0, 90, 180, 270]:

                rotated = img.rotate(angle, expand=True)

                w, h = rotated.size
                cropped = rotated.crop(
                    (int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95))
                )

                strip_width = cropped.size[0] // 6

                values = []

                for s in range(6):

                    left = s * strip_width
                    right = (s + 1) * strip_width if s < 5 else cropped.size[0]

                    strip = cropped.crop((left, 0, right, cropped.size[1]))

                    proc = ImageOps.autocontrast(strip)
                    proc = ImageEnhance.Contrast(proc).enhance(3)

                    txt = pytesseract.image_to_string(
                        proc,
                        config="--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789.-"
                    )

                    values.extend(extract_3p7_values(txt))

                if len(values) > len(best_vals):
                    best_vals = values

            # Only accept pages that contain real timing data
            if len(best_vals) >= 20:
                print(f"[MAGNET] Data page detected in {os.path.basename(pdf_path)}")
                return best_vals[:30]

        # If no page contains enough timing values → skip file
        print(f"[MAGNET] SKIP (no data page): {os.path.basename(pdf_path)}")
        return []

    finally:
        doc.close()



def run_magnet_analysis(q=None,cancel_flag=None):

    base=os.getcwd()

    pdfs=[f for f in os.listdir(base) if "magnet ring" in f.lower()]

    results=[]

    for f in pdfs:

        vals=extract_magnet_values(os.path.join(base,f))

        if len(vals)<20:
            continue

        arr=np.array(vals)

        mean=float(arr.mean())
        std=float(arr.std(ddof=1))

        cp,cpk=compute_cp_cpk(mean,std)

        results.append((f,mean,std,cpk))

    if not results:
        print("No magnet data found")
        return

    all_vals=np.concatenate([extract_magnet_values(os.path.join(base,f)) for f,_,_,_ in results])

    plt.figure(figsize=(6,4))
    plt.hist(all_vals,bins=20)
    plt.title("Magnet Timing Distribution")
    hist="mag_hist.png"
    plt.savefig(hist,dpi=300)
    plt.close()

    name=f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    out=versioned_filename(REPORT_FOLDER_MAGNET,name)

    pdf=FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"Rexair Magnet Timing CpK Report",ln=True)

    pdf.image(hist,x=20,y=40,w=160)

    pdf.add_page()
    pdf.set_font("Arial","",11)

    for r in results:
        pdf.cell(0,8,f"{r[0]}  Mean={r[1]:.4f}  Std={r[2]:.4f}  CpK={r[3]:.2f}",ln=True)

    pdf.output(out)

    os.remove(hist)

    print("Magnet report created:",out)
    open_file(out)

# ============================================================
# GUI
# ============================================================

def start_threaded(funcs):

    root=tk.Tk()
    root.title("Processing")
    root.geometry("400x120")

    ttk.Label(root,text="Processing...").pack(pady=10)

    bar=ttk.Progressbar(root,mode="indeterminate")
    bar.pack(fill="x",padx=20)
    bar.start()

    q=queue.Queue()

    def worker():
        for f in funcs:
            f(q,{})
        q.put("done")

    def check():
        try:
            if q.get_nowait()=="done":
                bar.stop()
                root.destroy()
                return
        except:
            pass
        root.after(200,check)

    root.after(200,check)

    threading.Thread(target=worker,daemon=True).start()

    root.mainloop()


def launch_gui():

    root=tk.Tk()
    root.title("Motor & Magnet Analysis")
    root.geometry("500x280")

    ttk.Label(root,text="Motor & Magnet Analysis",font=("Segoe UI",14,"bold")).pack(pady=15)

    ttk.Button(root,text="Run Motor Analysis",width=40,
               command=lambda:start_threaded([run_motor_analysis])).pack(pady=8)

    ttk.Button(root,text="Run Magnet Analysis",width=40,
               command=lambda:start_threaded([run_magnet_analysis])).pack(pady=8)

    ttk.Button(root,text="Run Both",width=40,
               command=lambda:start_threaded([run_motor_analysis,run_magnet_analysis])).pack(pady=8)

    root.mainloop()


if __name__=="__main__":
    launch_gui()