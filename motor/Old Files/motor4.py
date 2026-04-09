# ============================================================
# Rexair Motor & Magnet Analysis – CLEAN SCRIPT v14
# ============================================================

import os
import re
import platform
import subprocess
from datetime import datetime
import threading
import queue
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpdf import FPDF
import tkinter as tk
from tkinter import ttk

import fitz


REPORT_FOLDER_MOTOR="Motor Reports"
REPORT_FOLDER_MAGNET="Magnet Reports"

MAGNET_TARGET=3.7
MAGNET_TOL=0.5
LSL=MAGNET_TARGET-MAGNET_TOL
USL=MAGNET_TARGET+MAGNET_TOL

os.makedirs(REPORT_FOLDER_MOTOR,exist_ok=True)
os.makedirs(REPORT_FOLDER_MAGNET,exist_ok=True)

print("RUNNING CLEAN SCRIPT v14")


# ============================================================
# UTILITIES
# ============================================================

def open_file(filepath):

    try:
        if platform.system()=="Windows":
            os.startfile(filepath)
        elif platform.system()=="Darwin":
            subprocess.run(["open",filepath])
        else:
            subprocess.run(["xdg-open",filepath])
    except:
        pass


def versioned_filename(folder,name):

    path=os.path.join(folder,name)

    if not os.path.exists(path):
        return path

    base,ext=os.path.splitext(name)

    i=1
    while True:

        new=f"{base}_v{i}{ext}"
        p=os.path.join(folder,new)

        if not os.path.exists(p):
            return p

        i+=1


def compute_cp_cpk(mean,std):

    if std<=0:
        return 0,0

    cp=(USL-LSL)/(6*std)
    cpk=min((mean-LSL)/(3*std),(USL-mean)/(3*std))

    return cp,cpk


def sanitize(text):

    text=unicodedata.normalize("NFKD",str(text))
    text=text.replace("（","(").replace("）",")")
    return text.encode("latin-1","ignore").decode("latin-1")


# ============================================================
# MOTOR SECTION
# ============================================================

def extract_date_from_filename(fname):

    m=re.search(r"C6521(\d{6})",fname)

    if not m:
        return None

    return datetime.strptime(m.group(1),"%y%m%d")


def read_input_power(path):

    try:
        df=pd.read_excel(path,header=None)
    except:
        return []

    vals=[]

    for v in df.values.flatten():

        try:

            x=float(v)

            if 500<x<900:
                vals.append(x)

        except:
            pass

    return vals


def create_motor_pdf(out,months_dt,month_values):

    labels=[m.strftime("%b %Y") for m in months_dt]

    plt.figure(figsize=(11,5))
    plt.boxplot(month_values,tick_labels=labels,showfliers=False)
    plt.xticks(rotation=45)
    plt.title("Motor Input Power – Monthly Distribution")
    plt.tight_layout()

    box="motor_box.png"
    plt.savefig(box,dpi=300)
    plt.close()

    stds=[np.std(v,ddof=1) if len(v)>=2 else 0 for v in month_values]

    plt.figure(figsize=(11,4))
    plt.plot(range(1,len(stds)+1),stds,marker="o")
    plt.xticks(range(1,len(labels)+1),labels,rotation=45)
    plt.title("Motor Input Power – Std Deviation Trend")
    plt.tight_layout()

    stdimg="motor_std.png"
    plt.savefig(stdimg,dpi=300)
    plt.close()

    pdf=FPDF()

    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"Rexair Motor Test Trend Analysis",ln=True)
    pdf.image(box,x=10,y=30,w=190)

    pdf.add_page()
    pdf.image(stdimg,x=10,y=30,w=190)

    pdf.output(out)

    os.remove(box)
    os.remove(stdimg)


def run_motor_analysis(q=None,cancel_flag=None):

    base=os.getcwd()

    files=[f for f in os.listdir(base)
           if f.lower().endswith((".xlsx",".xls"))
           and "motor test data" in f.lower()]

    dated=[]

    for f in files:

        d=extract_date_from_filename(f)

        if d:
            dated.append((d,f))

    dated.sort(key=lambda x:x[0])

    monthly={}

    for d,f in dated:

        vals=read_input_power(os.path.join(base,f))

        if vals:
            monthly.setdefault(d.strftime("%Y-%m"),[]).extend(vals)

    if not monthly:
        print("No motor data found")
        return

    months=sorted(monthly.keys())
    months_dt=[datetime.strptime(m,"%Y-%m") for m in months]

    if len(months_dt)>12:
        months_dt=months_dt[-12:]

    month_values=[monthly[m.strftime("%Y-%m")] for m in months_dt]

    name=f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}.pdf"
    out=versioned_filename(REPORT_FOLDER_MOTOR,name)

    create_motor_pdf(out,months_dt,month_values)

    print("Motor report created:",out)
    open_file(out)


# ============================================================
# MAGNET SECTION
# ============================================================

def is_magnet_pdf(filepath):

    try:

        doc=fitz.open(filepath)

        text=""

        for page in doc:
            text+=page.get_text()

            if len(text)>2000:
                break

        doc.close()

        text=text.lower()

        keywords=["magnet","timing","cpk"]

        return any(k in text for k in keywords)

    except:
        return False


def extract_magnet_values(pdf):

    values=[]

    doc=fitz.open(pdf)

    try:

        for page in doc:

            text=page.get_text()

            matches=re.findall(r"3\.\d+",text)

            for m in matches:

                try:

                    v=float(m)

                    if 3.4<=v<=4.0:
                        values.append(v)

                except:
                    pass

    finally:

        doc.close()

    if len(values)>=10:

        print(f"[MAGNET] {os.path.basename(pdf)} -> {len(values)} values")

        return values

    return []


def create_magnet_pdf(output_path,results):

    labels=[]
    cpks=[]

    for i,r in enumerate(results):

        name=os.path.splitext(r["file"])[0]

        labels.append(sanitize(name))
        cpks.append(r["cpk"])

    plt.figure(figsize=(12,6))
    plt.plot(labels,cpks,marker="o",linewidth=2)

    plt.axhline(1.33,linestyle="--")
    plt.ylim(0,4)

    plt.ylabel("Cpk")
    plt.xlabel("File within Month")

    plt.title("Magnet Timing CpK Over Time (Per File)")

    plt.xticks(rotation=45)

    if min(cpks)>=1.33:
        status="CpK is Good"
        color="green"
    else:
        status="CpK is BAD"
        color="red"

    plt.text(len(labels)/2,2,status,fontsize=26,color=color,ha="center")

    plt.grid(True)

    trend_img="magnet_cpk_trend.png"

    plt.tight_layout()
    plt.savefig(trend_img,dpi=300)
    plt.close()

    all_values=[]

    for r in results:
        all_values.extend(r["values"])

    arr=np.array(all_values)

    plt.figure(figsize=(10,5))
    plt.hist(arr,bins=12)
    plt.axvline(MAGNET_TARGET,linestyle="--")
    plt.title("Magnet Timing Histogram")

    hist_img="magnet_hist.png"

    plt.tight_layout()
    plt.savefig(hist_img,dpi=300)
    plt.close()

    plt.figure(figsize=(10,4))
    plt.boxplot(arr,vert=False)
    plt.axvline(MAGNET_TARGET)
    plt.title("Magnet Timing Distribution")

    box_img="magnet_box.png"

    plt.tight_layout()
    plt.savefig(box_img,dpi=300)
    plt.close()

    pdf=FPDF()

    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"Rexair Magnet Timing CpK Report",ln=True)
    pdf.image(trend_img,x=10,y=40,w=180)

    pdf.add_page()
    pdf.image(hist_img,x=10,y=30,w=180)

    pdf.add_page()
    pdf.image(box_img,x=10,y=30,w=180)

    pdf.add_page()

    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"Per-File Summary Table",ln=True)

    pdf.set_font("Arial","",11)

    for i,r in enumerate(results):

        cp=(USL-LSL)/(6*r["std"])

        line=f"{labels[i]} Mean: {r['mean']:.4f} Std: {r['std']:.4f} Cp: {cp:.2f} CpK: {r['cpk']:.2f}"

        pdf.cell(0,8,sanitize(line),ln=True)

    pdf.output(output_path)

    os.remove(trend_img)
    os.remove(hist_img)
    os.remove(box_img)


def run_magnet_analysis(q=None,cancel_flag=None):

    base=os.getcwd()

    pdfs=[]

    for f in os.listdir(base):

        if not f.lower().endswith(".pdf"):
            continue

        if "rexair_magnet_timing_cpk_report" in f.lower():
            continue

        full=os.path.join(base,f)

        if is_magnet_pdf(full):
            pdfs.append(f)

    print("Magnet PDFs detected:",pdfs)

    results=[]

    for f in pdfs:

        vals=extract_magnet_values(os.path.join(base,f))

        if len(vals)<10:
            continue

        arr=np.array(vals)

        mean=float(arr.mean())
        std=float(arr.std(ddof=1))

        cp,cpk=compute_cp_cpk(mean,std)

        results.append({
            "file":f,
            "values":vals,
            "mean":mean,
            "std":std,
            "cpk":cpk
        })

    if not results:

        print("No magnet data found")
        return

    name=f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"

    out=versioned_filename(REPORT_FOLDER_MAGNET,name)

    create_magnet_pdf(out,results)

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

    ttk.Label(root,text="Motor & Magnet Analysis",
              font=("Segoe UI",14,"bold")).pack(pady=15)

    ttk.Button(root,text="Run Motor Analysis",width=40,
               command=lambda:start_threaded([run_motor_analysis])).pack(pady=8)

    ttk.Button(root,text="Run Magnet Analysis",width=40,
               command=lambda:start_threaded([run_magnet_analysis])).pack(pady=8)

    ttk.Button(root,text="Run Both",width=40,
               command=lambda:start_threaded([run_motor_analysis,
                                              run_magnet_analysis])).pack(pady=8)

    root.mainloop()


if __name__=="__main__":

    launch_gui()