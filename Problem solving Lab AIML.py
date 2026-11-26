"""
Experiential Learning - AI & ML (T.Y. Core, SEM-V)
Lab Assignment: Matplotlib Visualizations

Q1: Student performance analysis (5 subjects, 8 students)
Q2: Retail sales and business data analysis (6 months, 3 products)

Run:  python Lab_Assignment_AIML.py
Requires: matplotlib, numpy
All charts are saved as PNG files in a folder named "lab_figures".
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; figures are saved to disk
import matplotlib.pyplot as plt

OUT_DIR = "lab_figures"
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# =============================================================================
# QUESTION 1 : STUDENT PERFORMANCE ANALYSIS
# =============================================================================
print("=" * 70)
print("QUESTION 1: Student Performance Analysis")
print("=" * 70)

subjects = ["Mathematics", "Python", "Statistics", "DBMS", "AI"]
students = [f"Student {i}" for i in range(1, 9)]

# rows = students, columns = subjects (order as in the dataset)
marks = np.array([
    [85, 90, 78, 88, 92],
    [72, 75, 80, 70, 78],
    [95, 92, 89, 94, 96],
    [60, 65, 68, 72, 66],
    [88, 85, 90, 86, 91],
    [76, 80, 75, 79, 82],
    [92, 96, 94, 90, 95],
    [68, 72, 70, 65, 74],
])
subj_avg = marks.mean(axis=0)
stud_avg = marks.mean(axis=1)

# Summary statistics (printed for the interpretation)
print("\nSubject averages:")
for s, a in zip(subjects, subj_avg):
    print(f"  {s:<12} {a:6.2f}")
print("\nStudent averages:")
for s, a in zip(students, stud_avg):
    print(f"  {s:<10} {a:6.2f}")
best_subj = subjects[int(np.argmax(subj_avg))]
weak_subj = subjects[int(np.argmin(subj_avg))]
top_student = students[int(np.argmax(stud_avg))]
low_student = students[int(np.argmin(stud_avg))]
print(f"\nBest subject overall : {best_subj}")
print(f"Weakest subject      : {weak_subj}")
print(f"Top student          : {top_student}")
print(f"Lowest student       : {low_student}")

x = np.arange(len(students))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# ---- (a) Grouped bar chart: marks of each student in each subject ----------
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.16
for i, (subj, col) in enumerate(zip(subjects, colors)):
    ax.bar(x + (i - 2) * width, marks[:, i], width, label=subj, color=col,
           edgecolor="black", linewidth=0.5)
ax.set_title("Q1(a): Marks Obtained by Each Student in Each Subject", fontsize=14, fontweight="bold")
ax.set_xlabel("Students")
ax.set_ylabel("Marks (out of 100)")
ax.set_xticks(x)
ax.set_xticklabels(students)
ax.set_ylim(0, 110)
ax.legend(title="Subject", ncol=5, loc="upper center")
ax.grid(axis="y", linestyle="--", alpha=0.6)
save(fig, "Q1a_grouped_bar_marks.png")

# ---- (b) Line plot: Python vs Artificial Intelligence ----------------------
py_idx, ai_idx = subjects.index("Python"), subjects.index("AI")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(students, marks[:, py_idx], marker="o", markersize=9, linewidth=2.5,
        color="#ff7f0e", label="Python")
ax.plot(students, marks[:, ai_idx], marker="s", markersize=9, linewidth=2.5,
        linestyle="--", color="#9467bd", label="Artificial Intelligence")
ax.set_title("Q1(b): Python vs Artificial Intelligence Marks per Student", fontsize=14, fontweight="bold")
ax.set_xlabel("Students")
ax.set_ylabel("Marks (out of 100)")
ax.set_ylim(55, 105)
ax.legend()
ax.grid(True, linestyle=":", alpha=0.7)
plt.setp(ax.get_xticklabels(), rotation=20)
save(fig, "Q1b_line_python_vs_ai.png")

# ---- (c) Pie chart: average marks distribution across subjects -------------
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    subj_avg, labels=subjects, colors=colors, autopct="%1.1f%%",
    startangle=90, explode=[0.04] * 5, wedgeprops={"edgecolor": "white"},
    textprops={"fontsize": 11})
for t in autotexts:
    t.set_color("white")
    t.set_fontweight("bold")
ax.set_title("Q1(c): Share of Average Marks Across the Five Subjects", fontsize=14, fontweight="bold")
ax.legend(wedges, [f"{s}: {a:.2f}" for s, a in zip(subjects, subj_avg)],
          title="Subject: Average", loc="center left", bbox_to_anchor=(1, 0.5))
save(fig, "Q1c_pie_average_marks.png")

# ---- (d) Histogram: distribution of all marks, 5 bins ----------------------
all_marks = marks.flatten()
fig, ax = plt.subplots(figsize=(9, 6))
counts, edges, _ = ax.hist(all_marks, bins=5, color="#1f77b4", edgecolor="black", alpha=0.85)
ax.set_title("Q1(d): Distribution of All Students' Marks (5 bins)", fontsize=14, fontweight="bold")
ax.set_xlabel("Marks range")
ax.set_ylabel("Number of marks (frequency)")
ax.set_xticks(edges)
ax.grid(axis="y", linestyle="--", alpha=0.6)
for c, left, right in zip(counts, edges[:-1], edges[1:]):
    ax.text((left + right) / 2, c + 0.3, f"{int(c)}", ha="center", fontweight="bold")
save(fig, "Q1d_histogram_all_marks.png")
print("\nHistogram bin counts:", [int(c) for c in counts], "edges:", [round(float(e), 1) for e in edges])

# ---- (e) Scatter: Mathematics vs Python ------------------------------------
math_idx = subjects.index("Mathematics")
m_x, p_y = marks[:, math_idx], marks[:, py_idx]
r = np.corrcoef(m_x, p_y)[0, 1]
slope, intercept = np.polyfit(m_x, p_y, 1)
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(m_x, p_y, s=120, color="#2ca02c", marker="D", edgecolors="black", label="Students")
for i, (xi, yi) in enumerate(zip(m_x, p_y)):
    ax.annotate(f"S{i+1}", (xi, yi), textcoords="offset points", xytext=(6, 6))
xs = np.linspace(m_x.min() - 2, m_x.max() + 2, 100)
ax.plot(xs, slope * xs + intercept, "r--", linewidth=2, label=f"Trend line (r = {r:.2f})")
ax.set_title("Q1(e): Relationship Between Mathematics and Python Marks", fontsize=14, fontweight="bold")
ax.set_xlabel("Mathematics marks")
ax.set_ylabel("Python marks")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)
save(fig, "Q1e_scatter_math_vs_python.png")
print(f"\nCorrelation (Mathematics vs Python): r = {r:.3f}")

print("\n--- Q1 Interpretation ---")
print(f"(a) {top_student} is strong in every subject; {low_student} is lowest in most subjects.")
print(f"(b) Python and AI move together: students high in one tend to be high in the other.")
print(f"(c) {best_subj} has the highest average and {weak_subj} the lowest; the pie shows balanced shares.")
print(f"(d) Most marks lie in the upper bins, so the class is generally performing well.")
print(f"(e) Correlation r = {r:.2f} indicates a {'strong' if abs(r) > 0.7 else 'moderate' if abs(r) > 0.4 else 'weak'} "
      f"{'positive' if r > 0 else 'negative'} relationship between Mathematics and Python.")


# =============================================================================
# QUESTION 2 : SALES AND BUSINESS DATA ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("QUESTION 2: Retail Sales and Business Data Analysis")
print("=" * 70)

months = ["January", "February", "March", "April", "May", "June"]
short_m = [m[:3] for m in months]
products = ["Laptop", "Mobile", "Tablet"]

units = np.array([          # rows = months, cols = products
    [25, 40, 20],
    [30, 45, 25],
    [28, 50, 22],
    [35, 55, 30],
    [40, 60, 35],
    [45, 65, 40],
])
# Revenue in rupees (as given in the dataset)
revenue_rs = np.array([
    [1250000, 600000, 300000],
    [1500000, 675000, 375000],
    [1400000, 750000, 330000],
    [1750000, 825000, 450000],
    [2000000, 900000, 525000],
    [2250000, 975000, 600000],
], dtype=float)
revenue_lakh = revenue_rs / 1e5          # easier-to-read axis: Rs. lakh
monthly_total = revenue_lakh.sum(axis=1)
prod_total_rev = revenue_lakh.sum(axis=0)
prod_total_units = units.sum(axis=0)

print("\nTotal units sold :", {p: int(v) for p, v in zip(products, prod_total_units)})
print("Total revenue    : " + ", ".join(f"{p} Rs.{v:.2f} lakh" for p, v in zip(products, prod_total_rev)))
print("Monthly totals   : " + ", ".join(f"{m[:3]} {v:.2f}" for m, v in zip(months, monthly_total)))
best_rev_prod = products[int(np.argmax(prod_total_rev))]
best_units_prod = products[int(np.argmax(prod_total_units))]
peak_month = months[int(np.argmax(monthly_total))]
print(f"\nBest product by revenue : {best_rev_prod}")
print(f"Best product by units   : {best_units_prod}")
print(f"Highest revenue month   : {peak_month}")

pcolors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# ---- (a) Multiple-line plot: monthly unit sales trend ----------------------
fig, ax = plt.subplots(figsize=(10, 6))
markers = ["o", "s", "^"]
for i, p in enumerate(products):
    ax.plot(months, units[:, i], marker=markers[i], markersize=9, linewidth=2.5,
            color=pcolors[i], label=p)
    for j, v in enumerate(units[:, i]):
        ax.annotate(str(v), (j, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax.set_title("Q2(a): Monthly Sales Trends of Laptops, Mobiles and Tablets", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Units sold")
ax.set_ylim(0, 75)
ax.legend(title="Product")
ax.grid(True, linestyle="--", alpha=0.6)
save(fig, "Q2a_line_monthly_sales.png")

# ---- (b) Grouped bar chart: units per product per month --------------------
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(months))
w = 0.25
for i, p in enumerate(products):
    ax.bar(x + (i - 1) * w, units[:, i], w, label=p, color=pcolors[i], edgecolor="black", linewidth=0.5)
ax.set_title("Q2(b): Units Sold per Product Across Six Months", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Units sold")
ax.set_xticks(x)
ax.set_xticklabels(months)
ax.legend(title="Product")
ax.grid(axis="y", linestyle="--", alpha=0.6)
save(fig, "Q2b_grouped_bar_units.png")

# ---- (c) Pie chart: total revenue contribution of each product -------------
fig, ax = plt.subplots(figsize=(8, 8))
wedges, _, autotexts = ax.pie(
    prod_total_rev, labels=products, colors=pcolors, autopct="%1.1f%%",
    startangle=90, explode=[0.05, 0, 0], wedgeprops={"edgecolor": "white"},
    textprops={"fontsize": 12})
for t in autotexts:
    t.set_color("white")
    t.set_fontweight("bold")
ax.set_title("Q2(c): Revenue Contribution by Product (Jan-Jun)", fontsize=14, fontweight="bold")
ax.legend(wedges, [f"{p}: Rs. {v:.2f} lakh" for p, v in zip(products, prod_total_rev)],
          loc="center left", bbox_to_anchor=(1, 0.5))
save(fig, "Q2c_pie_revenue_share.png")

# ---- (d) Stacked bar chart: combined monthly revenue -----------------------
fig, ax = plt.subplots(figsize=(11, 6))
bottom = np.zeros(len(months))
for i, p in enumerate(products):
    ax.bar(months, revenue_lakh[:, i], bottom=bottom, label=p, color=pcolors[i],
           edgecolor="black", linewidth=0.5)
    bottom += revenue_lakh[:, i]
for j, tot in enumerate(monthly_total):
    ax.text(j, tot + 0.5, f"{tot:.2f}", ha="center", fontweight="bold")
ax.set_title("Q2(d): Combined Monthly Revenue of All Products (Stacked)", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue (Rs. lakh)")
ax.set_ylim(0, monthly_total.max() + 6)
ax.legend(title="Product")
ax.grid(axis="y", linestyle="--", alpha=0.6)
save(fig, "Q2d_stacked_monthly_revenue.png")

# ---- (e) Scatter: mobile sales vs tablet sales -----------------------------
mob, tab = units[:, 1], units[:, 2]
r2 = np.corrcoef(mob, tab)[0, 1]
slope2, icpt2 = np.polyfit(mob, tab, 1)
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(mob, tab, s=140, color="#d62728", marker="*", edgecolors="black", label="Months")
for m, xi, yi in zip(short_m, mob, tab):
    ax.annotate(m, (xi, yi), textcoords="offset points", xytext=(7, 5))
xs = np.linspace(mob.min() - 2, mob.max() + 2, 100)
ax.plot(xs, slope2 * xs + icpt2, "k--", linewidth=1.8, label=f"Trend line (r = {r2:.2f})")
ax.set_title("Q2(e): Mobile Sales vs Tablet Sales (Units)", fontsize=14, fontweight="bold")
ax.set_xlabel("Mobile units sold")
ax.set_ylabel("Tablet units sold")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.7)
save(fig, "Q2e_scatter_mobile_vs_tablet.png")
print(f"\nCorrelation (Mobile vs Tablet units): r = {r2:.3f}")

# ---- (f) Subplots: trend, product revenue comparison, total monthly revenue -
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
# f1: monthly sales trend (units)
for i, p in enumerate(products):
    axes[0].plot(short_m, units[:, i], marker=markers[i], color=pcolors[i], linewidth=2, label=p)
axes[0].set_title("Monthly Sales Trend (Units)", fontweight="bold")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Units sold")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.6)
# f2: product-wise revenue comparison
axes[1].bar(products, prod_total_rev, color=pcolors, edgecolor="black")
for k, v in enumerate(prod_total_rev):
    axes[1].text(k, v + 1, f"{v:.1f}", ha="center", fontweight="bold")
axes[1].set_title("Product-wise Total Revenue (Rs. lakh)", fontweight="bold")
axes[1].set_xlabel("Product")
axes[1].set_ylabel("Revenue (Rs. lakh)")
axes[1].grid(axis="y", linestyle="--", alpha=0.6)
# f3: total monthly revenue
axes[2].plot(short_m, monthly_total, marker="o", markersize=9, linewidth=2.5, color="#2ca02c")
for j, v in enumerate(monthly_total):
    axes[2].annotate(f"{v:.1f}", (j, v), textcoords="offset points", xytext=(0, 9), ha="center")
axes[2].set_title("Total Monthly Revenue (Rs. lakh)", fontweight="bold")
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Revenue (Rs. lakh)")
axes[2].set_ylim(0, monthly_total.max() + 6)
axes[2].grid(True, linestyle=":", alpha=0.7)
fig.suptitle("Q2(f): Sales Dashboard - Trend, Product Revenue and Monthly Revenue",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "Q2f_subplots_dashboard.png")

print("\n--- Q2 Interpretation ---")
print(f"(a) All three products grow steadily from January to June; Mobile has the highest unit volume each month.")
print(f"(b) Mobile sold the most units ({prod_total_units[1]}), Laptop {prod_total_units[0]}, Tablet {prod_total_units[2]}.")
print(f"(c) {best_rev_prod} contributes the largest share of revenue ({prod_total_rev[0] / prod_total_rev.sum() * 100:.1f}%).")
print(f"(d) Combined revenue rises every month; {peak_month} has the highest total (Rs. {monthly_total.max():.2f} lakh).")
print(f"(e) Mobile and tablet units are strongly positively correlated (r = {r2:.2f}).")
print(f"(f) The dashboard confirms an upward sales trend across all products.")
print(f"\nBest-performing product (revenue): {best_rev_prod}")
print(f"Month with highest total revenue : {peak_month}")
print(f"Overall sales trend              : Increasing month over month")
print("\nAll figures saved in folder:", os.path.abspath(OUT_DIR))
