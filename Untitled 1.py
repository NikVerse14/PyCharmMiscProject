# Create three dashboard wireframe images using matplotlib (no seaborn, no custom colors).
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_wireframe(title, panels, filename, figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title(title, fontsize=18, pad=20)

    # Outer frame
    ax.add_patch(Rectangle((1, 1), 98, 98, fill=False, linewidth=1.5))

    for p in panels:
        x, y, w, h, header, lines = p
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.2))
        ax.text(x + 2, y + h - 6, header, fontsize=12, va="top")
        if lines:
            for i, line in enumerate(lines):
                ax.text(x + 2, y + h - 14 - 8 * i, line, fontsize=9, va="top")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


# 1) Digital Investment ROI — Digital Value Dashboard
digital_panels = [
    # KPIs row
    (
        3,
        82,
        22,
        12,
        "Portfolio ROI (Actual vs Target)",
        ["ROI Actual, ROI Target, Variance", "Filters: Period, BU, Segment, Status"],
    ),
    (27, 82, 22, 12, "Operational Savings (YTD)", ["$ saved, STP rate trend"]),
    (51, 82, 22, 12, "CLV Uplift % (YTD)", ["Post vs Baseline CLV"]),
    (75, 82, 22, 12, "Digital Adoption Rate", ["% digital users"]),
    # Left column
    (3, 56, 30, 22, "Operational Efficiency", ["STP Rate (line)", "Cost-to-Income (sparkline)"]),
    # Middle
    (
        35,
        56,
        30,
        22,
        "Customer Value",
        ["CLV Distribution (histogram)", "Engagement vs CLV (scatter)"],
    ),
    # Right
    (67, 56, 30, 22, "Financial/Strategic", ["NPV by Project (bar)", "Scale / Fix / Stop tiles"]),
    # Bottom table
    (
        3,
        6,
        94,
        46,
        "Project Portfolio (drillable table)",
        [
            "Project | Owner | Spend | ROI | CLV Uplift | Op Savings | Status",
            "Alerts: ROI variance > 15%, Adoption ↓ 3 wks, NPV above target",
        ],
    ),
]

draw_wireframe(
    "Digital Value Dashboard (Wireframe)",
    digital_panels,
    "/mnt/data/digital_value_dashboard_wireframe.png",
)

# 2) AI Model Assurance — Credit Model Health Dashboard
ai_panels = [
    # KPIs row
    (3, 82, 22, 12, "Models with Red Bias Flags", ["Count of 'Red' in fairness_audit"]),
    (27, 82, 22, 12, "Models Breaching Drift", ["PSI/KS above threshold"]),
    (51, 82, 22, 12, "Avg AUC / Accuracy", ["Performance snapshot"]),
    (75, 82, 22, 12, "Compliance Pass Rate", ["% controls passed"]),
    # Left
    (3, 56, 30, 22, "Drift & Performance", ["Drift trend (line)", "Confusion metrics"]),
    # Middle
    (
        35,
        56,
        30,
        22,
        "Fairness & Outcomes",
        ["Approval by protected group (bar)", "Equalized odds & parity (bullets)"],
    ),
    # Right
    (
        67,
        56,
        30,
        22,
        "Governance & Compliance",
        ["Overdue reviews (list)", "Recent changes (timeline)"],
    ),
    # Bottom table
    (
        3,
        6,
        94,
        46,
        "Model Inventory (table)",
        [
            "Model | Owner | Version | AUC | Drift | Bias Flag | Last Retrain",
            "Alerts: Drift>0.2, Bias='Red', Review overdue >14d",
        ],
    ),
]

draw_wireframe(
    "AI Model Assurance Dashboard (Wireframe)",
    ai_panels,
    "/mnt/data/ai_model_assurance_wireframe.png",
)

# 3) Dynamic Hedging — Treasury Risk & Hedge Dashboard
hedge_panels = [
    # KPIs row
    (3, 82, 22, 12, "Current Regime", ["Stable/Trending/Volatile/Crisis"]),
    (27, 82, 22, 12, "Hedge Ratio (Actual vs Target)", ["Gauge / %"]),
    (51, 82, 22, 12, "Hedge P&L (YTD)", ["Unrealized + Realized"]),
    (75, 82, 22, 12, "Cost of Hedging (YTD)", ["Premiums / lost upside"]),
    # Left
    (
        3,
        56,
        30,
        22,
        "Market & Signals",
        ["Commodity price + AUD/USD (dual line)", "Volatility index (line)"],
    ),
    # Middle
    (
        35,
        56,
        30,
        22,
        "Coverage & Exposure",
        ["Target vs Actual ratio (gauge)", "Unhedged exposure by month (bar)"],
    ),
    # Right
    (
        67,
        56,
        30,
        22,
        "Performance & Cost",
        ["Hedge effectiveness (line)", "Static vs Dynamic benchmark (Δ bar)"],
    ),
    # Bottom table
    (
        3,
        6,
        94,
        46,
        "Positions (table)",
        [
            "Instrument | Strike | Maturity | Volume | MtM | Ratio",
            "Alerts: Ratio below target-10pp, Crisis regime, Unhedged ↑ 20% WoW",
        ],
    ),
]

draw_wireframe(
    "Treasury Risk & Hedge Dashboard (Wireframe)",
    hedge_panels,
    "/mnt/data/treasury_hedge_dashboard_wireframe.png",
)

"/mnt/data/digital_value_dashboard_wireframe.png", "/mnt/data/ai_model_assurance_wireframe.png", "/mnt/data/treasury_hedge_dashboard_wireframe.png"
# Return the file paths so the UI can offer downloads
