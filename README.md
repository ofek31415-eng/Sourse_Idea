# The Impact of the 1973 Global Energy Crisis on the Technological Development of Manufacturing Industries

**Author:** Ofek Be'eri · **Advisor:** Dana Vaknin Ganel · **Program:** IDEA, Hebrew University of Jerusalem

> This repository is the companion code and data archive for the above economics thesis. It contains all reproducible analyses, figures, and regression outputs used in the paper. A reviewer can navigate directly from this file to every figure, table, and numerical result cited in the paper.

---

## Research Question & Hypothesis

**Research question (§3.1.1):** Did the 1973 global energy crisis cause a lasting divergence in technological development between energy-intensive and non-energy-intensive US manufacturing industries?

**Hypothesis (§3.1.2):**
- *Short-term (confirmed):* The energy price shock immediately depressed Total Factor Productivity (TFP) and value-added in high-energy industries relative to low-energy industries.
- *Long-term (refuted → "Creative Destruction"):* Rather than a permanent decline, the shock forced radical adaptation. By the early 1980s, energy-intensive industries had reduced their energy intensity dramatically and ultimately *surpassed* the control group in TFP and economic growth — consistent with the Induced Innovation hypothesis and Schumpeter's theory of Creative Destruction.

---

## Key Findings

- **Short-Term Impact (Hypothesis Confirmed):** Immediately following the 1973 crisis, energy-intensive industries suffered a severe blow. The sudden spike in energy costs rendered existing capital stock obsolete and highly inefficient, resulting in a significant drop in TFP and value-added compared to the control group.

- **Long-Term Impact (Hypothesis Refuted — Creative Destruction):** Instead of declining permanently, the existential threat forced energy-intensive industries to adapt. By the early 1980s, these industries underwent radical structural changes, drastically reducing their energy intensity. Ultimately, they surpassed the low-energy control group in TFP and economic growth — providing strong empirical evidence for the Induced Innovation hypothesis and Joseph Schumpeter's theory of Creative Destruction.

---

## Data Source

| Item | Details |
|------|---------|
| Dataset | NBER-CES Manufacturing Industry Database |
| File | `data/nberces5818v1_n2012.csv` |
| Version | `nberces5818v1` |
| Coverage | 1958–2000 (6-digit NAICS, US manufacturing) |
| Why 2000? | Isolates the 1973 crisis effect from the 2008 financial shock |
| Source | [NBER-CES Manufacturing Industry Database](https://www.nber.org/research/data/nber-ces-manufacturing-industry-database) |

---

## Methodology

| Technique | Purpose |
|-----------|---------|
| **Dynamic Difference-in-Differences (DiD)** | Three OLS regression models that estimate the treatment × post-1973 interaction over every year, producing event-study plots (Graphs 6, 7, 8) |
| **Two-Way Fixed Effects** | `C(year)` controls for macroeconomic shocks common to all industries; `C(naics)` controls for unobserved time-invariant industry characteristics |
| **Propensity Score Matching (PSM)** | Year-by-year Nearest-Neighbor matching (logistic regression, caliper=0.05, k=1) on `log_emp + log_real_vadd + log_cap` — ensures an "apples-to-apples" treatment/control comparison before running the DiD |

**Key variables:** `tfp5` (Total Factor Productivity), `real_energy_intensity`, `log_vadd`, `is_high_energy`, `post_1973`, `treatment_above_median`, `log_cap`, `log_emp`.

**Models:**
- Model 1 — TFP (§4.2.1): N = 9,178 PSM-matched obs, R² = 0.848
- Model 2 — Real Energy Intensity, median 35–65% sub-sample (§4.2.2): N = 4,683 obs, R² = 0.836
- Model 3 — log VADD (§4.2.3): N = 9,178 PSM-matched obs, R² = 0.974

---

## Repository Layout

```
Sourse_Idea/
│
├── README.md                                        ← you are here
├── PAPER_MAP.md                                     ← figure/table → code/artifact mapping
├── .gitignore
│
├── data/
│   └── nberces5818v1_n2012.csv                      ← canonical NBER-CES dataset (1958–2000)
│
├── notebooks/
│   └── 01_idea_data_analysis.ipynb                  ← all analyses; annotated with paper §
│
├── figures/
│   ├── fig01_energy_price_shock.png                 ← §4.1 Graph 1: energy price index 1958–2000
│   ├── fig03_real_energy_intensity_by_group.png     ← §4.1 Graph 3: energy intensity by tercile
│   ├── fig04_tercile_vadd_by_output_and_emp.png     ← §4.1 Graph 4: VADD × output & employment, 1972 vs 1990
│   ├── fig05_tfp_by_tercile_long_run.png            ← §4.1 Graph 5: long-run TFP by tercile 1958–2000
│   ├── fig06_model1_tfp_event_study.png             ← §4.2.1 Graph 6: Model 1 TFP event study
│   ├── fig07_model2_energy_intensity_event_study.png← §4.2.2 Graph 7: Model 2 energy intensity event study
│   └── fig08_model3_log_vadd_event_study.png        ← §4.2.3 Graph 8: Model 3 log VADD event study
│
└── outputs/
    ├── table2_model1_tfp_regression.txt             ← §4.2.1 Table 2: Model 1 regression results
    ├── table3_model2_energy_intensity_regression.txt← §4.2.2 Table 3: Model 2 regression results
    ├── table4_model3_log_vadd_regression.txt        ← §4.2.3 Table 4: Model 3 regression results
    ├── appendix_a_psm_summary.txt                   ← Appendix A: PSM diagnostics & balance stats
    ├── industry_tercile_classification.csv          ← supplementary: per-NAICS energy tercile labels
    ├── extreme_robustness_results.txt               ← supplementary: extreme 5%/95% robustness analysis
    └── extreme_robustness_psm_log.txt               ← supplementary: PSM log for extreme robustness run
```

> **Notes:**
> - Table 1 (§4.1 descriptive statistics) and Graph 2 (§4.1 inline chart) are rendered as notebook outputs only — see the cell annotations inside `notebooks/01_idea_data_analysis.ipynb`.
> - The `outputs/` files are reviewer-facing canonical copies. The notebook code writes these files to `notebooks/` (hard-coded paths that must not be changed); the copies in `outputs/` are the tidy, labelled versions.
> - For the full paper → code mapping, see **[PAPER_MAP.md](PAPER_MAP.md)**.

---

## How to Reproduce

**Requirements:** Python 3.9+, Jupyter Notebook or JupyterLab.

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
```

Open and run the notebook:

```bash
jupyter notebook notebooks/01_idea_data_analysis.ipynb
```

> **Important:** The notebook uses an absolute path for the data file (`C:\Users\ofek3\...\data\nberces5818v1_n2012.csv`). Before running, update `file_path` in the first code cell to point to `data/nberces5818v1_n2012.csv` relative to your local clone.

Run all cells in order. The notebook is structured to mirror the paper chapters — each section has a markdown header citing the relevant paper section (§) and the figure/table it produces.

---

## Paper-to-Code Map

See **[PAPER_MAP.md](PAPER_MAP.md)** for a table mapping every paper artifact (Table 1–4, Graph 1–8, Appendix A) to the exact notebook cell and output file that produces it.

---

## Contact & Links

| | |
|-|-|
| **Author** | Ofek Be'eri |
| **Email** | ofek31415@gmail.com |
| **GitHub Repository** | https://github.com/ofek31415-eng/Sourse_Idea |
| **Advisor** | Dana Vaknin Ganel |
| **Institution** | IDEA Program, Hebrew University of Jerusalem |
