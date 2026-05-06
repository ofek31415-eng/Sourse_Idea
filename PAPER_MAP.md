# PAPER_MAP.md — Figure / Table → Code / Artifact Mapping

**Paper:** "The Impact of the 1973 Global Energy Crisis on the Technological Development of Manufacturing Industries"  
**Author:** אופק בארי (Ofek Be'eri) | **Advisor:** דנה וקנין גנאל | **Program:** IDEA, Hebrew University  
**Repository:** https://github.com/ofek31415-eng/Sourse_Idea.git  
**Notebook:** `notebooks/01_idea_data_analysis.ipynb` (single executable notebook, 38 cells)  
**Dataset:** `data/nberces5818v1_n2012.csv` (NBER-CES Manufacturing Industry Database, 1958–2018)

---

## How to Read This Map

| Column | Meaning |
|--------|---------|
| **Paper §** | Chapter / section number in the paper PDF |
| **Artifact** | Table or Graph number as it appears in the paper |
| **Caption (short)** | Short description of what the artifact shows |
| **Notebook cell(s)** | Cell index/indices in `notebooks/01_idea_data_analysis.ipynb` that produce this artifact |
| **Output file** | Path to the saved output file (relative to `Sourse_Idea/`) |
| **Data used** | Primary input variable(s) / dataframe |

---

## Tables

| Paper § | Artifact | Caption (short) | Notebook cell(s) | Output file | Data used |
|---------|----------|-----------------|------------------|-------------|-----------|
| §4.1 | **Table 1** — Descriptive Statistics | Summary stats of `tfp5`, `real_energy_intensity`, `log_vadd`, `log_emp`, `log_cap`, `vadd_growth`; pre-crisis (1958–1972) vs. post-crisis (1974–2000) × high/low-energy groups | Cells 6–16 (inline `df_final.groupby` output displayed in notebook) | *(No saved file — Table 1 is a displayed DataFrame in the notebook; see cells 6–16)* | `df_final` |
| §4.2.1 | **Table 2** — Model 1: TFP | Dynamic DiD of `tfp5`; PSM-matched full sample; N=9,178; R²=0.848; pre-1973 F p=0.724 | Cell 19 (writes result file); cells 32–34 (event-study infrastructure) | `outputs/table2_model1_tfp_regression.txt` | `matched_datasets['Dynamic']` |
| §4.2.2 | **Table 3** — Model 2: Real Energy Intensity | Dynamic DiD of `real_energy_intensity`; median 35–65th-pct sub-sample; N=4,683; R²=0.836; pre-1973 F p=0.006 | Cells 25–26 (median sub-sample creation + regression); cell 20 (full-tercile variant, supplementary) | `outputs/table3_model2_energy_intensity_regression.txt` | `regression_data_median` |
| §4.2.3 | **Table 4** — Model 3: log VADD | Dynamic DiD of `log_vadd`; PSM-matched full sample; N=9,178; R²=0.974; pre-1973 F p=0.101 | Cell 21 (writes result file); cells 32–34 (event-study infrastructure) | `outputs/table4_model3_log_vadd_regression.txt` | `matched_datasets['Dynamic']` |

---

## Graphs / Figures

| Paper § | Artifact | Caption (short) | Notebook cell(s) | Output file | Data used |
|---------|----------|-----------------|------------------|-------------|-----------|
| §1.1 | **Graph 1** — Oil share of world energy consumption | Share of oil in global energy mix, 1900s–1973. **External source** (Our World in Data 2024 / Energy Institute). Not produced by research code. | *(Not in notebook — static image from external source)* | *(Not in repo — external figure)* | External |
| §1.5.3 | **Graph 2** — Energy price shock, 1958–2000 | Normalised energy price index (`pien`) and output price index (`piship`), base year 1958=1; 1973 crisis marker; shows energy prices rose far more than other prices | Cells 8–9 (inline plot; no saved PNG — reproduced on notebook run) | `figures/fig01_energy_price_shock.png` *(saved version of the energy-price chart)* | `df_final` (`pien`, `piship`) |
| §4.1 | **Graph 3** — Real Energy Intensity by group, 1958–2000 | `real_energy_intensity` over time for high-energy vs. low-energy tercile; dramatic post-1973 convergence and divergence | Cell 10 (seaborn line plot with CI bands) | `figures/fig03_real_energy_intensity_by_group.png` | `df_final` (`real_energy_intensity`, `group_name`) |
| §4.1 | **Graph 4** — Tercile comparison: VADD × output and × employment | Value added per output and per employment, by energy-intensity tercile; comparing 1972 (pre-crisis) vs. 1990 (post-crisis) | Cell 14 (4-panel boxplot, Times New Roman styling) | `figures/fig04_tercile_vadd_by_output_and_emp.png` | `df_final` (`vadd_growth`, `group_name`, `period`) |
| §4.1 | **Graph 5** — Long-run TFP5 trends by tercile, 1958–2000 | `tfp5` trajectories for top and bottom energy-intensity terciles (middle omitted); post-1973 divergence and eventual catch-up | Cell 15 (pre/post TFP scatter and line plot by `group_name`) | `figures/fig05_tfp_by_tercile_long_run.png` | `df_final` (`tfp5`, `group_name`, `year`) |
| §4.2.1 | **Graph 6** — Model 1 Event Study: TFP coefficients | Dynamic DiD year-by-year interaction coefficients for `tfp5` with 95% CI, 1958–2000; pre-1973 coefficients ≈ 0 (parallel trends); sharp post-1973 decline | Cell 34 (`plot_trajectory_loop`, called for `dep_var='tfp5'`); infrastructure in cells 32–33 | `figures/fig06_model1_tfp_event_study.png` | `matched_datasets['Dynamic']` + `outputs/table2_model1_tfp_regression.txt` |
| §4.2.2 | **Graph 7** — Model 2 Event Study: Energy Intensity coefficients | Dynamic DiD year-by-year interaction coefficients for `real_energy_intensity` (median sub-sample), 1958–2000; short-run rigidity then long-run reversal | Cell 35 (median-sub-sample event study, reference year 1973) | `figures/fig07_model2_energy_intensity_event_study.png` | `regression_data_median` |
| §4.2.3 | **Graph 8** — Model 3 Event Study: log VADD coefficients | Dynamic DiD year-by-year interaction coefficients for `log_vadd`, 1958–2000; post-1973 decline then gradual recovery for high-energy industries | Cell 34 (`plot_trajectory_loop`, called for `dep_var='log_vadd'`); infrastructure in cells 32–33 | `figures/fig08_model3_log_vadd_event_study.png` | `matched_datasets['Dynamic']` + `outputs/table4_model3_log_vadd_regression.txt` |

> **Note — Graph 6 and Graph 8 share the same producer cell:** `plot_trajectory_loop` (cell 34) loops over dependent variables; it produces the `05_Dynamic_tfp5.png` output for Graph 6 and the `15_Dynamic_log_vadd.png` output for Graph 8 in the same call.

---

## Appendix A — PSM Summary

| Paper § | Artifact | Caption (short) | Notebook cell(s) | Output file | Data used |
|---------|----------|-----------------|------------------|-------------|-----------|
| Appendix A | **PSM Summary** — Dynamic PSM Matched Sample | N=9,178; 338 industries; 109.3 avg pairs/yr; caliper=0.05; covariates `log_emp` + `log_real_vadd` + `log_cap`; k=1 Nearest-Neighbour logistic | Cell 18 (runs `perform_psm`, writes log); cell 37 (prints summary statistics that verify Appendix A numbers) | `outputs/appendix_a_psm_summary.txt` | `df_final` |

---

## Supplementary / Additional Analyses (Appendix B)

Per Appendix B of the paper, the repository explicitly includes *"additional analyses not included in the paper body."* The following are retained as supplementary material:

| Artifact | Description | Notebook cell(s) | Output file(s) |
|----------|-------------|------------------|----------------|
| **Extreme 5%/95% Robustness Check** | PSM + Dynamic DiD on industries in the top 5% / bottom 5% of `real_energy_intensity` each year. Not cited in the paper body but retained per Appendix B. | Cell 23 (`classify_extreme_values` function def); cell 24 (runs extreme-groups PSM + DiD) | `outputs/extreme_robustness_results.txt`, `outputs/extreme_robustness_psm_log.txt` |
| **Industry Tercile Classification List** | Per-industry list of NAICS codes with assigned energy tercile (`group_name`) and pre-crisis average `real_energy_intensity`; useful for reviewer verification of Table 1 and Graph 4 industry groupings | Cell 5 (builds and saves the classification CSV) | `outputs/industry_tercile_classification.csv` |

---

## Data Source

| File | Description |
|------|-------------|
| `data/nberces5818v1_n2012.csv` | **NBER-CES Manufacturing Industry Database** (Bartelsman, Becker & Gray). Release `nberces5818v1`. Coverage: 459 4-digit NAICS manufacturing industries, 1958–2018. Variables used: `tfp5`, `energy`, `pien`, `vadd`, `piship`, `emp`, `cap`, `naics`, `year`. Loaded in cell 0. |

---

## Reproduction Notes

1. Open `notebooks/01_idea_data_analysis.ipynb` in Jupyter.
2. Ensure `data/nberces5818v1_n2012.csv` is present at the hard-coded path referenced in cell 0 (update the path string if the working directory differs).
3. Run all cells top-to-bottom. Cells 0–18 must complete before any model cell; cell 18 writes `psm_log.txt` and populates `matched_datasets`.
4. Output `.txt` regression files are written to the `notebooks/` subfolder by hard-coded paths in the code cells (not changed); canonical reviewer copies are in `outputs/`.
5. Dynamic event-study PNGs (Graphs 6 and 8) are written to `notebooks/trajectory_png_plots/` by `plot_trajectory_loop` (cell 34); canonical copies are in `figures/`.

See `README.md` for a full repository layout and dependency overview.
