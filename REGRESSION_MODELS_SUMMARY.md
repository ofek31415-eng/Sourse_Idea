# Regression Models Summary: Energy Crisis Impact Analysis (1973)

## Overview
This analysis examines how the 1973 energy crisis affected U.S. manufacturing industries using a **Difference-in-Differences (DiD)** research design. The analysis compares high-energy-intensive industries versus low-energy-intensive industries before and after 1973.

**Data Source:** NBER Census of Manufacturing (1958-2018, N=6,342 observations)

---

## Three Main Regression Models

### **MODEL 1: Productivity Response (Total Factor Productivity)**

**Research Question:** Did high-energy industries experience faster productivity growth after the crisis?

#### Dependent Variable (Y)
| Variable | Description | Type |
|----------|-------------|------|
| `tfp5` | Total Factor Productivity | **Raw Data** (from NBER Census) |

#### Independent Variables (X)

| Variable | Description | Type | Formula/Derivation |
|----------|-------------|------|-------------------|
| `is_high_energy` | Binary: 1 if industry in top energy tercile (pre-1973), 0 if bottom tercile | **Engineered** | Tercile classification based on avg `real_energy_intensity` (1958-1972) |
| `post_1973` | Binary: 1 if year ≥ 1973, 0 if year < 1973 | **Engineered** | Indicator for post-crisis period |
| `is_high_energy × post_1973` | **DiD Interaction:** Effect of crisis on high-energy industries | **Engineered** | Product of the two indicators above |
| `log_emp` | Log of employee count (control for firm size) | **Engineered** | `ln(emp)` for scale normalization |
| `C(year)` | Year fixed effects (dummy for each year 1958-1999) | **Engineered** | Categorical dummies capturing time trends |
| `C(naics)` | Industry fixed effects (dummy for each manufacturing sector) | **Engineered** | Categorical dummies capturing industry-specific characteristics |

#### Model Specification
```
tfp5 ~ is_high_energy * post_1973 + log_emp + C(year) + C(naics)
```

**Interpretation:** The coefficient on `is_high_energy:post_1973` tells us: *How much did high-energy industries' productivity diverge from low-energy industries after 1973?*

---

### **MODEL 2: Efficiency Response (Energy Intensity)**

**Research Question:** Did high-energy industries improve their energy efficiency more after the crisis?

#### Dependent Variable (Y)
| Variable | Description | Type | Formula |
|----------|-------------|------|---------|
| `real_energy_intensity` | Energy used per unit of output (adjusted for inflation) | **Engineered** | `real_energy / real_vadd` |

Where:
- `real_energy` = `energy / pien` (energy in real terms, deflated by energy price index)
- `real_vadd` = `vadd / piship` (value added in real terms, deflated by shipment price index)

#### Independent Variables (X)
Same as Model 1:
- `is_high_energy` 
- `post_1973` 
- `is_high_energy × post_1973` (DiD term)
- `log_emp`
- `C(year)` fixed effects
- `C(naics)` fixed effects

#### Model Specification
```
real_energy_intensity ~ is_high_energy * post_1973 + log_emp + C(year) + C(naics)
```

**Interpretation:** The coefficient on `is_high_energy:post_1973` tells us: *Did high-energy industries reduce their energy intensity more than low-energy industries after the 1973 crisis?*

---

### **MODEL 3: Economic Impact (Value Added Growth)**

**Research Question:** What was the short-term growth impact on high-energy industries after the crisis?

#### Dependent Variable (Y)
| Variable | Description | Type | Formula |
|----------|-------------|------|---------|
| `vadd_growth` | Annual percent change in real value added | **Engineered** | `pct_change(real_vadd) × 100` |

Where `real_vadd = vadd / piship`

#### Independent Variables (X)
Same as Models 1 & 2:
- `is_high_energy`
- `post_1973`
- `is_high_energy × post_1973` (DiD term)
- `log_emp`
- `C(year)` fixed effects
- `C(naics)` fixed effects

#### Model Specification
```
vadd_growth ~ is_high_energy * post_1973 + log_emp + C(year) + C(naics)
```

**Interpretation:** The coefficient on `is_high_energy:post_1973` tells us: *Did high-energy industries experience slower growth rates than low-energy industries after 1973?*

---

## Data Transformation Pipeline

### **Step 1: Raw Data Variables (From NBER Census)**
| Name | Description |
|------|-------------|
| `naics` | Industry classification code |
| `year` | Year (1958-2018) |
| `energy` | Energy consumption (nominal) |
| `pien` | Energy price index (deflationary) |
| `vadd` | Value of added output (nominal) |
| `piship` | Shipment/output price index (deflationary) |
| `tfp5` | Total Factor Productivity (pre-calculated) |
| `emp` | Number of employees |
| `cap` | Capital stock |

### **Step 2: Data Cleaning & Filtering**
- **Remove missing values:** Drop rows where `real_energy_intensity` or `tfp5` are null
- **Time period:** Keep only years 1958-2018 (focus period: 1958-1999 for main models)
- **Result:** 6,342 observations across ~80 manufacturing industries

### **Step 3: Create Real (Inflation-Adjusted) Variables**
| Engineered Variable | Formula | Purpose |
|-------------------|---------|---------|
| `real_energy` | `energy / pien` | Measure actual energy quantity (remove price inflation) |
| `real_vadd` | `vadd / piship` | Measure actual output (remove price inflation) |

### **Step 4: Create Key Energy Metrics**
| Engineered Variable | Formula | Purpose |
|-------------------|---------|---------|
| `real_energy_intensity` | `real_energy / real_vadd` | Energy efficiency metric (energy per unit output) |
| `vadd_growth` | `pct_change(real_vadd) × 100` | Annual growth rate of real output |

### **Step 5: Create Treatment Variables**
| Engineered Variable | Formula | Purpose |
|-------------------|---------|---------|
| `is_high_energy` | Binary classification (tercile-based) | Identify treatment group (before 1973) |
| `post_1973` | `year >= 1973` | Identify post-crisis period |
| `is_high_energy × post_1973` | Interaction product | **DiD effect:** Differential impact on treated group |

### **Step 6: Create Control & Specification Variables**
| Engineered Variable | Formula | Purpose |
|-------------------|---------|---------|
| `log_emp` | `ln(emp)` | Control for industry size (log scale) |
| `C(year)` | Categorical dummies | Absorb year-specific unobservable shocks |
| `C(naics)` | Categorical dummies | Absorb industry-specific characteristics |

---

## Sample Sizes & Model Properties

| Model | Dependent Variable | N Obs | Adj. R² | Matching Method |
|-------|------------------|-------|---------|-----------------|
| **Model 1** | `tfp5` | 6,342 | 0.799 | Propensity Score (Base 1968-72) |
| **Model 2** | `real_energy_intensity` | 6,342 | 0.890 | Propensity Score (Base 1968-72) |
| **Model 3** | `vadd_growth` | 4,832 | 0.203 | Propensity Score (Base 1968-72) |

**Note:** Propensity Score Matching (PSM) used to create balanced comparison groups (high-energy vs. low-energy industries) by controlling for baseline differences in firm size (employees, value added, capital).

---

## Key Findings Summary

### Model 1: Productivity (TFP)
- **DiD Coefficient:** Significant positive effect (results in `results_model1_tfp.txt`)
- **Interpretation:** High-energy industries achieved faster productivity growth post-1973 (likely driven by forced innovation/efficiency improvements)

### Model 2: Energy Intensity
- **DiD Coefficient:** Significant negative effect (results in `results_model2_intensity.txt`)
- **Interpretation:** High-energy industries reduced energy intensity more than low-energy industries post-1973 (structural adjustment to higher energy costs)

### Model 3: Growth (VADD Growth)
- **DiD Coefficient:** Mixed results by sector (results in `results_model3_growth.txt`)
- **Interpretation:** Short-term growth rates shown asymmetric impacts across different manufacturing sectors

---

## Robustness Checks & Alternative Specifications

Beyond the three main models, analysis includes:

1. **Event Study Design:** Dynamic DiD with year-by-year interactions
   - Formula: `outcome ~ is_high_energy * C(year) + controls`
   - Shows differential trajectories over time

2. **Alternative Matching Strategies:** PSM with different baseline periods
   - Base 1968-72
   - Base 1958-72
   - Base 1958 only
   - Results compiled in `results_extreme_robustness.txt`

3. **Sub-sample Analysis:** By industry size category
   - Small, Medium, Large industries
   - Shows heterogeneous treatment effects

---

## Variable Data Availability

### Raw Data from NBER (Directly Observed)
- ✓ `tfp5` — Pre-calculated by NBER researchers
- ✓ `energy` — Manufacturing energy consumption
- ✓ `pien` — Price index for energy
- ✓ `vadd` — Value of output added
- ✓ `piship` — Price index for shipments
- ✓ `emp` — Employment
- ✓ `cap` — Capital stock
- ✓ `year`, `naics` — Classification variables

### Created by Analysis (Engineered)
- ✓ `real_energy` — Deflated energy consumption
- ✓ `real_vadd` — Deflated value added
- ✓ `real_energy_intensity` — Energy per unit output
- ✓ `vadd_growth` — Annual growth rate
- ✓ `is_high_energy` — Treatment indicator (tercile classification)
- ✓ `post_1973` — Period indicator
- ✓ `log_emp` — Log-scale employment
- ✓ Fixed effects dummies (`C(year)`, `C(naics)`)

---

## Questions to Address in Meeting

1. **Causality:** How confident are we that the DiD design identifies causal effects?
   - *Parallel trends assumption tested in pre-crisis period (1958-1973)*

2. **Mechanism:** What explains the productivity improvement?
   - *Forced energy efficiency investment? Technological innovation? Industry composition shifts?*

3. **Heterogeneity:** Do effects differ by industry size or sub-sector?
   - *Partially addressed in robustness checks; more granular analysis possible*

4. **External Relevance:** Do these findings apply to other shocks or countries?
   - *Potential policy implications for climate/energy transitions*

---

## File References

- **Main Analysis Results:**
  - [results_model1_tfp.txt](results_model1_tfp.txt) — Full Model 1 output with all industry FEs
  - [results_model2_intensity.txt](results_model2_intensity.txt) — Full Model 2 output
  - [results_model3_growth.txt](results_model3_growth.txt) — Full Model 3 output
  
- **Robustness:**
  - [results_extreme_robustness.txt](results_extreme_robustness.txt) — Alternative specifications

- **Matching Quality:**
  - [psm_log.txt](psm_log.txt) — Propensity score matching diagnostics
  - [industry_classification_list.csv](industry_classification_list.csv) — Industry categorization

---

**Last Updated:** March 7, 2026
**Analysis Tool:** Python (statsmodels v0.13+), Jupyter Notebook
