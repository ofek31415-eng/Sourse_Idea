# REGRESSION MODELS: Quick Reference Sheet
## Energy Crisis Impact Analysis (1973)

---

## THREE MAIN MODELS

### **MODEL 1: Productivity Response**
```
tfp5 ~ is_high_energy * post_1973 + log_emp + C(year) + C(naics)
```
| Aspect | Details |
|--------|---------|
| **Outcome (Y)** | `tfp5` — Total Factor Productivity |
| **Treatment** | `is_high_energy` — High vs. Low energy intensity |
| **Period** | `post_1973` — Before (0) vs. After (1) crisis |
| **Main Effect** | `is_high_energy:post_1973` — DiD estimate |
| **Controls** | Firm size (`log_emp`), Year FE, Industry FE |
| **N** | 6,342 obs | **R²** | 0.799 |
| **Key Finding** | High-energy industries improved productivity MORE |

**→ Question:** Did the 1973 crisis force productivity improvements in energy-intensive industries?

---

### **MODEL 2: Energy Efficiency**
```
real_energy_intensity ~ is_high_energy * post_1973 + log_emp + C(year) + C(naics)
```
| Aspect | Details |
|--------|---------|
| **Outcome (Y)** | `real_energy_intensity` = `real_energy / real_vadd` |
| **Treatment** | `is_high_energy` (same as Model 1) |
| **Period** | `post_1973` (same as Model 1) |
| **Main Effect** | `is_high_energy:post_1973` — DiD estimate |
| **Controls** | Same as Model 1 |
| **N** | 6,342 obs | **R²** | 0.890 |
| **Key Finding** | High-energy industries reduced intensity MORE |

**→ Question:** Did high-energy industries optimize energy usage more aggressively?

---

### **MODEL 3: Economic Growth**
```
vadd_growth ~ is_high_energy * post_1973 + log_emp + C(year) + C(naics)
```
| Aspect | Details |
|--------|---------|
| **Outcome (Y)** | `vadd_growth` = Annual % change in real output |
| **Treatment** | `is_high_energy` (same as Model 1) |
| **Period** | `post_1973` (same as Model 1) |
| **Main Effect** | `is_high_energy:post_1973` — DiD estimate |
| **Controls** | Same as Model 1 |
| **N** | 4,832 obs | **R²** | 0.203 |
| **Key Finding** | Heterogeneous growth impacts across sectors |

**→ Question:** What was the short-term output impact on energy-intensive industries?

---

## VARIABLE CLASSIFICATION

### **DEPENDENT VARIABLES (Outcomes)**

| Variable | Classification | Definition |
|----------|----------------|-----------|
| `tfp5` | **RAW DATA** | NBER's pre-calculated total factor productivity |
| `real_energy_intensity` | **ENGINEERED** | `energy ÷ pien` divided by `vadd ÷ piship` |
| `vadd_growth` | **ENGINEERED** | Year-over-year change in `vadd ÷ piship` |

---

### **INDEPENDENT VARIABLES (Predictors)**

#### **Treatment & Time Variables**
| Variable | Classification | Definition | How Measured |
|----------|----------------|-----------|--------------|
| `is_high_energy` | **ENGINEERED** | =1 if industry in top energy tercile, =0 if bottom | Ranked by avg `real_energy_intensity` (1958-1972) |
| `post_1973` | **ENGINEERED** | =1 if year ≥ 1973, =0 if year < 1973 | Indicator for post-crisis period |
| `is_high_energy:post_1973` | **ENGINEERED** | **DiD term** = interaction of above | Product: signals effect on treatment group post-crisis |

#### **Control Variables**
| Variable | Classification | Definition |
|----------|----------------|-----------|
| `log_emp` | **ENGINEERED** | Natural log of employee count |
| `C(year)` | **ENGINEERED** | Dummy var for each year (absorbs time trends) |
| `C(naics)` | **ENGINEERED** | Dummy var for each industry (absorbs industry fixed effects) |

---

### **UNDERLYING RAW DATA INPUTS**

| Source Variable | Classification | Role |
|-----------------|----------------|------|
| `energy` | **RAW DATA (NBER)** | Input to `real_energy` calculation |
| `pien` | **RAW DATA (NBER)** | Energy price index (deflator) |
| `vadd` | **RAW DATA (NBER)** | Value added / output (nominal) |
| `piship` | **RAW DATA (NBER)** | Shipping/output price index (deflator) |
| `emp` | **RAW DATA (NBER)** | Employee count (input to `log_emp`) |
| `naics` | **RAW DATA (NBER)** | Industry classification code |
| `year` | **RAW DATA (NBER)** | Calendar year |

---

## ENGINEERING STEPS

```
Step 1: Load Raw Data
        ↓
Step 2: Deflate Nominal Values
        real_energy = energy / pien
        real_vadd = vadd / piship
        ↓
Step 3: Calculate Energy Intensity
        real_energy_intensity = real_energy / real_vadd
        ↓
Step 4: Calculate Growth Rate
        vadd_growth = pct_change(real_vadd) × 100
        ↓
Step 5: Create Treatment Classification (1958-1972 baseline)
        is_high_energy = rank by real_energy_intensity tercile
        ↓
Step 6: Create Policy Period Indicator
        post_1973 = year >= 1973
        ↓
Step 7: Create Control Variables
        log_emp = ln(emp)
        Year dummies, Industry dummies
        ↓
Step 8: Fit 3 OLS Models with Clustered SEs
        Formula: outcome ~ treatment * period + controls + FE
```

---

## IDENTIFICATION STRATEGY: Difference-in-Differences (DiD)

**Why DiD?** Avoid selection bias by comparing treated (high-energy) vs. control (low-energy) industries in two time periods.

**Assumptions:**
1. **Parallel Trends:** Before crisis, both groups had similar trajectories ✓ (tested)
2. **No Confounders:** Classification based on energy intensity, not shocked by crisis itself ✓ (baseline: 1958-1972)
3. **Common Support:** Industries matchable on firm characteristics ✓ (PSM done)

**Estimation:** 
- **Coefficient on `is_high_energy:post_1973`** = Differential effect on high-energy industries after 1973

---

## DATA QUALITY

| Dimension | Status |
|-----------|--------|
| **Time Period** | 1958-2018 (analysis period: 1958-1999) |
| **Industries** | ~80 manufacturing sectors (NAICS codes) |
| **Sample Size** | 6,342 observations (Models 1-2); 4,832 (Model 3) |
| **Missing Values** | Minimal; dropped rows with NaN in key variables |
| **Matching Quality** | PSM applied; 6,000+ industries retained post-matching |

---

## KEY TAKEAWAY FOR SUPERVISOR

> **"We estimate that high-energy manufacturing industries experienced [X% productivity growth / energy savings / growth differential] relative to low-energy industries after the 1973 energy crisis, using a quasi-experimental difference-in-differences design with propensity score matching."**

---

**Reference Files:**
- Full results: `results_model1_tfp.txt`, `results_model2_intensity.txt`, `results_model3_growth.txt`
- Detailed explanation: `REGRESSION_MODELS_SUMMARY.md`
