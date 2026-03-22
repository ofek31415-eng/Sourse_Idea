# COMPLETE VARIABLE GLOSSARY & DATA DICTIONARY

## DEPENDENT VARIABLES (Outcomes)

### **TFP5 - Total Factor Productivity**
- **Data Type:** Continuous (numerical)
- **Source:** RAW DATA from NBER Census of Manufactures
- **Definition:** Measure of how efficiently an industry converts inputs (labor, capital, materials) into outputs
- **Range:** Typically 0-1+ (indexed to a base year)
- **Used In:** Model 1
- **Missing Values:** <1% (dropped in analysis)

**Raw Data Variables Used to Compute It:**
- Value added, employment, capital stocks (NBER combined these)

---

### **REAL_ENERGY_INTENSITY - Energy per Unit Output**
- **Data Type:** Continuous (numerical)
- **Classification:** ENGINEERED from raw data
- **Formula:** 
  ```
  real_energy_intensity = (energy / pien) ÷ (vadd / piship)
  ```
- **Interpretation:** 
  - How much real energy (BTU) is required to produce one dollar of real output
  - Lower = More efficient
  - Higher = More energy-intensive
- **Range:** ~0.001 to 0.5+ (varies by industry)
- **Used In:** Model 2
- **Calculation Order:**
  1. First: Divide `energy` by `pien` → get real energy
  2. Then: Divide `vadd` by `piship` → get real value added
  3. Finally: Divide real energy by real value added

---

### **VADD_GROWTH - Annual Growth Rate of Output**
- **Data Type:** Continuous (%) 
- **Classification:** ENGINEERED from raw data
- **Formula:**
  ```
  vadd_growth = pct_change(real_vadd) × 100
  where real_vadd = vadd / piship
  ```
- **Interpretation:**
  - Annual percentage change in real (inflation-adjusted) value added
  - Positive = industry growing
  - Negative = industry declining
  - Typical range: -20% to +20% per year
- **Used In:** Model 3
- **Calculation Order:**
  1. First: Divide `vadd` by `piship` → get real value added
  2. Then: Calculate year-over-year percent change
  3. Finally: Multiply by 100 to express as percentage

---

## INDEPENDENT VARIABLES - KEY TREATMENT & POLICY VARIABLES

### **IS_HIGH_ENERGY - Treatment Group Indicator**
- **Data Type:** Binary (0 or 1)
- **Classification:** ENGINEERED from raw data
- **Definition:** 
  - =1 if industry classified as "High Energy Intensive"
  - =0 if industry classified as "Low Energy Intensive"
- **How Treatment Assigned:**
  - Calculated for EACH industry using baseline period (1958-1972)
  - Method: Tercile classification
    - Rank all industries by average `real_energy_intensity` during 1958-1972
    - Top 33% = "High Energy" (=1)
    - Bottom 33% = "Low Energy" (=0)
    - Middle 33% = Excluded from analysis
- **Why Baseline 1958-1972?**
  - Treatment must be assigned BEFORE crisis (1973)
  - Avoids reverse causality: crisis didn't determine high-energy status
- **Variation:** Time-invariant (same for each industry across all years)
- **Key Industries (Examples):**
  - **High Energy:** Steel, Chemicals, Petroleum, Paper & Pulp, Metal Smelting
  - **Low Energy:** Apparel, Leather Goods, Textiles (not bleached), Assembly sectors

---

### **POST_1973 - Crisis Period Indicator**
- **Data Type:** Binary (0 or 1)
- **Classification:** ENGINEERED from raw data
- **Definition:**
  - =1 if year ≥ 1973
  - =0 if year < 1973
- **Rationale:** Marks the 1973 oil embargo (OPEC embargo, October 1973)
- **Variation:** Changes with each year observation
- **Timeline:**
  - Pre-crisis: 1958-1972 (post_1973 = 0)
  - Crisis begins: 1973 (post_1973 = 1)
  - Post-crisis: 1974-1999+ (post_1973 = 1)

---

### **IS_HIGH_ENERGY × POST_1973 - Difference-in-Differences (DiD) Term**
- **Data Type:** Binary (0 or 1, derived)
- **Classification:** ENGINEERED (interaction)
- **Definition:** Product of two binary variables
  ```
  is_high_energy:post_1973 = is_high_energy × post_1973
  ```
- **Possible Values:**
  | is_high_energy | post_1973 | Interaction | Interpretation |
  |---|---|---|---|
  | 0 | 0 | **0** | Low-energy industry, pre-crisis |
  | 0 | 1 | **0** | Low-energy industry, post-crisis |
  | 1 | 0 | **0** | High-energy industry, pre-crisis |
  | 1 | 1 | **1** | High-energy industry, post-crisis ← TREATED |
- **Key Insight:** Only the last group (=1) experiences the "treatment" in econometric sense
- **DiD Coefficient Interpretation:** Difference-in-differences estimate
  - Controls for pre-existing differences between high/low energy industries
  - Controls for universal time trends (that affected both groups)
  - Isolates the crisis effect specific to high-energy industries

---

## INDEPENDENT VARIABLES - CONTROLS

### **LOG_EMP - Log of Employment**
- **Data Type:** Continuous (numeric)
- **Classification:** ENGINEERED from raw data
- **Formula:** `log_emp = ln(emp)` where emp = number of employees
- **Definition:** Natural logarithm transformation of industry workforce size
- **Why Log?** 
  - Employment varies orders of magnitude (100 to 100,000+ employees)
  - Log transformation linearizes relationship and stabilizes variance
  - Coefficient interpreted as elasticity: "1% increase in size → X% change in outcome"
- **Range:** Typically 5-12 (for ~150 to ~160,000 employees)
- **Purpose:** Control for industry scale/size effects
- **Missing Values:** Minimal; dropped if missing

---

### **C(YEAR) - Year Fixed Effects**
- **Data Type:** Categorical (factor variable with dummies)
- **Classification:** ENGINEERED (categorical)
- **Definition:** 
  - Separate dummy variable for each year 1958-1999
  - One year dropped as reference category (typically 1958)
  - Captures year-specific shocks common to all industries
- **Number of Dummies:** 41 variables (for 42 years, reference=1)
- **Interpretation:** 
  - Each year dummy = deviation from base year's productivity/efficiency
  - Example: `C(year)[T.1974]` = productivity change specific to 1974 vs. 1958
  - Absorbs macroeconomic shocks (recessions, tech trends), seasonal patterns
- **Why Included?** 
  - Controls for time trends that affect all industries equally
  - Improves causal identification of DiD effects
  - Critical for estimating `post_1973` effect separately

**Example Year Coefficients from Model 1 (TFP):**
```
C(year)[T.1973]   -0.0421 ***    (Productivity dip during crisis)
C(year)[T.1974]   -0.0584 ***    (Worse in immediate aftermath)
C(year)[T.1987]    0.0295 ***    (Recovery period)
```

---

### **C(NAICS) - Industry Fixed Effects**
- **Data Type:** Categorical (factor variable with dummies)
- **Classification:** ENGINEERED (categorical)
- **Definition:**
  - Separate dummy variable for each of ~80 manufacturing industries
  - One industry dropped as reference category
  - Captures industry-specific characteristics that don't change over time
- **Number of Dummies:** ~80 variables (one per industry sector)
- **Industry Classification:** NAICS 3-4 digit manufacturing codes
  - Examples: 3111 (Animal Slaughter), 3251 (Basic Chemicals), 3272 (Cement & Concrete)
- **Interpretation:**
  - Each industry dummy = baseline productivity/efficiency difference vs. reference sector
  - Example: `C(naics)[T.325120]` = chemical refining's base efficiency level
  - Can be large: some industries inherently more/less energy-intense
- **Why Included?**
  - Controls for permanent industry characteristics ("comparative advantage")
  - Prevents industry confounding (e.g., if high-energy industries are all chemicals)
  - Ensures within-industry variation identifies crisis effects

**Example Industry Coefficients from Model 2 (Energy Intensity):**
```
C(naics)[T.324121]    0.0349 ***    (Petroleum refining: high baseline intensity)
C(naics)[T.316110]    0.0142 ***    (Leather: moderate intensity)
C(naics)[T.311211]    0.0334 ***    (Flour milling: high intensity)
```

---

## UNDERLYING RAW DATA INPUTS (NBER Census Source)

### **ENERGY - Total Energy Consumption**
- **Data Type:** Continuous (numeric)
- **Classification:** RAW DATA from NBER
- **Units:** Typically British Thermal Units (BTU) or equivalent
- **Definition:** Total energy (electricity, natural gas, coal, oil) consumed by industry
- **Variation:** Across years and industries
- **Used To Calculate:** `real_energy = energy / pien`

---

### **PIEN - Energy Price Index**
- **Data Type:** Continuous (numeric, indexed to base year)
- **Classification:** RAW DATA from NBER
- **Definition:** Price index for energy inputs (controls for inflation in energy prices)
- **Base Year:** Typically 1958 = 100 or similar
- **Example Values:** 1958=100, 1973=150, 1980=300
- **Purpose:** Deflator to convert nominal energy into real (constant-dollar) energy
- **Used To Calculate:** `real_energy = energy / pien`

---

### **VADD - Value Added / Output**
- **Data Type:** Continuous (numeric)
- **Classification:** RAW DATA from NBER
- **Units:** Nominal dollars (not adjusted for inflation)
- **Definition:** Value of goods produced (gross output minus intermediate inputs)
- **Variation:** Across years and industries
- **Range:** Millions to billions of dollars (varies dramatically by industry)
- **Used To Calculate:** 
  - `real_vadd = vadd / piship`
  - `real_energy_intensity = real_energy / real_vadd`
  - `vadd_growth = pct_change(real_vadd)`

---

### **PISHIP - Shipment/Output Price Index**
- **Data Type:** Continuous (numeric, indexed)
- **Classification:** RAW DATA from NBER
- **Definition:** Price index for industry output (controls for output price inflation)
- **Base Year:** Typically 1958 = 100
- **Example Values:** 1958=100, 1973=110, 1980=150
- **Purpose:** Deflator to convert nominal value added into real (constant-dollar) output
- **Used To Calculate:** 
  - `real_vadd = vadd / piship`
  - `real_energy_intensity = real_energy / real_vadd`

---

### **EMP - Employment**
- **Data Type:** Continuous (integer count)
- **Classification:** RAW DATA from NBER
- **Units:** Number of employees
- **Definition:** Total employment in the industry in a given year
- **Range:** Hundreds to hundreds of thousands
- **Variation:** Across years (growth/decline) and industries
- **Used To Calculate:** `log_emp = ln(emp)`
- **Purpose in Model:** Control for firm/industry size

---

### **YEAR - Calendar Year**
- **Data Type:** Integer
- **Classification:** RAW DATA from NBER
- **Range:** 1958-2018 (analysis focuses on 1958-1999)
- **Used To Calculate:** `post_1973` indicator
- **Used To Create:** Year fixed effects `C(year)`

---

### **NAICS - Industry Classification Code**
- **Data Type:** Integer (categorical)
- **Classification:** RAW DATA from NBER
- **Definition:** North American Industry Classification System code
  - 3-4 digit code identifies specific manufacturing sub-sector
  - Standardized classification across U.S. economic data
- **Examples:**
  - 3111 = Animal Slaughter & Processing
  - 3251 = Basic Chemical Manufacturing
  - 3272 = Cement & Concrete Product Manufacturing
  - 3391 = Personal Care Product Manufacturing
- **Used To Calculate:** Industry fixed effects `C(naics)`

---

### **TFP5 - Total Factor Productivity (Raw)**
- **Data Type:** Continuous (numeric)
- **Classification:** RAW DATA - Pre-calculated by NBER researchers
- **Definition:** Measure of productive efficiency = (Output) / (Capital^α × Labor^β × Materials^γ)
- **Calculation Method:** NBER used growth accounting methodology
- **Interpretation:** 
  - Reflects technological progress, management efficiency, organization
  - Not directly energy-related; hence makes good outcome variable
- **Variation:** Across years and industries
- **Used In:** Model 1 as dependent variable

---

### **CAP - Capital Stock**
- **Data Type:** Continuous (numeric)
- **Classification:** RAW DATA from NBER
- **Units:** Dollar value (typically millions)
- **Definition:** Stock of productive capital (buildings, equipment, machinery)
- **Used For:** Propensity score matching (to create balanced control groups)
- **NOT directly in regressions** but used in preprocessing step

---

## DERIVED VARIABLES - IMPORTANT COMBINATIONS

### **REAL_ENERGY = ENERGY / PIEN**
- **Purpose:** Convert nominal energy to constant prices (real terms)
- **Example:** If nominal energy consumption doubled but energy prices also doubled, real_energy is unchanged
- **Use:** Input to calculate energy intensity and growth rates

---

### **REAL_VADD = VADD / PISHIP**
- **Purpose:** Convert nominal output to constant prices (real terms)
- **Example:** If industry output (VADD) grew 50% but output prices rose 30%, real growth is only 15%
- **Use:** Input to calculate energy intensity and growth rates

---

## SUMMARY TABLE: ALL VARIABLES

| # | Variable | Type | Source | Role in Models | Classification |
|---|----------|------|--------||---|---|
| 1 | `tfp5` | Continuous | NBER Census | Outcome (Model 1) | **RAW** |
| 2 | `real_energy_intensity` | Continuous | Engineered | Outcome (Model 2) | **ENGINEERED** |
| 3 | `vadd_growth` | Continuous | Engineered | Outcome (Model 3) | **ENGINEERED** |
| 4 | `is_high_energy` | Binary | Engineered | Treatment indicator | **ENGINEERED** |
| 5 | `post_1973` | Binary | Engineered | Period indicator | **ENGINEERED** |
| 6 | `is_high_energy:post_1973` | Binary | Engineered | DiD coefficient (main effect) | **ENGINEERED** |
| 7 | `log_emp` | Continuous | Engineered | Control variable | **ENGINEERED** |
| 8 | `C(year)` | Categorical | Engineered | Time fixed effects | **ENGINEERED** |
| 9 | `C(naics)` | Categorical | Engineered | Industry fixed effects | **ENGINEERED** |
| 10 | `energy` | Continuous | NBER Census | Input to real_energy | **RAW** |
| 11 | `pien` | Continuous | NBER Census | Deflator for energy | **RAW** |
| 12 | `vadd` | Continuous | NBER Census | Input to real_vadd | **RAW** |
| 13 | `piship` | Continuous | NBER Census | Deflator for output | **RAW** |
| 14 | `emp` | Continuous | NBER Census | Input to log_emp | **RAW** |
| 15 | `year` | Integer | NBER Census | Used for period classification | **RAW** |
| 16 | `naics` | Integer | NBER Census | Used for industry classification | **RAW** |

---

## NOTES ON DATA QUALITY & AVAILABILITY

- **Period Covered:** 1958-2018 (analysis focuses on 1958-1999 to capture immediate crisis effects)
- **Sample Size:** 6,342 observations (after removing missing values)
- **Industries:** ~80 manufacturing sectors
- **Time Variation:** Some variables vary each year for each industry; others are time-invariant per industry
- **Missing Data:** Minimal (<1% across indicators); rows with missing TFP, energy intensity, or growth dropped
- **Data Source:** NBER Census of Manufactures, a highly reputable long-term panel
- **Accessibility:** Data is publicly available at www.nber.org

