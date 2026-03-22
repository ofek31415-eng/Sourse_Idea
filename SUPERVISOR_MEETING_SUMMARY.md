# SUPERVISOR MEETING: ONE-PAGE VISUAL SUMMARY

## RESEARCH DESIGN: Difference-in-Differences Analysis
**Context:** 1973 Oil Crisis → Sought to understand impact on U.S. manufacturing

---

## MODEL COMPARISON TABLE

| Feature | **Model 1: Productivity** | **Model 2: Efficiency** | **Model 3: Growth** |
|---------|--------------------------|------------------------|-------------------|
| **Research Q.** | Did high-energy industries gain productivity? | Did they improve energy efficiency? | What was the growth impact? |
| **Dependent Variable (Y)** | `tfp5` | `real_energy_intensity` | `vadd_growth` |
| **Y Type** | RAW DATA | ENGINEERED | ENGINEERED |
| **Formula (if eng.)** | — | real_energy ÷ real_vadd | pct_change(real_vadd) × 100 |
| **Interpretation** | Level of productivity | Energy used per $ output | Annual % output change |
| **N Observations** | 6,342 | 6,342 | 4,832 |
| **Adj. R²** | 0.799 | 0.890 | 0.203 |
| **Expected Sign** | + (crisis drove innovation) | − (more efficiency post-crisis) | − (growth hit) |

---

## INDEPENDENT VARIABLES (All Models Use Same X's)

### **Treatment Variables**
| Variable | Type | Classification | How Computed | Interpretation |
|----------|------|-----------------|--------------|-----------------|
| `is_high_energy` | Binary | ENGINEERED | Top tercile by avg energy intensity (1958-72) | =1 if high-energy industry |
| `post_1973` | Binary | ENGINEERED | = 1 if year ≥ 1973 | =1 if post-crisis period |
| `is_high_energy × post_1973` | Binary | ENGINEERED | Product of above | **DiD Effect** ← Main coefficient |

### **Control Variables**
| Variable | Type | Classification | Formula | Purpose |
|----------|------|-----------------|---------|---------|
| `log_emp` | Continuous | ENGINEERED | ln(emp) | Control for industry size |
| `C(year)` | Categorical | ENGINEERED | Dummy for each 1958-1999 | Remove year-specific shocks |
| `C(naics)` | Categorical | ENGINEERED | Dummy for each industry | Remove industry fixed traits |

---

## DATA SOURCES & CLASSIFICATIONS

### **Where Data Comes From**

| Data Source | Components |
|------------|-----------|
| **NBER Census of Manufactures** (RAW) | tfp5, energy, pien, vadd, piship, emp, year, naics |
| **Your Analysis** (ENGINEERED) | real_energy, real_vadd, real_energy_intensity, vadd_growth, is_high_energy, post_1973, log_emp |

### **Quick Classification Checklist**

✓ **RAW DATA** (directly from NBER CSV):
- `tfp5` — Pre-calculated productivity
- `energy` — Energy consumption
- `pien` — Energy price index
- `vadd` — Value added (output)
- `piship` — Output price index
- `emp` — Employment
- `year`, `naics` — Classification codes

✓ **ENGINEERED/CREATED** (calculated in your code):
- `real_energy = energy / pien` (deflated energy)
- `real_vadd = vadd / piship` (deflated output)
- `real_energy_intensity = real_energy / real_vadd` (energy per unit output)
- `vadd_growth = pct_change(real_vadd) × 100` (annual % growth)
- `is_high_energy` = tercile dummy (based on 1958-72 ranking)
- `post_1973` = period dummy
- `log_emp = ln(emp)` (log scale)
- `C(year)`, `C(naics)` = fixed effect dummies

---

## CAUSAL STORY (DiD Logic)

```
┌─────────────────────────────────────────────────────────────────┐
│                     1973 OIL EMBARGO SHOCK                       │
│                          ↓                                       │
│            (Energy prices suddenly increased)                    │
│                          ↓                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  HIGH-ENERGY INDUSTRIES        LOW-ENERGY INDUSTRIES             │
│  (treatment group)              (control group)                   │
│  ↓                              ↓                                │
│  Expected Response:             Expected Response:              │
│  • Reduce energy use            • Minimal change                 │
│  • Invest in efficiency         • Steady state                   │
│  • Innovate (or shrink)         • Unaffected                     │
│                                                                   │
│  Observed Outcomes:             Observed Outcomes:              │
│  Model 1: +? in TFP            Model 1: ± TFP                   │
│  Model 2: −? Intensity         Model 2: ~ Intensity            │
│  Model 3: −? Growth            Model 3: ~ Growth              │
│                                                                   │
│  DiD = (High Post - High Pre) - (Low Post - Low Pre)            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## KEY REGRESSION EQUATIONS (Formal)

### Model 1
```
tfp5ᵢₜ = α + β₁(is_high_energyᵢ) + β₂(post_1973ₜ) 
         + β₃(is_high_energyᵢ × post_1973ₜ) 
         + β₄ log(empᵢₜ) 
         + γₜ + δᵢ + εᵢₜ

where: i = industry, t = year
       γₜ = year FE, δᵢ = industry FE
       β₃ = DiD coefficient (main effect of interest)
```

### Models 2 & 3
Same structure, different Y:
- Model 2: Y = real_energy_intensity
- Model 3: Y = vadd_growth

---

## INTERPRETATION GUIDE

### How to Read Results

**If DiD coefficient (β₃) is:**
- **Positive & Significant** (Model 1 TFP): High-energy industries improved productivity MORE post-crisis
- **Negative & Significant** (Model 2 Energy): High-energy industries reduced intensity MORE post-crisis
- **Negative & Significant** (Model 3 Growth): High-energy industries grew SLOWER post-crisis

**Magnitude Matters:**
- Small effect: Crisis had modest differential impact
- Large effect: Crisis severely disrupted high-energy industries OR forced major adaptation

---

## ROBUSTNESS CHECKS PERFORMED

| Check | Purpose |
|-------|---------|
| Propensity Score Matching (PSM) | Balance high/low energy groups on baseline size variables |
| Multiple baseline periods | Base 1968-72, 1958-72, 1958 only |
| Event study design | Year-by-year DiD estimates (dynamic effects) |
| Sub-sample analysis | By industry size (small/medium/large) |

---

## DATA PROCESSING FLOWCHART

```
1. Load NBER CSV
   ↓
2. Deflate nominal values
   • real_energy = energy / pien
   • real_vadd = vadd / piship
   ↓
3. Calculate intensity & growth
   • real_energy_intensity = real_energy / real_vadd
   • vadd_growth = pct_change(real_vadd) × 100
   ↓
4. Create treatment indicators
   • is_high_energy = rank & classify by pre-1973 intensity
   • post_1973 = year >= 1973
   ↓
5. Create controls
   • log_emp = ln(emp)
   • Year dummies, Industry dummies
   ↓
6. Apply Propensity Score Matching
   ↓
7. Fit 3 separate OLS models
   ↓
8. Results → Results files & visualizations
```

---

## FINAL CHECKLIST: What Your Supervisor Needs to Understand

✓ **Dependent Variables (What you're explaining):**
- [ ] Model 1: Productivity (TFP)
- [ ] Model 2: Energy efficiency (intensity)
- [ ] Model 3: Economic growth

✓ **Independent Variables (How you're explaining it):**
- [ ] Treatment: High vs. Low energy intensive industries
- [ ] Period: Before vs. After 1973
- [ ] Interaction (DiD): Differential effect on treated post-crisis
- [ ] Controls: Firm size, time trends, industry fixed effects

✓ **Raw vs. Engineered:**
- [ ] Can explain which variables came from NBER CSV
- [ ] Can explain which were calculated/transformed
- [ ] Can defend the transformations (e.g., why log energy/prices?)

✓ **Identification Strategy:**
- [ ] Can explain why DiD design is appropriate
- [ ] Can discuss parallel trends assumption
- [ ] Can mention PSM for balanced comparison

✓ **Results:**
- [ ] Know the signs & magnitudes of DiD coefficients
- [ ] Can interpret what they mean economically
- [ ] Can discuss significance & robustness

---

**For Your Supervisor Meeting, Print This + Bring:**
1. This one-page visual summary
2. QUICK_REFERENCE.md for detailed reference
3. Result files (results_model1_tfp.txt, etc.) to reference specific numbers

