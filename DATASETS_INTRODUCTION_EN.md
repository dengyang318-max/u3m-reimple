# Datasets Introduction and Data Processing Guide

This document provides a comprehensive introduction to the two datasets used in the ray-sweeping experiments: **Chicago Crimes** and **College Admission**. It covers data sources, preprocessing steps, and how the data is used in the experiments.

---

## 1. Overview

Two real-world datasets are used to evaluate the ray-sweeping algorithm for finding high-skew directions:

| Dataset | Source | Size | Features | Target | Use Case |
|---------|--------|------|----------|--------|----------|
| **Chicago Crimes** | Kaggle | ~510,372 records | Longitude, Latitude | Arrest | Large-scale spatial data |
| **College Admission** | Kaggle | 400 records | GRE, GPA | Admit | Small-scale educational data |

---

## 2. Chicago Crimes Dataset

### 2.1 Data Source

**Kaggle Dataset**: `currie32/crimes-in-chicago`  
**File**: `Chicago_Crimes_2012_to_2017.csv`  
**Download Method**: Automatic download via `kagglehub` if `--csv-path` is not provided

**Dataset Description**:
- Contains crime incident records from Chicago, Illinois
- Original dataset includes records from 2012 to 2017
- Includes various crime types, locations, dates, and outcomes

### 2.2 Data Preprocessing Pipeline

The preprocessing steps are implemented in `load_chicago_crimes_data()` function:

#### Step 1: Data Loading
```python
data = pd.read_csv(csv_path)
```

#### Step 2: Missing Value Handling
```python
data = data.dropna()  # Remove rows with any missing values
```

#### Step 3: Column Cleanup
- Remove identifier columns: `ID`, `Case Number`
- Remove metadata columns: `Updated On`

#### Step 4: Date Feature Extraction
```python
# Parse Date column and extract temporal features
data["date2"] = pd.to_datetime(data["Date"], errors="coerce")
data["Year"] = data["date2"].dt.year
data["Month"] = data["date2"].dt.month
data["Day"] = data["date2"].dt.day
data["Hour"] = data["date2"].dt.hour
data["Minute"] = data["date2"].dt.minute
data["Second"] = data["date2"].dt.second
data = data.drop(["Date", "date2"], axis=1)
```

#### Step 5: Time-Based Filtering
```python
# Keep only records from 2015-2017 (matching paper's experiment)
data = data[(data["Year"] >= 2015) & (data["Year"] <= 2017)]
```

**Rationale**: Focuses on recent data, reduces dataset size from ~1.4M to ~510K records, and aligns with the official experiment setup.

#### Step 6: Categorical Feature Encoding
```python
# Factorize categorical columns to numeric codes
for col in ["Block", "IUCR", "Description", "Location Description", 
            "FBI Code", "Location", "Primary Type"]:
    if col in data.columns:
        data[col] = pd.factorize(data[col])[0]
```

**Purpose**: Converts categorical features to numeric codes for compatibility with machine learning models.

#### Step 7: Geographic Coordinate Processing
```python
# Remove outliers (latitude > 40 filters out invalid coordinates)
data = data[data["Latitude"] > 40]

# Min-max normalization to [0, 1]
lon = data["Longitude"].astype(float)
lat = data["Latitude"].astype(float)
data["Longitude"] = (lon - lon.min()) / (lon.max() - lon.min())
data["Latitude"] = (lat - lat.min()) / (lat.max() - lat.min())

# Add convenience aliases
data["Lon"] = data["Longitude"]
data["Lat"] = data["Latitude"]
```

**Key Points**:
- **Outlier removal**: Filters out invalid coordinates (latitude ≤ 40)
- **Normalization**: Maps coordinates to [0, 1] range for consistent scaling
- **Preserves spatial relationships**: Relative positions are maintained

#### Step 8: Target Variable Setup
```python
if "Arrest" in data.columns:
    data["Target"] = data["Arrest"]  # Binary: 1 = arrested, 0 = not arrested
```

### 2.3 Point Set Construction for Experiments

After preprocessing, two 2D point sets are constructed for the ray-sweeping algorithm:

#### Primary Point Set
```python
x_train_new = np.asarray(data[["Lon", "Lat"]], dtype=float)
# Shape: (n_samples, 2)
# Each point: (normalized_longitude, normalized_latitude)
```

#### Rotated Point Set
```python
max_lat = float(np.max(x_train_new[:, 1]))
x_train_new_prime = np.column_stack((
    max_lat - x_train_new[:, 1],  # First coordinate: max_lat - lat
    x_train_new[:, 0]              # Second coordinate: lon
))
# Shape: (n_samples, 2)
# Each point: [max_lat - lat, lon]
```

**Purpose**: The rotated point set enables coverage of all directions in the dual space through the `vector_transfer` mechanism.

### 2.4 Sampling Strategy

**Default**: Sample 500 points (matching the paper's experiment)
```python
final_df = data.sample(n=500, random_state=1)
```

**Rationale**:
- Full dataset (~510K records) is too large for O(n²) intersection enumeration
- 500 points provide sufficient coverage while maintaining computational efficiency
- Fixed random seed ensures reproducibility

### 2.5 Final Dataset Characteristics

| Property | Value |
|----------|-------|
| **Original Size** | ~1.4M records (2012-2017) |
| **After Filtering** | ~510,372 records (2015-2017) |
| **After Sampling** | 500 points (default) |
| **Features Used** | Longitude (normalized), Latitude (normalized) |
| **Target Variable** | Arrest (binary: 0/1) |
| **Coordinate Range** | [0, 1] for both Lon and Lat |

---

## 3. College Admission Dataset

### 3.1 Data Source

**Kaggle Dataset**: `eswarchandt/admission`  
**File**: `Admission.xlsx`  
**Download Method**: Automatic download via `kagglehub` if `--excel-path` is not provided

**Dataset Description**:
- Contains graduate school admission records
- Includes student characteristics and admission outcomes
- Smaller, more focused dataset compared to Chicago Crimes

### 3.2 Data Preprocessing Pipeline

The preprocessing steps are implemented in `load_college_admission_data()` function:

#### Step 1: Data Loading
```python
data = pd.read_excel(excel_path)
```

#### Step 2: Column Validation
```python
# Verify required columns exist
if "gre" not in data.columns or "gpa" not in data.columns:
    raise ValueError("Expected columns `gre` and `gpa` in the dataset")
```

#### Step 3: Feature Normalization
```python
# Extract and normalize GRE scores
gre = data["gre"].astype(float)
data["gre"] = (gre - gre.min()) / (gre.max() - gre.min())

# Extract and normalize GPA scores
gpa = data["gpa"].astype(float)
data["gpa"] = (gpa - gpa.min()) / (gpa.max() - gpa.min())
```

**Normalization Method**: Min-max normalization to [0, 1] range
- **GRE**: `gre_norm = (gre - min(gre)) / (max(gre) - min(gre))`
- **GPA**: `gpa_norm = (gpa - min(gpa)) / (max(gpa) - min(gpa))`

**Purpose**: 
- Ensures both features are on the same scale
- Prevents features with larger ranges from dominating the algorithm
- Matches the official notebook's preprocessing

### 3.3 Point Set Construction for Experiments

After preprocessing, two 2D point sets are constructed:

#### Primary Point Set
```python
x_train_new = np.asarray(data[["gre", "gpa"]], dtype=float)
# Shape: (n_samples, 2)
# Each point: (normalized_GRE, normalized_GPA)
```

#### Rotated Point Set
```python
max_gpa = float(np.max(x_train_new[:, 1]))
x_train_new_prime = np.column_stack((
    max_gpa - x_train_new[:, 1],  # First coordinate: max_gpa - gpa
    x_train_new[:, 0]              # Second coordinate: gre
))
# Shape: (n_samples, 2)
# Each point: [max_gpa - gpa, gre]
```

**Purpose**: Same as Chicago Crimes - enables full direction space coverage.

### 3.4 Sampling Strategy

**Default**: Use all 400 records (no sampling)
```python
# Shuffle entire dataset with fixed random seed
final_df = data.sample(frac=1.0, random_state=0)
```

**Rationale**:
- Dataset is small enough (400 records) to process entirely
- No need for subsampling
- Fixed random seed ensures consistent ordering

**Optional Sampling**: If `--n-samples` is provided:
```python
if n_samples is not None:
    df_used = data.sample(n=n_samples, random_state=0)
```

### 3.5 Final Dataset Characteristics

| Property | Value |
|----------|-------|
| **Original Size** | 400 records |
| **After Preprocessing** | 400 records (all used) |
| **Features Used** | GRE (normalized), GPA (normalized) |
| **Target Variable** | Admit (binary: 0/1) |
| **Feature Range** | [0, 1] for both GRE and GPA |

---

## 4. Common Data Processing Patterns

### 4.1 Normalization Strategy

Both datasets use **min-max normalization** to map features to [0, 1]:

```python
normalized_value = (value - min) / (max - min)
```

**Benefits**:
- Consistent scale across features
- Preserves relative relationships
- Prevents numerical instability

### 4.2 Point Set Rotation Strategy

Both datasets construct rotated point sets using the same pattern:

**Primary**: `(feature1, feature2)`  
**Rotated**: `[max_feature2 - feature2, feature1]`

**Purpose**:
- Enables coverage of all directions in dual space
- Combined with `vector_transfer`, maps rotated directions back to original space
- Ensures no high-skew directions are missed

### 4.3 Random Seed Usage

**Chicago Crimes**: `random_state=1` for sampling  
**College Admission**: `random_state=0` for shuffling

**Purpose**: Ensures reproducibility across different runs.

---

## 5. Data Usage in Experiments

### 5.1 Experiment Workflow

1. **Data Loading**: Load raw data from CSV/Excel or download via kagglehub
2. **Preprocessing**: Apply normalization, filtering, and feature extraction
3. **Point Set Construction**: Build primary and rotated point sets
4. **Ray-Sweeping**: Run algorithm on point sets to find high-skew directions
5. **Tail Analysis**: Analyze tail regions for discovered directions
6. **Model Evaluation**: Evaluate model performance on tail regions

### 5.2 Key Differences Between Datasets

| Aspect | Chicago Crimes | College Admission |
|--------|----------------|-------------------|
| **Scale** | Large (~510K records) | Small (400 records) |
| **Sampling** | Yes (500 points default) | No (use all records) |
| **Features** | Geographic (Lon, Lat) | Academic (GRE, GPA) |
| **Domain** | Spatial/Geographic | Educational |
| **Complexity** | High (many features) | Low (focused features) |

### 5.3 Experimental Parameters

#### Chicago Crimes
- **Default samples**: 500 points
- **Percentiles for tail analysis**: `[1.0, 0.1, 0.01, 0.001, 0.0001]`
- **Model**: Logistic Regression trained on full dataset

#### College Admission
- **Default samples**: All 400 records
- **Percentiles for tail analysis**: `[1.0, 0.5, 0.2, 0.1, 0.08, 0.04]`
- **Model**: Logistic Regression trained on full dataset

---

## 6. Code References

### 6.1 Data Loading Functions

**Chicago Crimes**:
- File: `u3m_reimpl/experiments/experiment_ray_sweeping_2d_chicago_crimes_official_utils.py`
- Function: `load_chicago_crimes_data(csv_path)`
- Lines: ~75-167

**College Admission**:
- File: `u3m_reimpl/experiments/experiment_ray_sweeping_2d_college_admission_official_utils.py`
- Function: `load_college_admission_data(excel_path)`
- Lines: ~73-93

### 6.2 Point Set Construction Functions

**Chicago Crimes**:
- Function: `build_point_sets_from_data(data, n_samples=500)`
- Returns: `(x_train_new, x_train_new_prime, targets, final_df)`

**College Admission**:
- Function: `build_point_sets_from_data(data, n_samples=None)`
- Returns: `(x_train_new, x_train_new_prime, targets)`

### 6.3 Experiment Scripts

**Chicago Crimes**:
- `experiment_ray_sweeping_2d_chicago_crimes_official_utils.py`

**College Admission**:
- `experiment_ray_sweeping_2d_college_admission_official_utils.py`

---

## 7. Data Quality and Considerations

### 7.1 Chicago Crimes

**Potential Issues**:
- **Missing values**: Handled by dropping rows with any missing data
- **Invalid coordinates**: Filtered by latitude > 40
- **Temporal bias**: Only uses 2015-2017 data (may not represent all time periods)
- **Sampling bias**: 500-point sample may not represent full dataset

**Mitigation Strategies**:
- Fixed random seed for reproducibility
- Outlier filtering for geographic coordinates
- Time-based filtering matches official experiment

### 7.2 College Admission

**Potential Issues**:
- **Small sample size**: 400 records may limit statistical power
- **Feature simplicity**: Only 2 features (GRE, GPA) may not capture all factors
- **Potential bias**: Dataset may not represent all applicant populations

**Mitigation Strategies**:
- Uses all available records (no sampling)
- Normalization ensures fair feature weighting
- Fixed random seed for consistent ordering

---

## 8. Summary

### 8.1 Chicago Crimes Dataset

- **Source**: Kaggle (`currie32/crimes-in-chicago`)
- **Size**: ~510,372 records (2015-2017), sampled to 500 points
- **Features**: Normalized Longitude, Latitude
- **Target**: Arrest (binary)
- **Use Case**: Large-scale spatial data analysis

### 8.2 College Admission Dataset

- **Source**: Kaggle (`eswarchandt/admission`)
- **Size**: 400 records (all used)
- **Features**: Normalized GRE, GPA
- **Target**: Admit (binary)
- **Use Case**: Small-scale educational data analysis

### 8.3 Common Processing Steps

1. **Normalization**: Min-max to [0, 1]
2. **Point Set Construction**: Primary + Rotated sets
3. **Reproducibility**: Fixed random seeds
4. **Quality Control**: Outlier removal, missing value handling

Both datasets are processed consistently to ensure fair comparison and reproducible results across different experimental configurations.

---

**Document Generated**: 2025-12-08

