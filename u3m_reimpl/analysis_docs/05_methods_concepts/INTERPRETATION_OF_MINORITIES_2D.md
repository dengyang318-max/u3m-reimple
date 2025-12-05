## How Ray-sweeping Identifies Minority Groups in 2D (Chicago Crimes / College Admission)

This note explains how the 2D Ray-sweeping algorithm in the paper *“Mining the Minoria Unknown”* discovers
minority groups, why different data processing choices can lead to different tails, and how we should interpret
these differences in practice.

The goal is **not** to get a single fixed list of individuals, but to reveal **stable patterns of skewed behaviour**
under different experimental settings.

---

### 1. What the algorithm actually does

Given:
- a fixed dataset \(D\) (e.g. Chicago Crimes after preprocessing),
- a 2D feature space (e.g. \((\text{Lon}, \text{Lat})\) or \((\text{GRE}, \text{GPA})\)),
- a binary label (e.g. Arrest / admit),

the 2D Ray-sweeping algorithm:

1. Enumerates many candidate directions \(f\) in the 2D plane.
2. For each direction, projects all points to a scalar coordinate \(p_f = x \cdot f\).
3. Computes a **skewness** score measuring how asymmetric the projected distribution is around its median.
4. Picks the directions with the highest absolute skew.
5. For a chosen high-skew direction \(f\), takes the **tail** (e.g. top 1% or 0.1%) of the projected points as a
   candidate minority group.

Important:
- The algorithm does **not** pre-specify who is “minority”.
- It only says: “along this direction, the tail subset behaves very differently (label-wise) from the global data”.

---

### 2. Will different samples lead to different tails?

Yes – and this is expected.

- If we change the sample used for Ray-sweeping (e.g. 500 vs 1000 vs a larger subsample),
  the empirical distribution (mean / median / variance / density peaks) changes slightly.
- Ray-sweeping then finds **directions with maximal skew for that specific sample**.

Consequences:
- The **top-1 direction** can rotate by a noticeable amount when we change the sample.
- Even if the direction is similar, the exact set of points in the top 1% tail will generally differ
  between samples.

This does **not** mean the algorithm is invalid. It reflects the fact that we are solving an **extreme-value
problem on finite samples**: small changes in data can move the exact maximizer, but the underlying pattern
often remains similar.

---

### 3. Data processing choices = redefining the “problem”

Data pre-processing choices (filtering, normalization, feature selection) implicitly define:

> “In which space, and under which conditions, do we want to measure skewness and minority status?”

Examples:
- Choosing only \((\text{Lon}, \text{Lat})\) vs adding time or crime type changes the geometry of the space.
- Min–max normalization vs other scalings affects how distances and angles behave.
- Filtering years or outliers changes which regions are even eligible to be discovered.

Different choices ⇒ different geometric / statistical structure ⇒ the Ray-sweeping algorithm discovers
**different high-skew directions and tails**.

This is not a bug; it means that:
- **Each experiment corresponds to a specific “fairness / minority” viewpoint**, and the algorithm is exploring
  skewness under that viewpoint.

---

### 4. Why “minority groups” are not a fixed list of individuals

In the paper and the official notebooks, the authors care about **patterns**, not individual IDs:

- On Chicago Crimes, the key qualitative finding is that:
  - Some directions correspond to regions in North-side Chicago where the Arrest behaviour is highly skewed.
  - Different tails (under slightly different settings) still concentrate in similar geographic regions,
    even though the exact set of points changes.

Thus:
- At the **macro level**, different runs often highlight **the same type of region / pattern**:
  - same part of the city,
  - similar label distributions,
  - similar model error behaviour (e.g. baseline model is much less accurate on the tail than globally).
- At the **micro level**, each run may include slightly different individual points in the tail.

We should interpret “the minority group” as:

> A **structural pattern** (a region / direction / combination of attributes) that systematically exhibits
> skewed label behaviour, rather than an immutable set of individual data points.

---

### 5. Does the algorithm have intrinsic bias?

Under correct implementation, the algorithm itself does **not** encode prior bias towards any predefined group:

- It treats all directions in the chosen space symmetrically.
- It ranks them purely by statistical skewness of labels along the projection.
- Any “preference” it shows arises from:
  - the empirical data distribution,
  - the choice of features and pre-processing,
  - and in practice, from implementation details like dynamic vs fixed median, enumeration heuristics, etc.

So:
- If we keep the implementation stable but change the dataset or pre-processing, and the discovered tails change,
  this does **not** imply algorithmic unfairness.
- It means that under the new conditions, **different parts of the population exhibit the most extreme skew**.

---

### 6. How to explain “different minority groups” in experiments

When reporting experimental results, a good explanation structure is:

1. **State the setting**  
   - Which dataset, time range, filters?
   - Which 2D features, and how normalized?
   - Which label is used?

2. **Describe the discovered pattern, not just the raw tail set**  
   - Location of the high-skew region (e.g. North-side vs South-side).  
   - Label proportions in global vs tail (e.g. Arrest rate, admit rate).  
   - Baseline model performance in global vs tail (accuracy / F1).

3. **Compare across settings**  
   - If different pre-processing or sampling leads to different tails but consistently highlights
     similar regions / attribute combinations, we can say:
     > “Across multiple settings, the algorithm consistently identifies this region / group as an
     > under-performing minority.”
   - If radically different regions appear under different settings, then:
     > “Under setting A, the main minority pattern is X; under setting B, it is Y.
     > This shows that perceived minorities depend strongly on which features and scales we consider.”

4. **Clarify the meaning of “same group”**  
   - “Same group” ≠ same list of point IDs  
   - “Same group” ≈ same **semantic pattern**: similar geographic / attribute region, similar direction \(f\),
     similar skew and model error behaviour.

---

### 7. Practical takeaway for this reimplementation

For the reimplemented 2D experiments (College Admission & Chicago Crimes):

- If your Ray-sweeping implementation is correct (including dynamic median logic, intersection enumeration, etc.),
  and you observe that:
  - Under slightly different sampling or normalization, the **exact tail members** change,
  - But the **regions / directions / statistics** are similar,
  
  then this is consistent with the original algorithm design:
  - The method is robust at the pattern level, not at the individual ID level.

- When documenting results, focus on:
  - **where** the high-skew regions are,
  - **how** their label distributions differ from global,
  - **how** baseline models perform differently on these regions,
  
  rather than on perfectly matching every point with the official notebooks.


