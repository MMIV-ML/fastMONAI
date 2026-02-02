# ROR SELECT Statement Syntax Reference

## Grammar Overview

The SELECT statement filters DICOM studies and series using a SQL-like domain-specific language (DSL).

## Basic Syntax

```sql
SELECT <output_level> FROM study
WHERE series [NAMED "<label>"] HAS <conditions>
[ALSO WHERE series [NAMED "<label>"] HAS <conditions>]
[CHECK <comparison_rules>]
```

## Output Levels

| Level | Returns | Use Case |
|-------|---------|----------|
| `patient` | All studies for matching patients | Multi-study/longitudinal workflows |
| `study` | All series for matching studies | Study-level processing |
| `series` | Only matching series (default) | Single-series operations |
| `project` | Complete dataset | Model training |

## NAMED Clause

Labels a series for reference in workflow processing:

```sql
WHERE series NAMED "T1" HAS Modality == 'MR'
```

The label "T1" can be used later to reference this specific series in processing pipelines.

## Operators

### Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Exact match (for lists: any element matches) | `Modality == 'MR'` |
| `=` | Same as `==` | `Modality = 'MR'` |
| `containing` | List contains value | `ClassifyType containing DIFFUSION` |
| `regexp` | Regular expression (case-sensitive) | `SeriesDescription regexp '^T1.*3D'` |
| `>` | Greater than | `NumImages > 50` |
| `<` | Less than | `NumImages < 500` |
| `>=` | Greater than or equal | `NumImages >= 100` |
| `<=` | Less than or equal | `NumImages <= 200` |
| `approx` | Approximate equality (1e-4 tolerance) | `SliceThickness approx 1.0` |

### Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `AND` | All conditions must be true | `Modality == 'MR' AND NumImages > 50` |
| `and` | Same as AND (case-insensitive) | `Modality == 'MR' and NumImages > 50` |
| `OR` | Any condition can be true | `Modality == 'MR' OR Modality == 'CT'` |
| `or` | Same as OR (case-insensitive) | `Modality == 'MR' or Modality == 'CT'` |
| `NOT()` | Negate a condition | `NOT(NumImages > 200)` |
| `not()` | Same as NOT (case-insensitive) | `not(NumImages > 200)` |

## Queryable DICOM Tags

### Standard Tags

| Tag | Type | Description |
|-----|------|-------------|
| `Modality` | String | Imaging modality (MR, CT, US, PT, NM, etc.) |
| `SeriesDescription` | String | Series description text |
| `StudyDescription` | String | Study description text |
| `SequenceName` | String | MR sequence name |
| `PatientName` | String | Patient name |
| `PatientID` | String | Patient identifier |
| `StudyDate` | String | Date of study (YYYYMMDD format) |
| `SeriesNumber` | Integer | Series sequence number |
| `NumImages` | Integer | Number of images in series |
| `Manufacturer` | String | Equipment manufacturer |
| `ManufacturerModelName` | String | Equipment model |
| `FrameOfReferenceUID` | String | Spatial registration reference |

### Classification Tags

| Tag | Type | Description |
|-----|------|-------------|
| `ClassifyType` | List | Custom classification types |
| `ClassifyTypes` | List | Alias for ClassifyType |

Common ClassifyType values: `T1`, `T2`, `FLAIR`, `DIFFUSION`, `RESTING`, `PERFUSION`, `SWI`, `ASL`

### Custom DICOM Tags

Reference non-standard tags using hex notation:

```sql
WHERE series HAS ("0x0018","0x0050") > 1.0
```

Format: `("group","element")` where group and element are hex values.

## ALSO WHERE Clause

Requires additional series to be present in the same study/patient:

```sql
SELECT patient FROM study
WHERE series NAMED "T1" HAS
  Modality == 'MR' AND SeriesDescription regexp 'T1'
ALSO WHERE series NAMED "T2" HAS
  Modality == 'MR' AND SeriesDescription regexp 'T2'
ALSO WHERE series NAMED "DWI" HAS
  ClassifyType containing DIFFUSION
```

This returns only patients who have ALL three series types.

## CHECK Clause

Adds comparison rules between named series:

```sql
SELECT patient FROM study
WHERE series NAMED "pre" HAS SeriesDescription regexp 'PRE'
ALSO WHERE series NAMED "post" HAS SeriesDescription regexp 'POST'
CHECK pre@StudyDate < post@StudyDate
```

The `@` notation references a field from a named series.

Supported CHECK comparisons:
- `series1@field == series2@field` - Equality
- `series1@field < series2@field` - Less than (for dates/numbers)
- `series1@FrameOfReferenceUID = series2@FrameOfReferenceUID` - Spatial co-registration

## Complete Examples

### Example 1: Single Modality Filter

Find all MR series:

```sql
SELECT series FROM study
WHERE series HAS Modality == 'MR'
```

### Example 2: Description Pattern Matching

Find T1-weighted images with specific naming:

```sql
SELECT series FROM study
WHERE series NAMED "T1" HAS
  Modality == 'MR' AND SeriesDescription regexp 'T1.*3D.*MPRAGE'
```

### Example 3: Numeric Range Filter

Find series with specific image count range:

```sql
SELECT series FROM study
WHERE series HAS
  Modality == 'MR'
  AND NumImages > 50
  AND NumImages < 500
```

### Example 4: CT Series with Minimum Slices

```sql
SELECT series FROM study
WHERE series HAS
  Modality == 'CT'
  AND NumImages > 100
```

### Example 5: Multi-Series Research Workflow

Find patients with complete brain MRI protocol:

```sql
SELECT patient FROM study
WHERE series NAMED "T1" HAS
  ClassifyType containing T1
ALSO WHERE series NAMED "FLAIR" HAS
  SeriesDescription regexp 'FLAIR'
ALSO WHERE series NAMED "DWI" HAS
  ClassifyType containing DIFFUSION
ALSO WHERE series NAMED "REST" HAS
  ClassifyType containing RESTING
  AND NumImages > 10
  AND NOT(NumImages > 200)
```

### Example 6: Pattern Exclusion

Find T2 series but exclude FLAIR:

```sql
SELECT series FROM study
WHERE series HAS
  Modality == 'MR'
  AND SeriesDescription regexp 'T2'
  AND NOT(SeriesDescription regexp 'FLAIR')
```

### Example 7: Description Prefix Match

Find all series starting with "Anat":

```sql
SELECT series FROM study
WHERE series NAMED "Anat" HAS
  Modality == 'MR' AND SeriesDescription regexp '^Anat'
```

### Example 8: Co-registered Multi-Modal

Find T1 and diffusion with same frame of reference:

```sql
SELECT study FROM study
WHERE series NAMED "T1" HAS
  ClassifyType containing T1
ALSO WHERE series NAMED "DIFF" HAS
  ClassifyType containing DIFFUSION
CHECK T1@FrameOfReferenceUID == DIFF@FrameOfReferenceUID
```

### Example 9: Project-Level Training Data

Select all CT data for model training:

```sql
SELECT project
WHERE series HAS Modality == 'CT'
```

### Example 10: Longitudinal Study Selection

Find patients with pre and post treatment scans:

```sql
SELECT patient FROM study
WHERE series NAMED "pre" HAS
  SeriesDescription regexp 'baseline'
ALSO WHERE series NAMED "post" HAS
  SeriesDescription regexp 'followup'
CHECK pre@StudyDate < post@StudyDate
```

## Regular Expression Tips

| Pattern | Matches |
|---------|---------|
| `^T1` | Starts with "T1" |
| `T1$` | Ends with "T1" |
| `T1.*3D` | "T1" followed by "3D" (any chars between) |
| `T1\|T2` | "T1" or "T2" |
| `[Tt]1` | "T1" or "t1" (case variation) |
| `MPRAGE` | Contains "MPRAGE" anywhere |
| `.*FLAIR.*` | Contains "FLAIR" anywhere (explicit) |
| `^Ax.*T2` | Starts with "Ax", contains "T2" |
| `[0-9]+` | Contains one or more digits |

## String Value Quoting

String values can be quoted or unquoted:

```sql
-- Both are valid:
Modality == 'MR'
Modality == MR

-- For values with spaces, use quotes:
SeriesDescription regexp 'T1 weighted'
```

## Troubleshooting

### No results returned

1. Verify data has been added with `data_add` and parsing completed
2. Check field names are correct (case-sensitive)
3. Test with simpler query first (e.g., just Modality filter)
4. Use `get_series_tags` to see actual field values in the data
5. Verify regexp patterns are correct

### Unexpected results

1. Check `regexp` patterns carefully (case-sensitive by default)
2. Use `[Tt]1` for case-insensitive matching
3. Verify logical operator precedence (use parentheses)
4. Check NAMED labels match between WHERE and CHECK clauses

### Common Mistakes

```sql
-- Wrong: missing HAS keyword
WHERE series NAMED "T1" Modality == 'MR'

-- Correct:
WHERE series NAMED "T1" HAS Modality == 'MR'
```

```sql
-- Wrong: incorrect CHECK reference
CHECK T1.StudyDate < T2.StudyDate

-- Correct: use @ notation
CHECK T1@StudyDate < T2@StudyDate
```

```sql
-- Wrong: case mismatch in ClassifyType
ClassifyType containing diffusion

-- Correct: match exact case
ClassifyType containing DIFFUSION
```
