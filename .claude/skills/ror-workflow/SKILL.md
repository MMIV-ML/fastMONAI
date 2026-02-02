---
name: ror-workflow
description: Work with ROR (Research-Information-System) for DICOM data exploration, workflow setup, and series selection in research PACS environments. This skill should be used when exploring medical imaging datasets, setting up processing workflows, querying DICOM series/studies by modality or metadata, or filtering patients/studies for batch processing.
---

# ROR Workflow Tool

This skill provides guidance for working with ROR (Research-Information-System), a workflow tool for research PACS that enables processing pipelines for medical imaging data.

## When to Use This Skill

Use this skill when:

- Exploring DICOM datasets (listing patients, studies, series)
- Setting up data processing workflows for medical imaging
- Filtering DICOM series by modality, description, or other metadata
- Querying specific series for batch processing pipelines
- Working with research PACS data organization

## Available MCP Tools

ROR provides the following MCP tools for data interaction:

### Project Management

| Tool | Purpose |
|------|---------|
| `mcp__ror_mcp__project_init` | Initialize a new ROR project folder |
| `mcp__ror_mcp__data_add` | Add a DICOM data folder (triggers parsing) |
| `mcp__ror_mcp__data_clear` | Clear all imported data references |
| `mcp__ror_mcp__roots` | Manage ROR roots configuration |
| `mcp__ror_mcp__roots_list` | List currently configured roots |
| `mcp__ror_mcp__ror_info` | Get ROR tool information |

### Data Exploration

| Tool | Purpose |
|------|---------|
| `mcp__ror_mcp__get_patients_info` | List all patients/participants |
| `mcp__ror_mcp__get_study_info` | Get studies for a patient (or all if empty name) |
| `mcp__ror_mcp__get_series_info` | Get series for a study (requires StudyInstanceUID) |
| `mcp__ror_mcp__get_series_tags` | Get DICOM tags for a specific series |

### Series Selection

| Tool | Purpose |
|------|---------|
| `mcp__ror_mcp__get_current_select_statement` | View current filter query |
| `mcp__ror_mcp__set_new_select_statement` | Set new SELECT filter |

### MCP Resources

Access these resources via `ReadMcpResourceTool` with server `ror_mcp`:

| Resource URI | Description |
|--------------|-------------|
| `embedded:info` | ROR tool information |
| `embedded:data` | Current data summary (JSON) |
| `embedded:numparticipants` | Count of participants |
| `embedded:numstudies` | Count of studies |
| `embedded:numseries` | Count of series |
| `embedded:numimages` | Count of images |
| `embedded:tag` | Get DICOM tag name from group/element |
| `embedded:tagname` | Get DICOM tag group/element from name |

## Common Workflows

### Workflow 1: Initial Data Exploration

To explore a new DICOM dataset:

1. **Initialize project** (if needed):
   ```
   mcp__ror_mcp__project_init(path="/path/to/project")
   ```

2. **Add data folder**:
   ```
   mcp__ror_mcp__data_add(path="/path/to/dicom/data")
   ```
   Wait for parsing to complete before proceeding.

3. **Get overview**:
   - List patients: `mcp__ror_mcp__get_patients_info()`
   - Get studies for a patient: `mcp__ror_mcp__get_study_info(name="PatientName")`
   - Get all studies: `mcp__ror_mcp__get_study_info(name="")`

4. **Drill down to series**:
   - Get series for a study: `mcp__ror_mcp__get_series_info(study_instance_uid="...")`
   - Examine series tags: `mcp__ror_mcp__get_series_tags(series_instance_uid="...")`

### Workflow 2: Setting Up Series Selection

To filter DICOM series for processing:

1. **Understand the data** by exploring patients, studies, and series first

2. **Write a SELECT statement** using the query syntax (see `references/select-syntax.md`):
   ```sql
   SELECT series FROM study
   WHERE series NAMED "T1" HAS
     Modality == 'MR' AND SeriesDescription regexp 'T1'
   ```

3. **Set the filter**:
   ```
   mcp__ror_mcp__set_new_select_statement(select="SELECT series FROM study WHERE ...")
   ```

4. **Verify the selection**:
   ```
   mcp__ror_mcp__get_current_select_statement()
   ```

### Workflow 3: Multi-Series Workflow Setup

To select multiple related series (e.g., T1 + FLAIR + DWI for analysis):

```sql
SELECT patient FROM study
WHERE series NAMED "T1" HAS
  ClassifyType containing T1
ALSO WHERE series NAMED "FLAIR" HAS
  SeriesDescription regexp 'FLAIR'
ALSO WHERE series NAMED "DWI" HAS
  ClassifyType containing DIFFUSION
```

This returns patients that have all three series types available.

## SELECT Statement Quick Reference

Basic syntax:
```sql
SELECT <level> FROM study WHERE series [NAMED "name"] HAS <conditions>
```

**Output levels:** `patient`, `study`, `series`

**Operators:**
- `==`, `containing` - exact match
- `regexp` - regular expression
- `>`, `<`, `>=`, `<=` - numeric comparison
- `AND`, `OR`, `NOT()` - logical operators

**Common DICOM fields:**
- `Modality` - MR, CT, US, etc.
- `SeriesDescription` - Series name text
- `NumImages` - Image count
- `ClassifyType` - Custom classification

For complete syntax reference, read the `references/select-syntax.md` file.

## Best Practices

### Data Exploration

- Always explore the data structure before writing SELECT statements
- Use `get_series_tags` to discover available metadata fields
- Start with simple queries and refine iteratively

### SELECT Statements

- Use `NAMED` to label series for later reference in workflows
- Use `regexp` for flexible pattern matching on descriptions
- Use `ALSO WHERE` to require multiple series types per patient/study

### Performance

- Wait for `data_add` parsing to complete before querying
- Use specific patient names rather than empty string for large datasets
- Filter at the most specific level needed (series vs study vs patient)
