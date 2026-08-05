---
name: claude-md-updater
description: >
  Proactive CLAUDE.md maintenance skill that keeps project documentation in sync with code changes.
  This skill should be used when Claude detects that its session work has changed modules, transforms,
  dependencies, version numbers, test counts, or architecture components that are documented in
  CLAUDE.md. Scoped to CLAUDE.md only (not CLAUDE-*.md satellite files).
---

# CLAUDE.md Updater

Targeted, session-aware maintenance of CLAUDE.md to keep it accurate after code changes.

## When to Trigger

Activate this skill proactively at the end of a session (before commit) when the current session
included any of the following changes:

- Added, removed, or renamed a notebook or its `#| default_exp` target
- Added or removed public classes or functions in core modules (transforms, loss functions,
  metrics, patch components, inference engine, data classes)
- Bumped the version in `settings.ini`
- Changed dependency versions in `settings.ini` requirements
- Added or removed test cells (`test_eq`, `test_fail`, `test_close`) in notebooks
- Introduced new coding conventions or architectural patterns worth documenting
- Changed the patch-based workflow (new PatchConfig fields, new inference modes, ONNX support)
- Modified MLflow tracking behavior or dataset caching logic

Do NOT trigger for:
- Changes only to tutorial notebooks (11x-12x series) that do not affect module mapping
- Changes only to research/ directory files
- Pure documentation or comment-only changes within existing notebooks
- Changes to CLAUDE-*.md satellite files (use memory-bank-synchronizer agent instead)

## Update Process

### Step 1: Identify What Changed

Review the session's changes to determine which CLAUDE.md sections need updating. Use the
session context -- do not re-audit the entire codebase. Focus on what was just modified.

Categorize changes into:
- **Version/dependency**: settings.ini version or requirements changed
- **Module structure**: notebooks added/removed/renamed, module mapping changed
- **Public API**: new/removed classes, functions, or transforms
- **Architecture**: new subsystems, changed data flow, new design patterns
- **Tests**: test cells added or removed
- **Conventions**: new coding patterns worth documenting

### Step 2: Read Current CLAUDE.md Sections

Read only the specific sections of CLAUDE.md that need updating. Reference these section
markers to locate content:

| Change Type | Section to Update |
|-------------|-------------------|
| Version bump | `- **Version**:` line under `## Project Overview` |
| Dependency versions | `## Key Dependencies` table |
| New/removed notebook | `### Notebook-to-Module Mapping` code block |
| New/removed tutorial | `### Tutorial Notebooks` code block |
| Module purpose change | `### Core Module Purposes` table |
| New transform class | `### Available Transforms` lists |
| Patch workflow change | `### Patch-Based Workflow Components` table and design patterns |
| MLflow change | `### MLflow Artifact Management (utils)` section |
| Caching change | `### Dataset Metadata Caching (dataset_info)` section |
| Test count change | `## Testing` > `**Test Distribution:**` list |
| Line count change | Update `(~N lines)` references in module tables |

### Step 3: Apply Surgical Edits

For each identified change, edit only the affected lines in CLAUDE.md. Follow these rules:

1. **Match existing formatting exactly** -- table alignment, code block style, bullet style
2. **Preserve surrounding context** -- do not rewrite entire sections; edit only the specific
   lines, rows, or list items that changed
3. **Keep descriptions concise** -- module purpose descriptions should be one line; component
   descriptions should match the terse style of existing entries
4. **Update line counts only when significantly different** -- a change from ~1,260 to ~1,270
   is not worth updating; a change from ~1,260 to ~1,700 is

### Step 4: Sections to NEVER Modify

Do not touch these sections under any circumstances -- they contain strategic decisions,
stable conventions, or meta-instructions that are set by the project owner:

- `## AI Guidance` (meta-instructions for Claude)
- `## Memory Bank System` (structural, stable)
- `## Project Overview` description, Python versions, repository URL, docs URL
- `### Directory Structure`
- `## Development Commands`
- `### nbdev Workflow (Critical)`
- `### Medical Image Type Hierarchy`
- `### Entry Point`
- `## Git Workflow` and all sub-sections
- `### CI/CD Pipelines` and `### CI Build Configuration`
- `## Coding Conventions` and all sub-sections

If changes affect these sections (e.g., a fundamental architecture shift), flag the need
for a manual update to the user rather than editing directly.

### Step 5: Verify

After editing, verify the update by:

1. Re-reading the modified sections of CLAUDE.md to confirm correct formatting
2. Confirming that no protected sections were modified
3. Checking that the changes accurately reflect what was implemented in the session

## Quick Reference: Common Updates

### Version Bump
```
Check: settings.ini version field
Update: `- **Version**: X.Y.Z` in Project Overview
```

### New Transform Class
```
Check: new class in vision_augmentation.py inheriting DisplayedTransform/ItemTransform
Update: add to appropriate category in Available Transforms
  - Preprocessing, Mask Converters, or Augmentation
```

### New Dependency or Version Change
```
Check: settings.ini requirements line
Update: Key Dependencies table row (Library | Purpose)
```

### New Notebook-to-Module Mapping
```
Check: new .ipynb file with #| default_exp directive
Update: add line to Notebook-to-Module Mapping code block
```

### Test Distribution Change
```
Check: count test_eq/test_fail/test_close calls per notebook
Update: Test Distribution list (notebook - N tests)
  - Order by count descending
  - Add "(most comprehensive)" to highest count
```

### New Patch Workflow Component
```
Check: new public class/function in vision_patch.py
Update: Patch-Based Workflow Components table and/or Key Design Patterns
```
