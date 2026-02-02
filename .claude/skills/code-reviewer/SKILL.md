---
name: code-reviewer
description: >
  Proactive code review skill for comprehensive analysis covering code quality, security,
  and performance. Specialized for Python codebases with fastMONAI/nbdev/medical imaging
  focus. This skill should be used when Claude detects code changes, new implementations,
  or when asked to review code quality. Provides flexible output format adapting to context.
---

# Code Reviewer

Comprehensive code review analysis covering quality, security, and performance with Python/fastMONAI specialization.

## When to Use

Proactively activate this skill when:
- Reviewing code changes before commit
- Analyzing new implementations or features
- Detecting potential issues in modified files
- Asked to assess code quality or find problems
- Completing significant code writing tasks (self-review)

## Review Process

### 1. Understand Context First

Before reviewing, establish:
- Purpose of the code (feature, bugfix, refactor)
- Integration points with existing codebase
- Criticality level (production, research, prototype)

### 2. Apply Review Layers

Review code through three lenses, in order of priority:

#### Layer 1: Code Quality
- **Readability**: Clear naming, appropriate comments, logical organization
- **Maintainability**: Single responsibility, DRY principles, appropriate abstraction level
- **Correctness**: Logic errors, edge cases, error handling
- **Style**: PEP 8 compliance, consistent formatting, idiomatic Python

#### Layer 2: Security
- **Input validation**: Untrusted data handling, injection risks
- **Data exposure**: Sensitive data logging, hardcoded secrets
- **Resource safety**: File operations, network calls, subprocess usage
- **Dependencies**: Known vulnerabilities, version pinning

#### Layer 3: Performance
- **Algorithmic efficiency**: Time/space complexity, unnecessary iterations
- **Memory management**: Large object handling, memory leaks in loops
- **I/O optimization**: Batch operations, caching opportunities
- **Medical imaging specific**: Tensor operations, GPU memory, large volume handling

### 3. fastMONAI/nbdev Specific Checks

When reviewing fastMONAI code:
- Verify changes are in `.ipynb` notebooks, not `.py` files directly
- Check `#| export` directives are correct
- Ensure `#| notest` is used appropriately for expensive operations
- Validate padding_mode=0 usage with TorchIO CropOrPad
- Confirm preprocessing consistency between training and inference
- Check MedImage/MedMask type usage for medical data

Reference `references/python-patterns.md` for detailed Python anti-patterns and best practices.

## Output Format Guidelines

Adapt output based on context:

### For Small Changes (< 50 lines)
Use inline format with line references:
```
[file.py:42](file.py#L42) - Issue: Variable `x` shadows outer scope
[file.py:55-60](file.py#L55-L60) - Suggestion: Extract to helper function
```

### For Large Changes or Full Reviews
Use structured report:
```markdown
## Review Summary
Brief overview of findings

## Critical Issues
Must-fix problems (security, correctness)

## Improvements
Recommended changes (quality, performance)

## Positive Observations
Well-implemented patterns worth noting

## Questions
Clarifications needed from author
```

### For Self-Review (after writing code)
Brief checklist verification:
- Tested edge cases
- Error handling complete
- No obvious security issues
- Performance acceptable

## Severity Classification

- **Critical**: Security vulnerabilities, data loss risks, crash bugs
- **High**: Logic errors, significant performance issues, maintainability blockers
- **Medium**: Code smell, minor inefficiencies, style inconsistencies
- **Low**: Nitpicks, optional improvements, documentation gaps

## Review Tone

- Be constructive, not critical
- Explain the "why" behind suggestions
- Acknowledge good patterns alongside issues
- Provide concrete fix suggestions when possible
- Ask questions rather than assume intent
