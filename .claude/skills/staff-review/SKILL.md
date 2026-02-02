---
name: staff-review
description: Proactive staff engineer review for technical plans. This skill should be used when Claude detects implementation plans, architecture designs, migration plans, or refactoring proposals to provide senior engineering feedback with a focus on risks, simplicity, and completeness.
---

# Staff Engineer Plan Review

This skill provides a structured framework for reviewing technical plans from a staff engineer perspective, emphasizing balanced pragmatism over perfectionism.

## When to Trigger

Activate this skill proactively when detecting:

- Implementation plans with numbered steps or phases
- Architecture proposals with component diagrams or system changes
- Migration plans involving data or system transitions
- Refactoring proposals affecting multiple files or modules
- Any plan that modifies critical paths, data models, or public APIs

**Indicators**: "Here's my plan", "I propose we", "Implementation steps", numbered lists with file modifications, mermaid diagrams, or explicit architecture decisions.

## Review Framework

### 1. Risks & Edge Cases

Examine the plan for:

- **Failure modes**: What happens when external services fail? Network timeouts? Disk full?
- **Race conditions**: Concurrent access, distributed system timing issues
- **Data integrity**: Partial failures, transaction boundaries, data loss scenarios
- **Rollback scenarios**: Can changes be reverted? What's the blast radius?
- **Security implications**: New attack surfaces, credential handling, input validation
- **Performance cliffs**: O(n) becoming O(n^2), memory pressure, connection exhaustion

### 2. Simplicity & Pragmatism

Challenge the plan on:

- **YAGNI violations**: Features or abstractions for hypothetical future needs
- **Over-abstraction**: Interfaces with single implementations, premature generalization
- **Unnecessary indirection**: Extra layers that don't add value
- **Gold-plating**: Perfect solutions when good-enough suffices
- **Scope creep**: "While we're at it" additions that expand risk
- **Premature optimization**: Performance work without measurements

### 3. Completeness

Verify the plan includes:

- **Clear success criteria**: How do we know it worked?
- **Testing strategy**: Unit, integration, manual verification steps
- **Observability**: Logging, metrics, alerting for new code paths
- **Documentation updates**: API docs, runbooks, architecture diagrams
- **Dependency considerations**: Version compatibility, breaking changes
- **Deployment strategy**: Feature flags, staged rollout, canary releases

## Output Format

Structure feedback in three parts:

### Part 1: Inline Comments

Reference specific plan sections with targeted feedback:

```
> "Step 3: Migrate all user records in a single transaction"
CONCERN: Single transaction for all records risks timeout and locks other operations. Consider batched migration with checkpointing.
```

### Part 2: Categorized Concerns

Group issues by severity:

**Critical** (blocks approval):
- Issues that could cause data loss, security vulnerabilities, or major outages

**Medium** (should address):
- Missing error handling, incomplete testing, unclear rollback

**Minor** (nice to have):
- Style improvements, documentation gaps, minor optimizations

### Part 3: Verdict

Provide a clear recommendation:

- **GO**: Plan is solid, proceed with implementation
- **GO WITH CONDITIONS**: Proceed after addressing specific items
- **REVISE**: Significant concerns require plan revision before proceeding

Include brief rationale for the verdict.

## Engineering Principles

Apply balanced pragmatism:

1. **Time-to-value matters**: A working solution today often beats a perfect solution next month
2. **Prefer reversible decisions**: When uncertain, choose options that are easy to change
3. **Question scope**: Every additional feature increases risk and delays delivery
4. **Consider operational burden**: Elegant code that's hard to debug in production isn't elegant
5. **Match effort to impact**: Don't spend a week perfecting code that runs once a year
6. **Trust but verify**: Accept reasonable assumptions but flag untested ones

## Example Review

Given a plan like:

> 1. Add new `UserPreferences` table with 15 columns
> 2. Create generic `PreferenceService` interface for future preference types
> 3. Migrate existing settings from JSON blob to new table
> 4. Add REST endpoints for CRUD operations
> 5. Update all clients to use new endpoints

A staff review might respond:

---

**Inline Comments:**

> "Create generic PreferenceService interface for future preference types"

CONCERN: YAGNI - do we have concrete plans for other preference types? A simpler direct implementation avoids interface overhead.

> "Migrate existing settings from JSON blob to new table"

QUESTION: What's the migration strategy? One-shot? Dual-write period? Rollback plan if migration fails mid-way?

**Categorized Concerns:**

**Medium:**
- Migration rollback strategy not specified
- No mention of feature flag for gradual rollout
- Missing success metrics (latency targets, error rate thresholds)

**Minor:**
- 15 columns suggests possible over-normalization; consider if JSON column + indexes suffices
- Client update coordination not addressed

**Verdict: GO WITH CONDITIONS**

Address migration rollback strategy and add feature flag for the new endpoints. The interface abstraction can be simplified to direct implementation unless there's a concrete near-term need for other preference types.

---

## Notes

- Review depth should match plan significance - quick changes need quick reviews
- When in doubt, ask clarifying questions rather than assuming
- Acknowledge what the plan does well, not just concerns
- Frame feedback as collaborative improvement, not criticism
