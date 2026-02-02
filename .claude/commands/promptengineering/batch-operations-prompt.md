# Batch Operations Prompt

Optimize prompts for multiple file operations, parallel processing, and efficient bulk changes across a codebase. This helps Claude Code work more efficiently with TodoWrite patterns.

## Usage Examples

### Basic Usage
"Convert to batch: Update all test files to use new API"
"Batch prompt for: Rename variable across multiple files"
"Optimize for parallel: Add logging to all service files"

### With File Input
`/batch-operations-prompt @path/to/operation-request.md`
`/batch-operations-prompt @../refactoring-plan.txt`

### Complex Operations
"Batch refactor: Convert callbacks to async/await in all files"
"Parallel update: Add TypeScript types to all components"
"Bulk operation: Update import statements across the project"

## Instructions for Claude

When creating batch operation prompts:

### Input Handling
- If `$ARGUMENTS` is provided, read the file at that path to get the operation request to optimize
- If no `$ARGUMENTS`, use the user's direct input as the operation to optimize
- Support relative and absolute file paths

1. **Identify Parallelizable Tasks**: Determine what can be done simultaneously
2. **Group Related Operations**: Organize tasks by type and dependency
3. **Create Efficient Sequences**: Order operations to minimize conflicts
4. **Use TodoWrite Format**: Structure for Claude's task management
5. **Include Validation Steps**: Add checks between batch operations

### Batch Prompt Structure

#### 1. Overview
- Scope of changes
- Files/patterns affected
- Expected outcome

#### 2. Prerequisite Checks
- Required tools/dependencies
- Initial validation commands
- Backup recommendations

#### 3. Parallel Operations
- Independent tasks that can run simultaneously
- File groups that don't conflict
- Read operations for gathering information

#### 4. Sequential Operations
- Tasks with dependencies
- Operations that modify same files
- Final validation steps

### Optimization Strategies

#### File Grouping
```markdown
## Batch Operation: [Operation Name]

### Phase 1: Analysis (Parallel)
- Search for all affected files using Glob/Grep
- Read current implementations
- Identify patterns and dependencies

### Phase 2: Implementation (Grouped)
Group A (Independent files):
- File1.js: [specific change]
- File2.js: [specific change]

Group B (Related components):
- Component1.tsx: [change]
- Component1.test.tsx: [related change]

### Phase 3: Validation (Sequential)
1. Run linter on modified files
2. Execute test suite
3. Build verification
```

#### TodoWrite Integration
```markdown
### Task List Structure
1. Gather information (can parallelize):
   - Find all files matching pattern X
   - Read configuration files
   - Check current implementations

2. Batch updates (group by conflict potential):
   - Update non-conflicting files (parallel)
   - Update shared modules (sequential)
   - Update test files (parallel)

3. Verification (sequential):
   - Run type checking
   - Execute tests
   - Validate build
```

### Conversion Examples

#### Original Request:
"Update all API calls to use the new authentication header"

#### Batch-Optimized Version:
```markdown
## Batch Operation: Update API Authentication Headers

### Prerequisites
- Verify new auth header format
- Check all API call patterns in codebase

### Parallel Phase 1: Discovery
Execute simultaneously:
1. Grep for "fetch(" patterns
2. Grep for "axios." patterns
3. Grep for "api." patterns
4. Read auth configuration file

### Parallel Phase 2: Read Current Implementations
Read all files containing API calls (batch read):
- src/services/*.js
- src/api/*.js
- src/utils/api*.js

### Sequential Phase 3: Update by Pattern Type
Group 1 - Fetch calls:
- Update all fetch() calls with new header
- Pattern: Add "Authorization: Bearer ${token}"

Group 2 - Axios calls:
- Update axios config/interceptors
- Update individual axios calls

Group 3 - Custom API wrappers:
- Update wrapper functions
- Ensure backward compatibility

### Parallel Phase 4: Update Tests
Simultaneously update:
- Unit tests mocking API calls
- Integration tests with auth
- E2E test auth setup

### Sequential Phase 5: Validation
1. ESLint all modified files
2. Run test suite
3. Test one API call manually
4. Build project
```

### Output Format

Generate batch prompt as:

```markdown
## Batch Operation Prompt: [Operation Name]

### Efficiency Metrics
- Estimated sequential time: X operations
- Optimized parallel time: Y operations
- Parallelization factor: X/Y

### Execution Plan

#### Stage 1: Information Gathering (Parallel)
```bash
# Commands that can run simultaneously
[command 1] &
[command 2] &
[command 3] &
wait
```

#### Stage 2: Bulk Operations (Grouped)
**Parallel Group A:**
- Files: [list]
- Operation: [description]
- No conflicts with other groups

**Sequential Group B:**
- Files: [list]
- Operation: [description]
- Must complete before Group C

#### Stage 3: Verification (Sequential)
1. [Verification step 1]
2. [Verification step 2]
3. [Final validation]

### TodoWrite Task List
- [ ] Complete Stage 1 analysis (parallel)
- [ ] Execute Group A updates (parallel)
- [ ] Execute Group B updates (sequential)
- [ ] Run verification suite
- [ ] Document changes
```

Remember to:
- Maximize parallel operations
- Group by conflict potential
- Use TodoWrite's in_progress limitation wisely
- Include rollback strategies
- Provide specific file patterns