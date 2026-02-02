# Release Note Generator

Generate comprehensive release documentation from recent commits, producing two distinct outputs: a customer-facing release note and a technical engineering note.

## Interactive Workflow

When this command is triggered, **DO NOT** immediately generate release notes. Instead, present the user with two options:

### Mode Selection Prompt

Present this to the user:

```text
I can generate release notes in two ways:

**Mode 1: By Commit Count**
Generate notes for the last N commits (specify number or use default 10)
→ Quick generation when you know the commit count

**Mode 2: By Commit Hash Range (i.e. Last 24/48/72 Hours)**
Show all commits from the last 24/48/72 hours, then you select a starting commit
→ Precise control when you want to review recent commits first

Which mode would you like?
1. Commit count (provide number or use default)
2. Commit hash selection (show last 24/48/72 hours)

You can also provide an argument directly: /create-release-note 20
```

---

## Mode 1: By Commit Count

### Usage

```bash
/create-release-note          # Triggers mode selection
/create-release-note 20       # Directly uses Mode 1 with 20 commits
/create-release-note 50       # Directly uses Mode 1 with 50 commits
```

### Process

1. If `$ARGUMENTS` is provided, use it as commit count
2. If no `$ARGUMENTS`, ask user for commit count or default to 10
3. Set: `COMMIT_COUNT="${ARGUMENTS:-10}"`
4. Generate release notes immediately

---

## Mode 2: By Commit Hash Range

### Workflow

When user selects Mode 2, follow this process:

### Step 1: Retrieve Last 24 Hours of Commits

```bash
git log --since="24 hours ago" --pretty=format:"%h|%ai|%an|%s" --reverse
```

### Step 2: Present Commits to User

Format the output as a numbered list for easy selection:

```text
Commits from the last 24 hours (oldest to newest):

 1. a3f7e821 | 2025-10-15 09:23:45 | Alice Smith | Add OAuth provider configuration
 2. b4c8f932 | 2025-10-15 10:15:22 | Bob Jones | Implement token refresh flow
 3. c5d9e043 | 2025-10-15 11:42:18 | Alice Smith | Add provider UI components
 4. d6e1f154 | 2025-10-15 13:08:33 | Carol White | Database connection pooling
 5. e7f2g265 | 2025-10-15 14:55:47 | Alice Smith | Query optimization middleware
 6. f8g3h376 | 2025-10-15 16:20:12 | Bob Jones | Dark mode CSS variables
 7. g9h4i487 | 2025-10-15 17:10:55 | Carol White | Theme switching logic
 8. h0i5j598 | 2025-10-16 08:45:29 | Alice Smith | Error boundary implementation

Please provide the starting commit hash (8 characters) or number.
Release notes will be generated from your selection to HEAD (most recent).

Example: "a3f7e821" or "1" will generate notes for commits 1-8
Example: "d6e1f154" or "4" will generate notes for commits 4-8
```

### Step 3: Generate Notes from Selected Commit

Once user provides a commit hash or number:

```bash
# If user provided a number, extract the corresponding hash
SELECTED_HASH="<hash from user input>"

# Generate notes from selected commit to HEAD
git log ${SELECTED_HASH}..HEAD --stat --oneline
git log ${SELECTED_HASH}..HEAD --pretty=format:"%H|%s|%an|%ad" --date=short
```

**Important:** The range `${SELECTED_HASH}..HEAD` means "from the commit AFTER the selected hash to HEAD". If you want to include the selected commit itself, use `${SELECTED_HASH}^..HEAD` or count commits with `--ancestry-path`.

### Step 4: Confirm Range

Before generating, confirm with user:

```text
Generating release notes for N commits:
From: <hash> - <commit message>
To:   <HEAD hash> - <commit message>

Proceeding with generation...
```

---

## Core Requirements

### 1. Commit Analysis

**Determine commit source:**

- **Mode 1**: `COMMIT_COUNT="${ARGUMENTS:-10}"` → Use `git log -${COMMIT_COUNT}`
- **Mode 2**: User-selected hash → Use `git log ${SELECTED_HASH}..HEAD`

**Retrieve commits:**

- Use `git log <range> --stat --oneline`
- Use `git log <range> --pretty=format:"%H|%s|%an|%ad" --date=short`
- Analyze file changes to understand scope and impact
- Group related commits by feature/subsystem
- Identify major themes and primary focus areas

### 2. Traceability

- Every claim MUST be traceable to specific commit SHAs
- Reference actual files changed (e.g., src/config.ts, lib/utils.py)
- Use 8-character SHA prefixes for engineering notes (e.g., 0ca46028)
- Verify all technical details against actual commit content

### 3. Length Constraints

- Each section: ≤500 words (strict maximum)
- Aim for 150-180 words for optimal readability
- Prioritize most impactful changes if space constrained

---

## Section 1: Release Note (Customer-Facing)

### Purpose

Communicate value to end users without requiring deep technical knowledge. Audience varies by project type (system administrators, developers, product users, etc.).

### Tone and Style

- **Friendly & Clear**: Write as if explaining to a competent user of the software
- **Value-Focused**: Emphasize benefits and capabilities, not implementation details
- **Confident**: Use active voice and definitive statements
- **Professional**: Avoid jargon, explain acronyms on first use
- **Contextual**: Adapt language to the project type (infrastructure, web app, library, tool, etc.)

### Content Guidelines

**Include:**

- Major new features or functionality
- User-visible improvements
- Performance enhancements
- Security updates
- Dependency/component version upgrades
- Compatibility improvements
- Bug fixes affecting user experience

**Exclude:**

- Internal refactoring (unless it improves performance)
- Code organization changes
- Developer-only tooling
- Commit SHAs or file paths
- Implementation details
- Internal API changes (unless user-facing library)

### Structure Template

```markdown
## Release Note (Customer-Facing)

**[Project Name] [Version] - [Descriptive Title]**

[Opening paragraph: 1-2 sentences describing the primary focus/theme]

**Key improvements:**
- [Feature/improvement 1: benefit-focused description]
- [Feature/improvement 2: benefit-focused description]
- [Feature/improvement 3: benefit-focused description]
- [Feature/improvement 4: benefit-focused description]
- [etc.]

[Closing paragraph: 1-2 sentences about overall impact and use cases]
```

### Style Examples

✅ **Good (Customer-Facing):**
> "Enhanced authentication system with support for OAuth 2.0 and SAML providers"

❌ **Bad (Too Technical):**
> "Refactored src/auth/oauth.ts to implement RFC 6749 token refresh flow"

✅ **Good (Value-Focused):**
> "Improved database query performance, reducing page load times by 40%"

❌ **Bad (Implementation Details):**
> "Added connection pooling in db/connection.ts with configurable pool size"

✅ **Good (User Benefit):**
> "Added dark mode support with automatic system theme detection"

❌ **Bad (Technical Detail):**
> "Implemented CSS variables in styles/theme.css for runtime theme switching"

---

## Section 2: Engineering Note (Technical)

### Purpose

Provide developers/maintainers with precise technical details for code review, debugging, and future reference.

### Tone and Style

- **Precise & Technical**: Use exact terminology and technical language
- **Reference-Heavy**: Include SHAs, file paths, function names
- **Concise**: Information density over narrative
- **Structured**: Group by subsystem or feature area

### Content Guidelines

**Include:**

- 8-character SHA prefixes for every commit or commit group
- Exact file paths (src/components/App.tsx, lib/db/connection.py)
- Specific technical changes (version numbers, configuration changes)
- Module/function names when relevant
- Code organization changes
- All commits (even minor refactoring)
- Breaking changes or API modifications

**Structure:**

- Group related commits by subsystem
- List most significant changes first
- Use single-sentence summaries per commit/group
- Format: `SHA: description (file references)`

### Structure Template

```markdown
## Engineering Note (Technical)

**[Primary Focus/Theme]**

[Opening sentence: describe the main technical objective]

**[Subsystem/Feature Area 1]:**
- SHA1: brief technical description (file1, file2)
- SHA2: brief technical description (file3)
- SHA3, SHA4: grouped description (file4, file5, file6)

**[Subsystem/Feature Area 2]:**
- SHA5: brief technical description (file7, file8)
- SHA6: brief technical description (file9)

**[Subsystem/Feature Area 3]:**
- SHA7, SHA8, SHA9: grouped description (files10-15)
- SHA10: brief technical description (file16)

[Optional: List number of files affected if significant]
```

### Style Examples

✅ **Good (Technical):**
> "a3f7e821: OAuth 2.0 token refresh implementation in src/auth/oauth.ts, src/auth/tokens.ts"

❌ **Bad (Too Vague):**
> "Updated authentication system for better token handling"

✅ **Good (Grouped):**
> "c4d8a123, e5f9b234, a1c2d345: Database connection pooling (src/db/pool.ts, src/db/config.ts)"

❌ **Bad (No References):**
> "Fixed database connection issues"

✅ **Good (Precise):**
> "7b8c9d01: Upgrade react from 18.2.0 to 18.3.1 (package.json)"

❌ **Bad (Missing Context):**
> "Updated React dependency"

---

## Formatting Standards

### Markdown Requirements

- Use `##` for main section headers
- Use `**bold**` for subsection headers and project titles
- Use `-` for bullet lists
- Use `` `backticks` `` for file paths, commands, version numbers
- Use 8-character SHA prefixes: `0ca46028` not `0ca46028b9fa62bb995e41133036c9f0d6ac9fef`

### Horizontal Separator

Use `---` (three hyphens) to separate the two sections for visual clarity.

### Version Numbers

Format as: `version X.Y` or `version X.Y.Z` (e.g., "React 18.3", "Python 3.12.1")

### File Paths

- Use actual paths from repository: `src/components/App.tsx` not "main component"
- Multiple files: `(file1, file2, file3)` or `(files1-10)` for ranges
- Use project-appropriate path conventions (src/, lib/, app/, pkg/, etc.)

---

## Commit Grouping Strategy

### Group When

- Multiple commits modify the same file/subsystem
- Commits represent incremental work on same feature
- Space constraints require consolidation
- Related bug fixes or improvements

### Example Grouping

```text
Individual:
- c4d8a123: Add connection pool configuration
- e5f9b234: Implement pool lifecycle management
- a1c2d345: Add connection pool metrics

Grouped:
- c4d8a123, e5f9b234, a1c2d345: Database connection pooling (src/db/pool.ts, src/db/config.ts, src/db/metrics.ts)
```

### Don't Group

- Unrelated commits (different subsystems)
- Major features (deserve individual mention)
- Commits with significantly different file scopes
- Breaking changes (always call out separately)

---

## Quality Checklist

Before finalizing, verify:

- [ ] Mode selection presented (unless $ARGUMENTS provided)
- [ ] Commit range correctly determined (Mode 1: count, Mode 2: hash range)
- [ ] User confirmed commit range before generation
- [ ] Both sections ≤500 words
- [ ] Every claim traceable to specific commit(s)
- [ ] Customer note has no SHAs or file paths
- [ ] Engineering note has SHAs for all commits/groups
- [ ] File paths are accurate and complete
- [ ] Tone appropriate for each audience
- [ ] Markdown formatting consistent
- [ ] Version numbers accurate
- [ ] No typos or grammatical errors
- [ ] Primary focus clearly communicated in both sections
- [ ] Most significant changes prioritized first
- [ ] Language adapted to project type (not overly specific to one domain)

---

## Edge Cases

### If Fewer Commits Than Requested

- Generate notes for all available commits
- Note this at the beginning: "Release covering [N] commits"
- Example: "Release covering 7 commits (requested 10)"

### If No Commits in Last 24 Hours (Mode 2)

- Inform user: "No commits found in the last 24 hours"
- Offer alternatives:
  - Extend time range (48 hours, 7 days)
  - Switch to Mode 1 (commit count)
  - Manual hash range specification

### If Mostly Minor Changes

- Group aggressively by subsystem
- Lead with most significant changes
- Note: "Maintenance release with incremental improvements"

### If Single Major Feature Dominates

- Lead with that feature in both sections
- Group supporting commits under that theme
- Structure engineering note by feature components

### If Merge Commits Present

- Skip merge commits themselves
- Include the actual changes from merged branches
- Focus on functional changes, not merge mechanics

### If No Version Tag Available

- Use branch name or generic title: "Development Updates" or "Recent Improvements"
- Focus on change summary rather than version-specific language

### If User Provides Invalid Commit Hash

- Validate hash exists: `git cat-file -t ${HASH} 2>/dev/null`
- If invalid, show error and re-present commit list
- Suggest checking the hash or selecting by number instead

---

## Adapting to Project Types

### Infrastructure/DevOps Projects

- Focus on: deployment improvements, configuration management, monitoring, reliability
- Audience: sysadmins, DevOps engineers, SREs

### Web Applications

- Focus on: features, UX improvements, performance, security
- Audience: product users, stakeholders, QA teams

### Libraries/Frameworks

- Focus on: API changes, new capabilities, breaking changes, migration guides
- Audience: developers using the library

### CLI Tools

- Focus on: command changes, new options, output improvements, bug fixes
- Audience: command-line users, automation engineers

### Internal Tools

- Focus on: workflow improvements, bug fixes, integration updates
- Audience: team members, internal stakeholders

---

## Example Output Structure

```markdown
## Release Note (Customer-Facing)

**MyProject v2.4.0 - Authentication & Performance Update**

This release introduces comprehensive OAuth 2.0 support and significant performance improvements across the application.

**Key improvements:**
- OAuth 2.0 authentication with support for Google, GitHub, and Microsoft providers
- Improved database query performance with connection pooling, reducing response times by 40%
- Added dark mode support with automatic system theme detection
- Enhanced error handling and user feedback throughout the interface
- Security updates for dependency vulnerabilities

These enhancements provide a more secure, performant, and user-friendly experience across all application features.

---

## Engineering Note (Technical)

**OAuth 2.0 Integration and Performance Optimization**

Primary focus: authentication modernization and database performance improvements.

**Authentication System:**
- a3f7e821: OAuth 2.0 provider implementation (src/auth/oauth.ts, src/auth/providers/)
- b4c8f932: Token refresh flow and session management (src/auth/tokens.ts)
- c5d9e043: Provider registration UI components (src/components/auth/OAuthProviders.tsx)

**Performance Optimization:**
- d6e1f154: Database connection pooling (src/db/pool.ts, src/db/config.ts)
- e7f2g265: Query optimization middleware (src/db/middleware.ts)

**UI/UX Improvements:**
- f8g3h376, g9h4i487: Dark mode CSS variables and theme switching (src/styles/theme.css, src/components/ThemeProvider.tsx)
- h0i5j598: Error boundary implementation (src/components/ErrorBoundary.tsx)

**Security:**
- i1j6k609: Dependency updates for security patches (package.json, yarn.lock)
```

---

## Implementation Workflow

When executing this command, Claude should:

### If $ARGUMENTS Provided

1. Use `COMMIT_COUNT="${ARGUMENTS}"`
2. Run git commands with the determined count
3. Generate both sections immediately

### If No $ARGUMENTS

1. Present mode selection prompt to user
2. Wait for user response

**If user selects Mode 1:**
3. Ask for commit count or use default 10
4. Generate notes immediately

**If user selects Mode 2:**
3. Retrieve commits from last 24 hours
4. Present formatted list with numbers and hashes
5. Wait for user to provide hash or number
6. Validate selection
7. Confirm commit range
8. Generate notes from selected commit to HEAD

### Final Steps (Both Modes)

1. Analyze commits thoroughly
2. Generate both sections following all guidelines
3. Verify against quality checklist
4. Present both notes in the specified format
