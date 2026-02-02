# Security Audit

Perform a comprehensive security audit of the codebase to identify potential vulnerabilities, insecure patterns, and security best practice violations.

## Usage Examples

### Basic Usage
"Run a security audit on this project"
"Check for security vulnerabilities in the authentication module"
"Scan the API endpoints for security issues"

### Specific Audits
"Check for SQL injection vulnerabilities"
"Audit the file upload functionality for security risks"
"Review authentication and authorization implementation"
"Check for hardcoded secrets and API keys"

## Instructions for Claude

When performing a security audit:

1. **Systematic Scanning**: Examine the codebase systematically for common vulnerability patterns
2. **Use OWASP Guidelines**: Reference OWASP Top 10 and other security standards
3. **Check Multiple Layers**: Review frontend, backend, database, and infrastructure code
4. **Prioritize Findings**: Categorize issues by severity (Critical, High, Medium, Low)
5. **Provide Remediation**: Include specific fixes for each identified issue

### Security Checklist

#### Authentication & Authorization
- Password storage and hashing methods
- Session management security
- JWT implementation and validation
- Access control and permission checks
- Multi-factor authentication support

#### Input Validation & Sanitization
- SQL injection prevention
- XSS (Cross-Site Scripting) protection
- Command injection safeguards
- Path traversal prevention
- File upload validation

#### Data Protection
- Encryption in transit (HTTPS/TLS)
- Encryption at rest
- Sensitive data exposure
- API key and secret management
- PII handling compliance

#### Common Vulnerabilities
- CSRF protection
- Clickjacking prevention
- Security headers configuration
- Dependency vulnerabilities
- Insecure direct object references

#### API Security
- Rate limiting implementation
- API authentication methods
- Input validation on endpoints
- Error message information leakage
- CORS configuration

### Output Format

Provide a structured security report with:

```markdown
## Security Audit Report

### Summary
- Total issues found: X
- Critical: X, High: X, Medium: X, Low: X

### Critical Issues
#### 1. [Issue Name]
- **Location**: file.js:line
- **Description**: Detailed explanation
- **Impact**: Potential consequences
- **Remediation**: Specific fix with code example

### High Priority Issues
[Similar format]

### Medium Priority Issues
[Similar format]

### Low Priority Issues
[Similar format]

### Recommendations
- General security improvements
- Best practices to implement
- Tools and libraries to consider
```

Remember to:
- Be specific about file locations and line numbers
- Provide code examples for fixes
- Explain the security impact clearly
- Avoid false positives by understanding the context