# Check Best Practices

Analyze code against language-specific best practices, coding standards, and community conventions to improve code quality and maintainability.

## Usage Examples

### Basic Usage
"Check if this code follows Python best practices"
"Review JavaScript code for ES6+ best practices"
"Analyze React components for best practices"

### Specific Checks
"Check if this follows PEP 8 conventions"
"Review TypeScript code for proper type usage"
"Verify REST API design best practices"
"Check Git commit message conventions"

## Instructions for Claude

When checking best practices:

1. **Identify Language/Framework**: Detect the languages and frameworks being used
2. **Apply Relevant Standards**: Use appropriate style guides and conventions
3. **Context Awareness**: Consider project-specific patterns and existing conventions
4. **Actionable Feedback**: Provide specific examples of improvements
5. **Prioritize Issues**: Focus on impactful improvements over nitpicks

### Language-Specific Guidelines

#### Python
- PEP 8 style guide compliance
- PEP 484 type hints usage
- Pythonic idioms and patterns
- Proper exception handling
- Module and package structure

#### JavaScript/TypeScript
- Modern ES6+ features usage
- Async/await over callbacks
- Proper error handling
- Module organization
- TypeScript strict mode compliance

#### React/Vue/Angular
- Component structure and organization
- State management patterns
- Performance optimizations
- Accessibility considerations
- Testing patterns

#### API Design
- RESTful conventions
- Consistent naming patterns
- Proper HTTP status codes
- API versioning strategy
- Documentation standards

### Code Quality Aspects

#### Naming Conventions
- Variable and function names
- Class and module names
- Consistency across codebase
- Meaningful and descriptive names

#### Code Organization
- File and folder structure
- Separation of concerns
- DRY (Don't Repeat Yourself)
- Single Responsibility Principle
- Modular design

#### Error Handling
- Comprehensive error catching
- Meaningful error messages
- Proper logging practices
- Graceful degradation

#### Performance
- Efficient algorithms
- Proper caching strategies
- Lazy loading where appropriate
- Database query optimization
- Memory management

#### Testing
- Test coverage adequacy
- Test naming conventions
- Test organization
- Mock usage patterns
- Integration vs unit tests

### Output Format

Structure the analysis as:

```markdown
## Best Practices Review

### Summary
- Language/Framework: [Detected stack]
- Overall Score: X/10
- Key Areas for Improvement: [List]

### Excellent Practices Found
- [Practice 1]: Brief explanation
- [Practice 2]: Brief explanation

### Areas for Improvement

#### High Priority
1. **[Issue Name]**
   - Current: `code example`
   - Recommended: `improved code`
   - Reason: Explanation of why this matters

#### Medium Priority
[Similar format]

#### Low Priority
[Similar format]

### Quick Wins
- [Easy improvement 1]
- [Easy improvement 2]

### Long-term Recommendations
- [Strategic improvement 1]
- [Strategic improvement 2]
```

Focus on:
- Practical, implementable suggestions
- Learning opportunities
- Maintaining consistency with existing code
- Balancing perfection with pragmatism