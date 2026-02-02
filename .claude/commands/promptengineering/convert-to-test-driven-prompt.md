# Convert to Test-Driven Prompt

Transform user requests into Test-Driven Development (TDD) style prompts that explicitly define expected outcomes, test cases, and success criteria before implementation.

## Usage Examples

### Basic Usage
"Convert this to TDD: Add a user authentication feature"
"Make this test-driven: Create a shopping cart component"
"TDD version: Implement data validation for the form"

### With File Input
`/convert-to-test-driven-prompt @path/to/prompt-file.md`
`/convert-to-test-driven-prompt @../other-project/feature-request.txt`

### Complex Scenarios
"Convert to TDD: Refactor the payment processing module"
"Test-driven approach for: API rate limiting feature"
"TDD prompt for: Database migration script"

## Instructions for Claude

When converting to TDD prompts:

### Input Handling
- If `$ARGUMENTS` is provided, read the file at that path to get the prompt to convert
- If no `$ARGUMENTS`, use the user's direct input as the prompt to convert
- Support relative and absolute file paths

1. **Extract Requirements**: Identify core functionality from the original request
2. **Define Test Cases**: Create specific, measurable test scenarios
3. **Specify Expected Outcomes**: Clear success and failure criteria
4. **Structure for Implementation**: Organize prompt for red-green-refactor cycle
5. **Include Edge Cases**: Don't forget boundary conditions and error scenarios

### TDD Prompt Structure

#### 1. Objective Statement
Clear, concise description of what needs to be built

#### 2. Test Specifications
```
GIVEN: [Initial state/context]
WHEN: [Action performed]
THEN: [Expected outcome]
```

#### 3. Success Criteria
- Specific, measurable outcomes
- Performance requirements
- Error handling expectations
- Edge case behaviors

#### 4. Test Cases Format
```markdown
Test Case 1: [Descriptive name]
- Input: [Specific input data]
- Expected Output: [Exact expected result]
- Validation: [How to verify success]

Test Case 2: [Edge case name]
- Input: [Boundary/error condition]
- Expected Output: [Error handling result]
- Validation: [Error verification method]
```

### Conversion Examples

#### Original Request:
"Add user login functionality"

#### TDD Conversion:
```markdown
## Objective
Implement secure user login with email/password authentication

## Test Specifications

### Test 1: Successful Login
GIVEN: Valid user credentials exist in database
WHEN: User submits correct email and password
THEN: User receives auth token and is redirected to dashboard

### Test 2: Invalid Password
GIVEN: Valid email but incorrect password
WHEN: User submits login form
THEN: Return error "Invalid credentials" without revealing which field is wrong

### Test 3: Non-existent User
GIVEN: Email not in database
WHEN: User attempts login
THEN: Return same "Invalid credentials" error (prevent user enumeration)

### Test 4: Rate Limiting
GIVEN: User has failed 5 login attempts
WHEN: User attempts 6th login within 15 minutes
THEN: Block attempt and show "Too many attempts" error

## Success Criteria
- All tests pass
- Password is hashed using bcrypt
- Auth tokens expire after 24 hours
- Login attempts are logged
- Response time < 200ms
```

### Output Format

Generate TDD prompt as:

```markdown
## TDD Prompt: [Feature Name]

### Objective
[Clear description of the feature to implement]

### Test Suite

#### Happy Path Tests
[List of successful scenario tests]

#### Error Handling Tests
[List of failure scenario tests]

#### Edge Case Tests
[List of boundary condition tests]

### Implementation Requirements
- [ ] All tests must pass
- [ ] Code coverage > 80%
- [ ] Performance criteria met
- [ ] Security requirements satisfied

### Test-First Development Steps
1. Write failing test for [first requirement]
2. Implement minimal code to pass
3. Refactor while keeping tests green
4. Repeat for next requirement

### Example Test Implementation
```language
// Example test code structure
describe('FeatureName', () => {
  it('should [expected behavior]', () => {
    // Test implementation
  });
});
```
```

Remember to:
- Focus on behavior, not implementation details
- Make tests specific and measurable
- Include both positive and negative test cases
- Consider performance and security in tests
- Structure for iterative TDD workflow