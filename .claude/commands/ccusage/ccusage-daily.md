Please run the `ccusage daily -b` command and then provide a structured markdown summary of the Claude Code usage costs and statistics.

## Required Actions:
1. Execute `ccusage daily -b` using the Bash tool
2. Parse the output to extract key metrics and statistics
3. Generate a comprehensive markdown report

## Report Format Required:

### Executive Summary
- Total cost for the period
- Date range covered
- Number of usage days
- Average daily cost for sonnet
- Average daily cost for opus
- Average daily cost for sonnet + opus
- Peak usage day and cost for sonnet
- Peak usage day and cost for opus
- Peak usage day and cost for sonnet + opus
- Cache efficiency percentage

### Key Statistics Table
A markdown table with usage broken down by model and model totals:
- Total Tokens (in millions i.e. 1,234,567 reported as 1.2345 m)
- Input Tokens (in millions i.e. 1,234,567 reported as 1.2345 m)
- Output Tokens (in millions i.e. 1,234,567 reported as 1.2345 m)
- Cache Read Tokens (in millions i.e. 1,234,567 reported as 1.2345 m)
- Total Cost
- Average Daily Cost

### Daily Cost Summary Table
A compact markdown table showing:
- Date (in MM-DD format)
- Model used
- Input tokens (in millions i.e. 1,234,567 reported as 1.2345 m)
- Output tokens (in millions i.e. 1,234,567 reported as 1.2345 m)
- Cache read tokens (in millions i.e. 1,234,567 reported as 1.2345 m)
- Total cost for that day

Limit to the top 15-20 highest cost days to keep the table manageable.

### Usage Insights
Provide analysis including:
- Number of high usage days (over $20)
- Cache effectiveness assessment
- Token distribution breakdown
- Primary model identification
- Usage patterns and trends

### Recommendations
Based on the data, provide:
- Cost management insights
- Cache optimization observations
- Usage pattern analysis

Please format everything in clean, readable markdown with proper tables and bullet points. Focus on actionable insights and clear presentation of the cost and usage data.