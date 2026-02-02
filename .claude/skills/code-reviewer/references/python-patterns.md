# Python Code Review Patterns

Reference guide for common Python anti-patterns and best practices.

## Common Anti-Patterns

### Mutable Default Arguments
```python
# BAD
def append_to(element, to=[]):
    to.append(element)
    return to

# GOOD
def append_to(element, to=None):
    if to is None:
        to = []
    to.append(element)
    return to
```

### Bare Except Clauses
```python
# BAD
try:
    do_something()
except:
    pass

# GOOD
try:
    do_something()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    raise
```

### Using `type()` for Type Checking
```python
# BAD
if type(obj) == list:
    ...

# GOOD
if isinstance(obj, list):
    ...
```

### String Concatenation in Loops
```python
# BAD
result = ""
for item in items:
    result += str(item)

# GOOD
result = "".join(str(item) for item in items)
```

### Not Using Context Managers
```python
# BAD
f = open("file.txt")
data = f.read()
f.close()

# GOOD
with open("file.txt") as f:
    data = f.read()
```

## NumPy/PyTorch Specific

### Unnecessary Array Copies
```python
# BAD - creates copy
arr = np.array(tensor.numpy())

# GOOD - shares memory when possible
arr = tensor.numpy()
```

### Inefficient Tensor Operations
```python
# BAD - Python loop over tensors
for i in range(tensor.shape[0]):
    result[i] = tensor[i] * 2

# GOOD - vectorized operation
result = tensor * 2
```

### GPU Memory Leaks
```python
# BAD - keeps computation graph
outputs = []
for batch in dataloader:
    out = model(batch)
    outputs.append(out)  # Holds gradients

# GOOD - detach from graph
outputs = []
for batch in dataloader:
    with torch.no_grad():
        out = model(batch)
        outputs.append(out.cpu())
```

### Large Tensor Logging
```python
# BAD - prints entire tensor
print(f"Output: {large_tensor}")

# GOOD - print summary
print(f"Output shape: {large_tensor.shape}, mean: {large_tensor.mean():.4f}")
```

## Medical Imaging Specific

### Spacing/Orientation Mismatch
```python
# BAD - ignoring spacing
resampled = F.interpolate(image, size=target_size)

# GOOD - preserve physical space
resampled = tio.Resample(target_spacing)(image)
```

### Incorrect Axis Handling
```python
# BAD - assuming axis order
slice_2d = volume[:, :, middle_slice]

# GOOD - explicit axis
slice_2d = volume.index_select(dim=2, index=middle_slice)
```

### Missing Intensity Normalization
```python
# BAD - raw intensities vary by scanner
model(raw_image)

# GOOD - normalize first
normalized = tio.ZNormalization()(image)
model(normalized)
```

## Security Patterns

### Path Traversal
```python
# BAD - user input in path
filepath = os.path.join(base_dir, user_input)

# GOOD - validate path
filepath = os.path.join(base_dir, os.path.basename(user_input))
resolved = os.path.realpath(filepath)
if not resolved.startswith(os.path.realpath(base_dir)):
    raise ValueError("Invalid path")
```

### Subprocess Injection
```python
# BAD - shell injection risk
os.system(f"convert {user_file}")

# GOOD - use list arguments
subprocess.run(["convert", user_file], check=True)
```

### Pickle Security
```python
# BAD - arbitrary code execution
data = pickle.load(untrusted_file)

# GOOD - use safer alternatives or validate source
data = torch.load(trusted_file, weights_only=True)
```

## Performance Patterns

### Inefficient Comprehensions
```python
# BAD - nested loops in comprehension
[(x, y) for x in range(1000) for y in range(1000) if condition(x, y)]

# GOOD - use itertools or generators
from itertools import product
((x, y) for x, y in product(range(1000), repeat=2) if condition(x, y))
```

### Repeated Attribute Access
```python
# BAD - repeated lookup
for i in range(len(obj.data.items)):
    process(obj.data.items[i])

# GOOD - cache reference
items = obj.data.items
for i in range(len(items)):
    process(items[i])
```

### Large File Reading
```python
# BAD - loads entire file
content = open("huge_file.txt").read()
for line in content.split("\n"):
    process(line)

# GOOD - stream processing
with open("huge_file.txt") as f:
    for line in f:
        process(line)
```

## nbdev Specific

### Export Directive Placement
```python
# BAD - export after imports
import numpy as np
#| export

# GOOD - export at cell top
#| export
import numpy as np
```

### Test Cell Markers
```python
# Missing notest for expensive operations
#| export
def train_model():  # This will run during nbdev_test!
    ...

# GOOD - mark appropriately
#| notest
def train_model():
    ...
```

### Module-Level Side Effects
```python
# BAD - runs on import
#| export
model = load_heavy_model()  # Loads every import!

# GOOD - lazy loading
#| export
_model = None
def get_model():
    global _model
    if _model is None:
        _model = load_heavy_model()
    return _model
```
