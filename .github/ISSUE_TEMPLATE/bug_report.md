---
name: Bug report
about: 'Create a report to help us reproduce and improve. Remember to put the target
  repository in the title. '
title: "[Repository Name] Issue Title"
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is. Please try to add a minimal example to reproduce the error. Avoid any extra data and please include the imports to make the debug process efficient. For example : 

```
# All necessary imports at the beginning
import torch

# A succinct reproducing example trimmed down to the essential parts:
t = torch.rand(5, 10)  # Note: the bug is here, we should pass requires_grad=True
t.sum().backward()
```

**Expected behavior and Actual Behaviour**
A clear and concise description of what you expected to happen.

**Versions**
Please run the following and paste the output below.
```
wget https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/collect_env.py
# For security purposes, please check the contents of collect_env.py before running it.
python collect_env.py
```

**Additional context**
If the problem is some device specific, that could be mentioned over here along with some other context information.