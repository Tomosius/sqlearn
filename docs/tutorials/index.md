# Tutorials

Step-by-step guides that walk through complete sqlearn workflows from start to finish.
Each tutorial is self-contained, takes 10--15 minutes to read, and includes
copy-pasteable code examples with realistic data.

## Available Tutorials

| Tutorial | What you will learn |
|---|---|
| **[Your First Pipeline](basic-pipeline.md)** | Build a preprocessing pipeline with Imputer, StandardScaler, and OneHotEncoder. Inspect learned parameters, view generated SQL. |
| **[Custom Transformers](custom-transformers.md)** | Create your own transformers at three levels of complexity: one-liner expressions, template-based transforms with learned statistics, and full subclasses. |
| **[sqlearn vs sklearn](sklearn-comparison.md)** | Side-by-side comparison of preprocessing in sklearn and sqlearn. Verify identical output, see the SQL bonus. |

## Prerequisites

All tutorials assume you have sqlearn installed:

```bash
pip install sqlearn
```

Requires Python 3.10+. Core dependencies (`duckdb`, `numpy`, `sqlglot`) are installed
automatically.

## Conventions

Throughout the tutorials, sqlearn is imported as `sq`:

```python
import sqlearn as sq
```

Code examples use **tabbed blocks** to show the Python code, generated SQL, and
expected output side by side. You can click any tab to switch views.
