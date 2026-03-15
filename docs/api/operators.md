# Pipeline Operators

sqlearn supports composing pipelines and transformers using Python's `+` and `+=`
operators. The result is always a **flat** Pipeline -- no nesting.

## Composition with `+`

The `+` operator creates a new Pipeline by combining steps from both operands.
All steps are cloned, so the result is fully independent of the originals.

### Transformer + Transformer

```python
import sqlearn as sq

pipe = sq.Imputer() + sq.StandardScaler()
# Pipeline(step_00=Imputer, step_01=StandardScaler)
```

### Pipeline + Transformer

```python
pipe = sq.Pipeline([sq.Imputer()])
combined = pipe + sq.StandardScaler()
# Pipeline(step_00=Imputer, step_01=StandardScaler)
```

### Transformer + Pipeline

```python
pipe = sq.Pipeline([sq.StandardScaler()])
combined = sq.Imputer() + pipe
# Pipeline(step_00=Imputer, step_01=StandardScaler)
```

### Pipeline + Pipeline

```python
pipe1 = sq.Pipeline([sq.Imputer(), sq.StandardScaler()])
pipe2 = sq.Pipeline([sq.OneHotEncoder()])
combined = pipe1 + pipe2
# Pipeline(step_00=Imputer, step_01=StandardScaler, step_02=OneHotEncoder)
```

### Chaining

The `+` operator is left-associative, so chaining works naturally:

```python
pipe = sq.Imputer() + sq.StandardScaler() + sq.OneHotEncoder()
# Pipeline with 3 steps, fully flat
```

## In-place composition with `+=`

The `+=` operator is **non-mutating**. It returns a new Pipeline and rebinds the
variable, following Python's numeric convention.

```python
pipe = sq.Pipeline([sq.Imputer()])
pipe += sq.StandardScaler()
# pipe is now a NEW Pipeline with 2 steps
# The original Pipeline with just Imputer is gone
```

## Flattening

The `+` operator flattens one level deep. When combining two Pipelines, their
steps are concatenated into a single flat list:

```python
pipe1 = sq.Pipeline([sq.Imputer(), sq.StandardScaler()])
pipe2 = sq.Pipeline([sq.OneHotEncoder()])

combined = pipe1 + pipe2
len(combined.steps)  # 3, not 2 (no nested Pipeline)
```

## Step independence

All steps in the resulting Pipeline are **cloned** from the originals.
Modifying the original transformers or pipelines after composition does not
affect the combined Pipeline:

```python
scaler = sq.StandardScaler()
pipe = sq.Pipeline([sq.Imputer()])
combined = pipe + scaler

# Modifying scaler has no effect on combined
scaler.set_params(columns=["new_col"])
# combined still has the original scaler params
```

## Name collision handling

When combining Pipelines with auto-generated step names (like `step_00`),
collisions are automatically resolved by renumbering:

```python
pipe1 = sq.Pipeline([sq.Imputer()])        # step_00
pipe2 = sq.Pipeline([sq.StandardScaler()]) # step_00
combined = pipe1 + pipe2
# step_00=Imputer, step_01=StandardScaler (renumbered)
```

User-given names are never renamed. If two Pipelines share the same
user-given step name, an `InvalidStepError` is raised:

```python
pipe1 = sq.Pipeline([("scale", sq.StandardScaler())])
pipe2 = sq.Pipeline([("scale", sq.MinMaxScaler())])
combined = pipe1 + pipe2  # raises InvalidStepError
```

## API Reference

::: sqlearn.core.pipeline.Pipeline.__add__

::: sqlearn.core.pipeline.Pipeline.__radd__

::: sqlearn.core.pipeline.Pipeline.__iadd__

::: sqlearn.core.transformer.Transformer.__add__

::: sqlearn.core.transformer.Transformer.__iadd__
