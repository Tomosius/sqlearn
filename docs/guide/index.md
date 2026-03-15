# User Guide

This guide explains the concepts behind sqlearn --- how it works, why it works that way,
and how to get the most out of it. If you just want to get started quickly, see
[Getting Started](../getting-started.md).

## What is in this guide

**[Core Concepts](concepts.md)** --- The mental model for sqlearn. Covers the
fit/transform lifecycle, how transformers are classified as static or dynamic, how
columns are automatically routed by type, and how expressions compose into a single
SQL query.

**[How the Compiler Works](compiler.md)** --- A deep dive into the three-phase
compiler that turns your Python pipeline into SQL. Follows a concrete example
(Imputer + StandardScaler) step by step from Python code to the final SQL output.

**[Column Routing](column-routing.md)** --- Everything about controlling which columns
go to which transformers. From automatic type-based defaults to explicit selectors,
`Columns` groups, and `Union` branches.

**[Custom Transformers](custom-transformers.md)** --- Three levels of customization,
from one-line SQL expressions to full Transformer subclasses. Pick the level that
matches your needs.

## Who is this guide for

This guide is for anyone using sqlearn who wants to understand what happens under the
hood. You do not need to read it to use sqlearn productively --- the library is designed
so that `Pipeline([Imputer(), StandardScaler()])` just works. But when you need to
debug unexpected SQL output, write a custom transformer, or optimize a complex pipeline,
this guide gives you the knowledge to do so confidently.
