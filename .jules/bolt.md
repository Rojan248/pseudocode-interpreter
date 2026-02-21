## 2024-05-22 - Python Interpreter Optimization
**Learning:** In a Python-based AST interpreter, dictionary-based dispatch tables (especially with `lambda`s) for simple binary operations incur significant overhead due to function calls and dictionary lookups.
**Action:** For extremely hot paths like `evaluate_BinaryExpr`, an explicit `if/elif` chain is faster (~5% improvement observed) and eliminates the need for separate dispatch tables and helper functions.
