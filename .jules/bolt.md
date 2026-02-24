## 2024-05-22 - Python Function Call Overhead in Interpreter Loops
**Learning:** In a pure Python interpreter, the overhead of dictionary lookups combined with lambda function calls for every binary operation adds up significantly (~5-6% total runtime).
**Action:** Prefer inline if/elif chains for extremely hot paths like `evaluate_BinaryExpr` over cleaner but slower dispatch tables.
