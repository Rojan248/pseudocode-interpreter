## 2024-05-24 - Interpreter Loop Variable Lookup Overhead
**Learning:** The `Interpreter.visit_ForStmt` implementation was performing two full symbol table lookups (`get_cell`) per iteration—one to read the current value and one to update it. Even with O(1) cache, the function call and dictionary overhead became significant (26% of loop time) in tight loops.
**Action:** For interpreter loops, always cache the variable reference (e.g., `Cell` object) outside the loop body to bypass symbol table lookups during iteration.
