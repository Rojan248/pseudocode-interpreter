# Multi-dimensional Arrays and Enhanced Error Reporting

## Summary
Implemented full support for N-dimensional arrays in the interpreter and added robust runtime error reporting with line numbers.

## Features

### 1. Multi-dimensional Arrays
- **Implementation**: Used a flattened dictionary storage strategy with tuple keys `(i, j, ...)` to support sparse multi-dim arrays efficiently.
- **Symbol Table**: 
  - Updated `ArrayBounds` to store a list of dimension ranges `[(1, 10), (1, 10)]`.
  - Updated `Cell` to use tuple keys for `array_elements`.
  - Updated `get_array_element` and `set_array_element` to validate indices against all dimension bounds.
- **Interpreter**:
  - Updated `visit_DeclareStmt` to create multi-dim bounds.
  - Updated `visit_AssignStmt` and `evaluate_ArrayAccessExpr` to process list of indices.

### 2. Error Reporting
- **Parser**: Now attaches source line numbers to each statement node in the AST.
- **Interpreter**: Tracks `current_line` during execution.
- **Main**: Catches `InterpreterError` and reports `Runtime Error at line X: ...` for easier debugging.

## Verification
- **New Tests**:
  - `test_multidim.pse`: Verified 3x3 matrix multiplication and nested loops.
  - `test_errors.pse`: Verified runtime out-of-bounds error reporting with correct line number.
- **Regression**: Passed `test_array.pse` (1D), `test_records.pse` (User Types), and standard control flow tests.

## Cleanup
- Removed legacy `blueprint.md` documentation.
