# Commit: Full Cambridge 9618 Pseudocode Spec Compliance

Audited the interpreter against the official **Cambridge International AS & A Level Computer Science 9618 Pseudocode Guide for Teachers (2026)** and implemented all missing/incorrect syntax features.

---

## Fixes Applied

### 1. Record TYPE Field Syntax (CRITICAL)
- `DECLARE` keyword before each field inside `TYPE...ENDTYPE` blocks is now accepted (consumed if present, backward-compatible if omitted).

### 2. BYREF/BYVAL Mode Persistence (CRITICAL)
- The spec states the mode keyword need not be repeated for subsequent parameters. Fixed by moving `mode = "BYVAL"` outside the parameter parsing loop so the mode carries across comma-separated params.

### 3. OUTPUT Spacing (CRITICAL)
- Changed `" ".join(values)` → `"".join(values)`. The spec expects values to be concatenated directly; spacing is the programmer's responsibility via string literals.

### 4. Unicode Assignment Arrow (MODERATE)
- Lexer now recognizes `←` in addition to `<-` for assignment.

### 5. Enumerated TYPE Support (MODERATE)
- New syntax: `TYPE Season = (Spring, Summer, Autumn, Winter)`
- Enum values are stored as 0-indexed INTEGER constants.
- Added `enum_values` field to `TypeDecl` AST node.

### 6. Random File Handling (MODERATE)
- Added `RANDOM` mode to `OPENFILE`.
- Implemented `SEEK`, `GETRECORD`, `PUTRECORD` across lexer, parser, AST, and interpreter.
- Uses `pickle` for record serialization.

### 7. CASE Negative Labels (MINOR)
- Fixed `is_case_label_start` to detect `MINUS` + `INTEGER` peek for labels like `-3 : statement`.

### 8. OOP — Full CLASS Support (LARGE)
**Lexer:** Added tokens — `CLASS`, `ENDCLASS`, `INHERITS`, `PUBLIC`, `PRIVATE`, `NEW`, `SUPER`.

**AST Nodes Added:**
- `ClassDecl(name, parent, members)`
- `NewExpr(class_name, arguments)`
- `SuperExpr(method, arguments)`
- `MethodCallExpr(object_expr, method_name, arguments)`

**Parser:**
- `parse_class_decl()` — parses `CLASS...ENDCLASS` with optional `INHERITS`, `PUBLIC`/`PRIVATE` modifiers, bare field declarations (`PRIVATE Name : STRING`), and `PROCEDURE`/`FUNCTION` members.
- `PROCEDURE NEW(...)` — `NEW` keyword accepted as a procedure name (constructor).
- `SUPER.NEW(...)` / `SUPER.Method(...)` — parsed as standalone statements.
- `obj.Method(args)` — produces `MethodCallExpr` instead of crashing.
- `CALL obj.Method(args)` — dot-notation in CALL statements.

**Interpreter:**
- `PseudocodeObject` runtime class with `class_name`, `attributes`, `methods`, `parent_class`.
- `visit_ClassDecl` — stores class definitions for later instantiation.
- `evaluate_NewExpr` — creates objects, initializes inherited fields, calls constructor.
- `evaluate_SuperExpr` — dispatches to parent class methods.
- `evaluate_MethodCallExpr` — calls methods on object instances.
- `_call_method`, `_init_class_fields`, `_collect_methods` — OOP helper methods.
- `evaluate_MemberExpr` updated to handle both records (dict) and objects.
- Assignment to object attributes via `MemberExpr` target.
- Expression-statements (e.g., `SUPER.NEW()` as a statement) handled in `execute()`.

### 9. ReturnStmt Guard
- `visit_ReturnStmt` now handles bare `RETURN` (where `stmt.value` is `None`).

---

## Files Modified
| File | Changes |
|---|---|
| `lexer.py` | Unicode `←`, OOP tokens, random file tokens |
| `parser.py` | Enum types, BYREF persistence, CASE negatives, class parsing, method calls, SUPER statements |
| `ast_nodes.py` | `TypeDecl.enum_values`, `ClassDecl`, `NewExpr`, `SuperExpr`, `MethodCallExpr` |
| `interpreter.py` | OUTPUT spacing, enum types, random files, full OOP runtime, ReturnStmt guard |
| `symbol_table.py` | Unchanged |

## Tests
- All existing `.pse` test files pass with zero regressions.
- New `test_oop.pse` covers CLASS, INHERITS, PRIVATE fields, constructors, SUPER.NEW, method calls, and inherited method access.
