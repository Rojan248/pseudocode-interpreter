"""
Dry-Run / Trace Mode Interpreter for the 9618 Pseudocode Interpreter.

Cambridge 9618 A-Level exam-style dry-run interpreter:
  - Accepts pre-supplied input values (as given on an exam paper)
  - Records a trace table of selected variables at each executed step
  - Feeds inputs from a queue instead of prompting interactively

Extracted from interpreter.py to improve cohesion (LCOM4).
"""
from typing import Optional, Set, List
from ast_nodes import *
from symbol_table import SymbolTable, Cell, DataType
from interpreter import Interpreter, InterpreterError, PseudocodeObject


class DryRunInterpreter(Interpreter):
    """
    Cambridge 9618 A-Level exam-style dry-run interpreter.
    """

    def __init__(self, symbol_table: SymbolTable, input_queue=None,
                 trace_columns=None, max_steps=5000):
        super().__init__(symbol_table)
        self.trace = []
        self.step_count = 0
        self.max_steps = max_steps
        self.input_queue = list(input_queue) if input_queue else []
        self.input_index = 0
        
        # Parse custom column expressions if provided
        self.column_exprs = []
        self.char_col_indices = set() # Track which columns are Chars[...] for sorting/formatting
        
        if trace_columns:
            from lexer import Lexer
            from parser import Parser
            for idx, col_str in enumerate(trace_columns):
                try:
                    # Parse the column string into an AST node
                    l = Lexer(col_str)
                    tokens = l.tokenize()
                    p = Parser(tokens)
                    expr = p.parse_expression()
                    self.column_exprs.append((col_str, expr))
                except Exception:
                    # Fallback for invalid expressions - just treat as name
                    self.column_exprs.append((col_str, None))
        else:
            self.column_exprs = None  # None implies "auto-discover all variables"

        self.output_log = []

    # ══════════════════════════════════════════════════════
    #  AST Scanning (static, no execution)
    # ══════════════════════════════════════════════════════

    @staticmethod
    def scan_inputs(statements):
        """Walk AST to find all INPUT statements.
        Returns list of dicts: {'line': int, 'variable': str}
        """
        results = []
        DryRunInterpreter._walk_for_inputs(statements, results)
        return results

    @staticmethod
    def _walk_for_inputs(stmts, results):
        for stmt in stmts:
            if isinstance(stmt, InputStmt):
                var = _extract_input_target_name(stmt)
                results.append({'line': getattr(stmt, 'line', 0), 'variable': var})
            _walk_children(stmt, results)

    @staticmethod
    def scan_declares(statements):
        """Walk AST to find all DECLARE / CONSTANT statements.
        Returns list of dicts: {'name': str, 'type': str, 'is_array': bool}
        """
        results = []
        for stmt in statements:
            if isinstance(stmt, DeclareStmt):
                results.append({
                    'name': stmt.name,
                    'type': stmt.type_name,
                    'is_array': stmt.is_array,
                })
            elif isinstance(stmt, ConstantDecl):
                results.append({
                    'name': stmt.name,
                    'type': 'CONSTANT',
                    'is_array': False,
                })
        return results

    # ══════════════════════════════════════════════════════
    #  Snapshot & Trace Recording
    # ══════════════════════════════════════════════════════

    def _snapshot_vars(self):
        """Capture current values of traced columns."""
        snapshot = {}
        
        # Case 1: Custom defined columns (e.g. "I", "J", "Chars[J]")
        if self.column_exprs is not None:
            for col_name, node in self.column_exprs:
                if node:
                    try:
                        val = self.evaluate(node)
                        snapshot[col_name] = val
                    except Exception:
                        # Expression failed (e.g. J is out of bounds for Chars[J])
                        snapshot[col_name] = "" 
                else:
                    snapshot[col_name] = ""
            return snapshot

        # Case 2: Auto-discovery (Cambridge Format)
        for scope_level in range(self.symbol_table.scope_level + 1):
            scope = self.symbol_table.scopes[scope_level]
            for name, sym in scope.items():
                cell = sym.cell
                if cell.is_array and cell.array_elements:
                    # Expand array: Arr[1], Arr[2]...
                    for key, elem_cell in sorted(cell.array_elements.items()):
                        idx_str = ",".join(str(k) for k in key)
                        col_name = f"{name}[{idx_str}]"
                        try:
                            snapshot[col_name] = elem_cell.get()
                        except Exception:
                            snapshot[col_name] = elem_cell.value
                elif cell.is_array:
                    pass
                else:
                    snapshot[name] = _snapshot_cell(cell)
        return snapshot

    def _record(self, stmt, note=""):
        """Record one trace table row."""
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise InterpreterError(
                f"Dry-run stopped after {self.max_steps} steps "
                f"(possible infinite loop)")
        self.trace.append({
            'step': self.step_count,
            'line': getattr(stmt, 'line', self.current_line),
            'statement': _describe_stmt(stmt),
            'note': note,
            'variables': self._snapshot_vars(),
            'is_declare': isinstance(stmt, (DeclareStmt, ConstantDecl)),
            'is_definition': isinstance(stmt, (ProcedureDecl, FunctionDecl, TypeDecl, ClassDecl)),
        })

    # ══════════════════════════════════════════════════════
    #  Override execute to record trace
    # ══════════════════════════════════════════════════════

    def execute(self, stmt: Stmt):
        if hasattr(stmt, 'line') and stmt.line > 0:
            self.current_line = stmt.line

        # Definitions: store but don't trace body execution
        if isinstance(stmt, (ProcedureDecl, FunctionDecl, TypeDecl, ClassDecl)):
            self._record(stmt, "defined")
            return super().execute(stmt)

        # Expression-statements (e.g. SUPER.Method())
        if isinstance(stmt, Expr):
            self._record(stmt, "expr")
            return super().execute(stmt)

        # Execute the statement
        result = super().execute(stmt)

        # Build note about what changed
        note = self._build_execution_note(stmt)
        self._record(stmt, note)
        return result

    def _build_execution_note(self, stmt):
        """Build a human-readable note describing statement side-effects."""
        if isinstance(stmt, AssignStmt):
            return self._note_for_assign(stmt)
        if isinstance(stmt, DeclareStmt):
            return stmt.type_name
        if isinstance(stmt, OutputStmt):
            return self._note_for_output(stmt)
        if isinstance(stmt, InputStmt):
            return self._note_for_input(stmt)
        if isinstance(stmt, IfStmt):
            return "condition"
        if isinstance(stmt, ForStmt):
            return f"loop {stmt.identifier}"
        return ""

    def _note_for_assign(self, stmt):
        """Build note for an assignment statement."""
        t = stmt.target
        try:
            if isinstance(t, str):
                v = self.symbol_table.get_cell(t).get()
                return f"= {_fmt_value(v)}"
            if isinstance(t, ArrayAccessExpr):
                indices = [self.evaluate(idx) for idx in t.indices]
                v = self.symbol_table.array_access(t.array, indices).get()
                idx_s = ",".join(str(i) for i in indices)
                return f"[{idx_s}] = {_fmt_value(v)}"
        except Exception:
            pass
        return ""

    def _note_for_output(self, stmt):
        """Build note for an OUTPUT statement."""
        vals = [self._format_output(self.evaluate(a)) for a in stmt.values]
        out = "".join(vals)
        self.output_log.append(out)
        return f"OUT: {out}"

    def _note_for_input(self, stmt):
        """Build note for an INPUT statement."""
        if isinstance(stmt.target, VariableExpr):
            try:
                v = self.symbol_table.get_cell(stmt.target.name).get()
                return f"= {_fmt_value(v)}"
            except Exception:
                pass
        return ""

    # ══════════════════════════════════════════════════════
    #  Override INPUT to use pre-supplied queue
    # ══════════════════════════════════════════════════════

    def visit_InputStmt(self, stmt: InputStmt):
        val_str = self._consume_next_input()
        val, val_type = self._resolve_input_value(stmt, val_str)
        self._assign_input_to_target(stmt, val, val_type)

    def _consume_next_input(self):
        """Get next value from the pre-supplied input queue."""
        if self.input_index < len(self.input_queue):
            val_str = str(self.input_queue[self.input_index])
            self.input_index += 1
            return val_str
        raise InterpreterError(
            f"Dry-run ran out of pre-supplied input values. "
            f"Needed input #{self.input_index + 1} but only "
            f"{len(self.input_queue)} value(s) were provided.")

    def _resolve_input_value(self, stmt, val_str):
        """Coerce the raw input string to the target variable's type."""
        target_type = self._get_target_type(stmt)
        if target_type is not None:
            return self._coerce_input(val_str, target_type)
        return self._auto_parse_input(val_str)

    def _get_target_type(self, stmt):
        """Look up the declared type of the INPUT target variable."""
        try:
            if isinstance(stmt.target, VariableExpr):
                return self.symbol_table.get_cell(stmt.target.name).type
            if isinstance(stmt.target, str):
                return self.symbol_table.get_cell(stmt.target).type
            if isinstance(stmt.target, ArrayAccessExpr):
                cell = self.symbol_table.get_cell(stmt.target.array)
                if cell.is_array and cell.array_bounds:
                    return cell.array_bounds.element_type
        except Exception:
            pass
        return None

    def _assign_input_to_target(self, stmt, val, val_type):
        """Assign the parsed input value to the target variable/array."""
        if isinstance(stmt.target, VariableExpr):
            self.symbol_table.assign(stmt.target.name, val, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            self.symbol_table.array_assign(stmt.target.array, indices, val, val_type)
        elif isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, val, val_type)

    # ══════════════════════════════════════════════════════
    #  Results & Formatting
    # ══════════════════════════════════════════════════════

    def get_all_var_names(self):
        """Return list of column headers.
        If custom columns were provided, return them in defined order.
        Otherwise, return sorted auto-discovered variables.
        """
        if self.column_exprs is not None:
            return [name for name, _ in self.column_exprs]

        names = set()
        for entry in self.trace:
            names.update(entry['variables'].keys())
        names -= set(self.procedures.keys())
        names -= set(self.functions.keys())
        
        # Sort: scalars first alphabetically, then array elements
        scalars = sorted(n for n in names if '[' not in n)
        arrays = sorted((n for n in names if '[' in n),
                        key=lambda x: (x.split('[')[0], _parse_array_sort_key(x)))
        return scalars + arrays

    def get_cambridge_trace(self):
        """Return trace entries filtered for Cambridge exam format.
        Skips DECLARE/definition steps and only includes execution steps.
        """
        return [e for e in self.trace
                if not e.get('is_declare') and not e.get('is_definition')]

    def format_trace_text(self):
        """Format the trace as an ASCII table string."""
        if not self.trace:
            return "No trace data recorded."
        var_names = self.get_all_var_names()
        headers = ['Step', 'Line'] + var_names
        rows = _build_cambridge_rows(self.trace, var_names)
        return _format_ascii_table(headers, rows)


# ══════════════════════════════════════════════════════
#  Module-level helper functions (reduce nesting)
# ══════════════════════════════════════════════════════

def _extract_input_target_name(stmt):
    """Extract the variable name from an INPUT statement target."""
    if isinstance(stmt.target, VariableExpr):
        return stmt.target.name
    if isinstance(stmt.target, ArrayAccessExpr):
        return stmt.target.array + "[...]"
    if isinstance(stmt.target, str):
        return stmt.target
    return "?"


# ── Statement description dispatch table ──

_STMT_DESCRIPTIONS = {
    DeclareStmt:      lambda s: f"DECLARE {s.name} : {'ARRAY ' if s.is_array else ''}{s.type_name}",
    ConstantDecl:     lambda s: f"CONSTANT {s.name}",
    OutputStmt:       lambda s: "OUTPUT",
    IfStmt:           lambda s: "IF",
    WhileStmt:        lambda s: "WHILE",
    RepeatStmt:       lambda s: "REPEAT",
    ForStmt:          lambda s: f"FOR {s.identifier}",
    CaseStmt:         lambda s: "CASE OF",
    ProcedureCallStmt: lambda s: f"CALL {s.name}",
    ReturnStmt:       lambda s: "RETURN",
    ProcedureDecl:    lambda s: f"PROCEDURE {s.name}",
    FunctionDecl:     lambda s: f"FUNCTION {s.name}",
    TypeDecl:         lambda s: f"TYPE {s.name}",
    ClassDecl:        lambda s: f"CLASS {s.name}",
    FileStmt:         lambda s: f"{s.operation}FILE",
}


def _describe_assign(stmt):
    """Describe an AssignStmt target."""
    t = stmt.target
    if isinstance(t, str):
        return f"{t} \u2190 ..."
    if isinstance(t, ArrayAccessExpr):
        return f"{t.array}[...] \u2190 ..."
    if isinstance(t, MemberExpr):
        return f".{t.field} \u2190 ..."
    return "ASSIGN"


def _describe_input(stmt):
    """Describe an InputStmt."""
    if isinstance(stmt.target, VariableExpr):
        return f"INPUT {stmt.target.name}"
    if isinstance(stmt.target, ArrayAccessExpr):
        return f"INPUT {stmt.target.array}[...]"
    return "INPUT"


def _describe_stmt(stmt):
    """Short human-readable description of a statement."""
    if isinstance(stmt, AssignStmt):
        return _describe_assign(stmt)
    if isinstance(stmt, InputStmt):
        return _describe_input(stmt)

    desc_fn = _STMT_DESCRIPTIONS.get(type(stmt))
    if desc_fn:
        return desc_fn(stmt)
    return type(stmt).__name__


def _fmt_value(val):
    """Format a value for trace display."""
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def _snapshot_cell(cell):
    """Take a snapshot of a single Cell value."""
    if cell.is_array:
        return _snapshot_array(cell)
    if cell.type == DataType.RECORD and isinstance(cell.value, dict):
        return _snapshot_record(cell.value)
    if isinstance(cell.value, PseudocodeObject):
        return _snapshot_object(cell.value)
    try:
        return cell.get()
    except Exception:
        return cell.value


def _snapshot_array(cell):
    """Snapshot an array cell."""
    if not cell.array_elements:
        return ""
    arr = {}
    for key, ec in sorted(cell.array_elements.items()):
        idx = ",".join(str(k) for k in key)
        arr[f"[{idx}]"] = ec.get()
    return arr


def _snapshot_record(fields):
    """Snapshot a record (dict of Cells)."""
    return {
        fn: (fc.get() if isinstance(fc, Cell) else fc)
        for fn, fc in fields.items()
    }


def _snapshot_object(obj):
    """Snapshot a PseudocodeObject instance."""
    attrs = {
        an: (ac.get() if isinstance(ac, Cell) else ac)
        for an, ac in obj.attributes.items()
    }
    return f"<{obj.class_name}>{attrs}"


# ── AST child walker (replaces deeply nested if-elif chain) ──

_COMPOUND_CHILDREN = {
    IfStmt:     lambda s: [s.then_branch] + ([s.else_branch] if s.else_branch else []),
    WhileStmt:  lambda s: [s.body],
    RepeatStmt: lambda s: [s.body],
    ForStmt:    lambda s: [s.body],
}


def _walk_children(stmt, results):
    """Recurse into compound statement children for INPUT scanning."""
    getter = _COMPOUND_CHILDREN.get(type(stmt))
    if getter:
        for child_list in getter(stmt):
            DryRunInterpreter._walk_for_inputs(child_list, results)
        return

    if isinstance(stmt, CaseStmt):
        for branch in stmt.branches:
            DryRunInterpreter._walk_for_inputs(branch.statements, results)
        if stmt.otherwise_branch:
            DryRunInterpreter._walk_for_inputs(stmt.otherwise_branch, results)
    elif isinstance(stmt, (ProcedureDecl, FunctionDecl)):
        DryRunInterpreter._walk_for_inputs(stmt.body, results)
    elif isinstance(stmt, ClassDecl):
        for member in stmt.members:
            if isinstance(member, (ProcedureDecl, FunctionDecl)):
                DryRunInterpreter._walk_for_inputs(member.body, results)


# ── Trace table formatting helpers ──

def _parse_array_sort_key(name):
    """Parse array element name like 'Chars[2]' into a sortable key."""
    try:
        idx_part = name.split('[')[1].rstrip(']')
        return tuple(int(x) for x in idx_part.split(','))
    except (ValueError, IndexError):
        return (0,)


def _build_cambridge_rows(trace, var_names):
    """Build Cambridge exam-style trace rows.
    Only shows values when they change from the previous step.
    Skips DECLARE/definition steps.
    """
    rows = []
    prev_values = {}  # Track previous value for each variable
    step_num = 0
    for entry in trace:
        # Skip DECLARE and definition steps
        if entry.get('is_declare') or entry.get('is_definition'):
            continue
        step_num += 1
        row = [str(step_num), str(entry['line'])]
        for vn in var_names:
            val = entry['variables'].get(vn, None)
            if val is None:
                # Variable doesn't exist yet
                row.append('')
            else:
                formatted = _format_cambridge_cell(val)
                prev = prev_values.get(vn)
                if prev is None or prev != formatted:
                    row.append(formatted)
                    prev_values[vn] = formatted
                else:
                    row.append('')  # Unchanged — leave blank
        rows.append(row)
    return rows


def _format_cambridge_cell(val):
    """Format a single cell value in Cambridge exam style."""
    if val is None or val == '':
        return ''
    if isinstance(val, bool):
        return 'TRUE' if val else 'FALSE'
    if isinstance(val, str):
        # Show chars/strings with quotes like the exam paper
        if len(val) == 1:
            return f"'{val}'"
        return f'"{val}"'
    if isinstance(val, float):
        # Show clean floats
        if val == int(val):
            return str(int(val))
        return str(val)
    if isinstance(val, dict):
        return str(val)
    return str(val)


def _format_trace_cell(val):
    """Format a single cell value for the trace table."""
    if isinstance(val, dict):
        return str(val)
    if val == '':
        return ''
    if isinstance(val, str):
        return val
    return _fmt_value(val)


def _format_ascii_table(headers, rows):
    """Render headers + rows as a fixed-width ASCII table."""
    col_w = [len(h) for h in headers]
    for row in rows:
        for i, c in enumerate(row):
            col_w[i] = max(col_w[i], len(str(c)))
    col_w = [min(w, 30) for w in col_w]

    def pad(s, w):
        return str(s)[:w].ljust(w)

    sep = '+' + '+'.join('-' * (w + 2) for w in col_w) + '+'
    hdr = '|' + '|'.join(f" {pad(h, w)} " for h, w in zip(headers, col_w)) + '|'
    lines = [sep, hdr, sep]
    for row in rows:
        r = '|' + '|'.join(f" {pad(c, w)} " for c, w in zip(row, col_w)) + '|'
        lines.append(r)
    lines.append(sep)
    return '\n'.join(lines)
