# Code Citations

## License: unknown
https://github.com/cstroup/business-planning/blob/c48782917b7971b53d1174d4c05aa4463862e170/planning_app.py

```
Now I have full context. Let me build the complete solution. The approach:

1. **Scan the AST** to find all `InputStmt` nodes and extract variable names — no LLM needed
2. **Before dry-run starts**, show a setup dialog where the user supplies:
   - All input values upfront (comma-separated, like exam papers give them)
   - Which variables to trace (auto-detected from DECLARE statements, user can toggle)
3. **During execution**, feed inputs from the pre-supplied queue instead of prompting
4. **After execution**, show the trace table in exact A-Level format

Here are the three files to update:

**1. Append to the end of [interpreter.py](interpreter.py) — the `DryRunInterpreter` class:**

```python

# ──────────────── Dry-Run / Trace Mode ────────────────

class DryRunInterpreter(Interpreter):
    """
    Cambridge 9618-style dry-run interpreter.
    - Accepts pre-supplied input values (like exam papers provide)
    - Records a trace table of selected variables at each step
    - Feeds inputs from queue instead of prompting interactively
    """

    def __init__(self, symbol_table: SymbolTable, input_queue=None,
                 traced_vars=None, max_steps=5000):
        super().__init__(symbol_table)
        self.trace = []
        self.step_count = 0
        self.max_steps = max_steps
        self.input_queue = list(input_queue) if input_queue else []
        self.input_index = 0
        self.traced_vars = set(traced_vars) if traced_vars else None  # None = auto-detect all
        self.output_log = []  # record OUTPUT values too

    # ── AST scanning helpers (static, no execution needed) ──

    @staticmethod
    def scan_inputs(statements):
        """Walk the AST to find all INPUT statements and return info about them.
        Returns list of dicts: {'line': int, 'variable': str}
        """
        results = []
        DryRunInterpreter._walk_for_inputs(statements, results)
        return results

    @staticmethod
    def _walk_for_inputs(stmts, results):
        for stmt in stmts:
            if isinstance(stmt, InputStmt):
                var_name = "?"
                if isinstance(stmt.target, VariableExpr):
                    var_name = stmt.target.name
                elif isinstance(stmt.target, ArrayAccessExpr):
                    var_name = stmt.target.array + "[...]"
                elif isinstance(stmt.target, str):
                    var_name = stmt.target
                line = getattr(stmt, 'line', 0)
                results.append({'line': line, 'variable': var_name})
            # Recurse into compound statements
            if isinstance(stmt, IfStmt):
                DryRunInterpreter._walk_for_inputs(stmt.then_branch, results)
                if stmt.else_branch:
                    DryRunInterpreter._walk_for_inputs(stmt.else_branch, results)
            elif isinstance(stmt, WhileStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, RepeatStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, ForStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, CaseStmt):
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

    @staticmethod
    def scan_declares(statements):
        """Walk the AST to find all DECLARE statements.
        Returns list of dicts: {'name': str, 'type': str, 'is_array': bool}
        """
        results = []
        for stmt in statements:
            if isinstance(stmt, DeclareStmt):
                results.append({
                    'name': stmt.name,
                    'type': stmt.type_name,
                    'is_array': stmt.is_array
                })
            elif isinstance(stmt, ConstantDecl):
                results.append({
                    'name': stmt.name,
                    'type': 'CONSTANT',
                    'is_array': False
                })
        return results

    # ── Snapshot & trace recording ──

    def _snapshot_vars(self):
        """Capture values of traced variables only."""
        snapshot = {}
        for scope_level in range(self.symbol_table.scope_level + 1):
            scope = self.symbol_table.scopes.get(scope_level, {})
            for name, sym in scope.items():
                # If user specified which vars to trace, filter
                if self.traced_vars is not None and name not in self.traced_vars:
                    continue
                cell = sym.cell
                if cell.is_array:
                    if cell.array_elements:
                        arr = {}
                        for key, ec in sorted(cell.array_elements.items()):
                            idx = ",".join(str(k) for k in key)
                            arr[f"[{idx}]"] = ec.get()
                        snapshot[name] = arr
                    else:
                        snapshot[name] = ""
                elif cell.type == DataType.RECORD and isinstance(cell.value, dict):
                    rec = {}
                    for fn, fc in cell.value.items():
                        rec[fn] = fc.get() if isinstance(fc, Cell) else fc
                    snapshot[name] = rec
                elif isinstance(cell.value, PseudocodeObject):
                    attrs = {}
                    for an, ac in cell.value.attributes.items():
                        attrs[an] = ac.get() if isinstance(ac, Cell) else ac
                    snapshot[name] = f"<{cell.value.class_name}>{attrs}"
                else:
                    try:
                        snapshot[name] = cell.get()
                    except Exception:
                        snapshot[name] = cell.value
        return snapshot

    def _format_val(self, val):
        if isinstance(val, bool): return "TRUE" if val else "FALSE"
        if isinstance(val, str): return f'"{val}"'
        return str(val)

    def _describe_stmt(self, stmt):
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            if isinstance(t, str): return f"{t} \u2190 ..."
            if isinstance(t, ArrayAccessExpr): return f"{t.array}[...] \u2190 ..."
            if isinstance(t, MemberExpr): return f".{t.field} \u2190 ..."
            return "ASSIGN"
        if isinstance(stmt, DeclareStmt):
            arr = "ARRAY" if stmt.is_array else ""
            return f"DECLARE {stmt.name} : {arr} {stmt.type_name}".strip()
        if isinstance(stmt, ConstantDecl): return f"CONSTANT {stmt.name}"
        if isinstance(stmt, OutputStmt): return "OUTPUT"
        if isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr): return f"INPUT {stmt.target.name}"
            if isinstance(stmt.target, ArrayAccessExpr): return f"INPUT {stmt.target.array}[...]"
            return "INPUT"
        if isinstance(stmt, IfStmt): return "IF"
        if isinstance(stmt, WhileStmt): return "WHILE"
        if isinstance(stmt, RepeatStmt): return "REPEAT"
        if isinstance(stmt, ForStmt): return f"FOR {stmt.identifier}"
        if isinstance(stmt, CaseStmt): return "CASE OF"
        if isinstance(stmt, ProcedureCallStmt): return f"CALL {stmt.name}"
        if isinstance(stmt, ReturnStmt): return "RETURN"
        if isinstance(stmt, ProcedureDecl): return f"PROCEDURE {stmt.name}"
        if isinstance(stmt, FunctionDecl): return f"FUNCTION {stmt.name}"
        if isinstance(stmt, TypeDecl): return f"TYPE {stmt.name}"
        if isinstance(stmt, ClassDecl): return f"CLASS {stmt.name}"
        if isinstance(stmt, FileStmt): return f"{stmt.operation}FILE"
        return type(stmt).__name__

    def _record_step(self, stmt, note=""):
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise InterpreterError(
                f"Dry-run stopped after {self.max_steps} steps (possible infinite loop)")
        entry = {
            'step': self.step_count,
            'line': getattr(stmt, 'line', self.current_line),
            'statement': self._describe_stmt(stmt),
            'note': note,
            'variables': self._snapshot_vars(),
        }
        self.trace.append(entry)

    # ── Override execute to record trace ──

    def execute(self, stmt: Stmt):
        if hasattr(stmt, 'line') and stmt.line > 0:
            self.current_line = stmt.line

        # Definitions — store but don't trace body execution
        if isinstance(stmt, (ProcedureDecl, FunctionDecl, TypeDecl, ClassDecl)):
            self._record_step(stmt, "defined")
            return super().execute(stmt)

        if isinstance(stmt, Expr):
            self._record_step(stmt, "expr")
            return super().execute(stmt)

        # Execute
        result = super().execute(stmt)

        # Build note about what changed
        note = ""
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            try:
                if isinstance(t, str):
                    v = self.symbol_table.get_cell(t).get()
                    note = f"= {self._format_val(v)}"
                elif isinstance(t, ArrayAccessExpr):
                    indices = [self.evaluate(idx) for idx in t.indices]
                    v = self.symbol_table.array_access(t.array, indices).get()
                    idx_s = ",".join(str(i) for i in indices)
                    note = f"[{idx_s}] = {self._format_val(v)}"
            except Exception:
                pass
        elif isinstance(stmt, DeclareStmt):
            note = stmt.type_name
        elif isinstance(stmt, OutputStmt):
            # Record what was output
            vals = [self._format_output(self.evaluate(a)) for a in stmt.values]
            output_text = "".join(vals)
            self.output_log.append(output_text)
            note = f'OUT: {output_text}'
        elif isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr):
                try:
                    v = self.symbol_table.get_cell(stmt.target.name).get()
                    note = f"= {self._format_val(v)}"
                except Exception:
                    pass
        elif isinstance(stmt, IfStmt):
            note = "condition"
        elif isinstance(stmt, ForStmt):
            note = f"loop {stmt.identifier}"

        self._record_step(stmt, note)
        return result

    # ── Override INPUT to use pre-supplied queue ──

    def visit_InputStmt(self, stmt: InputStmt):
        """Use pre-supplied input values instead of prompting."""
        if self.input_index < len(self.input_queue):
            val_str = str(self.input_queue[self.input_index])
            self.input_index += 1
        else:
            raise InterpreterError(
                "Dry-run ran out of pre-supplied input values. "
                f"Needed input #{self.input_index + 1} but only "
                f"{len(self.input_queue)} values were provided.")

        # Determine target type for coercion
        target_type = None
        try:
            if isinstance(stmt.target, VariableExpr):
                target_type = self.symbol_table.get_cell(stmt.target.name).type
            elif isinstance(stmt.target, str):
                target_type = self.symbol_table.get_cell(stmt.target).type
            elif isinstance(stmt.target, ArrayAccessExpr):
                cell = self.symbol_table.get_cell(stmt.target.array)
                if cell.is_array and cell.array_bounds:
                    target_type = cell.array_bounds.element_type
        except Exception:
            pass

        if target_type is not None:
            val, val_type = self._coerce_input(val_str, target_type)
        else:
            val, val_type = self._auto_parse_input(val_str)

        if isinstance(stmt.target, VariableExpr):
            self.symbol_table.assign(stmt.target.name, val, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            self.symbol_table.array_assign(stmt.target.array, indices, val, val_type)
        elif isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, val, val_type)

    # ── Results ──

    def get_all_var_names(self):
        names = set()
        for entry in self.trace:
            names.update(entry['variables'].keys())
        names -= set(self.procedures.keys())
        names -= set(self.functions.keys())
        return sorted(names)

    def format_trace_text(self):
        if not self.trace:
            return "No trace data recorded."

        var_names = self.get_all_var_names()
        headers = ['Step', 'Line', 'Statement', 'Note'] + var_names
        rows = []
        for entry in self.trace:
            row = [str(entry['step']), str(entry['line']),
                   entry['statement'], entry['note']]
            for vn in var_names:
                val = entry['variables'].get(vn, '')
                if isinstance(val, dict):
                    val = str(val)
                elif val == '':
                    val = ''
                else:
                    val = self._format_val(val) if not isinstance(val, str) else val
                row.append(val)
            rows.append(row)

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, c in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(c)))
        col_widths = [min(w, 30) for w in col_widths]

        def pad(s, w):
            return str(s)[:w].ljust(w)

        sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
        hdr = '|' + '|'.join(f" {pad(h, w)} " for h, w in zip(headers, col_widths)) + '|'
        lines = [sep, hdr, sep]
        for row in rows:
            r = '|' + '|'.join(f" {pad(c, w)} " for c, w in zip(row, col_widths)) + '|'
            lines.append(r)
        lines.append(sep)
        return '\n'.join(lines)
```

**2. Replace the entire [ide.py](ide.py):**

```python
"""
9618 Pseudocode IDE - Modern Desktop Interface
Tkinter-based GUI with dark theme, syntax highlighting, line numbers, and integrated output.
Includes A-Level exam-style Dry Run mode with pre-supplied inputs and trace table.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import sys
import io
import os
import threading
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexer import Lexer, LexerError
from parser import Parser, ParserError
from interpreter import Interpreter, InterpreterError, DryRunInterpreter
from symbol_table import SymbolTable

# -- Theme Colors (Catppuccin Mocha) --
COLORS = {
    "bg":           "#1e1e2e",
    "bg_secondary": "#181825",
    "bg_tertiary":  "#11111b",
    "surface":      "#313244",
    "overlay":      "#45475a",
    "text":         "#cdd6f4",
    "subtext":      "#a6adc8",
    "blue":         "#89b4fa",
    "green":        "#a6e3a1",
    "red":          "#f38ba8",
    "yellow":       "#f9e2af",
    "mauve":        "#cba6f7",
    "peach":        "#fab387",
    "teal":         "#94e2d5",
    "pink":         "#f5c2e7",
    "lavender":     "#b4befe",
    "sky":          "#89dceb",
    "sapphire":     "#74c7ec",
    "line_num_fg":  "#585b70",
    "selection":    "#45475a",
    "cursor":       "#f5e0dc",
    "gutter":       "#282a3a",
    "output_bg":    "#11111b",
    "toolbar_bg":   "#181825",
    "status_bg":    "#181825",
    "tab_active":   "#1e1e2e",
    "accent":       "#89b4fa",
    "error":        "#f38ba8",
    "success":      "#a6e3a1",
    "warning":      "#f9e2af",
    "button_bg":    "#313244",
    "button_hover": "#45475a",
    "border":       "#313244",
}

# -- Keyword Lists for Highlighting --
KEYWORDS_CONTROL = {
    'IF', 'THEN', 'ELSE', 'ENDIF', 'CASE', 'OF', 'OTHERWISE', 'ENDCASE',
    'FOR', 'TO', 'STEP', 'NEXT', 'WHILE', 'DO', 'ENDWHILE',
    'REPEAT', 'UNTIL', 'RETURN',
}
KEYWORDS_DECL = {
    'DECLARE', 'CONSTANT', 'TYPE', 'ENDTYPE', 'ARRAY',
    'PROCEDURE', 'ENDPROCEDURE', 'FUNCTION', 'ENDFUNCTION',
    'RETURNS', 'BYREF', 'BYVAL', 'CALL',
}
KEYWORDS_TYPE = {'INTEGER', 'REAL', 'STRING', 'BOOLEAN', 'CHAR', 'DATE'}
KEYWORDS_IO = {
    'INPUT', 'OUTPUT', 'OPENFILE', 'READFILE', 'WRITEFILE', 'CLOSEFILE',
    'READ', 'WRITE', 'APPEND',
}
KEYWORDS_OP = {'AND', 'OR', 'NOT', 'DIV', 'MOD', 'TRUE', 'FALSE'}
BUILTINS = {
    'LENGTH', 'UCASE', 'LCASE', 'LEFT', 'RIGHT', 'MID',
    'INT', 'NUM_TO_STR', 'STR_TO_NUM', 'ASC', 'CHR', 'SQRT', 'RAND', 'EOF',
}


class LineNumbers(tk.Canvas):
    def __init__(self, parent, text_widget, **kwargs):
        super().__init__(parent, **kwargs)
        self.text_widget = text_widget
        self.font = None

    def redraw(self, *_args):
        self.delete("all")
        if self.text_widget is None:
            return
        i = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.create_text(
                self.winfo_width() - 8, y, anchor="ne", text=linenum,
                font=self.font, fill=COLORS["line_num_fg"],
            )
            i = self.text_widget.index(f"{i}+1line")
            if self.text_widget.compare(i, ">=", "end"):
                break


class CodeEditor(tk.Text):
    TAG_CONFIG = {
        "keyword_ctrl":  {"foreground": COLORS["mauve"],   "font_style": "bold"},
        "keyword_decl":  {"foreground": COLORS["blue"],    "font_style": "bold"},
        "keyword_type":  {"foreground": COLORS["yellow"]},
        "keyword_io":    {"foreground": COLORS["green"],   "font_style": "bold"},
        "keyword_op":    {"foreground": COLORS["peach"],   "font_style": "bold"},
        "builtin":       {"foreground": COLORS["sapphire"]},
        "string":        {"foreground": COLORS["green"]},
        "char":          {"foreground": COLORS["teal"]},
        "number":        {"foreground": COLORS["peach"]},
        "comment":       {"foreground": COLORS["overlay"], "font_style": "italic"},
        "operator":      {"foreground": COLORS["sky"]},
        "assign_arrow":  {"foreground": COLORS["red"]},
        "error_line":    {"background": "#3d2030"},
    }

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._setup_tags()
        self.bind("<<Modified>>", self._on_modify)
        self._highlight_job = None

    def _setup_tags(self):
        base_font = self.cget("font")
        for tag, cfg in self.TAG_CONFIG.items():
            opts = {"foreground": cfg.get("foreground", COLORS["text"])}
            if "background" in cfg:
                opts["background"] = cfg["background"]
            style = cfg.get("font_style", "")
            if style:
                fam = tkfont.Font(font=base_font).actual()["family"]
                sz = tkfont.Font(font=base_font).actual()["size"]
                weight = "bold" if "bold" in style else "normal"
                slant = "italic" if "italic" in style else "roman"
                opts["font"] = tkfont.Font(family=fam, size=sz, weight=weight, slant=slant)
            self.tag_configure(tag, **opts)
        self.tag_raise("error_line")

    def _on_modify(self, _event=None):
        if self.edit_modified():
            if self._highlight_job:
                self.after_cancel(self._highlight_job)
            self._highlight_job = self.after(80, self.highlight_syntax)
            self.edit_modified(False)
            self.event_generate("<<ContentChanged>>")

    def highlight_syntax(self):
        for tag in self.TAG_CONFIG:
            self.tag_remove(tag, "1.0", "end")
        code = self.get("1.0", "end-1c")
        patterns = [
            ("comment",      r'//[^\n]*'),
            ("string",       r'"[^"]*"'),
            ("char",         r"'[^']*'"),
            ("assign_arrow", r'<-'),
            ("number",       r'\b\d+\.\d+\b|\b\d+\b'),
            ("operator",     r'<>|<=|>=|[<>=+\-*/&]'),
        ]
        for tag, pat in patterns:
            for m in re.finditer(pat, code):
                start = f"1.0+{m.start()}c"
                end = f"1.0+{m.end()}c"
                self.tag_add(tag, start, end)
        word_groups = [
            ("keyword_ctrl", KEYWORDS_CONTROL),
            ("keyword_decl", KEYWORDS_DECL),
            ("keyword_type", KEYWORDS_TYPE),
            ("keyword_io",   KEYWORDS_IO),
            ("keyword_op",   KEYWORDS_OP),
            ("builtin",      BUILTINS),
        ]
        for tag, words in word_groups:
            for w in words:
                pat = rf'\b{w}\b'
                for m in re.finditer(pat, code):
                    start = f"1.0+{m.start()}c"
                    end = f"1.0+{m.end()}c"
                    tags_at = self.tag_names(start)
                    if "string" not in tags_at and "comment" not in tags_at and "char" not in tags_at:
                        self.tag_add(tag, start, end)

    def mark_error_line(self, line_num):
        self.tag_remove("error_line", "1.0", "end")
        if line_num and line_num > 0:
            self.tag_add("error_line", f"{line_num}.0", f"{line_num}.end+1c")
            self.see(f"{line_num}.0")


class OutputPanel(tk.Text):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(state="disabled")
        self.tag_configure("error",        foreground=COLORS["error"])
        self.tag_configure("success",      foreground=COLORS["success"])
        self.tag_configure("info",         foreground=COLORS["subtext"])
        self.tag_configure("output",       foreground=COLORS["text"])
        self.tag_configure("input_prompt", foreground=COLORS["yellow"])

    def append(self, text, tag="output"):
        self.configure(state="normal")
        self.insert("end", text, tag)
        self.see("end")
        self.configure(state="disabled")

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class InputDialog(tk.Toplevel):
    def __init__(self, parent, prompt):
        super().__init__(parent)
        self.title("Input Required")
        self.result = None
        self.configure(bg=COLORS["bg"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 400, 160
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        frame = tk.Frame(self, bg=COLORS["bg"], padx=20, pady=15)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text=prompt, font=("Segoe UI", 11),
                 fg=COLORS["yellow"], bg=COLORS["bg"], anchor="w").pack(fill="x", pady=(0, 8))

        self.entry = tk.Entry(frame, font=("Cascadia Code", 12),
                              bg=COLORS["surface"], fg=COLORS["text"],
                              insertbackground=COLORS["cursor"], relief="flat", bd=0)
        self.entry.pack(fill="x", ipady=6)
        self.entry.focus_set()
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Escape>", lambda e: self._cancel())

        btn_frame = tk.Frame(frame, bg=COLORS["bg"])
        btn_frame.pack(fill="x", pady=(12, 0))

        tk.Button(btn_frame, text="OK", font=("Segoe UI", 10),
                  bg=COLORS["accent"], fg=COLORS["bg_tertiary"],
                  activebackground=COLORS["lavender"],
                  relief="flat", bd=0, padx=20, pady=4,
                  command=self._submit).pack(side="right")
        tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 10),
                  bg=COLORS["surface"], fg=COLORS["text"],
                  activebackground=COLORS["overlay"],
                  relief="flat", bd=0, padx=16, pady=4,
                  command=self._cancel).pack(side="right", padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _submit(self, _event=None):
        self.result = self.entry.get()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class RedirectOutput:
    def __init__(self, panel, tag="output"):
        self.panel = panel
        self.tag = tag

    def write(self, text):
        if text:
            self.panel.after(0, self.panel.append, text, self.tag)

    def flush(self):
        pass


# ──────────────── Dry Run Setup Dialog ────────────────

class DryRunSetupDialog(tk.Toplevel):
    """
    A-Level exam-style dry run setup.
    Shows detected INPUT statements, lets user supply values upfront,
    and choose which variables to include in the trace table.
    """

    def __init__(self, parent, input_info, declare_info):
        super().__init__(parent)
        self.title("Dry Run Setup")
        self.result = None  # Will be {'inputs': [...], 'traced_vars': set or None}
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 520, 560
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        self.minsize(400, 400)

        self.input_info = input_info    # [{'line': int, 'variable': str}, ...]
        self.declare_info = declare_info  # [{'name': str, 'type': str, 'is_array': bool}, ...]
        self.input_entries = []         # tk.Entry widgets for each input
        self.var_checkboxes = {}        # {name: tk.BooleanVar}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _build_ui(self):
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # ── Title ──
        tk.Label(main, text="Dry Run Setup",
                 font=("Segoe UI", 14, "bold"), fg=COLORS["accent"],
                 bg=COLORS["bg"]).pack(anchor="w", pady=(0, 4))
        tk.Label(main, text="Supply input values and select variables to trace,\n"
                            "just like a Cambridge 9618 exam trace table.",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"], justify="left").pack(anchor="w", pady=(0, 12))

        # ── Input Values Section ──
        input_frame = tk.LabelFrame(main, text=" Input Values ",
                                     font=("Segoe UI", 10, "bold"),
                                     fg=COLORS["yellow"], bg=COLORS["bg"],
                                     bd=1, relief="solid",
                                     highlightbackground=COLORS["surface"])
        input_frame.pack(fill="x", pady=(0, 12))

        if self.input_info:
            tk.Label(input_frame,
                     text="Enter values in order (as given on an exam paper):",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

            for i, info in enumerate(self.input_info):
                row = tk.Frame(input_frame, bg=COLORS["bg"])
                row.pack(fill="x", padx=10, pady=2)

                label_text = f"Input #{i+1}  (Line {info['line']}: {info['variable']})"
                tk.Label(row, text=label_text, font=("Segoe UI", 9),
                         fg=COLORS["text"], bg=COLORS["bg"], width=35,
                         anchor="w").pack(side="left")

                entry = tk.Entry(row, font=("Cascadia Code", 11),
                                 bg=COLORS["surface"], fg=COLORS["text"],
                                 insertbackground=COLORS["cursor"],
                                 relief="flat", bd=0, width=20)
                entry.pack(side="left", padx=(4, 0), ipady=3)
                self.input_entries.append(entry)

            # Also allow a bulk entry box
            tk.Label(input_frame,
                     text="Or enter all values comma-separated:",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(8, 2))

            self.bulk_entry = tk.Entry(input_frame, font=("Cascadia Code", 11),
                                        bg=COLORS["surface"], fg=COLORS["text"],
                                        insertbackground=COLORS["cursor"],
                                        relief="flat", bd=0)
            self.bulk_entry.pack(fill="x", padx=10, pady=(0, 8), ipady=4)
            self.bulk_entry.insert(0, "e.g. 5, 3, 8")
            self.bulk_entry.bind("<FocusIn>", self._clear_placeholder)
        else:
            tk.Label(input_frame,
                     text="No INPUT statements detected in this program.",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(padx=10, pady=8)
            self.bulk_entry = None

        # ── Variables to Trace Section ──
        var_frame = tk.LabelFrame(main, text=" Trace Table Columns ",
                                   font=("Segoe UI", 10, "bold"),
                                   fg=COLORS["teal"], bg=COLORS["bg"],
                                   bd=1, relief="solid",
                                   highlightbackground=COLORS["surface"])
        var_frame.pack(fill="both", expand=True, pady=(0, 12))

        tk.Label(var_frame,
                 text="Select which variables to show in the trace table:",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

        # Scrollable checkbox area
        canvas_frame = tk.Frame(var_frame, bg=COLORS["bg"])
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        canvas = tk.Canvas(canvas_frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        cb
```


## License: unknown
https://github.com/cstroup/business-planning/blob/c48782917b7971b53d1174d4c05aa4463862e170/planning_app.py

```
Now I have full context. Let me build the complete solution. The approach:

1. **Scan the AST** to find all `InputStmt` nodes and extract variable names — no LLM needed
2. **Before dry-run starts**, show a setup dialog where the user supplies:
   - All input values upfront (comma-separated, like exam papers give them)
   - Which variables to trace (auto-detected from DECLARE statements, user can toggle)
3. **During execution**, feed inputs from the pre-supplied queue instead of prompting
4. **After execution**, show the trace table in exact A-Level format

Here are the three files to update:

**1. Append to the end of [interpreter.py](interpreter.py) — the `DryRunInterpreter` class:**

```python

# ──────────────── Dry-Run / Trace Mode ────────────────

class DryRunInterpreter(Interpreter):
    """
    Cambridge 9618-style dry-run interpreter.
    - Accepts pre-supplied input values (like exam papers provide)
    - Records a trace table of selected variables at each step
    - Feeds inputs from queue instead of prompting interactively
    """

    def __init__(self, symbol_table: SymbolTable, input_queue=None,
                 traced_vars=None, max_steps=5000):
        super().__init__(symbol_table)
        self.trace = []
        self.step_count = 0
        self.max_steps = max_steps
        self.input_queue = list(input_queue) if input_queue else []
        self.input_index = 0
        self.traced_vars = set(traced_vars) if traced_vars else None  # None = auto-detect all
        self.output_log = []  # record OUTPUT values too

    # ── AST scanning helpers (static, no execution needed) ──

    @staticmethod
    def scan_inputs(statements):
        """Walk the AST to find all INPUT statements and return info about them.
        Returns list of dicts: {'line': int, 'variable': str}
        """
        results = []
        DryRunInterpreter._walk_for_inputs(statements, results)
        return results

    @staticmethod
    def _walk_for_inputs(stmts, results):
        for stmt in stmts:
            if isinstance(stmt, InputStmt):
                var_name = "?"
                if isinstance(stmt.target, VariableExpr):
                    var_name = stmt.target.name
                elif isinstance(stmt.target, ArrayAccessExpr):
                    var_name = stmt.target.array + "[...]"
                elif isinstance(stmt.target, str):
                    var_name = stmt.target
                line = getattr(stmt, 'line', 0)
                results.append({'line': line, 'variable': var_name})
            # Recurse into compound statements
            if isinstance(stmt, IfStmt):
                DryRunInterpreter._walk_for_inputs(stmt.then_branch, results)
                if stmt.else_branch:
                    DryRunInterpreter._walk_for_inputs(stmt.else_branch, results)
            elif isinstance(stmt, WhileStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, RepeatStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, ForStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, CaseStmt):
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

    @staticmethod
    def scan_declares(statements):
        """Walk the AST to find all DECLARE statements.
        Returns list of dicts: {'name': str, 'type': str, 'is_array': bool}
        """
        results = []
        for stmt in statements:
            if isinstance(stmt, DeclareStmt):
                results.append({
                    'name': stmt.name,
                    'type': stmt.type_name,
                    'is_array': stmt.is_array
                })
            elif isinstance(stmt, ConstantDecl):
                results.append({
                    'name': stmt.name,
                    'type': 'CONSTANT',
                    'is_array': False
                })
        return results

    # ── Snapshot & trace recording ──

    def _snapshot_vars(self):
        """Capture values of traced variables only."""
        snapshot = {}
        for scope_level in range(self.symbol_table.scope_level + 1):
            scope = self.symbol_table.scopes.get(scope_level, {})
            for name, sym in scope.items():
                # If user specified which vars to trace, filter
                if self.traced_vars is not None and name not in self.traced_vars:
                    continue
                cell = sym.cell
                if cell.is_array:
                    if cell.array_elements:
                        arr = {}
                        for key, ec in sorted(cell.array_elements.items()):
                            idx = ",".join(str(k) for k in key)
                            arr[f"[{idx}]"] = ec.get()
                        snapshot[name] = arr
                    else:
                        snapshot[name] = ""
                elif cell.type == DataType.RECORD and isinstance(cell.value, dict):
                    rec = {}
                    for fn, fc in cell.value.items():
                        rec[fn] = fc.get() if isinstance(fc, Cell) else fc
                    snapshot[name] = rec
                elif isinstance(cell.value, PseudocodeObject):
                    attrs = {}
                    for an, ac in cell.value.attributes.items():
                        attrs[an] = ac.get() if isinstance(ac, Cell) else ac
                    snapshot[name] = f"<{cell.value.class_name}>{attrs}"
                else:
                    try:
                        snapshot[name] = cell.get()
                    except Exception:
                        snapshot[name] = cell.value
        return snapshot

    def _format_val(self, val):
        if isinstance(val, bool): return "TRUE" if val else "FALSE"
        if isinstance(val, str): return f'"{val}"'
        return str(val)

    def _describe_stmt(self, stmt):
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            if isinstance(t, str): return f"{t} \u2190 ..."
            if isinstance(t, ArrayAccessExpr): return f"{t.array}[...] \u2190 ..."
            if isinstance(t, MemberExpr): return f".{t.field} \u2190 ..."
            return "ASSIGN"
        if isinstance(stmt, DeclareStmt):
            arr = "ARRAY" if stmt.is_array else ""
            return f"DECLARE {stmt.name} : {arr} {stmt.type_name}".strip()
        if isinstance(stmt, ConstantDecl): return f"CONSTANT {stmt.name}"
        if isinstance(stmt, OutputStmt): return "OUTPUT"
        if isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr): return f"INPUT {stmt.target.name}"
            if isinstance(stmt.target, ArrayAccessExpr): return f"INPUT {stmt.target.array}[...]"
            return "INPUT"
        if isinstance(stmt, IfStmt): return "IF"
        if isinstance(stmt, WhileStmt): return "WHILE"
        if isinstance(stmt, RepeatStmt): return "REPEAT"
        if isinstance(stmt, ForStmt): return f"FOR {stmt.identifier}"
        if isinstance(stmt, CaseStmt): return "CASE OF"
        if isinstance(stmt, ProcedureCallStmt): return f"CALL {stmt.name}"
        if isinstance(stmt, ReturnStmt): return "RETURN"
        if isinstance(stmt, ProcedureDecl): return f"PROCEDURE {stmt.name}"
        if isinstance(stmt, FunctionDecl): return f"FUNCTION {stmt.name}"
        if isinstance(stmt, TypeDecl): return f"TYPE {stmt.name}"
        if isinstance(stmt, ClassDecl): return f"CLASS {stmt.name}"
        if isinstance(stmt, FileStmt): return f"{stmt.operation}FILE"
        return type(stmt).__name__

    def _record_step(self, stmt, note=""):
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise InterpreterError(
                f"Dry-run stopped after {self.max_steps} steps (possible infinite loop)")
        entry = {
            'step': self.step_count,
            'line': getattr(stmt, 'line', self.current_line),
            'statement': self._describe_stmt(stmt),
            'note': note,
            'variables': self._snapshot_vars(),
        }
        self.trace.append(entry)

    # ── Override execute to record trace ──

    def execute(self, stmt: Stmt):
        if hasattr(stmt, 'line') and stmt.line > 0:
            self.current_line = stmt.line

        # Definitions — store but don't trace body execution
        if isinstance(stmt, (ProcedureDecl, FunctionDecl, TypeDecl, ClassDecl)):
            self._record_step(stmt, "defined")
            return super().execute(stmt)

        if isinstance(stmt, Expr):
            self._record_step(stmt, "expr")
            return super().execute(stmt)

        # Execute
        result = super().execute(stmt)

        # Build note about what changed
        note = ""
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            try:
                if isinstance(t, str):
                    v = self.symbol_table.get_cell(t).get()
                    note = f"= {self._format_val(v)}"
                elif isinstance(t, ArrayAccessExpr):
                    indices = [self.evaluate(idx) for idx in t.indices]
                    v = self.symbol_table.array_access(t.array, indices).get()
                    idx_s = ",".join(str(i) for i in indices)
                    note = f"[{idx_s}] = {self._format_val(v)}"
            except Exception:
                pass
        elif isinstance(stmt, DeclareStmt):
            note = stmt.type_name
        elif isinstance(stmt, OutputStmt):
            # Record what was output
            vals = [self._format_output(self.evaluate(a)) for a in stmt.values]
            output_text = "".join(vals)
            self.output_log.append(output_text)
            note = f'OUT: {output_text}'
        elif isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr):
                try:
                    v = self.symbol_table.get_cell(stmt.target.name).get()
                    note = f"= {self._format_val(v)}"
                except Exception:
                    pass
        elif isinstance(stmt, IfStmt):
            note = "condition"
        elif isinstance(stmt, ForStmt):
            note = f"loop {stmt.identifier}"

        self._record_step(stmt, note)
        return result

    # ── Override INPUT to use pre-supplied queue ──

    def visit_InputStmt(self, stmt: InputStmt):
        """Use pre-supplied input values instead of prompting."""
        if self.input_index < len(self.input_queue):
            val_str = str(self.input_queue[self.input_index])
            self.input_index += 1
        else:
            raise InterpreterError(
                "Dry-run ran out of pre-supplied input values. "
                f"Needed input #{self.input_index + 1} but only "
                f"{len(self.input_queue)} values were provided.")

        # Determine target type for coercion
        target_type = None
        try:
            if isinstance(stmt.target, VariableExpr):
                target_type = self.symbol_table.get_cell(stmt.target.name).type
            elif isinstance(stmt.target, str):
                target_type = self.symbol_table.get_cell(stmt.target).type
            elif isinstance(stmt.target, ArrayAccessExpr):
                cell = self.symbol_table.get_cell(stmt.target.array)
                if cell.is_array and cell.array_bounds:
                    target_type = cell.array_bounds.element_type
        except Exception:
            pass

        if target_type is not None:
            val, val_type = self._coerce_input(val_str, target_type)
        else:
            val, val_type = self._auto_parse_input(val_str)

        if isinstance(stmt.target, VariableExpr):
            self.symbol_table.assign(stmt.target.name, val, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            self.symbol_table.array_assign(stmt.target.array, indices, val, val_type)
        elif isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, val, val_type)

    # ── Results ──

    def get_all_var_names(self):
        names = set()
        for entry in self.trace:
            names.update(entry['variables'].keys())
        names -= set(self.procedures.keys())
        names -= set(self.functions.keys())
        return sorted(names)

    def format_trace_text(self):
        if not self.trace:
            return "No trace data recorded."

        var_names = self.get_all_var_names()
        headers = ['Step', 'Line', 'Statement', 'Note'] + var_names
        rows = []
        for entry in self.trace:
            row = [str(entry['step']), str(entry['line']),
                   entry['statement'], entry['note']]
            for vn in var_names:
                val = entry['variables'].get(vn, '')
                if isinstance(val, dict):
                    val = str(val)
                elif val == '':
                    val = ''
                else:
                    val = self._format_val(val) if not isinstance(val, str) else val
                row.append(val)
            rows.append(row)

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, c in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(c)))
        col_widths = [min(w, 30) for w in col_widths]

        def pad(s, w):
            return str(s)[:w].ljust(w)

        sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
        hdr = '|' + '|'.join(f" {pad(h, w)} " for h, w in zip(headers, col_widths)) + '|'
        lines = [sep, hdr, sep]
        for row in rows:
            r = '|' + '|'.join(f" {pad(c, w)} " for c, w in zip(row, col_widths)) + '|'
            lines.append(r)
        lines.append(sep)
        return '\n'.join(lines)
```

**2. Replace the entire [ide.py](ide.py):**

```python
"""
9618 Pseudocode IDE - Modern Desktop Interface
Tkinter-based GUI with dark theme, syntax highlighting, line numbers, and integrated output.
Includes A-Level exam-style Dry Run mode with pre-supplied inputs and trace table.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import sys
import io
import os
import threading
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexer import Lexer, LexerError
from parser import Parser, ParserError
from interpreter import Interpreter, InterpreterError, DryRunInterpreter
from symbol_table import SymbolTable

# -- Theme Colors (Catppuccin Mocha) --
COLORS = {
    "bg":           "#1e1e2e",
    "bg_secondary": "#181825",
    "bg_tertiary":  "#11111b",
    "surface":      "#313244",
    "overlay":      "#45475a",
    "text":         "#cdd6f4",
    "subtext":      "#a6adc8",
    "blue":         "#89b4fa",
    "green":        "#a6e3a1",
    "red":          "#f38ba8",
    "yellow":       "#f9e2af",
    "mauve":        "#cba6f7",
    "peach":        "#fab387",
    "teal":         "#94e2d5",
    "pink":         "#f5c2e7",
    "lavender":     "#b4befe",
    "sky":          "#89dceb",
    "sapphire":     "#74c7ec",
    "line_num_fg":  "#585b70",
    "selection":    "#45475a",
    "cursor":       "#f5e0dc",
    "gutter":       "#282a3a",
    "output_bg":    "#11111b",
    "toolbar_bg":   "#181825",
    "status_bg":    "#181825",
    "tab_active":   "#1e1e2e",
    "accent":       "#89b4fa",
    "error":        "#f38ba8",
    "success":      "#a6e3a1",
    "warning":      "#f9e2af",
    "button_bg":    "#313244",
    "button_hover": "#45475a",
    "border":       "#313244",
}

# -- Keyword Lists for Highlighting --
KEYWORDS_CONTROL = {
    'IF', 'THEN', 'ELSE', 'ENDIF', 'CASE', 'OF', 'OTHERWISE', 'ENDCASE',
    'FOR', 'TO', 'STEP', 'NEXT', 'WHILE', 'DO', 'ENDWHILE',
    'REPEAT', 'UNTIL', 'RETURN',
}
KEYWORDS_DECL = {
    'DECLARE', 'CONSTANT', 'TYPE', 'ENDTYPE', 'ARRAY',
    'PROCEDURE', 'ENDPROCEDURE', 'FUNCTION', 'ENDFUNCTION',
    'RETURNS', 'BYREF', 'BYVAL', 'CALL',
}
KEYWORDS_TYPE = {'INTEGER', 'REAL', 'STRING', 'BOOLEAN', 'CHAR', 'DATE'}
KEYWORDS_IO = {
    'INPUT', 'OUTPUT', 'OPENFILE', 'READFILE', 'WRITEFILE', 'CLOSEFILE',
    'READ', 'WRITE', 'APPEND',
}
KEYWORDS_OP = {'AND', 'OR', 'NOT', 'DIV', 'MOD', 'TRUE', 'FALSE'}
BUILTINS = {
    'LENGTH', 'UCASE', 'LCASE', 'LEFT', 'RIGHT', 'MID',
    'INT', 'NUM_TO_STR', 'STR_TO_NUM', 'ASC', 'CHR', 'SQRT', 'RAND', 'EOF',
}


class LineNumbers(tk.Canvas):
    def __init__(self, parent, text_widget, **kwargs):
        super().__init__(parent, **kwargs)
        self.text_widget = text_widget
        self.font = None

    def redraw(self, *_args):
        self.delete("all")
        if self.text_widget is None:
            return
        i = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.create_text(
                self.winfo_width() - 8, y, anchor="ne", text=linenum,
                font=self.font, fill=COLORS["line_num_fg"],
            )
            i = self.text_widget.index(f"{i}+1line")
            if self.text_widget.compare(i, ">=", "end"):
                break


class CodeEditor(tk.Text):
    TAG_CONFIG = {
        "keyword_ctrl":  {"foreground": COLORS["mauve"],   "font_style": "bold"},
        "keyword_decl":  {"foreground": COLORS["blue"],    "font_style": "bold"},
        "keyword_type":  {"foreground": COLORS["yellow"]},
        "keyword_io":    {"foreground": COLORS["green"],   "font_style": "bold"},
        "keyword_op":    {"foreground": COLORS["peach"],   "font_style": "bold"},
        "builtin":       {"foreground": COLORS["sapphire"]},
        "string":        {"foreground": COLORS["green"]},
        "char":          {"foreground": COLORS["teal"]},
        "number":        {"foreground": COLORS["peach"]},
        "comment":       {"foreground": COLORS["overlay"], "font_style": "italic"},
        "operator":      {"foreground": COLORS["sky"]},
        "assign_arrow":  {"foreground": COLORS["red"]},
        "error_line":    {"background": "#3d2030"},
    }

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._setup_tags()
        self.bind("<<Modified>>", self._on_modify)
        self._highlight_job = None

    def _setup_tags(self):
        base_font = self.cget("font")
        for tag, cfg in self.TAG_CONFIG.items():
            opts = {"foreground": cfg.get("foreground", COLORS["text"])}
            if "background" in cfg:
                opts["background"] = cfg["background"]
            style = cfg.get("font_style", "")
            if style:
                fam = tkfont.Font(font=base_font).actual()["family"]
                sz = tkfont.Font(font=base_font).actual()["size"]
                weight = "bold" if "bold" in style else "normal"
                slant = "italic" if "italic" in style else "roman"
                opts["font"] = tkfont.Font(family=fam, size=sz, weight=weight, slant=slant)
            self.tag_configure(tag, **opts)
        self.tag_raise("error_line")

    def _on_modify(self, _event=None):
        if self.edit_modified():
            if self._highlight_job:
                self.after_cancel(self._highlight_job)
            self._highlight_job = self.after(80, self.highlight_syntax)
            self.edit_modified(False)
            self.event_generate("<<ContentChanged>>")

    def highlight_syntax(self):
        for tag in self.TAG_CONFIG:
            self.tag_remove(tag, "1.0", "end")
        code = self.get("1.0", "end-1c")
        patterns = [
            ("comment",      r'//[^\n]*'),
            ("string",       r'"[^"]*"'),
            ("char",         r"'[^']*'"),
            ("assign_arrow", r'<-'),
            ("number",       r'\b\d+\.\d+\b|\b\d+\b'),
            ("operator",     r'<>|<=|>=|[<>=+\-*/&]'),
        ]
        for tag, pat in patterns:
            for m in re.finditer(pat, code):
                start = f"1.0+{m.start()}c"
                end = f"1.0+{m.end()}c"
                self.tag_add(tag, start, end)
        word_groups = [
            ("keyword_ctrl", KEYWORDS_CONTROL),
            ("keyword_decl", KEYWORDS_DECL),
            ("keyword_type", KEYWORDS_TYPE),
            ("keyword_io",   KEYWORDS_IO),
            ("keyword_op",   KEYWORDS_OP),
            ("builtin",      BUILTINS),
        ]
        for tag, words in word_groups:
            for w in words:
                pat = rf'\b{w}\b'
                for m in re.finditer(pat, code):
                    start = f"1.0+{m.start()}c"
                    end = f"1.0+{m.end()}c"
                    tags_at = self.tag_names(start)
                    if "string" not in tags_at and "comment" not in tags_at and "char" not in tags_at:
                        self.tag_add(tag, start, end)

    def mark_error_line(self, line_num):
        self.tag_remove("error_line", "1.0", "end")
        if line_num and line_num > 0:
            self.tag_add("error_line", f"{line_num}.0", f"{line_num}.end+1c")
            self.see(f"{line_num}.0")


class OutputPanel(tk.Text):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(state="disabled")
        self.tag_configure("error",        foreground=COLORS["error"])
        self.tag_configure("success",      foreground=COLORS["success"])
        self.tag_configure("info",         foreground=COLORS["subtext"])
        self.tag_configure("output",       foreground=COLORS["text"])
        self.tag_configure("input_prompt", foreground=COLORS["yellow"])

    def append(self, text, tag="output"):
        self.configure(state="normal")
        self.insert("end", text, tag)
        self.see("end")
        self.configure(state="disabled")

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class InputDialog(tk.Toplevel):
    def __init__(self, parent, prompt):
        super().__init__(parent)
        self.title("Input Required")
        self.result = None
        self.configure(bg=COLORS["bg"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 400, 160
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        frame = tk.Frame(self, bg=COLORS["bg"], padx=20, pady=15)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text=prompt, font=("Segoe UI", 11),
                 fg=COLORS["yellow"], bg=COLORS["bg"], anchor="w").pack(fill="x", pady=(0, 8))

        self.entry = tk.Entry(frame, font=("Cascadia Code", 12),
                              bg=COLORS["surface"], fg=COLORS["text"],
                              insertbackground=COLORS["cursor"], relief="flat", bd=0)
        self.entry.pack(fill="x", ipady=6)
        self.entry.focus_set()
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Escape>", lambda e: self._cancel())

        btn_frame = tk.Frame(frame, bg=COLORS["bg"])
        btn_frame.pack(fill="x", pady=(12, 0))

        tk.Button(btn_frame, text="OK", font=("Segoe UI", 10),
                  bg=COLORS["accent"], fg=COLORS["bg_tertiary"],
                  activebackground=COLORS["lavender"],
                  relief="flat", bd=0, padx=20, pady=4,
                  command=self._submit).pack(side="right")
        tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 10),
                  bg=COLORS["surface"], fg=COLORS["text"],
                  activebackground=COLORS["overlay"],
                  relief="flat", bd=0, padx=16, pady=4,
                  command=self._cancel).pack(side="right", padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _submit(self, _event=None):
        self.result = self.entry.get()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class RedirectOutput:
    def __init__(self, panel, tag="output"):
        self.panel = panel
        self.tag = tag

    def write(self, text):
        if text:
            self.panel.after(0, self.panel.append, text, self.tag)

    def flush(self):
        pass


# ──────────────── Dry Run Setup Dialog ────────────────

class DryRunSetupDialog(tk.Toplevel):
    """
    A-Level exam-style dry run setup.
    Shows detected INPUT statements, lets user supply values upfront,
    and choose which variables to include in the trace table.
    """

    def __init__(self, parent, input_info, declare_info):
        super().__init__(parent)
        self.title("Dry Run Setup")
        self.result = None  # Will be {'inputs': [...], 'traced_vars': set or None}
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 520, 560
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        self.minsize(400, 400)

        self.input_info = input_info    # [{'line': int, 'variable': str}, ...]
        self.declare_info = declare_info  # [{'name': str, 'type': str, 'is_array': bool}, ...]
        self.input_entries = []         # tk.Entry widgets for each input
        self.var_checkboxes = {}        # {name: tk.BooleanVar}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _build_ui(self):
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # ── Title ──
        tk.Label(main, text="Dry Run Setup",
                 font=("Segoe UI", 14, "bold"), fg=COLORS["accent"],
                 bg=COLORS["bg"]).pack(anchor="w", pady=(0, 4))
        tk.Label(main, text="Supply input values and select variables to trace,\n"
                            "just like a Cambridge 9618 exam trace table.",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"], justify="left").pack(anchor="w", pady=(0, 12))

        # ── Input Values Section ──
        input_frame = tk.LabelFrame(main, text=" Input Values ",
                                     font=("Segoe UI", 10, "bold"),
                                     fg=COLORS["yellow"], bg=COLORS["bg"],
                                     bd=1, relief="solid",
                                     highlightbackground=COLORS["surface"])
        input_frame.pack(fill="x", pady=(0, 12))

        if self.input_info:
            tk.Label(input_frame,
                     text="Enter values in order (as given on an exam paper):",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

            for i, info in enumerate(self.input_info):
                row = tk.Frame(input_frame, bg=COLORS["bg"])
                row.pack(fill="x", padx=10, pady=2)

                label_text = f"Input #{i+1}  (Line {info['line']}: {info['variable']})"
                tk.Label(row, text=label_text, font=("Segoe UI", 9),
                         fg=COLORS["text"], bg=COLORS["bg"], width=35,
                         anchor="w").pack(side="left")

                entry = tk.Entry(row, font=("Cascadia Code", 11),
                                 bg=COLORS["surface"], fg=COLORS["text"],
                                 insertbackground=COLORS["cursor"],
                                 relief="flat", bd=0, width=20)
                entry.pack(side="left", padx=(4, 0), ipady=3)
                self.input_entries.append(entry)

            # Also allow a bulk entry box
            tk.Label(input_frame,
                     text="Or enter all values comma-separated:",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(8, 2))

            self.bulk_entry = tk.Entry(input_frame, font=("Cascadia Code", 11),
                                        bg=COLORS["surface"], fg=COLORS["text"],
                                        insertbackground=COLORS["cursor"],
                                        relief="flat", bd=0)
            self.bulk_entry.pack(fill="x", padx=10, pady=(0, 8), ipady=4)
            self.bulk_entry.insert(0, "e.g. 5, 3, 8")
            self.bulk_entry.bind("<FocusIn>", self._clear_placeholder)
        else:
            tk.Label(input_frame,
                     text="No INPUT statements detected in this program.",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(padx=10, pady=8)
            self.bulk_entry = None

        # ── Variables to Trace Section ──
        var_frame = tk.LabelFrame(main, text=" Trace Table Columns ",
                                   font=("Segoe UI", 10, "bold"),
                                   fg=COLORS["teal"], bg=COLORS["bg"],
                                   bd=1, relief="solid",
                                   highlightbackground=COLORS["surface"])
        var_frame.pack(fill="both", expand=True, pady=(0, 12))

        tk.Label(var_frame,
                 text="Select which variables to show in the trace table:",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

        # Scrollable checkbox area
        canvas_frame = tk.Frame(var_frame, bg=COLORS["bg"])
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        canvas = tk.Canvas(canvas_frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        cb
```


## License: unknown
https://github.com/cstroup/business-planning/blob/c48782917b7971b53d1174d4c05aa4463862e170/planning_app.py

```
Now I have full context. Let me build the complete solution. The approach:

1. **Scan the AST** to find all `InputStmt` nodes and extract variable names — no LLM needed
2. **Before dry-run starts**, show a setup dialog where the user supplies:
   - All input values upfront (comma-separated, like exam papers give them)
   - Which variables to trace (auto-detected from DECLARE statements, user can toggle)
3. **During execution**, feed inputs from the pre-supplied queue instead of prompting
4. **After execution**, show the trace table in exact A-Level format

Here are the three files to update:

**1. Append to the end of [interpreter.py](interpreter.py) — the `DryRunInterpreter` class:**

```python

# ──────────────── Dry-Run / Trace Mode ────────────────

class DryRunInterpreter(Interpreter):
    """
    Cambridge 9618-style dry-run interpreter.
    - Accepts pre-supplied input values (like exam papers provide)
    - Records a trace table of selected variables at each step
    - Feeds inputs from queue instead of prompting interactively
    """

    def __init__(self, symbol_table: SymbolTable, input_queue=None,
                 traced_vars=None, max_steps=5000):
        super().__init__(symbol_table)
        self.trace = []
        self.step_count = 0
        self.max_steps = max_steps
        self.input_queue = list(input_queue) if input_queue else []
        self.input_index = 0
        self.traced_vars = set(traced_vars) if traced_vars else None  # None = auto-detect all
        self.output_log = []  # record OUTPUT values too

    # ── AST scanning helpers (static, no execution needed) ──

    @staticmethod
    def scan_inputs(statements):
        """Walk the AST to find all INPUT statements and return info about them.
        Returns list of dicts: {'line': int, 'variable': str}
        """
        results = []
        DryRunInterpreter._walk_for_inputs(statements, results)
        return results

    @staticmethod
    def _walk_for_inputs(stmts, results):
        for stmt in stmts:
            if isinstance(stmt, InputStmt):
                var_name = "?"
                if isinstance(stmt.target, VariableExpr):
                    var_name = stmt.target.name
                elif isinstance(stmt.target, ArrayAccessExpr):
                    var_name = stmt.target.array + "[...]"
                elif isinstance(stmt.target, str):
                    var_name = stmt.target
                line = getattr(stmt, 'line', 0)
                results.append({'line': line, 'variable': var_name})
            # Recurse into compound statements
            if isinstance(stmt, IfStmt):
                DryRunInterpreter._walk_for_inputs(stmt.then_branch, results)
                if stmt.else_branch:
                    DryRunInterpreter._walk_for_inputs(stmt.else_branch, results)
            elif isinstance(stmt, WhileStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, RepeatStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, ForStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, CaseStmt):
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

    @staticmethod
    def scan_declares(statements):
        """Walk the AST to find all DECLARE statements.
        Returns list of dicts: {'name': str, 'type': str, 'is_array': bool}
        """
        results = []
        for stmt in statements:
            if isinstance(stmt, DeclareStmt):
                results.append({
                    'name': stmt.name,
                    'type': stmt.type_name,
                    'is_array': stmt.is_array
                })
            elif isinstance(stmt, ConstantDecl):
                results.append({
                    'name': stmt.name,
                    'type': 'CONSTANT',
                    'is_array': False
                })
        return results

    # ── Snapshot & trace recording ──

    def _snapshot_vars(self):
        """Capture values of traced variables only."""
        snapshot = {}
        for scope_level in range(self.symbol_table.scope_level + 1):
            scope = self.symbol_table.scopes.get(scope_level, {})
            for name, sym in scope.items():
                # If user specified which vars to trace, filter
                if self.traced_vars is not None and name not in self.traced_vars:
                    continue
                cell = sym.cell
                if cell.is_array:
                    if cell.array_elements:
                        arr = {}
                        for key, ec in sorted(cell.array_elements.items()):
                            idx = ",".join(str(k) for k in key)
                            arr[f"[{idx}]"] = ec.get()
                        snapshot[name] = arr
                    else:
                        snapshot[name] = ""
                elif cell.type == DataType.RECORD and isinstance(cell.value, dict):
                    rec = {}
                    for fn, fc in cell.value.items():
                        rec[fn] = fc.get() if isinstance(fc, Cell) else fc
                    snapshot[name] = rec
                elif isinstance(cell.value, PseudocodeObject):
                    attrs = {}
                    for an, ac in cell.value.attributes.items():
                        attrs[an] = ac.get() if isinstance(ac, Cell) else ac
                    snapshot[name] = f"<{cell.value.class_name}>{attrs}"
                else:
                    try:
                        snapshot[name] = cell.get()
                    except Exception:
                        snapshot[name] = cell.value
        return snapshot

    def _format_val(self, val):
        if isinstance(val, bool): return "TRUE" if val else "FALSE"
        if isinstance(val, str): return f'"{val}"'
        return str(val)

    def _describe_stmt(self, stmt):
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            if isinstance(t, str): return f"{t} \u2190 ..."
            if isinstance(t, ArrayAccessExpr): return f"{t.array}[...] \u2190 ..."
            if isinstance(t, MemberExpr): return f".{t.field} \u2190 ..."
            return "ASSIGN"
        if isinstance(stmt, DeclareStmt):
            arr = "ARRAY" if stmt.is_array else ""
            return f"DECLARE {stmt.name} : {arr} {stmt.type_name}".strip()
        if isinstance(stmt, ConstantDecl): return f"CONSTANT {stmt.name}"
        if isinstance(stmt, OutputStmt): return "OUTPUT"
        if isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr): return f"INPUT {stmt.target.name}"
            if isinstance(stmt.target, ArrayAccessExpr): return f"INPUT {stmt.target.array}[...]"
            return "INPUT"
        if isinstance(stmt, IfStmt): return "IF"
        if isinstance(stmt, WhileStmt): return "WHILE"
        if isinstance(stmt, RepeatStmt): return "REPEAT"
        if isinstance(stmt, ForStmt): return f"FOR {stmt.identifier}"
        if isinstance(stmt, CaseStmt): return "CASE OF"
        if isinstance(stmt, ProcedureCallStmt): return f"CALL {stmt.name}"
        if isinstance(stmt, ReturnStmt): return "RETURN"
        if isinstance(stmt, ProcedureDecl): return f"PROCEDURE {stmt.name}"
        if isinstance(stmt, FunctionDecl): return f"FUNCTION {stmt.name}"
        if isinstance(stmt, TypeDecl): return f"TYPE {stmt.name}"
        if isinstance(stmt, ClassDecl): return f"CLASS {stmt.name}"
        if isinstance(stmt, FileStmt): return f"{stmt.operation}FILE"
        return type(stmt).__name__

    def _record_step(self, stmt, note=""):
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise InterpreterError(
                f"Dry-run stopped after {self.max_steps} steps (possible infinite loop)")
        entry = {
            'step': self.step_count,
            'line': getattr(stmt, 'line', self.current_line),
            'statement': self._describe_stmt(stmt),
            'note': note,
            'variables': self._snapshot_vars(),
        }
        self.trace.append(entry)

    # ── Override execute to record trace ──

    def execute(self, stmt: Stmt):
        if hasattr(stmt, 'line') and stmt.line > 0:
            self.current_line = stmt.line

        # Definitions — store but don't trace body execution
        if isinstance(stmt, (ProcedureDecl, FunctionDecl, TypeDecl, ClassDecl)):
            self._record_step(stmt, "defined")
            return super().execute(stmt)

        if isinstance(stmt, Expr):
            self._record_step(stmt, "expr")
            return super().execute(stmt)

        # Execute
        result = super().execute(stmt)

        # Build note about what changed
        note = ""
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            try:
                if isinstance(t, str):
                    v = self.symbol_table.get_cell(t).get()
                    note = f"= {self._format_val(v)}"
                elif isinstance(t, ArrayAccessExpr):
                    indices = [self.evaluate(idx) for idx in t.indices]
                    v = self.symbol_table.array_access(t.array, indices).get()
                    idx_s = ",".join(str(i) for i in indices)
                    note = f"[{idx_s}] = {self._format_val(v)}"
            except Exception:
                pass
        elif isinstance(stmt, DeclareStmt):
            note = stmt.type_name
        elif isinstance(stmt, OutputStmt):
            # Record what was output
            vals = [self._format_output(self.evaluate(a)) for a in stmt.values]
            output_text = "".join(vals)
            self.output_log.append(output_text)
            note = f'OUT: {output_text}'
        elif isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr):
                try:
                    v = self.symbol_table.get_cell(stmt.target.name).get()
                    note = f"= {self._format_val(v)}"
                except Exception:
                    pass
        elif isinstance(stmt, IfStmt):
            note = "condition"
        elif isinstance(stmt, ForStmt):
            note = f"loop {stmt.identifier}"

        self._record_step(stmt, note)
        return result

    # ── Override INPUT to use pre-supplied queue ──

    def visit_InputStmt(self, stmt: InputStmt):
        """Use pre-supplied input values instead of prompting."""
        if self.input_index < len(self.input_queue):
            val_str = str(self.input_queue[self.input_index])
            self.input_index += 1
        else:
            raise InterpreterError(
                "Dry-run ran out of pre-supplied input values. "
                f"Needed input #{self.input_index + 1} but only "
                f"{len(self.input_queue)} values were provided.")

        # Determine target type for coercion
        target_type = None
        try:
            if isinstance(stmt.target, VariableExpr):
                target_type = self.symbol_table.get_cell(stmt.target.name).type
            elif isinstance(stmt.target, str):
                target_type = self.symbol_table.get_cell(stmt.target).type
            elif isinstance(stmt.target, ArrayAccessExpr):
                cell = self.symbol_table.get_cell(stmt.target.array)
                if cell.is_array and cell.array_bounds:
                    target_type = cell.array_bounds.element_type
        except Exception:
            pass

        if target_type is not None:
            val, val_type = self._coerce_input(val_str, target_type)
        else:
            val, val_type = self._auto_parse_input(val_str)

        if isinstance(stmt.target, VariableExpr):
            self.symbol_table.assign(stmt.target.name, val, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            self.symbol_table.array_assign(stmt.target.array, indices, val, val_type)
        elif isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, val, val_type)

    # ── Results ──

    def get_all_var_names(self):
        names = set()
        for entry in self.trace:
            names.update(entry['variables'].keys())
        names -= set(self.procedures.keys())
        names -= set(self.functions.keys())
        return sorted(names)

    def format_trace_text(self):
        if not self.trace:
            return "No trace data recorded."

        var_names = self.get_all_var_names()
        headers = ['Step', 'Line', 'Statement', 'Note'] + var_names
        rows = []
        for entry in self.trace:
            row = [str(entry['step']), str(entry['line']),
                   entry['statement'], entry['note']]
            for vn in var_names:
                val = entry['variables'].get(vn, '')
                if isinstance(val, dict):
                    val = str(val)
                elif val == '':
                    val = ''
                else:
                    val = self._format_val(val) if not isinstance(val, str) else val
                row.append(val)
            rows.append(row)

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, c in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(c)))
        col_widths = [min(w, 30) for w in col_widths]

        def pad(s, w):
            return str(s)[:w].ljust(w)

        sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
        hdr = '|' + '|'.join(f" {pad(h, w)} " for h, w in zip(headers, col_widths)) + '|'
        lines = [sep, hdr, sep]
        for row in rows:
            r = '|' + '|'.join(f" {pad(c, w)} " for c, w in zip(row, col_widths)) + '|'
            lines.append(r)
        lines.append(sep)
        return '\n'.join(lines)
```

**2. Replace the entire [ide.py](ide.py):**

```python
"""
9618 Pseudocode IDE - Modern Desktop Interface
Tkinter-based GUI with dark theme, syntax highlighting, line numbers, and integrated output.
Includes A-Level exam-style Dry Run mode with pre-supplied inputs and trace table.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import sys
import io
import os
import threading
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexer import Lexer, LexerError
from parser import Parser, ParserError
from interpreter import Interpreter, InterpreterError, DryRunInterpreter
from symbol_table import SymbolTable

# -- Theme Colors (Catppuccin Mocha) --
COLORS = {
    "bg":           "#1e1e2e",
    "bg_secondary": "#181825",
    "bg_tertiary":  "#11111b",
    "surface":      "#313244",
    "overlay":      "#45475a",
    "text":         "#cdd6f4",
    "subtext":      "#a6adc8",
    "blue":         "#89b4fa",
    "green":        "#a6e3a1",
    "red":          "#f38ba8",
    "yellow":       "#f9e2af",
    "mauve":        "#cba6f7",
    "peach":        "#fab387",
    "teal":         "#94e2d5",
    "pink":         "#f5c2e7",
    "lavender":     "#b4befe",
    "sky":          "#89dceb",
    "sapphire":     "#74c7ec",
    "line_num_fg":  "#585b70",
    "selection":    "#45475a",
    "cursor":       "#f5e0dc",
    "gutter":       "#282a3a",
    "output_bg":    "#11111b",
    "toolbar_bg":   "#181825",
    "status_bg":    "#181825",
    "tab_active":   "#1e1e2e",
    "accent":       "#89b4fa",
    "error":        "#f38ba8",
    "success":      "#a6e3a1",
    "warning":      "#f9e2af",
    "button_bg":    "#313244",
    "button_hover": "#45475a",
    "border":       "#313244",
}

# -- Keyword Lists for Highlighting --
KEYWORDS_CONTROL = {
    'IF', 'THEN', 'ELSE', 'ENDIF', 'CASE', 'OF', 'OTHERWISE', 'ENDCASE',
    'FOR', 'TO', 'STEP', 'NEXT', 'WHILE', 'DO', 'ENDWHILE',
    'REPEAT', 'UNTIL', 'RETURN',
}
KEYWORDS_DECL = {
    'DECLARE', 'CONSTANT', 'TYPE', 'ENDTYPE', 'ARRAY',
    'PROCEDURE', 'ENDPROCEDURE', 'FUNCTION', 'ENDFUNCTION',
    'RETURNS', 'BYREF', 'BYVAL', 'CALL',
}
KEYWORDS_TYPE = {'INTEGER', 'REAL', 'STRING', 'BOOLEAN', 'CHAR', 'DATE'}
KEYWORDS_IO = {
    'INPUT', 'OUTPUT', 'OPENFILE', 'READFILE', 'WRITEFILE', 'CLOSEFILE',
    'READ', 'WRITE', 'APPEND',
}
KEYWORDS_OP = {'AND', 'OR', 'NOT', 'DIV', 'MOD', 'TRUE', 'FALSE'}
BUILTINS = {
    'LENGTH', 'UCASE', 'LCASE', 'LEFT', 'RIGHT', 'MID',
    'INT', 'NUM_TO_STR', 'STR_TO_NUM', 'ASC', 'CHR', 'SQRT', 'RAND', 'EOF',
}


class LineNumbers(tk.Canvas):
    def __init__(self, parent, text_widget, **kwargs):
        super().__init__(parent, **kwargs)
        self.text_widget = text_widget
        self.font = None

    def redraw(self, *_args):
        self.delete("all")
        if self.text_widget is None:
            return
        i = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.create_text(
                self.winfo_width() - 8, y, anchor="ne", text=linenum,
                font=self.font, fill=COLORS["line_num_fg"],
            )
            i = self.text_widget.index(f"{i}+1line")
            if self.text_widget.compare(i, ">=", "end"):
                break


class CodeEditor(tk.Text):
    TAG_CONFIG = {
        "keyword_ctrl":  {"foreground": COLORS["mauve"],   "font_style": "bold"},
        "keyword_decl":  {"foreground": COLORS["blue"],    "font_style": "bold"},
        "keyword_type":  {"foreground": COLORS["yellow"]},
        "keyword_io":    {"foreground": COLORS["green"],   "font_style": "bold"},
        "keyword_op":    {"foreground": COLORS["peach"],   "font_style": "bold"},
        "builtin":       {"foreground": COLORS["sapphire"]},
        "string":        {"foreground": COLORS["green"]},
        "char":          {"foreground": COLORS["teal"]},
        "number":        {"foreground": COLORS["peach"]},
        "comment":       {"foreground": COLORS["overlay"], "font_style": "italic"},
        "operator":      {"foreground": COLORS["sky"]},
        "assign_arrow":  {"foreground": COLORS["red"]},
        "error_line":    {"background": "#3d2030"},
    }

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._setup_tags()
        self.bind("<<Modified>>", self._on_modify)
        self._highlight_job = None

    def _setup_tags(self):
        base_font = self.cget("font")
        for tag, cfg in self.TAG_CONFIG.items():
            opts = {"foreground": cfg.get("foreground", COLORS["text"])}
            if "background" in cfg:
                opts["background"] = cfg["background"]
            style = cfg.get("font_style", "")
            if style:
                fam = tkfont.Font(font=base_font).actual()["family"]
                sz = tkfont.Font(font=base_font).actual()["size"]
                weight = "bold" if "bold" in style else "normal"
                slant = "italic" if "italic" in style else "roman"
                opts["font"] = tkfont.Font(family=fam, size=sz, weight=weight, slant=slant)
            self.tag_configure(tag, **opts)
        self.tag_raise("error_line")

    def _on_modify(self, _event=None):
        if self.edit_modified():
            if self._highlight_job:
                self.after_cancel(self._highlight_job)
            self._highlight_job = self.after(80, self.highlight_syntax)
            self.edit_modified(False)
            self.event_generate("<<ContentChanged>>")

    def highlight_syntax(self):
        for tag in self.TAG_CONFIG:
            self.tag_remove(tag, "1.0", "end")
        code = self.get("1.0", "end-1c")
        patterns = [
            ("comment",      r'//[^\n]*'),
            ("string",       r'"[^"]*"'),
            ("char",         r"'[^']*'"),
            ("assign_arrow", r'<-'),
            ("number",       r'\b\d+\.\d+\b|\b\d+\b'),
            ("operator",     r'<>|<=|>=|[<>=+\-*/&]'),
        ]
        for tag, pat in patterns:
            for m in re.finditer(pat, code):
                start = f"1.0+{m.start()}c"
                end = f"1.0+{m.end()}c"
                self.tag_add(tag, start, end)
        word_groups = [
            ("keyword_ctrl", KEYWORDS_CONTROL),
            ("keyword_decl", KEYWORDS_DECL),
            ("keyword_type", KEYWORDS_TYPE),
            ("keyword_io",   KEYWORDS_IO),
            ("keyword_op",   KEYWORDS_OP),
            ("builtin",      BUILTINS),
        ]
        for tag, words in word_groups:
            for w in words:
                pat = rf'\b{w}\b'
                for m in re.finditer(pat, code):
                    start = f"1.0+{m.start()}c"
                    end = f"1.0+{m.end()}c"
                    tags_at = self.tag_names(start)
                    if "string" not in tags_at and "comment" not in tags_at and "char" not in tags_at:
                        self.tag_add(tag, start, end)

    def mark_error_line(self, line_num):
        self.tag_remove("error_line", "1.0", "end")
        if line_num and line_num > 0:
            self.tag_add("error_line", f"{line_num}.0", f"{line_num}.end+1c")
            self.see(f"{line_num}.0")


class OutputPanel(tk.Text):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(state="disabled")
        self.tag_configure("error",        foreground=COLORS["error"])
        self.tag_configure("success",      foreground=COLORS["success"])
        self.tag_configure("info",         foreground=COLORS["subtext"])
        self.tag_configure("output",       foreground=COLORS["text"])
        self.tag_configure("input_prompt", foreground=COLORS["yellow"])

    def append(self, text, tag="output"):
        self.configure(state="normal")
        self.insert("end", text, tag)
        self.see("end")
        self.configure(state="disabled")

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class InputDialog(tk.Toplevel):
    def __init__(self, parent, prompt):
        super().__init__(parent)
        self.title("Input Required")
        self.result = None
        self.configure(bg=COLORS["bg"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 400, 160
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        frame = tk.Frame(self, bg=COLORS["bg"], padx=20, pady=15)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text=prompt, font=("Segoe UI", 11),
                 fg=COLORS["yellow"], bg=COLORS["bg"], anchor="w").pack(fill="x", pady=(0, 8))

        self.entry = tk.Entry(frame, font=("Cascadia Code", 12),
                              bg=COLORS["surface"], fg=COLORS["text"],
                              insertbackground=COLORS["cursor"], relief="flat", bd=0)
        self.entry.pack(fill="x", ipady=6)
        self.entry.focus_set()
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Escape>", lambda e: self._cancel())

        btn_frame = tk.Frame(frame, bg=COLORS["bg"])
        btn_frame.pack(fill="x", pady=(12, 0))

        tk.Button(btn_frame, text="OK", font=("Segoe UI", 10),
                  bg=COLORS["accent"], fg=COLORS["bg_tertiary"],
                  activebackground=COLORS["lavender"],
                  relief="flat", bd=0, padx=20, pady=4,
                  command=self._submit).pack(side="right")
        tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 10),
                  bg=COLORS["surface"], fg=COLORS["text"],
                  activebackground=COLORS["overlay"],
                  relief="flat", bd=0, padx=16, pady=4,
                  command=self._cancel).pack(side="right", padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _submit(self, _event=None):
        self.result = self.entry.get()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class RedirectOutput:
    def __init__(self, panel, tag="output"):
        self.panel = panel
        self.tag = tag

    def write(self, text):
        if text:
            self.panel.after(0, self.panel.append, text, self.tag)

    def flush(self):
        pass


# ──────────────── Dry Run Setup Dialog ────────────────

class DryRunSetupDialog(tk.Toplevel):
    """
    A-Level exam-style dry run setup.
    Shows detected INPUT statements, lets user supply values upfront,
    and choose which variables to include in the trace table.
    """

    def __init__(self, parent, input_info, declare_info):
        super().__init__(parent)
        self.title("Dry Run Setup")
        self.result = None  # Will be {'inputs': [...], 'traced_vars': set or None}
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 520, 560
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        self.minsize(400, 400)

        self.input_info = input_info    # [{'line': int, 'variable': str}, ...]
        self.declare_info = declare_info  # [{'name': str, 'type': str, 'is_array': bool}, ...]
        self.input_entries = []         # tk.Entry widgets for each input
        self.var_checkboxes = {}        # {name: tk.BooleanVar}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _build_ui(self):
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # ── Title ──
        tk.Label(main, text="Dry Run Setup",
                 font=("Segoe UI", 14, "bold"), fg=COLORS["accent"],
                 bg=COLORS["bg"]).pack(anchor="w", pady=(0, 4))
        tk.Label(main, text="Supply input values and select variables to trace,\n"
                            "just like a Cambridge 9618 exam trace table.",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"], justify="left").pack(anchor="w", pady=(0, 12))

        # ── Input Values Section ──
        input_frame = tk.LabelFrame(main, text=" Input Values ",
                                     font=("Segoe UI", 10, "bold"),
                                     fg=COLORS["yellow"], bg=COLORS["bg"],
                                     bd=1, relief="solid",
                                     highlightbackground=COLORS["surface"])
        input_frame.pack(fill="x", pady=(0, 12))

        if self.input_info:
            tk.Label(input_frame,
                     text="Enter values in order (as given on an exam paper):",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

            for i, info in enumerate(self.input_info):
                row = tk.Frame(input_frame, bg=COLORS["bg"])
                row.pack(fill="x", padx=10, pady=2)

                label_text = f"Input #{i+1}  (Line {info['line']}: {info['variable']})"
                tk.Label(row, text=label_text, font=("Segoe UI", 9),
                         fg=COLORS["text"], bg=COLORS["bg"], width=35,
                         anchor="w").pack(side="left")

                entry = tk.Entry(row, font=("Cascadia Code", 11),
                                 bg=COLORS["surface"], fg=COLORS["text"],
                                 insertbackground=COLORS["cursor"],
                                 relief="flat", bd=0, width=20)
                entry.pack(side="left", padx=(4, 0), ipady=3)
                self.input_entries.append(entry)

            # Also allow a bulk entry box
            tk.Label(input_frame,
                     text="Or enter all values comma-separated:",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(8, 2))

            self.bulk_entry = tk.Entry(input_frame, font=("Cascadia Code", 11),
                                        bg=COLORS["surface"], fg=COLORS["text"],
                                        insertbackground=COLORS["cursor"],
                                        relief="flat", bd=0)
            self.bulk_entry.pack(fill="x", padx=10, pady=(0, 8), ipady=4)
            self.bulk_entry.insert(0, "e.g. 5, 3, 8")
            self.bulk_entry.bind("<FocusIn>", self._clear_placeholder)
        else:
            tk.Label(input_frame,
                     text="No INPUT statements detected in this program.",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(padx=10, pady=8)
            self.bulk_entry = None

        # ── Variables to Trace Section ──
        var_frame = tk.LabelFrame(main, text=" Trace Table Columns ",
                                   font=("Segoe UI", 10, "bold"),
                                   fg=COLORS["teal"], bg=COLORS["bg"],
                                   bd=1, relief="solid",
                                   highlightbackground=COLORS["surface"])
        var_frame.pack(fill="both", expand=True, pady=(0, 12))

        tk.Label(var_frame,
                 text="Select which variables to show in the trace table:",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

        # Scrollable checkbox area
        canvas_frame = tk.Frame(var_frame, bg=COLORS["bg"])
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        canvas = tk.Canvas(canvas_frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        cb
```


## License: unknown
https://github.com/cstroup/business-planning/blob/c48782917b7971b53d1174d4c05aa4463862e170/planning_app.py

```
Now I have full context. Let me build the complete solution. The approach:

1. **Scan the AST** to find all `InputStmt` nodes and extract variable names — no LLM needed
2. **Before dry-run starts**, show a setup dialog where the user supplies:
   - All input values upfront (comma-separated, like exam papers give them)
   - Which variables to trace (auto-detected from DECLARE statements, user can toggle)
3. **During execution**, feed inputs from the pre-supplied queue instead of prompting
4. **After execution**, show the trace table in exact A-Level format

Here are the three files to update:

**1. Append to the end of [interpreter.py](interpreter.py) — the `DryRunInterpreter` class:**

```python

# ──────────────── Dry-Run / Trace Mode ────────────────

class DryRunInterpreter(Interpreter):
    """
    Cambridge 9618-style dry-run interpreter.
    - Accepts pre-supplied input values (like exam papers provide)
    - Records a trace table of selected variables at each step
    - Feeds inputs from queue instead of prompting interactively
    """

    def __init__(self, symbol_table: SymbolTable, input_queue=None,
                 traced_vars=None, max_steps=5000):
        super().__init__(symbol_table)
        self.trace = []
        self.step_count = 0
        self.max_steps = max_steps
        self.input_queue = list(input_queue) if input_queue else []
        self.input_index = 0
        self.traced_vars = set(traced_vars) if traced_vars else None  # None = auto-detect all
        self.output_log = []  # record OUTPUT values too

    # ── AST scanning helpers (static, no execution needed) ──

    @staticmethod
    def scan_inputs(statements):
        """Walk the AST to find all INPUT statements and return info about them.
        Returns list of dicts: {'line': int, 'variable': str}
        """
        results = []
        DryRunInterpreter._walk_for_inputs(statements, results)
        return results

    @staticmethod
    def _walk_for_inputs(stmts, results):
        for stmt in stmts:
            if isinstance(stmt, InputStmt):
                var_name = "?"
                if isinstance(stmt.target, VariableExpr):
                    var_name = stmt.target.name
                elif isinstance(stmt.target, ArrayAccessExpr):
                    var_name = stmt.target.array + "[...]"
                elif isinstance(stmt.target, str):
                    var_name = stmt.target
                line = getattr(stmt, 'line', 0)
                results.append({'line': line, 'variable': var_name})
            # Recurse into compound statements
            if isinstance(stmt, IfStmt):
                DryRunInterpreter._walk_for_inputs(stmt.then_branch, results)
                if stmt.else_branch:
                    DryRunInterpreter._walk_for_inputs(stmt.else_branch, results)
            elif isinstance(stmt, WhileStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, RepeatStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, ForStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, CaseStmt):
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

    @staticmethod
    def scan_declares(statements):
        """Walk the AST to find all DECLARE statements.
        Returns list of dicts: {'name': str, 'type': str, 'is_array': bool}
        """
        results = []
        for stmt in statements:
            if isinstance(stmt, DeclareStmt):
                results.append({
                    'name': stmt.name,
                    'type': stmt.type_name,
                    'is_array': stmt.is_array
                })
            elif isinstance(stmt, ConstantDecl):
                results.append({
                    'name': stmt.name,
                    'type': 'CONSTANT',
                    'is_array': False
                })
        return results

    # ── Snapshot & trace recording ──

    def _snapshot_vars(self):
        """Capture values of traced variables only."""
        snapshot = {}
        for scope_level in range(self.symbol_table.scope_level + 1):
            scope = self.symbol_table.scopes.get(scope_level, {})
            for name, sym in scope.items():
                # If user specified which vars to trace, filter
                if self.traced_vars is not None and name not in self.traced_vars:
                    continue
                cell = sym.cell
                if cell.is_array:
                    if cell.array_elements:
                        arr = {}
                        for key, ec in sorted(cell.array_elements.items()):
                            idx = ",".join(str(k) for k in key)
                            arr[f"[{idx}]"] = ec.get()
                        snapshot[name] = arr
                    else:
                        snapshot[name] = ""
                elif cell.type == DataType.RECORD and isinstance(cell.value, dict):
                    rec = {}
                    for fn, fc in cell.value.items():
                        rec[fn] = fc.get() if isinstance(fc, Cell) else fc
                    snapshot[name] = rec
                elif isinstance(cell.value, PseudocodeObject):
                    attrs = {}
                    for an, ac in cell.value.attributes.items():
                        attrs[an] = ac.get() if isinstance(ac, Cell) else ac
                    snapshot[name] = f"<{cell.value.class_name}>{attrs}"
                else:
                    try:
                        snapshot[name] = cell.get()
                    except Exception:
                        snapshot[name] = cell.value
        return snapshot

    def _format_val(self, val):
        if isinstance(val, bool): return "TRUE" if val else "FALSE"
        if isinstance(val, str): return f'"{val}"'
        return str(val)

    def _describe_stmt(self, stmt):
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            if isinstance(t, str): return f"{t} \u2190 ..."
            if isinstance(t, ArrayAccessExpr): return f"{t.array}[...] \u2190 ..."
            if isinstance(t, MemberExpr): return f".{t.field} \u2190 ..."
            return "ASSIGN"
        if isinstance(stmt, DeclareStmt):
            arr = "ARRAY" if stmt.is_array else ""
            return f"DECLARE {stmt.name} : {arr} {stmt.type_name}".strip()
        if isinstance(stmt, ConstantDecl): return f"CONSTANT {stmt.name}"
        if isinstance(stmt, OutputStmt): return "OUTPUT"
        if isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr): return f"INPUT {stmt.target.name}"
            if isinstance(stmt.target, ArrayAccessExpr): return f"INPUT {stmt.target.array}[...]"
            return "INPUT"
        if isinstance(stmt, IfStmt): return "IF"
        if isinstance(stmt, WhileStmt): return "WHILE"
        if isinstance(stmt, RepeatStmt): return "REPEAT"
        if isinstance(stmt, ForStmt): return f"FOR {stmt.identifier}"
        if isinstance(stmt, CaseStmt): return "CASE OF"
        if isinstance(stmt, ProcedureCallStmt): return f"CALL {stmt.name}"
        if isinstance(stmt, ReturnStmt): return "RETURN"
        if isinstance(stmt, ProcedureDecl): return f"PROCEDURE {stmt.name}"
        if isinstance(stmt, FunctionDecl): return f"FUNCTION {stmt.name}"
        if isinstance(stmt, TypeDecl): return f"TYPE {stmt.name}"
        if isinstance(stmt, ClassDecl): return f"CLASS {stmt.name}"
        if isinstance(stmt, FileStmt): return f"{stmt.operation}FILE"
        return type(stmt).__name__

    def _record_step(self, stmt, note=""):
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise InterpreterError(
                f"Dry-run stopped after {self.max_steps} steps (possible infinite loop)")
        entry = {
            'step': self.step_count,
            'line': getattr(stmt, 'line', self.current_line),
            'statement': self._describe_stmt(stmt),
            'note': note,
            'variables': self._snapshot_vars(),
        }
        self.trace.append(entry)

    # ── Override execute to record trace ──

    def execute(self, stmt: Stmt):
        if hasattr(stmt, 'line') and stmt.line > 0:
            self.current_line = stmt.line

        # Definitions — store but don't trace body execution
        if isinstance(stmt, (ProcedureDecl, FunctionDecl, TypeDecl, ClassDecl)):
            self._record_step(stmt, "defined")
            return super().execute(stmt)

        if isinstance(stmt, Expr):
            self._record_step(stmt, "expr")
            return super().execute(stmt)

        # Execute
        result = super().execute(stmt)

        # Build note about what changed
        note = ""
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            try:
                if isinstance(t, str):
                    v = self.symbol_table.get_cell(t).get()
                    note = f"= {self._format_val(v)}"
                elif isinstance(t, ArrayAccessExpr):
                    indices = [self.evaluate(idx) for idx in t.indices]
                    v = self.symbol_table.array_access(t.array, indices).get()
                    idx_s = ",".join(str(i) for i in indices)
                    note = f"[{idx_s}] = {self._format_val(v)}"
            except Exception:
                pass
        elif isinstance(stmt, DeclareStmt):
            note = stmt.type_name
        elif isinstance(stmt, OutputStmt):
            # Record what was output
            vals = [self._format_output(self.evaluate(a)) for a in stmt.values]
            output_text = "".join(vals)
            self.output_log.append(output_text)
            note = f'OUT: {output_text}'
        elif isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr):
                try:
                    v = self.symbol_table.get_cell(stmt.target.name).get()
                    note = f"= {self._format_val(v)}"
                except Exception:
                    pass
        elif isinstance(stmt, IfStmt):
            note = "condition"
        elif isinstance(stmt, ForStmt):
            note = f"loop {stmt.identifier}"

        self._record_step(stmt, note)
        return result

    # ── Override INPUT to use pre-supplied queue ──

    def visit_InputStmt(self, stmt: InputStmt):
        """Use pre-supplied input values instead of prompting."""
        if self.input_index < len(self.input_queue):
            val_str = str(self.input_queue[self.input_index])
            self.input_index += 1
        else:
            raise InterpreterError(
                "Dry-run ran out of pre-supplied input values. "
                f"Needed input #{self.input_index + 1} but only "
                f"{len(self.input_queue)} values were provided.")

        # Determine target type for coercion
        target_type = None
        try:
            if isinstance(stmt.target, VariableExpr):
                target_type = self.symbol_table.get_cell(stmt.target.name).type
            elif isinstance(stmt.target, str):
                target_type = self.symbol_table.get_cell(stmt.target).type
            elif isinstance(stmt.target, ArrayAccessExpr):
                cell = self.symbol_table.get_cell(stmt.target.array)
                if cell.is_array and cell.array_bounds:
                    target_type = cell.array_bounds.element_type
        except Exception:
            pass

        if target_type is not None:
            val, val_type = self._coerce_input(val_str, target_type)
        else:
            val, val_type = self._auto_parse_input(val_str)

        if isinstance(stmt.target, VariableExpr):
            self.symbol_table.assign(stmt.target.name, val, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            self.symbol_table.array_assign(stmt.target.array, indices, val, val_type)
        elif isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, val, val_type)

    # ── Results ──

    def get_all_var_names(self):
        names = set()
        for entry in self.trace:
            names.update(entry['variables'].keys())
        names -= set(self.procedures.keys())
        names -= set(self.functions.keys())
        return sorted(names)

    def format_trace_text(self):
        if not self.trace:
            return "No trace data recorded."

        var_names = self.get_all_var_names()
        headers = ['Step', 'Line', 'Statement', 'Note'] + var_names
        rows = []
        for entry in self.trace:
            row = [str(entry['step']), str(entry['line']),
                   entry['statement'], entry['note']]
            for vn in var_names:
                val = entry['variables'].get(vn, '')
                if isinstance(val, dict):
                    val = str(val)
                elif val == '':
                    val = ''
                else:
                    val = self._format_val(val) if not isinstance(val, str) else val
                row.append(val)
            rows.append(row)

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, c in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(c)))
        col_widths = [min(w, 30) for w in col_widths]

        def pad(s, w):
            return str(s)[:w].ljust(w)

        sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
        hdr = '|' + '|'.join(f" {pad(h, w)} " for h, w in zip(headers, col_widths)) + '|'
        lines = [sep, hdr, sep]
        for row in rows:
            r = '|' + '|'.join(f" {pad(c, w)} " for c, w in zip(row, col_widths)) + '|'
            lines.append(r)
        lines.append(sep)
        return '\n'.join(lines)
```

**2. Replace the entire [ide.py](ide.py):**

```python
"""
9618 Pseudocode IDE - Modern Desktop Interface
Tkinter-based GUI with dark theme, syntax highlighting, line numbers, and integrated output.
Includes A-Level exam-style Dry Run mode with pre-supplied inputs and trace table.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import sys
import io
import os
import threading
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexer import Lexer, LexerError
from parser import Parser, ParserError
from interpreter import Interpreter, InterpreterError, DryRunInterpreter
from symbol_table import SymbolTable

# -- Theme Colors (Catppuccin Mocha) --
COLORS = {
    "bg":           "#1e1e2e",
    "bg_secondary": "#181825",
    "bg_tertiary":  "#11111b",
    "surface":      "#313244",
    "overlay":      "#45475a",
    "text":         "#cdd6f4",
    "subtext":      "#a6adc8",
    "blue":         "#89b4fa",
    "green":        "#a6e3a1",
    "red":          "#f38ba8",
    "yellow":       "#f9e2af",
    "mauve":        "#cba6f7",
    "peach":        "#fab387",
    "teal":         "#94e2d5",
    "pink":         "#f5c2e7",
    "lavender":     "#b4befe",
    "sky":          "#89dceb",
    "sapphire":     "#74c7ec",
    "line_num_fg":  "#585b70",
    "selection":    "#45475a",
    "cursor":       "#f5e0dc",
    "gutter":       "#282a3a",
    "output_bg":    "#11111b",
    "toolbar_bg":   "#181825",
    "status_bg":    "#181825",
    "tab_active":   "#1e1e2e",
    "accent":       "#89b4fa",
    "error":        "#f38ba8",
    "success":      "#a6e3a1",
    "warning":      "#f9e2af",
    "button_bg":    "#313244",
    "button_hover": "#45475a",
    "border":       "#313244",
}

# -- Keyword Lists for Highlighting --
KEYWORDS_CONTROL = {
    'IF', 'THEN', 'ELSE', 'ENDIF', 'CASE', 'OF', 'OTHERWISE', 'ENDCASE',
    'FOR', 'TO', 'STEP', 'NEXT', 'WHILE', 'DO', 'ENDWHILE',
    'REPEAT', 'UNTIL', 'RETURN',
}
KEYWORDS_DECL = {
    'DECLARE', 'CONSTANT', 'TYPE', 'ENDTYPE', 'ARRAY',
    'PROCEDURE', 'ENDPROCEDURE', 'FUNCTION', 'ENDFUNCTION',
    'RETURNS', 'BYREF', 'BYVAL', 'CALL',
}
KEYWORDS_TYPE = {'INTEGER', 'REAL', 'STRING', 'BOOLEAN', 'CHAR', 'DATE'}
KEYWORDS_IO = {
    'INPUT', 'OUTPUT', 'OPENFILE', 'READFILE', 'WRITEFILE', 'CLOSEFILE',
    'READ', 'WRITE', 'APPEND',
}
KEYWORDS_OP = {'AND', 'OR', 'NOT', 'DIV', 'MOD', 'TRUE', 'FALSE'}
BUILTINS = {
    'LENGTH', 'UCASE', 'LCASE', 'LEFT', 'RIGHT', 'MID',
    'INT', 'NUM_TO_STR', 'STR_TO_NUM', 'ASC', 'CHR', 'SQRT', 'RAND', 'EOF',
}


class LineNumbers(tk.Canvas):
    def __init__(self, parent, text_widget, **kwargs):
        super().__init__(parent, **kwargs)
        self.text_widget = text_widget
        self.font = None

    def redraw(self, *_args):
        self.delete("all")
        if self.text_widget is None:
            return
        i = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.create_text(
                self.winfo_width() - 8, y, anchor="ne", text=linenum,
                font=self.font, fill=COLORS["line_num_fg"],
            )
            i = self.text_widget.index(f"{i}+1line")
            if self.text_widget.compare(i, ">=", "end"):
                break


class CodeEditor(tk.Text):
    TAG_CONFIG = {
        "keyword_ctrl":  {"foreground": COLORS["mauve"],   "font_style": "bold"},
        "keyword_decl":  {"foreground": COLORS["blue"],    "font_style": "bold"},
        "keyword_type":  {"foreground": COLORS["yellow"]},
        "keyword_io":    {"foreground": COLORS["green"],   "font_style": "bold"},
        "keyword_op":    {"foreground": COLORS["peach"],   "font_style": "bold"},
        "builtin":       {"foreground": COLORS["sapphire"]},
        "string":        {"foreground": COLORS["green"]},
        "char":          {"foreground": COLORS["teal"]},
        "number":        {"foreground": COLORS["peach"]},
        "comment":       {"foreground": COLORS["overlay"], "font_style": "italic"},
        "operator":      {"foreground": COLORS["sky"]},
        "assign_arrow":  {"foreground": COLORS["red"]},
        "error_line":    {"background": "#3d2030"},
    }

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._setup_tags()
        self.bind("<<Modified>>", self._on_modify)
        self._highlight_job = None

    def _setup_tags(self):
        base_font = self.cget("font")
        for tag, cfg in self.TAG_CONFIG.items():
            opts = {"foreground": cfg.get("foreground", COLORS["text"])}
            if "background" in cfg:
                opts["background"] = cfg["background"]
            style = cfg.get("font_style", "")
            if style:
                fam = tkfont.Font(font=base_font).actual()["family"]
                sz = tkfont.Font(font=base_font).actual()["size"]
                weight = "bold" if "bold" in style else "normal"
                slant = "italic" if "italic" in style else "roman"
                opts["font"] = tkfont.Font(family=fam, size=sz, weight=weight, slant=slant)
            self.tag_configure(tag, **opts)
        self.tag_raise("error_line")

    def _on_modify(self, _event=None):
        if self.edit_modified():
            if self._highlight_job:
                self.after_cancel(self._highlight_job)
            self._highlight_job = self.after(80, self.highlight_syntax)
            self.edit_modified(False)
            self.event_generate("<<ContentChanged>>")

    def highlight_syntax(self):
        for tag in self.TAG_CONFIG:
            self.tag_remove(tag, "1.0", "end")
        code = self.get("1.0", "end-1c")
        patterns = [
            ("comment",      r'//[^\n]*'),
            ("string",       r'"[^"]*"'),
            ("char",         r"'[^']*'"),
            ("assign_arrow", r'<-'),
            ("number",       r'\b\d+\.\d+\b|\b\d+\b'),
            ("operator",     r'<>|<=|>=|[<>=+\-*/&]'),
        ]
        for tag, pat in patterns:
            for m in re.finditer(pat, code):
                start = f"1.0+{m.start()}c"
                end = f"1.0+{m.end()}c"
                self.tag_add(tag, start, end)
        word_groups = [
            ("keyword_ctrl", KEYWORDS_CONTROL),
            ("keyword_decl", KEYWORDS_DECL),
            ("keyword_type", KEYWORDS_TYPE),
            ("keyword_io",   KEYWORDS_IO),
            ("keyword_op",   KEYWORDS_OP),
            ("builtin",      BUILTINS),
        ]
        for tag, words in word_groups:
            for w in words:
                pat = rf'\b{w}\b'
                for m in re.finditer(pat, code):
                    start = f"1.0+{m.start()}c"
                    end = f"1.0+{m.end()}c"
                    tags_at = self.tag_names(start)
                    if "string" not in tags_at and "comment" not in tags_at and "char" not in tags_at:
                        self.tag_add(tag, start, end)

    def mark_error_line(self, line_num):
        self.tag_remove("error_line", "1.0", "end")
        if line_num and line_num > 0:
            self.tag_add("error_line", f"{line_num}.0", f"{line_num}.end+1c")
            self.see(f"{line_num}.0")


class OutputPanel(tk.Text):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(state="disabled")
        self.tag_configure("error",        foreground=COLORS["error"])
        self.tag_configure("success",      foreground=COLORS["success"])
        self.tag_configure("info",         foreground=COLORS["subtext"])
        self.tag_configure("output",       foreground=COLORS["text"])
        self.tag_configure("input_prompt", foreground=COLORS["yellow"])

    def append(self, text, tag="output"):
        self.configure(state="normal")
        self.insert("end", text, tag)
        self.see("end")
        self.configure(state="disabled")

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class InputDialog(tk.Toplevel):
    def __init__(self, parent, prompt):
        super().__init__(parent)
        self.title("Input Required")
        self.result = None
        self.configure(bg=COLORS["bg"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 400, 160
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        frame = tk.Frame(self, bg=COLORS["bg"], padx=20, pady=15)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text=prompt, font=("Segoe UI", 11),
                 fg=COLORS["yellow"], bg=COLORS["bg"], anchor="w").pack(fill="x", pady=(0, 8))

        self.entry = tk.Entry(frame, font=("Cascadia Code", 12),
                              bg=COLORS["surface"], fg=COLORS["text"],
                              insertbackground=COLORS["cursor"], relief="flat", bd=0)
        self.entry.pack(fill="x", ipady=6)
        self.entry.focus_set()
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Escape>", lambda e: self._cancel())

        btn_frame = tk.Frame(frame, bg=COLORS["bg"])
        btn_frame.pack(fill="x", pady=(12, 0))

        tk.Button(btn_frame, text="OK", font=("Segoe UI", 10),
                  bg=COLORS["accent"], fg=COLORS["bg_tertiary"],
                  activebackground=COLORS["lavender"],
                  relief="flat", bd=0, padx=20, pady=4,
                  command=self._submit).pack(side="right")
        tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 10),
                  bg=COLORS["surface"], fg=COLORS["text"],
                  activebackground=COLORS["overlay"],
                  relief="flat", bd=0, padx=16, pady=4,
                  command=self._cancel).pack(side="right", padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _submit(self, _event=None):
        self.result = self.entry.get()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class RedirectOutput:
    def __init__(self, panel, tag="output"):
        self.panel = panel
        self.tag = tag

    def write(self, text):
        if text:
            self.panel.after(0, self.panel.append, text, self.tag)

    def flush(self):
        pass


# ──────────────── Dry Run Setup Dialog ────────────────

class DryRunSetupDialog(tk.Toplevel):
    """
    A-Level exam-style dry run setup.
    Shows detected INPUT statements, lets user supply values upfront,
    and choose which variables to include in the trace table.
    """

    def __init__(self, parent, input_info, declare_info):
        super().__init__(parent)
        self.title("Dry Run Setup")
        self.result = None  # Will be {'inputs': [...], 'traced_vars': set or None}
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 520, 560
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        self.minsize(400, 400)

        self.input_info = input_info    # [{'line': int, 'variable': str}, ...]
        self.declare_info = declare_info  # [{'name': str, 'type': str, 'is_array': bool}, ...]
        self.input_entries = []         # tk.Entry widgets for each input
        self.var_checkboxes = {}        # {name: tk.BooleanVar}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _build_ui(self):
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # ── Title ──
        tk.Label(main, text="Dry Run Setup",
                 font=("Segoe UI", 14, "bold"), fg=COLORS["accent"],
                 bg=COLORS["bg"]).pack(anchor="w", pady=(0, 4))
        tk.Label(main, text="Supply input values and select variables to trace,\n"
                            "just like a Cambridge 9618 exam trace table.",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"], justify="left").pack(anchor="w", pady=(0, 12))

        # ── Input Values Section ──
        input_frame = tk.LabelFrame(main, text=" Input Values ",
                                     font=("Segoe UI", 10, "bold"),
                                     fg=COLORS["yellow"], bg=COLORS["bg"],
                                     bd=1, relief="solid",
                                     highlightbackground=COLORS["surface"])
        input_frame.pack(fill="x", pady=(0, 12))

        if self.input_info:
            tk.Label(input_frame,
                     text="Enter values in order (as given on an exam paper):",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

            for i, info in enumerate(self.input_info):
                row = tk.Frame(input_frame, bg=COLORS["bg"])
                row.pack(fill="x", padx=10, pady=2)

                label_text = f"Input #{i+1}  (Line {info['line']}: {info['variable']})"
                tk.Label(row, text=label_text, font=("Segoe UI", 9),
                         fg=COLORS["text"], bg=COLORS["bg"], width=35,
                         anchor="w").pack(side="left")

                entry = tk.Entry(row, font=("Cascadia Code", 11),
                                 bg=COLORS["surface"], fg=COLORS["text"],
                                 insertbackground=COLORS["cursor"],
                                 relief="flat", bd=0, width=20)
                entry.pack(side="left", padx=(4, 0), ipady=3)
                self.input_entries.append(entry)

            # Also allow a bulk entry box
            tk.Label(input_frame,
                     text="Or enter all values comma-separated:",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(8, 2))

            self.bulk_entry = tk.Entry(input_frame, font=("Cascadia Code", 11),
                                        bg=COLORS["surface"], fg=COLORS["text"],
                                        insertbackground=COLORS["cursor"],
                                        relief="flat", bd=0)
            self.bulk_entry.pack(fill="x", padx=10, pady=(0, 8), ipady=4)
            self.bulk_entry.insert(0, "e.g. 5, 3, 8")
            self.bulk_entry.bind("<FocusIn>", self._clear_placeholder)
        else:
            tk.Label(input_frame,
                     text="No INPUT statements detected in this program.",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(padx=10, pady=8)
            self.bulk_entry = None

        # ── Variables to Trace Section ──
        var_frame = tk.LabelFrame(main, text=" Trace Table Columns ",
                                   font=("Segoe UI", 10, "bold"),
                                   fg=COLORS["teal"], bg=COLORS["bg"],
                                   bd=1, relief="solid",
                                   highlightbackground=COLORS["surface"])
        var_frame.pack(fill="both", expand=True, pady=(0, 12))

        tk.Label(var_frame,
                 text="Select which variables to show in the trace table:",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

        # Scrollable checkbox area
        canvas_frame = tk.Frame(var_frame, bg=COLORS["bg"])
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        canvas = tk.Canvas(canvas_frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        cb
```


## License: unknown
https://github.com/cstroup/business-planning/blob/c48782917b7971b53d1174d4c05aa4463862e170/planning_app.py

```
Now I have full context. Let me build the complete solution. The approach:

1. **Scan the AST** to find all `InputStmt` nodes and extract variable names — no LLM needed
2. **Before dry-run starts**, show a setup dialog where the user supplies:
   - All input values upfront (comma-separated, like exam papers give them)
   - Which variables to trace (auto-detected from DECLARE statements, user can toggle)
3. **During execution**, feed inputs from the pre-supplied queue instead of prompting
4. **After execution**, show the trace table in exact A-Level format

Here are the three files to update:

**1. Append to the end of [interpreter.py](interpreter.py) — the `DryRunInterpreter` class:**

```python

# ──────────────── Dry-Run / Trace Mode ────────────────

class DryRunInterpreter(Interpreter):
    """
    Cambridge 9618-style dry-run interpreter.
    - Accepts pre-supplied input values (like exam papers provide)
    - Records a trace table of selected variables at each step
    - Feeds inputs from queue instead of prompting interactively
    """

    def __init__(self, symbol_table: SymbolTable, input_queue=None,
                 traced_vars=None, max_steps=5000):
        super().__init__(symbol_table)
        self.trace = []
        self.step_count = 0
        self.max_steps = max_steps
        self.input_queue = list(input_queue) if input_queue else []
        self.input_index = 0
        self.traced_vars = set(traced_vars) if traced_vars else None  # None = auto-detect all
        self.output_log = []  # record OUTPUT values too

    # ── AST scanning helpers (static, no execution needed) ──

    @staticmethod
    def scan_inputs(statements):
        """Walk the AST to find all INPUT statements and return info about them.
        Returns list of dicts: {'line': int, 'variable': str}
        """
        results = []
        DryRunInterpreter._walk_for_inputs(statements, results)
        return results

    @staticmethod
    def _walk_for_inputs(stmts, results):
        for stmt in stmts:
            if isinstance(stmt, InputStmt):
                var_name = "?"
                if isinstance(stmt.target, VariableExpr):
                    var_name = stmt.target.name
                elif isinstance(stmt.target, ArrayAccessExpr):
                    var_name = stmt.target.array + "[...]"
                elif isinstance(stmt.target, str):
                    var_name = stmt.target
                line = getattr(stmt, 'line', 0)
                results.append({'line': line, 'variable': var_name})
            # Recurse into compound statements
            if isinstance(stmt, IfStmt):
                DryRunInterpreter._walk_for_inputs(stmt.then_branch, results)
                if stmt.else_branch:
                    DryRunInterpreter._walk_for_inputs(stmt.else_branch, results)
            elif isinstance(stmt, WhileStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, RepeatStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, ForStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, CaseStmt):
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

    @staticmethod
    def scan_declares(statements):
        """Walk the AST to find all DECLARE statements.
        Returns list of dicts: {'name': str, 'type': str, 'is_array': bool}
        """
        results = []
        for stmt in statements:
            if isinstance(stmt, DeclareStmt):
                results.append({
                    'name': stmt.name,
                    'type': stmt.type_name,
                    'is_array': stmt.is_array
                })
            elif isinstance(stmt, ConstantDecl):
                results.append({
                    'name': stmt.name,
                    'type': 'CONSTANT',
                    'is_array': False
                })
        return results

    # ── Snapshot & trace recording ──

    def _snapshot_vars(self):
        """Capture values of traced variables only."""
        snapshot = {}
        for scope_level in range(self.symbol_table.scope_level + 1):
            scope = self.symbol_table.scopes.get(scope_level, {})
            for name, sym in scope.items():
                # If user specified which vars to trace, filter
                if self.traced_vars is not None and name not in self.traced_vars:
                    continue
                cell = sym.cell
                if cell.is_array:
                    if cell.array_elements:
                        arr = {}
                        for key, ec in sorted(cell.array_elements.items()):
                            idx = ",".join(str(k) for k in key)
                            arr[f"[{idx}]"] = ec.get()
                        snapshot[name] = arr
                    else:
                        snapshot[name] = ""
                elif cell.type == DataType.RECORD and isinstance(cell.value, dict):
                    rec = {}
                    for fn, fc in cell.value.items():
                        rec[fn] = fc.get() if isinstance(fc, Cell) else fc
                    snapshot[name] = rec
                elif isinstance(cell.value, PseudocodeObject):
                    attrs = {}
                    for an, ac in cell.value.attributes.items():
                        attrs[an] = ac.get() if isinstance(ac, Cell) else ac
                    snapshot[name] = f"<{cell.value.class_name}>{attrs}"
                else:
                    try:
                        snapshot[name] = cell.get()
                    except Exception:
                        snapshot[name] = cell.value
        return snapshot

    def _format_val(self, val):
        if isinstance(val, bool): return "TRUE" if val else "FALSE"
        if isinstance(val, str): return f'"{val}"'
        return str(val)

    def _describe_stmt(self, stmt):
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            if isinstance(t, str): return f"{t} \u2190 ..."
            if isinstance(t, ArrayAccessExpr): return f"{t.array}[...] \u2190 ..."
            if isinstance(t, MemberExpr): return f".{t.field} \u2190 ..."
            return "ASSIGN"
        if isinstance(stmt, DeclareStmt):
            arr = "ARRAY" if stmt.is_array else ""
            return f"DECLARE {stmt.name} : {arr} {stmt.type_name}".strip()
        if isinstance(stmt, ConstantDecl): return f"CONSTANT {stmt.name}"
        if isinstance(stmt, OutputStmt): return "OUTPUT"
        if isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr): return f"INPUT {stmt.target.name}"
            if isinstance(stmt.target, ArrayAccessExpr): return f"INPUT {stmt.target.array}[...]"
            return "INPUT"
        if isinstance(stmt, IfStmt): return "IF"
        if isinstance(stmt, WhileStmt): return "WHILE"
        if isinstance(stmt, RepeatStmt): return "REPEAT"
        if isinstance(stmt, ForStmt): return f"FOR {stmt.identifier}"
        if isinstance(stmt, CaseStmt): return "CASE OF"
        if isinstance(stmt, ProcedureCallStmt): return f"CALL {stmt.name}"
        if isinstance(stmt, ReturnStmt): return "RETURN"
        if isinstance(stmt, ProcedureDecl): return f"PROCEDURE {stmt.name}"
        if isinstance(stmt, FunctionDecl): return f"FUNCTION {stmt.name}"
        if isinstance(stmt, TypeDecl): return f"TYPE {stmt.name}"
        if isinstance(stmt, ClassDecl): return f"CLASS {stmt.name}"
        if isinstance(stmt, FileStmt): return f"{stmt.operation}FILE"
        return type(stmt).__name__

    def _record_step(self, stmt, note=""):
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise InterpreterError(
                f"Dry-run stopped after {self.max_steps} steps (possible infinite loop)")
        entry = {
            'step': self.step_count,
            'line': getattr(stmt, 'line', self.current_line),
            'statement': self._describe_stmt(stmt),
            'note': note,
            'variables': self._snapshot_vars(),
        }
        self.trace.append(entry)

    # ── Override execute to record trace ──

    def execute(self, stmt: Stmt):
        if hasattr(stmt, 'line') and stmt.line > 0:
            self.current_line = stmt.line

        # Definitions — store but don't trace body execution
        if isinstance(stmt, (ProcedureDecl, FunctionDecl, TypeDecl, ClassDecl)):
            self._record_step(stmt, "defined")
            return super().execute(stmt)

        if isinstance(stmt, Expr):
            self._record_step(stmt, "expr")
            return super().execute(stmt)

        # Execute
        result = super().execute(stmt)

        # Build note about what changed
        note = ""
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            try:
                if isinstance(t, str):
                    v = self.symbol_table.get_cell(t).get()
                    note = f"= {self._format_val(v)}"
                elif isinstance(t, ArrayAccessExpr):
                    indices = [self.evaluate(idx) for idx in t.indices]
                    v = self.symbol_table.array_access(t.array, indices).get()
                    idx_s = ",".join(str(i) for i in indices)
                    note = f"[{idx_s}] = {self._format_val(v)}"
            except Exception:
                pass
        elif isinstance(stmt, DeclareStmt):
            note = stmt.type_name
        elif isinstance(stmt, OutputStmt):
            # Record what was output
            vals = [self._format_output(self.evaluate(a)) for a in stmt.values]
            output_text = "".join(vals)
            self.output_log.append(output_text)
            note = f'OUT: {output_text}'
        elif isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr):
                try:
                    v = self.symbol_table.get_cell(stmt.target.name).get()
                    note = f"= {self._format_val(v)}"
                except Exception:
                    pass
        elif isinstance(stmt, IfStmt):
            note = "condition"
        elif isinstance(stmt, ForStmt):
            note = f"loop {stmt.identifier}"

        self._record_step(stmt, note)
        return result

    # ── Override INPUT to use pre-supplied queue ──

    def visit_InputStmt(self, stmt: InputStmt):
        """Use pre-supplied input values instead of prompting."""
        if self.input_index < len(self.input_queue):
            val_str = str(self.input_queue[self.input_index])
            self.input_index += 1
        else:
            raise InterpreterError(
                "Dry-run ran out of pre-supplied input values. "
                f"Needed input #{self.input_index + 1} but only "
                f"{len(self.input_queue)} values were provided.")

        # Determine target type for coercion
        target_type = None
        try:
            if isinstance(stmt.target, VariableExpr):
                target_type = self.symbol_table.get_cell(stmt.target.name).type
            elif isinstance(stmt.target, str):
                target_type = self.symbol_table.get_cell(stmt.target).type
            elif isinstance(stmt.target, ArrayAccessExpr):
                cell = self.symbol_table.get_cell(stmt.target.array)
                if cell.is_array and cell.array_bounds:
                    target_type = cell.array_bounds.element_type
        except Exception:
            pass

        if target_type is not None:
            val, val_type = self._coerce_input(val_str, target_type)
        else:
            val, val_type = self._auto_parse_input(val_str)

        if isinstance(stmt.target, VariableExpr):
            self.symbol_table.assign(stmt.target.name, val, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            self.symbol_table.array_assign(stmt.target.array, indices, val, val_type)
        elif isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, val, val_type)

    # ── Results ──

    def get_all_var_names(self):
        names = set()
        for entry in self.trace:
            names.update(entry['variables'].keys())
        names -= set(self.procedures.keys())
        names -= set(self.functions.keys())
        return sorted(names)

    def format_trace_text(self):
        if not self.trace:
            return "No trace data recorded."

        var_names = self.get_all_var_names()
        headers = ['Step', 'Line', 'Statement', 'Note'] + var_names
        rows = []
        for entry in self.trace:
            row = [str(entry['step']), str(entry['line']),
                   entry['statement'], entry['note']]
            for vn in var_names:
                val = entry['variables'].get(vn, '')
                if isinstance(val, dict):
                    val = str(val)
                elif val == '':
                    val = ''
                else:
                    val = self._format_val(val) if not isinstance(val, str) else val
                row.append(val)
            rows.append(row)

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, c in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(c)))
        col_widths = [min(w, 30) for w in col_widths]

        def pad(s, w):
            return str(s)[:w].ljust(w)

        sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
        hdr = '|' + '|'.join(f" {pad(h, w)} " for h, w in zip(headers, col_widths)) + '|'
        lines = [sep, hdr, sep]
        for row in rows:
            r = '|' + '|'.join(f" {pad(c, w)} " for c, w in zip(row, col_widths)) + '|'
            lines.append(r)
        lines.append(sep)
        return '\n'.join(lines)
```

**2. Replace the entire [ide.py](ide.py):**

```python
"""
9618 Pseudocode IDE - Modern Desktop Interface
Tkinter-based GUI with dark theme, syntax highlighting, line numbers, and integrated output.
Includes A-Level exam-style Dry Run mode with pre-supplied inputs and trace table.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import sys
import io
import os
import threading
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexer import Lexer, LexerError
from parser import Parser, ParserError
from interpreter import Interpreter, InterpreterError, DryRunInterpreter
from symbol_table import SymbolTable

# -- Theme Colors (Catppuccin Mocha) --
COLORS = {
    "bg":           "#1e1e2e",
    "bg_secondary": "#181825",
    "bg_tertiary":  "#11111b",
    "surface":      "#313244",
    "overlay":      "#45475a",
    "text":         "#cdd6f4",
    "subtext":      "#a6adc8",
    "blue":         "#89b4fa",
    "green":        "#a6e3a1",
    "red":          "#f38ba8",
    "yellow":       "#f9e2af",
    "mauve":        "#cba6f7",
    "peach":        "#fab387",
    "teal":         "#94e2d5",
    "pink":         "#f5c2e7",
    "lavender":     "#b4befe",
    "sky":          "#89dceb",
    "sapphire":     "#74c7ec",
    "line_num_fg":  "#585b70",
    "selection":    "#45475a",
    "cursor":       "#f5e0dc",
    "gutter":       "#282a3a",
    "output_bg":    "#11111b",
    "toolbar_bg":   "#181825",
    "status_bg":    "#181825",
    "tab_active":   "#1e1e2e",
    "accent":       "#89b4fa",
    "error":        "#f38ba8",
    "success":      "#a6e3a1",
    "warning":      "#f9e2af",
    "button_bg":    "#313244",
    "button_hover": "#45475a",
    "border":       "#313244",
}

# -- Keyword Lists for Highlighting --
KEYWORDS_CONTROL = {
    'IF', 'THEN', 'ELSE', 'ENDIF', 'CASE', 'OF', 'OTHERWISE', 'ENDCASE',
    'FOR', 'TO', 'STEP', 'NEXT', 'WHILE', 'DO', 'ENDWHILE',
    'REPEAT', 'UNTIL', 'RETURN',
}
KEYWORDS_DECL = {
    'DECLARE', 'CONSTANT', 'TYPE', 'ENDTYPE', 'ARRAY',
    'PROCEDURE', 'ENDPROCEDURE', 'FUNCTION', 'ENDFUNCTION',
    'RETURNS', 'BYREF', 'BYVAL', 'CALL',
}
KEYWORDS_TYPE = {'INTEGER', 'REAL', 'STRING', 'BOOLEAN', 'CHAR', 'DATE'}
KEYWORDS_IO = {
    'INPUT', 'OUTPUT', 'OPENFILE', 'READFILE', 'WRITEFILE', 'CLOSEFILE',
    'READ', 'WRITE', 'APPEND',
}
KEYWORDS_OP = {'AND', 'OR', 'NOT', 'DIV', 'MOD', 'TRUE', 'FALSE'}
BUILTINS = {
    'LENGTH', 'UCASE', 'LCASE', 'LEFT', 'RIGHT', 'MID',
    'INT', 'NUM_TO_STR', 'STR_TO_NUM', 'ASC', 'CHR', 'SQRT', 'RAND', 'EOF',
}


class LineNumbers(tk.Canvas):
    def __init__(self, parent, text_widget, **kwargs):
        super().__init__(parent, **kwargs)
        self.text_widget = text_widget
        self.font = None

    def redraw(self, *_args):
        self.delete("all")
        if self.text_widget is None:
            return
        i = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.create_text(
                self.winfo_width() - 8, y, anchor="ne", text=linenum,
                font=self.font, fill=COLORS["line_num_fg"],
            )
            i = self.text_widget.index(f"{i}+1line")
            if self.text_widget.compare(i, ">=", "end"):
                break


class CodeEditor(tk.Text):
    TAG_CONFIG = {
        "keyword_ctrl":  {"foreground": COLORS["mauve"],   "font_style": "bold"},
        "keyword_decl":  {"foreground": COLORS["blue"],    "font_style": "bold"},
        "keyword_type":  {"foreground": COLORS["yellow"]},
        "keyword_io":    {"foreground": COLORS["green"],   "font_style": "bold"},
        "keyword_op":    {"foreground": COLORS["peach"],   "font_style": "bold"},
        "builtin":       {"foreground": COLORS["sapphire"]},
        "string":        {"foreground": COLORS["green"]},
        "char":          {"foreground": COLORS["teal"]},
        "number":        {"foreground": COLORS["peach"]},
        "comment":       {"foreground": COLORS["overlay"], "font_style": "italic"},
        "operator":      {"foreground": COLORS["sky"]},
        "assign_arrow":  {"foreground": COLORS["red"]},
        "error_line":    {"background": "#3d2030"},
    }

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._setup_tags()
        self.bind("<<Modified>>", self._on_modify)
        self._highlight_job = None

    def _setup_tags(self):
        base_font = self.cget("font")
        for tag, cfg in self.TAG_CONFIG.items():
            opts = {"foreground": cfg.get("foreground", COLORS["text"])}
            if "background" in cfg:
                opts["background"] = cfg["background"]
            style = cfg.get("font_style", "")
            if style:
                fam = tkfont.Font(font=base_font).actual()["family"]
                sz = tkfont.Font(font=base_font).actual()["size"]
                weight = "bold" if "bold" in style else "normal"
                slant = "italic" if "italic" in style else "roman"
                opts["font"] = tkfont.Font(family=fam, size=sz, weight=weight, slant=slant)
            self.tag_configure(tag, **opts)
        self.tag_raise("error_line")

    def _on_modify(self, _event=None):
        if self.edit_modified():
            if self._highlight_job:
                self.after_cancel(self._highlight_job)
            self._highlight_job = self.after(80, self.highlight_syntax)
            self.edit_modified(False)
            self.event_generate("<<ContentChanged>>")

    def highlight_syntax(self):
        for tag in self.TAG_CONFIG:
            self.tag_remove(tag, "1.0", "end")
        code = self.get("1.0", "end-1c")
        patterns = [
            ("comment",      r'//[^\n]*'),
            ("string",       r'"[^"]*"'),
            ("char",         r"'[^']*'"),
            ("assign_arrow", r'<-'),
            ("number",       r'\b\d+\.\d+\b|\b\d+\b'),
            ("operator",     r'<>|<=|>=|[<>=+\-*/&]'),
        ]
        for tag, pat in patterns:
            for m in re.finditer(pat, code):
                start = f"1.0+{m.start()}c"
                end = f"1.0+{m.end()}c"
                self.tag_add(tag, start, end)
        word_groups = [
            ("keyword_ctrl", KEYWORDS_CONTROL),
            ("keyword_decl", KEYWORDS_DECL),
            ("keyword_type", KEYWORDS_TYPE),
            ("keyword_io",   KEYWORDS_IO),
            ("keyword_op",   KEYWORDS_OP),
            ("builtin",      BUILTINS),
        ]
        for tag, words in word_groups:
            for w in words:
                pat = rf'\b{w}\b'
                for m in re.finditer(pat, code):
                    start = f"1.0+{m.start()}c"
                    end = f"1.0+{m.end()}c"
                    tags_at = self.tag_names(start)
                    if "string" not in tags_at and "comment" not in tags_at and "char" not in tags_at:
                        self.tag_add(tag, start, end)

    def mark_error_line(self, line_num):
        self.tag_remove("error_line", "1.0", "end")
        if line_num and line_num > 0:
            self.tag_add("error_line", f"{line_num}.0", f"{line_num}.end+1c")
            self.see(f"{line_num}.0")


class OutputPanel(tk.Text):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(state="disabled")
        self.tag_configure("error",        foreground=COLORS["error"])
        self.tag_configure("success",      foreground=COLORS["success"])
        self.tag_configure("info",         foreground=COLORS["subtext"])
        self.tag_configure("output",       foreground=COLORS["text"])
        self.tag_configure("input_prompt", foreground=COLORS["yellow"])

    def append(self, text, tag="output"):
        self.configure(state="normal")
        self.insert("end", text, tag)
        self.see("end")
        self.configure(state="disabled")

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class InputDialog(tk.Toplevel):
    def __init__(self, parent, prompt):
        super().__init__(parent)
        self.title("Input Required")
        self.result = None
        self.configure(bg=COLORS["bg"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 400, 160
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        frame = tk.Frame(self, bg=COLORS["bg"], padx=20, pady=15)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text=prompt, font=("Segoe UI", 11),
                 fg=COLORS["yellow"], bg=COLORS["bg"], anchor="w").pack(fill="x", pady=(0, 8))

        self.entry = tk.Entry(frame, font=("Cascadia Code", 12),
                              bg=COLORS["surface"], fg=COLORS["text"],
                              insertbackground=COLORS["cursor"], relief="flat", bd=0)
        self.entry.pack(fill="x", ipady=6)
        self.entry.focus_set()
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Escape>", lambda e: self._cancel())

        btn_frame = tk.Frame(frame, bg=COLORS["bg"])
        btn_frame.pack(fill="x", pady=(12, 0))

        tk.Button(btn_frame, text="OK", font=("Segoe UI", 10),
                  bg=COLORS["accent"], fg=COLORS["bg_tertiary"],
                  activebackground=COLORS["lavender"],
                  relief="flat", bd=0, padx=20, pady=4,
                  command=self._submit).pack(side="right")
        tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 10),
                  bg=COLORS["surface"], fg=COLORS["text"],
                  activebackground=COLORS["overlay"],
                  relief="flat", bd=0, padx=16, pady=4,
                  command=self._cancel).pack(side="right", padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _submit(self, _event=None):
        self.result = self.entry.get()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class RedirectOutput:
    def __init__(self, panel, tag="output"):
        self.panel = panel
        self.tag = tag

    def write(self, text):
        if text:
            self.panel.after(0, self.panel.append, text, self.tag)

    def flush(self):
        pass


# ──────────────── Dry Run Setup Dialog ────────────────

class DryRunSetupDialog(tk.Toplevel):
    """
    A-Level exam-style dry run setup.
    Shows detected INPUT statements, lets user supply values upfront,
    and choose which variables to include in the trace table.
    """

    def __init__(self, parent, input_info, declare_info):
        super().__init__(parent)
        self.title("Dry Run Setup")
        self.result = None  # Will be {'inputs': [...], 'traced_vars': set or None}
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 520, 560
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        self.minsize(400, 400)

        self.input_info = input_info    # [{'line': int, 'variable': str}, ...]
        self.declare_info = declare_info  # [{'name': str, 'type': str, 'is_array': bool}, ...]
        self.input_entries = []         # tk.Entry widgets for each input
        self.var_checkboxes = {}        # {name: tk.BooleanVar}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _build_ui(self):
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # ── Title ──
        tk.Label(main, text="Dry Run Setup",
                 font=("Segoe UI", 14, "bold"), fg=COLORS["accent"],
                 bg=COLORS["bg"]).pack(anchor="w", pady=(0, 4))
        tk.Label(main, text="Supply input values and select variables to trace,\n"
                            "just like a Cambridge 9618 exam trace table.",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"], justify="left").pack(anchor="w", pady=(0, 12))

        # ── Input Values Section ──
        input_frame = tk.LabelFrame(main, text=" Input Values ",
                                     font=("Segoe UI", 10, "bold"),
                                     fg=COLORS["yellow"], bg=COLORS["bg"],
                                     bd=1, relief="solid",
                                     highlightbackground=COLORS["surface"])
        input_frame.pack(fill="x", pady=(0, 12))

        if self.input_info:
            tk.Label(input_frame,
                     text="Enter values in order (as given on an exam paper):",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

            for i, info in enumerate(self.input_info):
                row = tk.Frame(input_frame, bg=COLORS["bg"])
                row.pack(fill="x", padx=10, pady=2)

                label_text = f"Input #{i+1}  (Line {info['line']}: {info['variable']})"
                tk.Label(row, text=label_text, font=("Segoe UI", 9),
                         fg=COLORS["text"], bg=COLORS["bg"], width=35,
                         anchor="w").pack(side="left")

                entry = tk.Entry(row, font=("Cascadia Code", 11),
                                 bg=COLORS["surface"], fg=COLORS["text"],
                                 insertbackground=COLORS["cursor"],
                                 relief="flat", bd=0, width=20)
                entry.pack(side="left", padx=(4, 0), ipady=3)
                self.input_entries.append(entry)

            # Also allow a bulk entry box
            tk.Label(input_frame,
                     text="Or enter all values comma-separated:",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(8, 2))

            self.bulk_entry = tk.Entry(input_frame, font=("Cascadia Code", 11),
                                        bg=COLORS["surface"], fg=COLORS["text"],
                                        insertbackground=COLORS["cursor"],
                                        relief="flat", bd=0)
            self.bulk_entry.pack(fill="x", padx=10, pady=(0, 8), ipady=4)
            self.bulk_entry.insert(0, "e.g. 5, 3, 8")
            self.bulk_entry.bind("<FocusIn>", self._clear_placeholder)
        else:
            tk.Label(input_frame,
                     text="No INPUT statements detected in this program.",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(padx=10, pady=8)
            self.bulk_entry = None

        # ── Variables to Trace Section ──
        var_frame = tk.LabelFrame(main, text=" Trace Table Columns ",
                                   font=("Segoe UI", 10, "bold"),
                                   fg=COLORS["teal"], bg=COLORS["bg"],
                                   bd=1, relief="solid",
                                   highlightbackground=COLORS["surface"])
        var_frame.pack(fill="both", expand=True, pady=(0, 12))

        tk.Label(var_frame,
                 text="Select which variables to show in the trace table:",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

        # Scrollable checkbox area
        canvas_frame = tk.Frame(var_frame, bg=COLORS["bg"])
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        canvas = tk.Canvas(canvas_frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        cb
```


## License: unknown
https://github.com/cstroup/business-planning/blob/c48782917b7971b53d1174d4c05aa4463862e170/planning_app.py

```
Now I have full context. Let me build the complete solution. The approach:

1. **Scan the AST** to find all `InputStmt` nodes and extract variable names — no LLM needed
2. **Before dry-run starts**, show a setup dialog where the user supplies:
   - All input values upfront (comma-separated, like exam papers give them)
   - Which variables to trace (auto-detected from DECLARE statements, user can toggle)
3. **During execution**, feed inputs from the pre-supplied queue instead of prompting
4. **After execution**, show the trace table in exact A-Level format

Here are the three files to update:

**1. Append to the end of [interpreter.py](interpreter.py) — the `DryRunInterpreter` class:**

```python

# ──────────────── Dry-Run / Trace Mode ────────────────

class DryRunInterpreter(Interpreter):
    """
    Cambridge 9618-style dry-run interpreter.
    - Accepts pre-supplied input values (like exam papers provide)
    - Records a trace table of selected variables at each step
    - Feeds inputs from queue instead of prompting interactively
    """

    def __init__(self, symbol_table: SymbolTable, input_queue=None,
                 traced_vars=None, max_steps=5000):
        super().__init__(symbol_table)
        self.trace = []
        self.step_count = 0
        self.max_steps = max_steps
        self.input_queue = list(input_queue) if input_queue else []
        self.input_index = 0
        self.traced_vars = set(traced_vars) if traced_vars else None  # None = auto-detect all
        self.output_log = []  # record OUTPUT values too

    # ── AST scanning helpers (static, no execution needed) ──

    @staticmethod
    def scan_inputs(statements):
        """Walk the AST to find all INPUT statements and return info about them.
        Returns list of dicts: {'line': int, 'variable': str}
        """
        results = []
        DryRunInterpreter._walk_for_inputs(statements, results)
        return results

    @staticmethod
    def _walk_for_inputs(stmts, results):
        for stmt in stmts:
            if isinstance(stmt, InputStmt):
                var_name = "?"
                if isinstance(stmt.target, VariableExpr):
                    var_name = stmt.target.name
                elif isinstance(stmt.target, ArrayAccessExpr):
                    var_name = stmt.target.array + "[...]"
                elif isinstance(stmt.target, str):
                    var_name = stmt.target
                line = getattr(stmt, 'line', 0)
                results.append({'line': line, 'variable': var_name})
            # Recurse into compound statements
            if isinstance(stmt, IfStmt):
                DryRunInterpreter._walk_for_inputs(stmt.then_branch, results)
                if stmt.else_branch:
                    DryRunInterpreter._walk_for_inputs(stmt.else_branch, results)
            elif isinstance(stmt, WhileStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, RepeatStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, ForStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, CaseStmt):
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

    @staticmethod
    def scan_declares(statements):
        """Walk the AST to find all DECLARE statements.
        Returns list of dicts: {'name': str, 'type': str, 'is_array': bool}
        """
        results = []
        for stmt in statements:
            if isinstance(stmt, DeclareStmt):
                results.append({
                    'name': stmt.name,
                    'type': stmt.type_name,
                    'is_array': stmt.is_array
                })
            elif isinstance(stmt, ConstantDecl):
                results.append({
                    'name': stmt.name,
                    'type': 'CONSTANT',
                    'is_array': False
                })
        return results

    # ── Snapshot & trace recording ──

    def _snapshot_vars(self):
        """Capture values of traced variables only."""
        snapshot = {}
        for scope_level in range(self.symbol_table.scope_level + 1):
            scope = self.symbol_table.scopes.get(scope_level, {})
            for name, sym in scope.items():
                # If user specified which vars to trace, filter
                if self.traced_vars is not None and name not in self.traced_vars:
                    continue
                cell = sym.cell
                if cell.is_array:
                    if cell.array_elements:
                        arr = {}
                        for key, ec in sorted(cell.array_elements.items()):
                            idx = ",".join(str(k) for k in key)
                            arr[f"[{idx}]"] = ec.get()
                        snapshot[name] = arr
                    else:
                        snapshot[name] = ""
                elif cell.type == DataType.RECORD and isinstance(cell.value, dict):
                    rec = {}
                    for fn, fc in cell.value.items():
                        rec[fn] = fc.get() if isinstance(fc, Cell) else fc
                    snapshot[name] = rec
                elif isinstance(cell.value, PseudocodeObject):
                    attrs = {}
                    for an, ac in cell.value.attributes.items():
                        attrs[an] = ac.get() if isinstance(ac, Cell) else ac
                    snapshot[name] = f"<{cell.value.class_name}>{attrs}"
                else:
                    try:
                        snapshot[name] = cell.get()
                    except Exception:
                        snapshot[name] = cell.value
        return snapshot

    def _format_val(self, val):
        if isinstance(val, bool): return "TRUE" if val else "FALSE"
        if isinstance(val, str): return f'"{val}"'
        return str(val)

    def _describe_stmt(self, stmt):
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            if isinstance(t, str): return f"{t} \u2190 ..."
            if isinstance(t, ArrayAccessExpr): return f"{t.array}[...] \u2190 ..."
            if isinstance(t, MemberExpr): return f".{t.field} \u2190 ..."
            return "ASSIGN"
        if isinstance(stmt, DeclareStmt):
            arr = "ARRAY" if stmt.is_array else ""
            return f"DECLARE {stmt.name} : {arr} {stmt.type_name}".strip()
        if isinstance(stmt, ConstantDecl): return f"CONSTANT {stmt.name}"
        if isinstance(stmt, OutputStmt): return "OUTPUT"
        if isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr): return f"INPUT {stmt.target.name}"
            if isinstance(stmt.target, ArrayAccessExpr): return f"INPUT {stmt.target.array}[...]"
            return "INPUT"
        if isinstance(stmt, IfStmt): return "IF"
        if isinstance(stmt, WhileStmt): return "WHILE"
        if isinstance(stmt, RepeatStmt): return "REPEAT"
        if isinstance(stmt, ForStmt): return f"FOR {stmt.identifier}"
        if isinstance(stmt, CaseStmt): return "CASE OF"
        if isinstance(stmt, ProcedureCallStmt): return f"CALL {stmt.name}"
        if isinstance(stmt, ReturnStmt): return "RETURN"
        if isinstance(stmt, ProcedureDecl): return f"PROCEDURE {stmt.name}"
        if isinstance(stmt, FunctionDecl): return f"FUNCTION {stmt.name}"
        if isinstance(stmt, TypeDecl): return f"TYPE {stmt.name}"
        if isinstance(stmt, ClassDecl): return f"CLASS {stmt.name}"
        if isinstance(stmt, FileStmt): return f"{stmt.operation}FILE"
        return type(stmt).__name__

    def _record_step(self, stmt, note=""):
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise InterpreterError(
                f"Dry-run stopped after {self.max_steps} steps (possible infinite loop)")
        entry = {
            'step': self.step_count,
            'line': getattr(stmt, 'line', self.current_line),
            'statement': self._describe_stmt(stmt),
            'note': note,
            'variables': self._snapshot_vars(),
        }
        self.trace.append(entry)

    # ── Override execute to record trace ──

    def execute(self, stmt: Stmt):
        if hasattr(stmt, 'line') and stmt.line > 0:
            self.current_line = stmt.line

        # Definitions — store but don't trace body execution
        if isinstance(stmt, (ProcedureDecl, FunctionDecl, TypeDecl, ClassDecl)):
            self._record_step(stmt, "defined")
            return super().execute(stmt)

        if isinstance(stmt, Expr):
            self._record_step(stmt, "expr")
            return super().execute(stmt)

        # Execute
        result = super().execute(stmt)

        # Build note about what changed
        note = ""
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            try:
                if isinstance(t, str):
                    v = self.symbol_table.get_cell(t).get()
                    note = f"= {self._format_val(v)}"
                elif isinstance(t, ArrayAccessExpr):
                    indices = [self.evaluate(idx) for idx in t.indices]
                    v = self.symbol_table.array_access(t.array, indices).get()
                    idx_s = ",".join(str(i) for i in indices)
                    note = f"[{idx_s}] = {self._format_val(v)}"
            except Exception:
                pass
        elif isinstance(stmt, DeclareStmt):
            note = stmt.type_name
        elif isinstance(stmt, OutputStmt):
            # Record what was output
            vals = [self._format_output(self.evaluate(a)) for a in stmt.values]
            output_text = "".join(vals)
            self.output_log.append(output_text)
            note = f'OUT: {output_text}'
        elif isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr):
                try:
                    v = self.symbol_table.get_cell(stmt.target.name).get()
                    note = f"= {self._format_val(v)}"
                except Exception:
                    pass
        elif isinstance(stmt, IfStmt):
            note = "condition"
        elif isinstance(stmt, ForStmt):
            note = f"loop {stmt.identifier}"

        self._record_step(stmt, note)
        return result

    # ── Override INPUT to use pre-supplied queue ──

    def visit_InputStmt(self, stmt: InputStmt):
        """Use pre-supplied input values instead of prompting."""
        if self.input_index < len(self.input_queue):
            val_str = str(self.input_queue[self.input_index])
            self.input_index += 1
        else:
            raise InterpreterError(
                "Dry-run ran out of pre-supplied input values. "
                f"Needed input #{self.input_index + 1} but only "
                f"{len(self.input_queue)} values were provided.")

        # Determine target type for coercion
        target_type = None
        try:
            if isinstance(stmt.target, VariableExpr):
                target_type = self.symbol_table.get_cell(stmt.target.name).type
            elif isinstance(stmt.target, str):
                target_type = self.symbol_table.get_cell(stmt.target).type
            elif isinstance(stmt.target, ArrayAccessExpr):
                cell = self.symbol_table.get_cell(stmt.target.array)
                if cell.is_array and cell.array_bounds:
                    target_type = cell.array_bounds.element_type
        except Exception:
            pass

        if target_type is not None:
            val, val_type = self._coerce_input(val_str, target_type)
        else:
            val, val_type = self._auto_parse_input(val_str)

        if isinstance(stmt.target, VariableExpr):
            self.symbol_table.assign(stmt.target.name, val, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            self.symbol_table.array_assign(stmt.target.array, indices, val, val_type)
        elif isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, val, val_type)

    # ── Results ──

    def get_all_var_names(self):
        names = set()
        for entry in self.trace:
            names.update(entry['variables'].keys())
        names -= set(self.procedures.keys())
        names -= set(self.functions.keys())
        return sorted(names)

    def format_trace_text(self):
        if not self.trace:
            return "No trace data recorded."

        var_names = self.get_all_var_names()
        headers = ['Step', 'Line', 'Statement', 'Note'] + var_names
        rows = []
        for entry in self.trace:
            row = [str(entry['step']), str(entry['line']),
                   entry['statement'], entry['note']]
            for vn in var_names:
                val = entry['variables'].get(vn, '')
                if isinstance(val, dict):
                    val = str(val)
                elif val == '':
                    val = ''
                else:
                    val = self._format_val(val) if not isinstance(val, str) else val
                row.append(val)
            rows.append(row)

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, c in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(c)))
        col_widths = [min(w, 30) for w in col_widths]

        def pad(s, w):
            return str(s)[:w].ljust(w)

        sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
        hdr = '|' + '|'.join(f" {pad(h, w)} " for h, w in zip(headers, col_widths)) + '|'
        lines = [sep, hdr, sep]
        for row in rows:
            r = '|' + '|'.join(f" {pad(c, w)} " for c, w in zip(row, col_widths)) + '|'
            lines.append(r)
        lines.append(sep)
        return '\n'.join(lines)
```

**2. Replace the entire [ide.py](ide.py):**

```python
"""
9618 Pseudocode IDE - Modern Desktop Interface
Tkinter-based GUI with dark theme, syntax highlighting, line numbers, and integrated output.
Includes A-Level exam-style Dry Run mode with pre-supplied inputs and trace table.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import sys
import io
import os
import threading
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexer import Lexer, LexerError
from parser import Parser, ParserError
from interpreter import Interpreter, InterpreterError, DryRunInterpreter
from symbol_table import SymbolTable

# -- Theme Colors (Catppuccin Mocha) --
COLORS = {
    "bg":           "#1e1e2e",
    "bg_secondary": "#181825",
    "bg_tertiary":  "#11111b",
    "surface":      "#313244",
    "overlay":      "#45475a",
    "text":         "#cdd6f4",
    "subtext":      "#a6adc8",
    "blue":         "#89b4fa",
    "green":        "#a6e3a1",
    "red":          "#f38ba8",
    "yellow":       "#f9e2af",
    "mauve":        "#cba6f7",
    "peach":        "#fab387",
    "teal":         "#94e2d5",
    "pink":         "#f5c2e7",
    "lavender":     "#b4befe",
    "sky":          "#89dceb",
    "sapphire":     "#74c7ec",
    "line_num_fg":  "#585b70",
    "selection":    "#45475a",
    "cursor":       "#f5e0dc",
    "gutter":       "#282a3a",
    "output_bg":    "#11111b",
    "toolbar_bg":   "#181825",
    "status_bg":    "#181825",
    "tab_active":   "#1e1e2e",
    "accent":       "#89b4fa",
    "error":        "#f38ba8",
    "success":      "#a6e3a1",
    "warning":      "#f9e2af",
    "button_bg":    "#313244",
    "button_hover": "#45475a",
    "border":       "#313244",
}

# -- Keyword Lists for Highlighting --
KEYWORDS_CONTROL = {
    'IF', 'THEN', 'ELSE', 'ENDIF', 'CASE', 'OF', 'OTHERWISE', 'ENDCASE',
    'FOR', 'TO', 'STEP', 'NEXT', 'WHILE', 'DO', 'ENDWHILE',
    'REPEAT', 'UNTIL', 'RETURN',
}
KEYWORDS_DECL = {
    'DECLARE', 'CONSTANT', 'TYPE', 'ENDTYPE', 'ARRAY',
    'PROCEDURE', 'ENDPROCEDURE', 'FUNCTION', 'ENDFUNCTION',
    'RETURNS', 'BYREF', 'BYVAL', 'CALL',
}
KEYWORDS_TYPE = {'INTEGER', 'REAL', 'STRING', 'BOOLEAN', 'CHAR', 'DATE'}
KEYWORDS_IO = {
    'INPUT', 'OUTPUT', 'OPENFILE', 'READFILE', 'WRITEFILE', 'CLOSEFILE',
    'READ', 'WRITE', 'APPEND',
}
KEYWORDS_OP = {'AND', 'OR', 'NOT', 'DIV', 'MOD', 'TRUE', 'FALSE'}
BUILTINS = {
    'LENGTH', 'UCASE', 'LCASE', 'LEFT', 'RIGHT', 'MID',
    'INT', 'NUM_TO_STR', 'STR_TO_NUM', 'ASC', 'CHR', 'SQRT', 'RAND', 'EOF',
}


class LineNumbers(tk.Canvas):
    def __init__(self, parent, text_widget, **kwargs):
        super().__init__(parent, **kwargs)
        self.text_widget = text_widget
        self.font = None

    def redraw(self, *_args):
        self.delete("all")
        if self.text_widget is None:
            return
        i = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.create_text(
                self.winfo_width() - 8, y, anchor="ne", text=linenum,
                font=self.font, fill=COLORS["line_num_fg"],
            )
            i = self.text_widget.index(f"{i}+1line")
            if self.text_widget.compare(i, ">=", "end"):
                break


class CodeEditor(tk.Text):
    TAG_CONFIG = {
        "keyword_ctrl":  {"foreground": COLORS["mauve"],   "font_style": "bold"},
        "keyword_decl":  {"foreground": COLORS["blue"],    "font_style": "bold"},
        "keyword_type":  {"foreground": COLORS["yellow"]},
        "keyword_io":    {"foreground": COLORS["green"],   "font_style": "bold"},
        "keyword_op":    {"foreground": COLORS["peach"],   "font_style": "bold"},
        "builtin":       {"foreground": COLORS["sapphire"]},
        "string":        {"foreground": COLORS["green"]},
        "char":          {"foreground": COLORS["teal"]},
        "number":        {"foreground": COLORS["peach"]},
        "comment":       {"foreground": COLORS["overlay"], "font_style": "italic"},
        "operator":      {"foreground": COLORS["sky"]},
        "assign_arrow":  {"foreground": COLORS["red"]},
        "error_line":    {"background": "#3d2030"},
    }

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._setup_tags()
        self.bind("<<Modified>>", self._on_modify)
        self._highlight_job = None

    def _setup_tags(self):
        base_font = self.cget("font")
        for tag, cfg in self.TAG_CONFIG.items():
            opts = {"foreground": cfg.get("foreground", COLORS["text"])}
            if "background" in cfg:
                opts["background"] = cfg["background"]
            style = cfg.get("font_style", "")
            if style:
                fam = tkfont.Font(font=base_font).actual()["family"]
                sz = tkfont.Font(font=base_font).actual()["size"]
                weight = "bold" if "bold" in style else "normal"
                slant = "italic" if "italic" in style else "roman"
                opts["font"] = tkfont.Font(family=fam, size=sz, weight=weight, slant=slant)
            self.tag_configure(tag, **opts)
        self.tag_raise("error_line")

    def _on_modify(self, _event=None):
        if self.edit_modified():
            if self._highlight_job:
                self.after_cancel(self._highlight_job)
            self._highlight_job = self.after(80, self.highlight_syntax)
            self.edit_modified(False)
            self.event_generate("<<ContentChanged>>")

    def highlight_syntax(self):
        for tag in self.TAG_CONFIG:
            self.tag_remove(tag, "1.0", "end")
        code = self.get("1.0", "end-1c")
        patterns = [
            ("comment",      r'//[^\n]*'),
            ("string",       r'"[^"]*"'),
            ("char",         r"'[^']*'"),
            ("assign_arrow", r'<-'),
            ("number",       r'\b\d+\.\d+\b|\b\d+\b'),
            ("operator",     r'<>|<=|>=|[<>=+\-*/&]'),
        ]
        for tag, pat in patterns:
            for m in re.finditer(pat, code):
                start = f"1.0+{m.start()}c"
                end = f"1.0+{m.end()}c"
                self.tag_add(tag, start, end)
        word_groups = [
            ("keyword_ctrl", KEYWORDS_CONTROL),
            ("keyword_decl", KEYWORDS_DECL),
            ("keyword_type", KEYWORDS_TYPE),
            ("keyword_io",   KEYWORDS_IO),
            ("keyword_op",   KEYWORDS_OP),
            ("builtin",      BUILTINS),
        ]
        for tag, words in word_groups:
            for w in words:
                pat = rf'\b{w}\b'
                for m in re.finditer(pat, code):
                    start = f"1.0+{m.start()}c"
                    end = f"1.0+{m.end()}c"
                    tags_at = self.tag_names(start)
                    if "string" not in tags_at and "comment" not in tags_at and "char" not in tags_at:
                        self.tag_add(tag, start, end)

    def mark_error_line(self, line_num):
        self.tag_remove("error_line", "1.0", "end")
        if line_num and line_num > 0:
            self.tag_add("error_line", f"{line_num}.0", f"{line_num}.end+1c")
            self.see(f"{line_num}.0")


class OutputPanel(tk.Text):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(state="disabled")
        self.tag_configure("error",        foreground=COLORS["error"])
        self.tag_configure("success",      foreground=COLORS["success"])
        self.tag_configure("info",         foreground=COLORS["subtext"])
        self.tag_configure("output",       foreground=COLORS["text"])
        self.tag_configure("input_prompt", foreground=COLORS["yellow"])

    def append(self, text, tag="output"):
        self.configure(state="normal")
        self.insert("end", text, tag)
        self.see("end")
        self.configure(state="disabled")

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class InputDialog(tk.Toplevel):
    def __init__(self, parent, prompt):
        super().__init__(parent)
        self.title("Input Required")
        self.result = None
        self.configure(bg=COLORS["bg"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 400, 160
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        frame = tk.Frame(self, bg=COLORS["bg"], padx=20, pady=15)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text=prompt, font=("Segoe UI", 11),
                 fg=COLORS["yellow"], bg=COLORS["bg"], anchor="w").pack(fill="x", pady=(0, 8))

        self.entry = tk.Entry(frame, font=("Cascadia Code", 12),
                              bg=COLORS["surface"], fg=COLORS["text"],
                              insertbackground=COLORS["cursor"], relief="flat", bd=0)
        self.entry.pack(fill="x", ipady=6)
        self.entry.focus_set()
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Escape>", lambda e: self._cancel())

        btn_frame = tk.Frame(frame, bg=COLORS["bg"])
        btn_frame.pack(fill="x", pady=(12, 0))

        tk.Button(btn_frame, text="OK", font=("Segoe UI", 10),
                  bg=COLORS["accent"], fg=COLORS["bg_tertiary"],
                  activebackground=COLORS["lavender"],
                  relief="flat", bd=0, padx=20, pady=4,
                  command=self._submit).pack(side="right")
        tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 10),
                  bg=COLORS["surface"], fg=COLORS["text"],
                  activebackground=COLORS["overlay"],
                  relief="flat", bd=0, padx=16, pady=4,
                  command=self._cancel).pack(side="right", padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _submit(self, _event=None):
        self.result = self.entry.get()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class RedirectOutput:
    def __init__(self, panel, tag="output"):
        self.panel = panel
        self.tag = tag

    def write(self, text):
        if text:
            self.panel.after(0, self.panel.append, text, self.tag)

    def flush(self):
        pass


# ──────────────── Dry Run Setup Dialog ────────────────

class DryRunSetupDialog(tk.Toplevel):
    """
    A-Level exam-style dry run setup.
    Shows detected INPUT statements, lets user supply values upfront,
    and choose which variables to include in the trace table.
    """

    def __init__(self, parent, input_info, declare_info):
        super().__init__(parent)
        self.title("Dry Run Setup")
        self.result = None  # Will be {'inputs': [...], 'traced_vars': set or None}
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 520, 560
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        self.minsize(400, 400)

        self.input_info = input_info    # [{'line': int, 'variable': str}, ...]
        self.declare_info = declare_info  # [{'name': str, 'type': str, 'is_array': bool}, ...]
        self.input_entries = []         # tk.Entry widgets for each input
        self.var_checkboxes = {}        # {name: tk.BooleanVar}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _build_ui(self):
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # ── Title ──
        tk.Label(main, text="Dry Run Setup",
                 font=("Segoe UI", 14, "bold"), fg=COLORS["accent"],
                 bg=COLORS["bg"]).pack(anchor="w", pady=(0, 4))
        tk.Label(main, text="Supply input values and select variables to trace,\n"
                            "just like a Cambridge 9618 exam trace table.",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"], justify="left").pack(anchor="w", pady=(0, 12))

        # ── Input Values Section ──
        input_frame = tk.LabelFrame(main, text=" Input Values ",
                                     font=("Segoe UI", 10, "bold"),
                                     fg=COLORS["yellow"], bg=COLORS["bg"],
                                     bd=1, relief="solid",
                                     highlightbackground=COLORS["surface"])
        input_frame.pack(fill="x", pady=(0, 12))

        if self.input_info:
            tk.Label(input_frame,
                     text="Enter values in order (as given on an exam paper):",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

            for i, info in enumerate(self.input_info):
                row = tk.Frame(input_frame, bg=COLORS["bg"])
                row.pack(fill="x", padx=10, pady=2)

                label_text = f"Input #{i+1}  (Line {info['line']}: {info['variable']})"
                tk.Label(row, text=label_text, font=("Segoe UI", 9),
                         fg=COLORS["text"], bg=COLORS["bg"], width=35,
                         anchor="w").pack(side="left")

                entry = tk.Entry(row, font=("Cascadia Code", 11),
                                 bg=COLORS["surface"], fg=COLORS["text"],
                                 insertbackground=COLORS["cursor"],
                                 relief="flat", bd=0, width=20)
                entry.pack(side="left", padx=(4, 0), ipady=3)
                self.input_entries.append(entry)

            # Also allow a bulk entry box
            tk.Label(input_frame,
                     text="Or enter all values comma-separated:",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(8, 2))

            self.bulk_entry = tk.Entry(input_frame, font=("Cascadia Code", 11),
                                        bg=COLORS["surface"], fg=COLORS["text"],
                                        insertbackground=COLORS["cursor"],
                                        relief="flat", bd=0)
            self.bulk_entry.pack(fill="x", padx=10, pady=(0, 8), ipady=4)
            self.bulk_entry.insert(0, "e.g. 5, 3, 8")
            self.bulk_entry.bind("<FocusIn>", self._clear_placeholder)
        else:
            tk.Label(input_frame,
                     text="No INPUT statements detected in this program.",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(padx=10, pady=8)
            self.bulk_entry = None

        # ── Variables to Trace Section ──
        var_frame = tk.LabelFrame(main, text=" Trace Table Columns ",
                                   font=("Segoe UI", 10, "bold"),
                                   fg=COLORS["teal"], bg=COLORS["bg"],
                                   bd=1, relief="solid",
                                   highlightbackground=COLORS["surface"])
        var_frame.pack(fill="both", expand=True, pady=(0, 12))

        tk.Label(var_frame,
                 text="Select which variables to show in the trace table:",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

        # Scrollable checkbox area
        canvas_frame = tk.Frame(var_frame, bg=COLORS["bg"])
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        canvas = tk.Canvas(canvas_frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        cb
```


## License: unknown
https://github.com/cstroup/business-planning/blob/c48782917b7971b53d1174d4c05aa4463862e170/planning_app.py

```
Now I have full context. Let me build the complete solution. The approach:

1. **Scan the AST** to find all `InputStmt` nodes and extract variable names — no LLM needed
2. **Before dry-run starts**, show a setup dialog where the user supplies:
   - All input values upfront (comma-separated, like exam papers give them)
   - Which variables to trace (auto-detected from DECLARE statements, user can toggle)
3. **During execution**, feed inputs from the pre-supplied queue instead of prompting
4. **After execution**, show the trace table in exact A-Level format

Here are the three files to update:

**1. Append to the end of [interpreter.py](interpreter.py) — the `DryRunInterpreter` class:**

```python

# ──────────────── Dry-Run / Trace Mode ────────────────

class DryRunInterpreter(Interpreter):
    """
    Cambridge 9618-style dry-run interpreter.
    - Accepts pre-supplied input values (like exam papers provide)
    - Records a trace table of selected variables at each step
    - Feeds inputs from queue instead of prompting interactively
    """

    def __init__(self, symbol_table: SymbolTable, input_queue=None,
                 traced_vars=None, max_steps=5000):
        super().__init__(symbol_table)
        self.trace = []
        self.step_count = 0
        self.max_steps = max_steps
        self.input_queue = list(input_queue) if input_queue else []
        self.input_index = 0
        self.traced_vars = set(traced_vars) if traced_vars else None  # None = auto-detect all
        self.output_log = []  # record OUTPUT values too

    # ── AST scanning helpers (static, no execution needed) ──

    @staticmethod
    def scan_inputs(statements):
        """Walk the AST to find all INPUT statements and return info about them.
        Returns list of dicts: {'line': int, 'variable': str}
        """
        results = []
        DryRunInterpreter._walk_for_inputs(statements, results)
        return results

    @staticmethod
    def _walk_for_inputs(stmts, results):
        for stmt in stmts:
            if isinstance(stmt, InputStmt):
                var_name = "?"
                if isinstance(stmt.target, VariableExpr):
                    var_name = stmt.target.name
                elif isinstance(stmt.target, ArrayAccessExpr):
                    var_name = stmt.target.array + "[...]"
                elif isinstance(stmt.target, str):
                    var_name = stmt.target
                line = getattr(stmt, 'line', 0)
                results.append({'line': line, 'variable': var_name})
            # Recurse into compound statements
            if isinstance(stmt, IfStmt):
                DryRunInterpreter._walk_for_inputs(stmt.then_branch, results)
                if stmt.else_branch:
                    DryRunInterpreter._walk_for_inputs(stmt.else_branch, results)
            elif isinstance(stmt, WhileStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, RepeatStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, ForStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, CaseStmt):
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

    @staticmethod
    def scan_declares(statements):
        """Walk the AST to find all DECLARE statements.
        Returns list of dicts: {'name': str, 'type': str, 'is_array': bool}
        """
        results = []
        for stmt in statements:
            if isinstance(stmt, DeclareStmt):
                results.append({
                    'name': stmt.name,
                    'type': stmt.type_name,
                    'is_array': stmt.is_array
                })
            elif isinstance(stmt, ConstantDecl):
                results.append({
                    'name': stmt.name,
                    'type': 'CONSTANT',
                    'is_array': False
                })
        return results

    # ── Snapshot & trace recording ──

    def _snapshot_vars(self):
        """Capture values of traced variables only."""
        snapshot = {}
        for scope_level in range(self.symbol_table.scope_level + 1):
            scope = self.symbol_table.scopes.get(scope_level, {})
            for name, sym in scope.items():
                # If user specified which vars to trace, filter
                if self.traced_vars is not None and name not in self.traced_vars:
                    continue
                cell = sym.cell
                if cell.is_array:
                    if cell.array_elements:
                        arr = {}
                        for key, ec in sorted(cell.array_elements.items()):
                            idx = ",".join(str(k) for k in key)
                            arr[f"[{idx}]"] = ec.get()
                        snapshot[name] = arr
                    else:
                        snapshot[name] = ""
                elif cell.type == DataType.RECORD and isinstance(cell.value, dict):
                    rec = {}
                    for fn, fc in cell.value.items():
                        rec[fn] = fc.get() if isinstance(fc, Cell) else fc
                    snapshot[name] = rec
                elif isinstance(cell.value, PseudocodeObject):
                    attrs = {}
                    for an, ac in cell.value.attributes.items():
                        attrs[an] = ac.get() if isinstance(ac, Cell) else ac
                    snapshot[name] = f"<{cell.value.class_name}>{attrs}"
                else:
                    try:
                        snapshot[name] = cell.get()
                    except Exception:
                        snapshot[name] = cell.value
        return snapshot

    def _format_val(self, val):
        if isinstance(val, bool): return "TRUE" if val else "FALSE"
        if isinstance(val, str): return f'"{val}"'
        return str(val)

    def _describe_stmt(self, stmt):
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            if isinstance(t, str): return f"{t} \u2190 ..."
            if isinstance(t, ArrayAccessExpr): return f"{t.array}[...] \u2190 ..."
            if isinstance(t, MemberExpr): return f".{t.field} \u2190 ..."
            return "ASSIGN"
        if isinstance(stmt, DeclareStmt):
            arr = "ARRAY" if stmt.is_array else ""
            return f"DECLARE {stmt.name} : {arr} {stmt.type_name}".strip()
        if isinstance(stmt, ConstantDecl): return f"CONSTANT {stmt.name}"
        if isinstance(stmt, OutputStmt): return "OUTPUT"
        if isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr): return f"INPUT {stmt.target.name}"
            if isinstance(stmt.target, ArrayAccessExpr): return f"INPUT {stmt.target.array}[...]"
            return "INPUT"
        if isinstance(stmt, IfStmt): return "IF"
        if isinstance(stmt, WhileStmt): return "WHILE"
        if isinstance(stmt, RepeatStmt): return "REPEAT"
        if isinstance(stmt, ForStmt): return f"FOR {stmt.identifier}"
        if isinstance(stmt, CaseStmt): return "CASE OF"
        if isinstance(stmt, ProcedureCallStmt): return f"CALL {stmt.name}"
        if isinstance(stmt, ReturnStmt): return "RETURN"
        if isinstance(stmt, ProcedureDecl): return f"PROCEDURE {stmt.name}"
        if isinstance(stmt, FunctionDecl): return f"FUNCTION {stmt.name}"
        if isinstance(stmt, TypeDecl): return f"TYPE {stmt.name}"
        if isinstance(stmt, ClassDecl): return f"CLASS {stmt.name}"
        if isinstance(stmt, FileStmt): return f"{stmt.operation}FILE"
        return type(stmt).__name__

    def _record_step(self, stmt, note=""):
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise InterpreterError(
                f"Dry-run stopped after {self.max_steps} steps (possible infinite loop)")
        entry = {
            'step': self.step_count,
            'line': getattr(stmt, 'line', self.current_line),
            'statement': self._describe_stmt(stmt),
            'note': note,
            'variables': self._snapshot_vars(),
        }
        self.trace.append(entry)

    # ── Override execute to record trace ──

    def execute(self, stmt: Stmt):
        if hasattr(stmt, 'line') and stmt.line > 0:
            self.current_line = stmt.line

        # Definitions — store but don't trace body execution
        if isinstance(stmt, (ProcedureDecl, FunctionDecl, TypeDecl, ClassDecl)):
            self._record_step(stmt, "defined")
            return super().execute(stmt)

        if isinstance(stmt, Expr):
            self._record_step(stmt, "expr")
            return super().execute(stmt)

        # Execute
        result = super().execute(stmt)

        # Build note about what changed
        note = ""
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            try:
                if isinstance(t, str):
                    v = self.symbol_table.get_cell(t).get()
                    note = f"= {self._format_val(v)}"
                elif isinstance(t, ArrayAccessExpr):
                    indices = [self.evaluate(idx) for idx in t.indices]
                    v = self.symbol_table.array_access(t.array, indices).get()
                    idx_s = ",".join(str(i) for i in indices)
                    note = f"[{idx_s}] = {self._format_val(v)}"
            except Exception:
                pass
        elif isinstance(stmt, DeclareStmt):
            note = stmt.type_name
        elif isinstance(stmt, OutputStmt):
            # Record what was output
            vals = [self._format_output(self.evaluate(a)) for a in stmt.values]
            output_text = "".join(vals)
            self.output_log.append(output_text)
            note = f'OUT: {output_text}'
        elif isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr):
                try:
                    v = self.symbol_table.get_cell(stmt.target.name).get()
                    note = f"= {self._format_val(v)}"
                except Exception:
                    pass
        elif isinstance(stmt, IfStmt):
            note = "condition"
        elif isinstance(stmt, ForStmt):
            note = f"loop {stmt.identifier}"

        self._record_step(stmt, note)
        return result

    # ── Override INPUT to use pre-supplied queue ──

    def visit_InputStmt(self, stmt: InputStmt):
        """Use pre-supplied input values instead of prompting."""
        if self.input_index < len(self.input_queue):
            val_str = str(self.input_queue[self.input_index])
            self.input_index += 1
        else:
            raise InterpreterError(
                "Dry-run ran out of pre-supplied input values. "
                f"Needed input #{self.input_index + 1} but only "
                f"{len(self.input_queue)} values were provided.")

        # Determine target type for coercion
        target_type = None
        try:
            if isinstance(stmt.target, VariableExpr):
                target_type = self.symbol_table.get_cell(stmt.target.name).type
            elif isinstance(stmt.target, str):
                target_type = self.symbol_table.get_cell(stmt.target).type
            elif isinstance(stmt.target, ArrayAccessExpr):
                cell = self.symbol_table.get_cell(stmt.target.array)
                if cell.is_array and cell.array_bounds:
                    target_type = cell.array_bounds.element_type
        except Exception:
            pass

        if target_type is not None:
            val, val_type = self._coerce_input(val_str, target_type)
        else:
            val, val_type = self._auto_parse_input(val_str)

        if isinstance(stmt.target, VariableExpr):
            self.symbol_table.assign(stmt.target.name, val, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            self.symbol_table.array_assign(stmt.target.array, indices, val, val_type)
        elif isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, val, val_type)

    # ── Results ──

    def get_all_var_names(self):
        names = set()
        for entry in self.trace:
            names.update(entry['variables'].keys())
        names -= set(self.procedures.keys())
        names -= set(self.functions.keys())
        return sorted(names)

    def format_trace_text(self):
        if not self.trace:
            return "No trace data recorded."

        var_names = self.get_all_var_names()
        headers = ['Step', 'Line', 'Statement', 'Note'] + var_names
        rows = []
        for entry in self.trace:
            row = [str(entry['step']), str(entry['line']),
                   entry['statement'], entry['note']]
            for vn in var_names:
                val = entry['variables'].get(vn, '')
                if isinstance(val, dict):
                    val = str(val)
                elif val == '':
                    val = ''
                else:
                    val = self._format_val(val) if not isinstance(val, str) else val
                row.append(val)
            rows.append(row)

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, c in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(c)))
        col_widths = [min(w, 30) for w in col_widths]

        def pad(s, w):
            return str(s)[:w].ljust(w)

        sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
        hdr = '|' + '|'.join(f" {pad(h, w)} " for h, w in zip(headers, col_widths)) + '|'
        lines = [sep, hdr, sep]
        for row in rows:
            r = '|' + '|'.join(f" {pad(c, w)} " for c, w in zip(row, col_widths)) + '|'
            lines.append(r)
        lines.append(sep)
        return '\n'.join(lines)
```

**2. Replace the entire [ide.py](ide.py):**

```python
"""
9618 Pseudocode IDE - Modern Desktop Interface
Tkinter-based GUI with dark theme, syntax highlighting, line numbers, and integrated output.
Includes A-Level exam-style Dry Run mode with pre-supplied inputs and trace table.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import sys
import io
import os
import threading
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexer import Lexer, LexerError
from parser import Parser, ParserError
from interpreter import Interpreter, InterpreterError, DryRunInterpreter
from symbol_table import SymbolTable

# -- Theme Colors (Catppuccin Mocha) --
COLORS = {
    "bg":           "#1e1e2e",
    "bg_secondary": "#181825",
    "bg_tertiary":  "#11111b",
    "surface":      "#313244",
    "overlay":      "#45475a",
    "text":         "#cdd6f4",
    "subtext":      "#a6adc8",
    "blue":         "#89b4fa",
    "green":        "#a6e3a1",
    "red":          "#f38ba8",
    "yellow":       "#f9e2af",
    "mauve":        "#cba6f7",
    "peach":        "#fab387",
    "teal":         "#94e2d5",
    "pink":         "#f5c2e7",
    "lavender":     "#b4befe",
    "sky":          "#89dceb",
    "sapphire":     "#74c7ec",
    "line_num_fg":  "#585b70",
    "selection":    "#45475a",
    "cursor":       "#f5e0dc",
    "gutter":       "#282a3a",
    "output_bg":    "#11111b",
    "toolbar_bg":   "#181825",
    "status_bg":    "#181825",
    "tab_active":   "#1e1e2e",
    "accent":       "#89b4fa",
    "error":        "#f38ba8",
    "success":      "#a6e3a1",
    "warning":      "#f9e2af",
    "button_bg":    "#313244",
    "button_hover": "#45475a",
    "border":       "#313244",
}

# -- Keyword Lists for Highlighting --
KEYWORDS_CONTROL = {
    'IF', 'THEN', 'ELSE', 'ENDIF', 'CASE', 'OF', 'OTHERWISE', 'ENDCASE',
    'FOR', 'TO', 'STEP', 'NEXT', 'WHILE', 'DO', 'ENDWHILE',
    'REPEAT', 'UNTIL', 'RETURN',
}
KEYWORDS_DECL = {
    'DECLARE', 'CONSTANT', 'TYPE', 'ENDTYPE', 'ARRAY',
    'PROCEDURE', 'ENDPROCEDURE', 'FUNCTION', 'ENDFUNCTION',
    'RETURNS', 'BYREF', 'BYVAL', 'CALL',
}
KEYWORDS_TYPE = {'INTEGER', 'REAL', 'STRING', 'BOOLEAN', 'CHAR', 'DATE'}
KEYWORDS_IO = {
    'INPUT', 'OUTPUT', 'OPENFILE', 'READFILE', 'WRITEFILE', 'CLOSEFILE',
    'READ', 'WRITE', 'APPEND',
}
KEYWORDS_OP = {'AND', 'OR', 'NOT', 'DIV', 'MOD', 'TRUE', 'FALSE'}
BUILTINS = {
    'LENGTH', 'UCASE', 'LCASE', 'LEFT', 'RIGHT', 'MID',
    'INT', 'NUM_TO_STR', 'STR_TO_NUM', 'ASC', 'CHR', 'SQRT', 'RAND', 'EOF',
}


class LineNumbers(tk.Canvas):
    def __init__(self, parent, text_widget, **kwargs):
        super().__init__(parent, **kwargs)
        self.text_widget = text_widget
        self.font = None

    def redraw(self, *_args):
        self.delete("all")
        if self.text_widget is None:
            return
        i = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.create_text(
                self.winfo_width() - 8, y, anchor="ne", text=linenum,
                font=self.font, fill=COLORS["line_num_fg"],
            )
            i = self.text_widget.index(f"{i}+1line")
            if self.text_widget.compare(i, ">=", "end"):
                break


class CodeEditor(tk.Text):
    TAG_CONFIG = {
        "keyword_ctrl":  {"foreground": COLORS["mauve"],   "font_style": "bold"},
        "keyword_decl":  {"foreground": COLORS["blue"],    "font_style": "bold"},
        "keyword_type":  {"foreground": COLORS["yellow"]},
        "keyword_io":    {"foreground": COLORS["green"],   "font_style": "bold"},
        "keyword_op":    {"foreground": COLORS["peach"],   "font_style": "bold"},
        "builtin":       {"foreground": COLORS["sapphire"]},
        "string":        {"foreground": COLORS["green"]},
        "char":          {"foreground": COLORS["teal"]},
        "number":        {"foreground": COLORS["peach"]},
        "comment":       {"foreground": COLORS["overlay"], "font_style": "italic"},
        "operator":      {"foreground": COLORS["sky"]},
        "assign_arrow":  {"foreground": COLORS["red"]},
        "error_line":    {"background": "#3d2030"},
    }

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._setup_tags()
        self.bind("<<Modified>>", self._on_modify)
        self._highlight_job = None

    def _setup_tags(self):
        base_font = self.cget("font")
        for tag, cfg in self.TAG_CONFIG.items():
            opts = {"foreground": cfg.get("foreground", COLORS["text"])}
            if "background" in cfg:
                opts["background"] = cfg["background"]
            style = cfg.get("font_style", "")
            if style:
                fam = tkfont.Font(font=base_font).actual()["family"]
                sz = tkfont.Font(font=base_font).actual()["size"]
                weight = "bold" if "bold" in style else "normal"
                slant = "italic" if "italic" in style else "roman"
                opts["font"] = tkfont.Font(family=fam, size=sz, weight=weight, slant=slant)
            self.tag_configure(tag, **opts)
        self.tag_raise("error_line")

    def _on_modify(self, _event=None):
        if self.edit_modified():
            if self._highlight_job:
                self.after_cancel(self._highlight_job)
            self._highlight_job = self.after(80, self.highlight_syntax)
            self.edit_modified(False)
            self.event_generate("<<ContentChanged>>")

    def highlight_syntax(self):
        for tag in self.TAG_CONFIG:
            self.tag_remove(tag, "1.0", "end")
        code = self.get("1.0", "end-1c")
        patterns = [
            ("comment",      r'//[^\n]*'),
            ("string",       r'"[^"]*"'),
            ("char",         r"'[^']*'"),
            ("assign_arrow", r'<-'),
            ("number",       r'\b\d+\.\d+\b|\b\d+\b'),
            ("operator",     r'<>|<=|>=|[<>=+\-*/&]'),
        ]
        for tag, pat in patterns:
            for m in re.finditer(pat, code):
                start = f"1.0+{m.start()}c"
                end = f"1.0+{m.end()}c"
                self.tag_add(tag, start, end)
        word_groups = [
            ("keyword_ctrl", KEYWORDS_CONTROL),
            ("keyword_decl", KEYWORDS_DECL),
            ("keyword_type", KEYWORDS_TYPE),
            ("keyword_io",   KEYWORDS_IO),
            ("keyword_op",   KEYWORDS_OP),
            ("builtin",      BUILTINS),
        ]
        for tag, words in word_groups:
            for w in words:
                pat = rf'\b{w}\b'
                for m in re.finditer(pat, code):
                    start = f"1.0+{m.start()}c"
                    end = f"1.0+{m.end()}c"
                    tags_at = self.tag_names(start)
                    if "string" not in tags_at and "comment" not in tags_at and "char" not in tags_at:
                        self.tag_add(tag, start, end)

    def mark_error_line(self, line_num):
        self.tag_remove("error_line", "1.0", "end")
        if line_num and line_num > 0:
            self.tag_add("error_line", f"{line_num}.0", f"{line_num}.end+1c")
            self.see(f"{line_num}.0")


class OutputPanel(tk.Text):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(state="disabled")
        self.tag_configure("error",        foreground=COLORS["error"])
        self.tag_configure("success",      foreground=COLORS["success"])
        self.tag_configure("info",         foreground=COLORS["subtext"])
        self.tag_configure("output",       foreground=COLORS["text"])
        self.tag_configure("input_prompt", foreground=COLORS["yellow"])

    def append(self, text, tag="output"):
        self.configure(state="normal")
        self.insert("end", text, tag)
        self.see("end")
        self.configure(state="disabled")

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class InputDialog(tk.Toplevel):
    def __init__(self, parent, prompt):
        super().__init__(parent)
        self.title("Input Required")
        self.result = None
        self.configure(bg=COLORS["bg"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 400, 160
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        frame = tk.Frame(self, bg=COLORS["bg"], padx=20, pady=15)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text=prompt, font=("Segoe UI", 11),
                 fg=COLORS["yellow"], bg=COLORS["bg"], anchor="w").pack(fill="x", pady=(0, 8))

        self.entry = tk.Entry(frame, font=("Cascadia Code", 12),
                              bg=COLORS["surface"], fg=COLORS["text"],
                              insertbackground=COLORS["cursor"], relief="flat", bd=0)
        self.entry.pack(fill="x", ipady=6)
        self.entry.focus_set()
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Escape>", lambda e: self._cancel())

        btn_frame = tk.Frame(frame, bg=COLORS["bg"])
        btn_frame.pack(fill="x", pady=(12, 0))

        tk.Button(btn_frame, text="OK", font=("Segoe UI", 10),
                  bg=COLORS["accent"], fg=COLORS["bg_tertiary"],
                  activebackground=COLORS["lavender"],
                  relief="flat", bd=0, padx=20, pady=4,
                  command=self._submit).pack(side="right")
        tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 10),
                  bg=COLORS["surface"], fg=COLORS["text"],
                  activebackground=COLORS["overlay"],
                  relief="flat", bd=0, padx=16, pady=4,
                  command=self._cancel).pack(side="right", padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _submit(self, _event=None):
        self.result = self.entry.get()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class RedirectOutput:
    def __init__(self, panel, tag="output"):
        self.panel = panel
        self.tag = tag

    def write(self, text):
        if text:
            self.panel.after(0, self.panel.append, text, self.tag)

    def flush(self):
        pass


# ──────────────── Dry Run Setup Dialog ────────────────

class DryRunSetupDialog(tk.Toplevel):
    """
    A-Level exam-style dry run setup.
    Shows detected INPUT statements, lets user supply values upfront,
    and choose which variables to include in the trace table.
    """

    def __init__(self, parent, input_info, declare_info):
        super().__init__(parent)
        self.title("Dry Run Setup")
        self.result = None  # Will be {'inputs': [...], 'traced_vars': set or None}
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 520, 560
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        self.minsize(400, 400)

        self.input_info = input_info    # [{'line': int, 'variable': str}, ...]
        self.declare_info = declare_info  # [{'name': str, 'type': str, 'is_array': bool}, ...]
        self.input_entries = []         # tk.Entry widgets for each input
        self.var_checkboxes = {}        # {name: tk.BooleanVar}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _build_ui(self):
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # ── Title ──
        tk.Label(main, text="Dry Run Setup",
                 font=("Segoe UI", 14, "bold"), fg=COLORS["accent"],
                 bg=COLORS["bg"]).pack(anchor="w", pady=(0, 4))
        tk.Label(main, text="Supply input values and select variables to trace,\n"
                            "just like a Cambridge 9618 exam trace table.",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"], justify="left").pack(anchor="w", pady=(0, 12))

        # ── Input Values Section ──
        input_frame = tk.LabelFrame(main, text=" Input Values ",
                                     font=("Segoe UI", 10, "bold"),
                                     fg=COLORS["yellow"], bg=COLORS["bg"],
                                     bd=1, relief="solid",
                                     highlightbackground=COLORS["surface"])
        input_frame.pack(fill="x", pady=(0, 12))

        if self.input_info:
            tk.Label(input_frame,
                     text="Enter values in order (as given on an exam paper):",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

            for i, info in enumerate(self.input_info):
                row = tk.Frame(input_frame, bg=COLORS["bg"])
                row.pack(fill="x", padx=10, pady=2)

                label_text = f"Input #{i+1}  (Line {info['line']}: {info['variable']})"
                tk.Label(row, text=label_text, font=("Segoe UI", 9),
                         fg=COLORS["text"], bg=COLORS["bg"], width=35,
                         anchor="w").pack(side="left")

                entry = tk.Entry(row, font=("Cascadia Code", 11),
                                 bg=COLORS["surface"], fg=COLORS["text"],
                                 insertbackground=COLORS["cursor"],
                                 relief="flat", bd=0, width=20)
                entry.pack(side="left", padx=(4, 0), ipady=3)
                self.input_entries.append(entry)

            # Also allow a bulk entry box
            tk.Label(input_frame,
                     text="Or enter all values comma-separated:",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(8, 2))

            self.bulk_entry = tk.Entry(input_frame, font=("Cascadia Code", 11),
                                        bg=COLORS["surface"], fg=COLORS["text"],
                                        insertbackground=COLORS["cursor"],
                                        relief="flat", bd=0)
            self.bulk_entry.pack(fill="x", padx=10, pady=(0, 8), ipady=4)
            self.bulk_entry.insert(0, "e.g. 5, 3, 8")
            self.bulk_entry.bind("<FocusIn>", self._clear_placeholder)
        else:
            tk.Label(input_frame,
                     text="No INPUT statements detected in this program.",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(padx=10, pady=8)
            self.bulk_entry = None

        # ── Variables to Trace Section ──
        var_frame = tk.LabelFrame(main, text=" Trace Table Columns ",
                                   font=("Segoe UI", 10, "bold"),
                                   fg=COLORS["teal"], bg=COLORS["bg"],
                                   bd=1, relief="solid",
                                   highlightbackground=COLORS["surface"])
        var_frame.pack(fill="both", expand=True, pady=(0, 12))

        tk.Label(var_frame,
                 text="Select which variables to show in the trace table:",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

        # Scrollable checkbox area
        canvas_frame = tk.Frame(var_frame, bg=COLORS["bg"])
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        canvas = tk.Canvas(canvas_frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        cb
```


## License: unknown
https://github.com/cstroup/business-planning/blob/c48782917b7971b53d1174d4c05aa4463862e170/planning_app.py

```
Now I have full context. Let me build the complete solution. The approach:

1. **Scan the AST** to find all `InputStmt` nodes and extract variable names — no LLM needed
2. **Before dry-run starts**, show a setup dialog where the user supplies:
   - All input values upfront (comma-separated, like exam papers give them)
   - Which variables to trace (auto-detected from DECLARE statements, user can toggle)
3. **During execution**, feed inputs from the pre-supplied queue instead of prompting
4. **After execution**, show the trace table in exact A-Level format

Here are the three files to update:

**1. Append to the end of [interpreter.py](interpreter.py) — the `DryRunInterpreter` class:**

```python

# ──────────────── Dry-Run / Trace Mode ────────────────

class DryRunInterpreter(Interpreter):
    """
    Cambridge 9618-style dry-run interpreter.
    - Accepts pre-supplied input values (like exam papers provide)
    - Records a trace table of selected variables at each step
    - Feeds inputs from queue instead of prompting interactively
    """

    def __init__(self, symbol_table: SymbolTable, input_queue=None,
                 traced_vars=None, max_steps=5000):
        super().__init__(symbol_table)
        self.trace = []
        self.step_count = 0
        self.max_steps = max_steps
        self.input_queue = list(input_queue) if input_queue else []
        self.input_index = 0
        self.traced_vars = set(traced_vars) if traced_vars else None  # None = auto-detect all
        self.output_log = []  # record OUTPUT values too

    # ── AST scanning helpers (static, no execution needed) ──

    @staticmethod
    def scan_inputs(statements):
        """Walk the AST to find all INPUT statements and return info about them.
        Returns list of dicts: {'line': int, 'variable': str}
        """
        results = []
        DryRunInterpreter._walk_for_inputs(statements, results)
        return results

    @staticmethod
    def _walk_for_inputs(stmts, results):
        for stmt in stmts:
            if isinstance(stmt, InputStmt):
                var_name = "?"
                if isinstance(stmt.target, VariableExpr):
                    var_name = stmt.target.name
                elif isinstance(stmt.target, ArrayAccessExpr):
                    var_name = stmt.target.array + "[...]"
                elif isinstance(stmt.target, str):
                    var_name = stmt.target
                line = getattr(stmt, 'line', 0)
                results.append({'line': line, 'variable': var_name})
            # Recurse into compound statements
            if isinstance(stmt, IfStmt):
                DryRunInterpreter._walk_for_inputs(stmt.then_branch, results)
                if stmt.else_branch:
                    DryRunInterpreter._walk_for_inputs(stmt.else_branch, results)
            elif isinstance(stmt, WhileStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, RepeatStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, ForStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, CaseStmt):
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

    @staticmethod
    def scan_declares(statements):
        """Walk the AST to find all DECLARE statements.
        Returns list of dicts: {'name': str, 'type': str, 'is_array': bool}
        """
        results = []
        for stmt in statements:
            if isinstance(stmt, DeclareStmt):
                results.append({
                    'name': stmt.name,
                    'type': stmt.type_name,
                    'is_array': stmt.is_array
                })
            elif isinstance(stmt, ConstantDecl):
                results.append({
                    'name': stmt.name,
                    'type': 'CONSTANT',
                    'is_array': False
                })
        return results

    # ── Snapshot & trace recording ──

    def _snapshot_vars(self):
        """Capture values of traced variables only."""
        snapshot = {}
        for scope_level in range(self.symbol_table.scope_level + 1):
            scope = self.symbol_table.scopes.get(scope_level, {})
            for name, sym in scope.items():
                # If user specified which vars to trace, filter
                if self.traced_vars is not None and name not in self.traced_vars:
                    continue
                cell = sym.cell
                if cell.is_array:
                    if cell.array_elements:
                        arr = {}
                        for key, ec in sorted(cell.array_elements.items()):
                            idx = ",".join(str(k) for k in key)
                            arr[f"[{idx}]"] = ec.get()
                        snapshot[name] = arr
                    else:
                        snapshot[name] = ""
                elif cell.type == DataType.RECORD and isinstance(cell.value, dict):
                    rec = {}
                    for fn, fc in cell.value.items():
                        rec[fn] = fc.get() if isinstance(fc, Cell) else fc
                    snapshot[name] = rec
                elif isinstance(cell.value, PseudocodeObject):
                    attrs = {}
                    for an, ac in cell.value.attributes.items():
                        attrs[an] = ac.get() if isinstance(ac, Cell) else ac
                    snapshot[name] = f"<{cell.value.class_name}>{attrs}"
                else:
                    try:
                        snapshot[name] = cell.get()
                    except Exception:
                        snapshot[name] = cell.value
        return snapshot

    def _format_val(self, val):
        if isinstance(val, bool): return "TRUE" if val else "FALSE"
        if isinstance(val, str): return f'"{val}"'
        return str(val)

    def _describe_stmt(self, stmt):
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            if isinstance(t, str): return f"{t} \u2190 ..."
            if isinstance(t, ArrayAccessExpr): return f"{t.array}[...] \u2190 ..."
            if isinstance(t, MemberExpr): return f".{t.field} \u2190 ..."
            return "ASSIGN"
        if isinstance(stmt, DeclareStmt):
            arr = "ARRAY" if stmt.is_array else ""
            return f"DECLARE {stmt.name} : {arr} {stmt.type_name}".strip()
        if isinstance(stmt, ConstantDecl): return f"CONSTANT {stmt.name}"
        if isinstance(stmt, OutputStmt): return "OUTPUT"
        if isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr): return f"INPUT {stmt.target.name}"
            if isinstance(stmt.target, ArrayAccessExpr): return f"INPUT {stmt.target.array}[...]"
            return "INPUT"
        if isinstance(stmt, IfStmt): return "IF"
        if isinstance(stmt, WhileStmt): return "WHILE"
        if isinstance(stmt, RepeatStmt): return "REPEAT"
        if isinstance(stmt, ForStmt): return f"FOR {stmt.identifier}"
        if isinstance(stmt, CaseStmt): return "CASE OF"
        if isinstance(stmt, ProcedureCallStmt): return f"CALL {stmt.name}"
        if isinstance(stmt, ReturnStmt): return "RETURN"
        if isinstance(stmt, ProcedureDecl): return f"PROCEDURE {stmt.name}"
        if isinstance(stmt, FunctionDecl): return f"FUNCTION {stmt.name}"
        if isinstance(stmt, TypeDecl): return f"TYPE {stmt.name}"
        if isinstance(stmt, ClassDecl): return f"CLASS {stmt.name}"
        if isinstance(stmt, FileStmt): return f"{stmt.operation}FILE"
        return type(stmt).__name__

    def _record_step(self, stmt, note=""):
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise InterpreterError(
                f"Dry-run stopped after {self.max_steps} steps (possible infinite loop)")
        entry = {
            'step': self.step_count,
            'line': getattr(stmt, 'line', self.current_line),
            'statement': self._describe_stmt(stmt),
            'note': note,
            'variables': self._snapshot_vars(),
        }
        self.trace.append(entry)

    # ── Override execute to record trace ──

    def execute(self, stmt: Stmt):
        if hasattr(stmt, 'line') and stmt.line > 0:
            self.current_line = stmt.line

        # Definitions — store but don't trace body execution
        if isinstance(stmt, (ProcedureDecl, FunctionDecl, TypeDecl, ClassDecl)):
            self._record_step(stmt, "defined")
            return super().execute(stmt)

        if isinstance(stmt, Expr):
            self._record_step(stmt, "expr")
            return super().execute(stmt)

        # Execute
        result = super().execute(stmt)

        # Build note about what changed
        note = ""
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            try:
                if isinstance(t, str):
                    v = self.symbol_table.get_cell(t).get()
                    note = f"= {self._format_val(v)}"
                elif isinstance(t, ArrayAccessExpr):
                    indices = [self.evaluate(idx) for idx in t.indices]
                    v = self.symbol_table.array_access(t.array, indices).get()
                    idx_s = ",".join(str(i) for i in indices)
                    note = f"[{idx_s}] = {self._format_val(v)}"
            except Exception:
                pass
        elif isinstance(stmt, DeclareStmt):
            note = stmt.type_name
        elif isinstance(stmt, OutputStmt):
            # Record what was output
            vals = [self._format_output(self.evaluate(a)) for a in stmt.values]
            output_text = "".join(vals)
            self.output_log.append(output_text)
            note = f'OUT: {output_text}'
        elif isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr):
                try:
                    v = self.symbol_table.get_cell(stmt.target.name).get()
                    note = f"= {self._format_val(v)}"
                except Exception:
                    pass
        elif isinstance(stmt, IfStmt):
            note = "condition"
        elif isinstance(stmt, ForStmt):
            note = f"loop {stmt.identifier}"

        self._record_step(stmt, note)
        return result

    # ── Override INPUT to use pre-supplied queue ──

    def visit_InputStmt(self, stmt: InputStmt):
        """Use pre-supplied input values instead of prompting."""
        if self.input_index < len(self.input_queue):
            val_str = str(self.input_queue[self.input_index])
            self.input_index += 1
        else:
            raise InterpreterError(
                "Dry-run ran out of pre-supplied input values. "
                f"Needed input #{self.input_index + 1} but only "
                f"{len(self.input_queue)} values were provided.")

        # Determine target type for coercion
        target_type = None
        try:
            if isinstance(stmt.target, VariableExpr):
                target_type = self.symbol_table.get_cell(stmt.target.name).type
            elif isinstance(stmt.target, str):
                target_type = self.symbol_table.get_cell(stmt.target).type
            elif isinstance(stmt.target, ArrayAccessExpr):
                cell = self.symbol_table.get_cell(stmt.target.array)
                if cell.is_array and cell.array_bounds:
                    target_type = cell.array_bounds.element_type
        except Exception:
            pass

        if target_type is not None:
            val, val_type = self._coerce_input(val_str, target_type)
        else:
            val, val_type = self._auto_parse_input(val_str)

        if isinstance(stmt.target, VariableExpr):
            self.symbol_table.assign(stmt.target.name, val, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            self.symbol_table.array_assign(stmt.target.array, indices, val, val_type)
        elif isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, val, val_type)

    # ── Results ──

    def get_all_var_names(self):
        names = set()
        for entry in self.trace:
            names.update(entry['variables'].keys())
        names -= set(self.procedures.keys())
        names -= set(self.functions.keys())
        return sorted(names)

    def format_trace_text(self):
        if not self.trace:
            return "No trace data recorded."

        var_names = self.get_all_var_names()
        headers = ['Step', 'Line', 'Statement', 'Note'] + var_names
        rows = []
        for entry in self.trace:
            row = [str(entry['step']), str(entry['line']),
                   entry['statement'], entry['note']]
            for vn in var_names:
                val = entry['variables'].get(vn, '')
                if isinstance(val, dict):
                    val = str(val)
                elif val == '':
                    val = ''
                else:
                    val = self._format_val(val) if not isinstance(val, str) else val
                row.append(val)
            rows.append(row)

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, c in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(c)))
        col_widths = [min(w, 30) for w in col_widths]

        def pad(s, w):
            return str(s)[:w].ljust(w)

        sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
        hdr = '|' + '|'.join(f" {pad(h, w)} " for h, w in zip(headers, col_widths)) + '|'
        lines = [sep, hdr, sep]
        for row in rows:
            r = '|' + '|'.join(f" {pad(c, w)} " for c, w in zip(row, col_widths)) + '|'
            lines.append(r)
        lines.append(sep)
        return '\n'.join(lines)
```

**2. Replace the entire [ide.py](ide.py):**

```python
"""
9618 Pseudocode IDE - Modern Desktop Interface
Tkinter-based GUI with dark theme, syntax highlighting, line numbers, and integrated output.
Includes A-Level exam-style Dry Run mode with pre-supplied inputs and trace table.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import sys
import io
import os
import threading
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexer import Lexer, LexerError
from parser import Parser, ParserError
from interpreter import Interpreter, InterpreterError, DryRunInterpreter
from symbol_table import SymbolTable

# -- Theme Colors (Catppuccin Mocha) --
COLORS = {
    "bg":           "#1e1e2e",
    "bg_secondary": "#181825",
    "bg_tertiary":  "#11111b",
    "surface":      "#313244",
    "overlay":      "#45475a",
    "text":         "#cdd6f4",
    "subtext":      "#a6adc8",
    "blue":         "#89b4fa",
    "green":        "#a6e3a1",
    "red":          "#f38ba8",
    "yellow":       "#f9e2af",
    "mauve":        "#cba6f7",
    "peach":        "#fab387",
    "teal":         "#94e2d5",
    "pink":         "#f5c2e7",
    "lavender":     "#b4befe",
    "sky":          "#89dceb",
    "sapphire":     "#74c7ec",
    "line_num_fg":  "#585b70",
    "selection":    "#45475a",
    "cursor":       "#f5e0dc",
    "gutter":       "#282a3a",
    "output_bg":    "#11111b",
    "toolbar_bg":   "#181825",
    "status_bg":    "#181825",
    "tab_active":   "#1e1e2e",
    "accent":       "#89b4fa",
    "error":        "#f38ba8",
    "success":      "#a6e3a1",
    "warning":      "#f9e2af",
    "button_bg":    "#313244",
    "button_hover": "#45475a",
    "border":       "#313244",
}

# -- Keyword Lists for Highlighting --
KEYWORDS_CONTROL = {
    'IF', 'THEN', 'ELSE', 'ENDIF', 'CASE', 'OF', 'OTHERWISE', 'ENDCASE',
    'FOR', 'TO', 'STEP', 'NEXT', 'WHILE', 'DO', 'ENDWHILE',
    'REPEAT', 'UNTIL', 'RETURN',
}
KEYWORDS_DECL = {
    'DECLARE', 'CONSTANT', 'TYPE', 'ENDTYPE', 'ARRAY',
    'PROCEDURE', 'ENDPROCEDURE', 'FUNCTION', 'ENDFUNCTION',
    'RETURNS', 'BYREF', 'BYVAL', 'CALL',
}
KEYWORDS_TYPE = {'INTEGER', 'REAL', 'STRING', 'BOOLEAN', 'CHAR', 'DATE'}
KEYWORDS_IO = {
    'INPUT', 'OUTPUT', 'OPENFILE', 'READFILE', 'WRITEFILE', 'CLOSEFILE',
    'READ', 'WRITE', 'APPEND',
}
KEYWORDS_OP = {'AND', 'OR', 'NOT', 'DIV', 'MOD', 'TRUE', 'FALSE'}
BUILTINS = {
    'LENGTH', 'UCASE', 'LCASE', 'LEFT', 'RIGHT', 'MID',
    'INT', 'NUM_TO_STR', 'STR_TO_NUM', 'ASC', 'CHR', 'SQRT', 'RAND', 'EOF',
}


class LineNumbers(tk.Canvas):
    def __init__(self, parent, text_widget, **kwargs):
        super().__init__(parent, **kwargs)
        self.text_widget = text_widget
        self.font = None

    def redraw(self, *_args):
        self.delete("all")
        if self.text_widget is None:
            return
        i = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.create_text(
                self.winfo_width() - 8, y, anchor="ne", text=linenum,
                font=self.font, fill=COLORS["line_num_fg"],
            )
            i = self.text_widget.index(f"{i}+1line")
            if self.text_widget.compare(i, ">=", "end"):
                break


class CodeEditor(tk.Text):
    TAG_CONFIG = {
        "keyword_ctrl":  {"foreground": COLORS["mauve"],   "font_style": "bold"},
        "keyword_decl":  {"foreground": COLORS["blue"],    "font_style": "bold"},
        "keyword_type":  {"foreground": COLORS["yellow"]},
        "keyword_io":    {"foreground": COLORS["green"],   "font_style": "bold"},
        "keyword_op":    {"foreground": COLORS["peach"],   "font_style": "bold"},
        "builtin":       {"foreground": COLORS["sapphire"]},
        "string":        {"foreground": COLORS["green"]},
        "char":          {"foreground": COLORS["teal"]},
        "number":        {"foreground": COLORS["peach"]},
        "comment":       {"foreground": COLORS["overlay"], "font_style": "italic"},
        "operator":      {"foreground": COLORS["sky"]},
        "assign_arrow":  {"foreground": COLORS["red"]},
        "error_line":    {"background": "#3d2030"},
    }

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._setup_tags()
        self.bind("<<Modified>>", self._on_modify)
        self._highlight_job = None

    def _setup_tags(self):
        base_font = self.cget("font")
        for tag, cfg in self.TAG_CONFIG.items():
            opts = {"foreground": cfg.get("foreground", COLORS["text"])}
            if "background" in cfg:
                opts["background"] = cfg["background"]
            style = cfg.get("font_style", "")
            if style:
                fam = tkfont.Font(font=base_font).actual()["family"]
                sz = tkfont.Font(font=base_font).actual()["size"]
                weight = "bold" if "bold" in style else "normal"
                slant = "italic" if "italic" in style else "roman"
                opts["font"] = tkfont.Font(family=fam, size=sz, weight=weight, slant=slant)
            self.tag_configure(tag, **opts)
        self.tag_raise("error_line")

    def _on_modify(self, _event=None):
        if self.edit_modified():
            if self._highlight_job:
                self.after_cancel(self._highlight_job)
            self._highlight_job = self.after(80, self.highlight_syntax)
            self.edit_modified(False)
            self.event_generate("<<ContentChanged>>")

    def highlight_syntax(self):
        for tag in self.TAG_CONFIG:
            self.tag_remove(tag, "1.0", "end")
        code = self.get("1.0", "end-1c")
        patterns = [
            ("comment",      r'//[^\n]*'),
            ("string",       r'"[^"]*"'),
            ("char",         r"'[^']*'"),
            ("assign_arrow", r'<-'),
            ("number",       r'\b\d+\.\d+\b|\b\d+\b'),
            ("operator",     r'<>|<=|>=|[<>=+\-*/&]'),
        ]
        for tag, pat in patterns:
            for m in re.finditer(pat, code):
                start = f"1.0+{m.start()}c"
                end = f"1.0+{m.end()}c"
                self.tag_add(tag, start, end)
        word_groups = [
            ("keyword_ctrl", KEYWORDS_CONTROL),
            ("keyword_decl", KEYWORDS_DECL),
            ("keyword_type", KEYWORDS_TYPE),
            ("keyword_io",   KEYWORDS_IO),
            ("keyword_op",   KEYWORDS_OP),
            ("builtin",      BUILTINS),
        ]
        for tag, words in word_groups:
            for w in words:
                pat = rf'\b{w}\b'
                for m in re.finditer(pat, code):
                    start = f"1.0+{m.start()}c"
                    end = f"1.0+{m.end()}c"
                    tags_at = self.tag_names(start)
                    if "string" not in tags_at and "comment" not in tags_at and "char" not in tags_at:
                        self.tag_add(tag, start, end)

    def mark_error_line(self, line_num):
        self.tag_remove("error_line", "1.0", "end")
        if line_num and line_num > 0:
            self.tag_add("error_line", f"{line_num}.0", f"{line_num}.end+1c")
            self.see(f"{line_num}.0")


class OutputPanel(tk.Text):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(state="disabled")
        self.tag_configure("error",        foreground=COLORS["error"])
        self.tag_configure("success",      foreground=COLORS["success"])
        self.tag_configure("info",         foreground=COLORS["subtext"])
        self.tag_configure("output",       foreground=COLORS["text"])
        self.tag_configure("input_prompt", foreground=COLORS["yellow"])

    def append(self, text, tag="output"):
        self.configure(state="normal")
        self.insert("end", text, tag)
        self.see("end")
        self.configure(state="disabled")

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class InputDialog(tk.Toplevel):
    def __init__(self, parent, prompt):
        super().__init__(parent)
        self.title("Input Required")
        self.result = None
        self.configure(bg=COLORS["bg"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 400, 160
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        frame = tk.Frame(self, bg=COLORS["bg"], padx=20, pady=15)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text=prompt, font=("Segoe UI", 11),
                 fg=COLORS["yellow"], bg=COLORS["bg"], anchor="w").pack(fill="x", pady=(0, 8))

        self.entry = tk.Entry(frame, font=("Cascadia Code", 12),
                              bg=COLORS["surface"], fg=COLORS["text"],
                              insertbackground=COLORS["cursor"], relief="flat", bd=0)
        self.entry.pack(fill="x", ipady=6)
        self.entry.focus_set()
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Escape>", lambda e: self._cancel())

        btn_frame = tk.Frame(frame, bg=COLORS["bg"])
        btn_frame.pack(fill="x", pady=(12, 0))

        tk.Button(btn_frame, text="OK", font=("Segoe UI", 10),
                  bg=COLORS["accent"], fg=COLORS["bg_tertiary"],
                  activebackground=COLORS["lavender"],
                  relief="flat", bd=0, padx=20, pady=4,
                  command=self._submit).pack(side="right")
        tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 10),
                  bg=COLORS["surface"], fg=COLORS["text"],
                  activebackground=COLORS["overlay"],
                  relief="flat", bd=0, padx=16, pady=4,
                  command=self._cancel).pack(side="right", padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _submit(self, _event=None):
        self.result = self.entry.get()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class RedirectOutput:
    def __init__(self, panel, tag="output"):
        self.panel = panel
        self.tag = tag

    def write(self, text):
        if text:
            self.panel.after(0, self.panel.append, text, self.tag)

    def flush(self):
        pass


# ──────────────── Dry Run Setup Dialog ────────────────

class DryRunSetupDialog(tk.Toplevel):
    """
    A-Level exam-style dry run setup.
    Shows detected INPUT statements, lets user supply values upfront,
    and choose which variables to include in the trace table.
    """

    def __init__(self, parent, input_info, declare_info):
        super().__init__(parent)
        self.title("Dry Run Setup")
        self.result = None  # Will be {'inputs': [...], 'traced_vars': set or None}
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 520, 560
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        self.minsize(400, 400)

        self.input_info = input_info    # [{'line': int, 'variable': str}, ...]
        self.declare_info = declare_info  # [{'name': str, 'type': str, 'is_array': bool}, ...]
        self.input_entries = []         # tk.Entry widgets for each input
        self.var_checkboxes = {}        # {name: tk.BooleanVar}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _build_ui(self):
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # ── Title ──
        tk.Label(main, text="Dry Run Setup",
                 font=("Segoe UI", 14, "bold"), fg=COLORS["accent"],
                 bg=COLORS["bg"]).pack(anchor="w", pady=(0, 4))
        tk.Label(main, text="Supply input values and select variables to trace,\n"
                            "just like a Cambridge 9618 exam trace table.",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"], justify="left").pack(anchor="w", pady=(0, 12))

        # ── Input Values Section ──
        input_frame = tk.LabelFrame(main, text=" Input Values ",
                                     font=("Segoe UI", 10, "bold"),
                                     fg=COLORS["yellow"], bg=COLORS["bg"],
                                     bd=1, relief="solid",
                                     highlightbackground=COLORS["surface"])
        input_frame.pack(fill="x", pady=(0, 12))

        if self.input_info:
            tk.Label(input_frame,
                     text="Enter values in order (as given on an exam paper):",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

            for i, info in enumerate(self.input_info):
                row = tk.Frame(input_frame, bg=COLORS["bg"])
                row.pack(fill="x", padx=10, pady=2)

                label_text = f"Input #{i+1}  (Line {info['line']}: {info['variable']})"
                tk.Label(row, text=label_text, font=("Segoe UI", 9),
                         fg=COLORS["text"], bg=COLORS["bg"], width=35,
                         anchor="w").pack(side="left")

                entry = tk.Entry(row, font=("Cascadia Code", 11),
                                 bg=COLORS["surface"], fg=COLORS["text"],
                                 insertbackground=COLORS["cursor"],
                                 relief="flat", bd=0, width=20)
                entry.pack(side="left", padx=(4, 0), ipady=3)
                self.input_entries.append(entry)

            # Also allow a bulk entry box
            tk.Label(input_frame,
                     text="Or enter all values comma-separated:",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(8, 2))

            self.bulk_entry = tk.Entry(input_frame, font=("Cascadia Code", 11),
                                        bg=COLORS["surface"], fg=COLORS["text"],
                                        insertbackground=COLORS["cursor"],
                                        relief="flat", bd=0)
            self.bulk_entry.pack(fill="x", padx=10, pady=(0, 8), ipady=4)
            self.bulk_entry.insert(0, "e.g. 5, 3, 8")
            self.bulk_entry.bind("<FocusIn>", self._clear_placeholder)
        else:
            tk.Label(input_frame,
                     text="No INPUT statements detected in this program.",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(padx=10, pady=8)
            self.bulk_entry = None

        # ── Variables to Trace Section ──
        var_frame = tk.LabelFrame(main, text=" Trace Table Columns ",
                                   font=("Segoe UI", 10, "bold"),
                                   fg=COLORS["teal"], bg=COLORS["bg"],
                                   bd=1, relief="solid",
                                   highlightbackground=COLORS["surface"])
        var_frame.pack(fill="both", expand=True, pady=(0, 12))

        tk.Label(var_frame,
                 text="Select which variables to show in the trace table:",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

        # Scrollable checkbox area
        canvas_frame = tk.Frame(var_frame, bg=COLORS["bg"])
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        canvas = tk.Canvas(canvas_frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        cb
```


## License: unknown
https://github.com/cstroup/business-planning/blob/c48782917b7971b53d1174d4c05aa4463862e170/planning_app.py

```
Now I have full context. Let me build the complete solution. The approach:

1. **Scan the AST** to find all `InputStmt` nodes and extract variable names — no LLM needed
2. **Before dry-run starts**, show a setup dialog where the user supplies:
   - All input values upfront (comma-separated, like exam papers give them)
   - Which variables to trace (auto-detected from DECLARE statements, user can toggle)
3. **During execution**, feed inputs from the pre-supplied queue instead of prompting
4. **After execution**, show the trace table in exact A-Level format

Here are the three files to update:

**1. Append to the end of [interpreter.py](interpreter.py) — the `DryRunInterpreter` class:**

```python

# ──────────────── Dry-Run / Trace Mode ────────────────

class DryRunInterpreter(Interpreter):
    """
    Cambridge 9618-style dry-run interpreter.
    - Accepts pre-supplied input values (like exam papers provide)
    - Records a trace table of selected variables at each step
    - Feeds inputs from queue instead of prompting interactively
    """

    def __init__(self, symbol_table: SymbolTable, input_queue=None,
                 traced_vars=None, max_steps=5000):
        super().__init__(symbol_table)
        self.trace = []
        self.step_count = 0
        self.max_steps = max_steps
        self.input_queue = list(input_queue) if input_queue else []
        self.input_index = 0
        self.traced_vars = set(traced_vars) if traced_vars else None  # None = auto-detect all
        self.output_log = []  # record OUTPUT values too

    # ── AST scanning helpers (static, no execution needed) ──

    @staticmethod
    def scan_inputs(statements):
        """Walk the AST to find all INPUT statements and return info about them.
        Returns list of dicts: {'line': int, 'variable': str}
        """
        results = []
        DryRunInterpreter._walk_for_inputs(statements, results)
        return results

    @staticmethod
    def _walk_for_inputs(stmts, results):
        for stmt in stmts:
            if isinstance(stmt, InputStmt):
                var_name = "?"
                if isinstance(stmt.target, VariableExpr):
                    var_name = stmt.target.name
                elif isinstance(stmt.target, ArrayAccessExpr):
                    var_name = stmt.target.array + "[...]"
                elif isinstance(stmt.target, str):
                    var_name = stmt.target
                line = getattr(stmt, 'line', 0)
                results.append({'line': line, 'variable': var_name})
            # Recurse into compound statements
            if isinstance(stmt, IfStmt):
                DryRunInterpreter._walk_for_inputs(stmt.then_branch, results)
                if stmt.else_branch:
                    DryRunInterpreter._walk_for_inputs(stmt.else_branch, results)
            elif isinstance(stmt, WhileStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, RepeatStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, ForStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, CaseStmt):
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

    @staticmethod
    def scan_declares(statements):
        """Walk the AST to find all DECLARE statements.
        Returns list of dicts: {'name': str, 'type': str, 'is_array': bool}
        """
        results = []
        for stmt in statements:
            if isinstance(stmt, DeclareStmt):
                results.append({
                    'name': stmt.name,
                    'type': stmt.type_name,
                    'is_array': stmt.is_array
                })
            elif isinstance(stmt, ConstantDecl):
                results.append({
                    'name': stmt.name,
                    'type': 'CONSTANT',
                    'is_array': False
                })
        return results

    # ── Snapshot & trace recording ──

    def _snapshot_vars(self):
        """Capture values of traced variables only."""
        snapshot = {}
        for scope_level in range(self.symbol_table.scope_level + 1):
            scope = self.symbol_table.scopes.get(scope_level, {})
            for name, sym in scope.items():
                # If user specified which vars to trace, filter
                if self.traced_vars is not None and name not in self.traced_vars:
                    continue
                cell = sym.cell
                if cell.is_array:
                    if cell.array_elements:
                        arr = {}
                        for key, ec in sorted(cell.array_elements.items()):
                            idx = ",".join(str(k) for k in key)
                            arr[f"[{idx}]"] = ec.get()
                        snapshot[name] = arr
                    else:
                        snapshot[name] = ""
                elif cell.type == DataType.RECORD and isinstance(cell.value, dict):
                    rec = {}
                    for fn, fc in cell.value.items():
                        rec[fn] = fc.get() if isinstance(fc, Cell) else fc
                    snapshot[name] = rec
                elif isinstance(cell.value, PseudocodeObject):
                    attrs = {}
                    for an, ac in cell.value.attributes.items():
                        attrs[an] = ac.get() if isinstance(ac, Cell) else ac
                    snapshot[name] = f"<{cell.value.class_name}>{attrs}"
                else:
                    try:
                        snapshot[name] = cell.get()
                    except Exception:
                        snapshot[name] = cell.value
        return snapshot

    def _format_val(self, val):
        if isinstance(val, bool): return "TRUE" if val else "FALSE"
        if isinstance(val, str): return f'"{val}"'
        return str(val)

    def _describe_stmt(self, stmt):
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            if isinstance(t, str): return f"{t} \u2190 ..."
            if isinstance(t, ArrayAccessExpr): return f"{t.array}[...] \u2190 ..."
            if isinstance(t, MemberExpr): return f".{t.field} \u2190 ..."
            return "ASSIGN"
        if isinstance(stmt, DeclareStmt):
            arr = "ARRAY" if stmt.is_array else ""
            return f"DECLARE {stmt.name} : {arr} {stmt.type_name}".strip()
        if isinstance(stmt, ConstantDecl): return f"CONSTANT {stmt.name}"
        if isinstance(stmt, OutputStmt): return "OUTPUT"
        if isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr): return f"INPUT {stmt.target.name}"
            if isinstance(stmt.target, ArrayAccessExpr): return f"INPUT {stmt.target.array}[...]"
            return "INPUT"
        if isinstance(stmt, IfStmt): return "IF"
        if isinstance(stmt, WhileStmt): return "WHILE"
        if isinstance(stmt, RepeatStmt): return "REPEAT"
        if isinstance(stmt, ForStmt): return f"FOR {stmt.identifier}"
        if isinstance(stmt, CaseStmt): return "CASE OF"
        if isinstance(stmt, ProcedureCallStmt): return f"CALL {stmt.name}"
        if isinstance(stmt, ReturnStmt): return "RETURN"
        if isinstance(stmt, ProcedureDecl): return f"PROCEDURE {stmt.name}"
        if isinstance(stmt, FunctionDecl): return f"FUNCTION {stmt.name}"
        if isinstance(stmt, TypeDecl): return f"TYPE {stmt.name}"
        if isinstance(stmt, ClassDecl): return f"CLASS {stmt.name}"
        if isinstance(stmt, FileStmt): return f"{stmt.operation}FILE"
        return type(stmt).__name__

    def _record_step(self, stmt, note=""):
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise InterpreterError(
                f"Dry-run stopped after {self.max_steps} steps (possible infinite loop)")
        entry = {
            'step': self.step_count,
            'line': getattr(stmt, 'line', self.current_line),
            'statement': self._describe_stmt(stmt),
            'note': note,
            'variables': self._snapshot_vars(),
        }
        self.trace.append(entry)

    # ── Override execute to record trace ──

    def execute(self, stmt: Stmt):
        if hasattr(stmt, 'line') and stmt.line > 0:
            self.current_line = stmt.line

        # Definitions — store but don't trace body execution
        if isinstance(stmt, (ProcedureDecl, FunctionDecl, TypeDecl, ClassDecl)):
            self._record_step(stmt, "defined")
            return super().execute(stmt)

        if isinstance(stmt, Expr):
            self._record_step(stmt, "expr")
            return super().execute(stmt)

        # Execute
        result = super().execute(stmt)

        # Build note about what changed
        note = ""
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            try:
                if isinstance(t, str):
                    v = self.symbol_table.get_cell(t).get()
                    note = f"= {self._format_val(v)}"
                elif isinstance(t, ArrayAccessExpr):
                    indices = [self.evaluate(idx) for idx in t.indices]
                    v = self.symbol_table.array_access(t.array, indices).get()
                    idx_s = ",".join(str(i) for i in indices)
                    note = f"[{idx_s}] = {self._format_val(v)}"
            except Exception:
                pass
        elif isinstance(stmt, DeclareStmt):
            note = stmt.type_name
        elif isinstance(stmt, OutputStmt):
            # Record what was output
            vals = [self._format_output(self.evaluate(a)) for a in stmt.values]
            output_text = "".join(vals)
            self.output_log.append(output_text)
            note = f'OUT: {output_text}'
        elif isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr):
                try:
                    v = self.symbol_table.get_cell(stmt.target.name).get()
                    note = f"= {self._format_val(v)}"
                except Exception:
                    pass
        elif isinstance(stmt, IfStmt):
            note = "condition"
        elif isinstance(stmt, ForStmt):
            note = f"loop {stmt.identifier}"

        self._record_step(stmt, note)
        return result

    # ── Override INPUT to use pre-supplied queue ──

    def visit_InputStmt(self, stmt: InputStmt):
        """Use pre-supplied input values instead of prompting."""
        if self.input_index < len(self.input_queue):
            val_str = str(self.input_queue[self.input_index])
            self.input_index += 1
        else:
            raise InterpreterError(
                "Dry-run ran out of pre-supplied input values. "
                f"Needed input #{self.input_index + 1} but only "
                f"{len(self.input_queue)} values were provided.")

        # Determine target type for coercion
        target_type = None
        try:
            if isinstance(stmt.target, VariableExpr):
                target_type = self.symbol_table.get_cell(stmt.target.name).type
            elif isinstance(stmt.target, str):
                target_type = self.symbol_table.get_cell(stmt.target).type
            elif isinstance(stmt.target, ArrayAccessExpr):
                cell = self.symbol_table.get_cell(stmt.target.array)
                if cell.is_array and cell.array_bounds:
                    target_type = cell.array_bounds.element_type
        except Exception:
            pass

        if target_type is not None:
            val, val_type = self._coerce_input(val_str, target_type)
        else:
            val, val_type = self._auto_parse_input(val_str)

        if isinstance(stmt.target, VariableExpr):
            self.symbol_table.assign(stmt.target.name, val, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            self.symbol_table.array_assign(stmt.target.array, indices, val, val_type)
        elif isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, val, val_type)

    # ── Results ──

    def get_all_var_names(self):
        names = set()
        for entry in self.trace:
            names.update(entry['variables'].keys())
        names -= set(self.procedures.keys())
        names -= set(self.functions.keys())
        return sorted(names)

    def format_trace_text(self):
        if not self.trace:
            return "No trace data recorded."

        var_names = self.get_all_var_names()
        headers = ['Step', 'Line', 'Statement', 'Note'] + var_names
        rows = []
        for entry in self.trace:
            row = [str(entry['step']), str(entry['line']),
                   entry['statement'], entry['note']]
            for vn in var_names:
                val = entry['variables'].get(vn, '')
                if isinstance(val, dict):
                    val = str(val)
                elif val == '':
                    val = ''
                else:
                    val = self._format_val(val) if not isinstance(val, str) else val
                row.append(val)
            rows.append(row)

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, c in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(c)))
        col_widths = [min(w, 30) for w in col_widths]

        def pad(s, w):
            return str(s)[:w].ljust(w)

        sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
        hdr = '|' + '|'.join(f" {pad(h, w)} " for h, w in zip(headers, col_widths)) + '|'
        lines = [sep, hdr, sep]
        for row in rows:
            r = '|' + '|'.join(f" {pad(c, w)} " for c, w in zip(row, col_widths)) + '|'
            lines.append(r)
        lines.append(sep)
        return '\n'.join(lines)
```

**2. Replace the entire [ide.py](ide.py):**

```python
"""
9618 Pseudocode IDE - Modern Desktop Interface
Tkinter-based GUI with dark theme, syntax highlighting, line numbers, and integrated output.
Includes A-Level exam-style Dry Run mode with pre-supplied inputs and trace table.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import sys
import io
import os
import threading
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexer import Lexer, LexerError
from parser import Parser, ParserError
from interpreter import Interpreter, InterpreterError, DryRunInterpreter
from symbol_table import SymbolTable

# -- Theme Colors (Catppuccin Mocha) --
COLORS = {
    "bg":           "#1e1e2e",
    "bg_secondary": "#181825",
    "bg_tertiary":  "#11111b",
    "surface":      "#313244",
    "overlay":      "#45475a",
    "text":         "#cdd6f4",
    "subtext":      "#a6adc8",
    "blue":         "#89b4fa",
    "green":        "#a6e3a1",
    "red":          "#f38ba8",
    "yellow":       "#f9e2af",
    "mauve":        "#cba6f7",
    "peach":        "#fab387",
    "teal":         "#94e2d5",
    "pink":         "#f5c2e7",
    "lavender":     "#b4befe",
    "sky":          "#89dceb",
    "sapphire":     "#74c7ec",
    "line_num_fg":  "#585b70",
    "selection":    "#45475a",
    "cursor":       "#f5e0dc",
    "gutter":       "#282a3a",
    "output_bg":    "#11111b",
    "toolbar_bg":   "#181825",
    "status_bg":    "#181825",
    "tab_active":   "#1e1e2e",
    "accent":       "#89b4fa",
    "error":        "#f38ba8",
    "success":      "#a6e3a1",
    "warning":      "#f9e2af",
    "button_bg":    "#313244",
    "button_hover": "#45475a",
    "border":       "#313244",
}

# -- Keyword Lists for Highlighting --
KEYWORDS_CONTROL = {
    'IF', 'THEN', 'ELSE', 'ENDIF', 'CASE', 'OF', 'OTHERWISE', 'ENDCASE',
    'FOR', 'TO', 'STEP', 'NEXT', 'WHILE', 'DO', 'ENDWHILE',
    'REPEAT', 'UNTIL', 'RETURN',
}
KEYWORDS_DECL = {
    'DECLARE', 'CONSTANT', 'TYPE', 'ENDTYPE', 'ARRAY',
    'PROCEDURE', 'ENDPROCEDURE', 'FUNCTION', 'ENDFUNCTION',
    'RETURNS', 'BYREF', 'BYVAL', 'CALL',
}
KEYWORDS_TYPE = {'INTEGER', 'REAL', 'STRING', 'BOOLEAN', 'CHAR', 'DATE'}
KEYWORDS_IO = {
    'INPUT', 'OUTPUT', 'OPENFILE', 'READFILE', 'WRITEFILE', 'CLOSEFILE',
    'READ', 'WRITE', 'APPEND',
}
KEYWORDS_OP = {'AND', 'OR', 'NOT', 'DIV', 'MOD', 'TRUE', 'FALSE'}
BUILTINS = {
    'LENGTH', 'UCASE', 'LCASE', 'LEFT', 'RIGHT', 'MID',
    'INT', 'NUM_TO_STR', 'STR_TO_NUM', 'ASC', 'CHR', 'SQRT', 'RAND', 'EOF',
}


class LineNumbers(tk.Canvas):
    def __init__(self, parent, text_widget, **kwargs):
        super().__init__(parent, **kwargs)
        self.text_widget = text_widget
        self.font = None

    def redraw(self, *_args):
        self.delete("all")
        if self.text_widget is None:
            return
        i = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.create_text(
                self.winfo_width() - 8, y, anchor="ne", text=linenum,
                font=self.font, fill=COLORS["line_num_fg"],
            )
            i = self.text_widget.index(f"{i}+1line")
            if self.text_widget.compare(i, ">=", "end"):
                break


class CodeEditor(tk.Text):
    TAG_CONFIG = {
        "keyword_ctrl":  {"foreground": COLORS["mauve"],   "font_style": "bold"},
        "keyword_decl":  {"foreground": COLORS["blue"],    "font_style": "bold"},
        "keyword_type":  {"foreground": COLORS["yellow"]},
        "keyword_io":    {"foreground": COLORS["green"],   "font_style": "bold"},
        "keyword_op":    {"foreground": COLORS["peach"],   "font_style": "bold"},
        "builtin":       {"foreground": COLORS["sapphire"]},
        "string":        {"foreground": COLORS["green"]},
        "char":          {"foreground": COLORS["teal"]},
        "number":        {"foreground": COLORS["peach"]},
        "comment":       {"foreground": COLORS["overlay"], "font_style": "italic"},
        "operator":      {"foreground": COLORS["sky"]},
        "assign_arrow":  {"foreground": COLORS["red"]},
        "error_line":    {"background": "#3d2030"},
    }

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._setup_tags()
        self.bind("<<Modified>>", self._on_modify)
        self._highlight_job = None

    def _setup_tags(self):
        base_font = self.cget("font")
        for tag, cfg in self.TAG_CONFIG.items():
            opts = {"foreground": cfg.get("foreground", COLORS["text"])}
            if "background" in cfg:
                opts["background"] = cfg["background"]
            style = cfg.get("font_style", "")
            if style:
                fam = tkfont.Font(font=base_font).actual()["family"]
                sz = tkfont.Font(font=base_font).actual()["size"]
                weight = "bold" if "bold" in style else "normal"
                slant = "italic" if "italic" in style else "roman"
                opts["font"] = tkfont.Font(family=fam, size=sz, weight=weight, slant=slant)
            self.tag_configure(tag, **opts)
        self.tag_raise("error_line")

    def _on_modify(self, _event=None):
        if self.edit_modified():
            if self._highlight_job:
                self.after_cancel(self._highlight_job)
            self._highlight_job = self.after(80, self.highlight_syntax)
            self.edit_modified(False)
            self.event_generate("<<ContentChanged>>")

    def highlight_syntax(self):
        for tag in self.TAG_CONFIG:
            self.tag_remove(tag, "1.0", "end")
        code = self.get("1.0", "end-1c")
        patterns = [
            ("comment",      r'//[^\n]*'),
            ("string",       r'"[^"]*"'),
            ("char",         r"'[^']*'"),
            ("assign_arrow", r'<-'),
            ("number",       r'\b\d+\.\d+\b|\b\d+\b'),
            ("operator",     r'<>|<=|>=|[<>=+\-*/&]'),
        ]
        for tag, pat in patterns:
            for m in re.finditer(pat, code):
                start = f"1.0+{m.start()}c"
                end = f"1.0+{m.end()}c"
                self.tag_add(tag, start, end)
        word_groups = [
            ("keyword_ctrl", KEYWORDS_CONTROL),
            ("keyword_decl", KEYWORDS_DECL),
            ("keyword_type", KEYWORDS_TYPE),
            ("keyword_io",   KEYWORDS_IO),
            ("keyword_op",   KEYWORDS_OP),
            ("builtin",      BUILTINS),
        ]
        for tag, words in word_groups:
            for w in words:
                pat = rf'\b{w}\b'
                for m in re.finditer(pat, code):
                    start = f"1.0+{m.start()}c"
                    end = f"1.0+{m.end()}c"
                    tags_at = self.tag_names(start)
                    if "string" not in tags_at and "comment" not in tags_at and "char" not in tags_at:
                        self.tag_add(tag, start, end)

    def mark_error_line(self, line_num):
        self.tag_remove("error_line", "1.0", "end")
        if line_num and line_num > 0:
            self.tag_add("error_line", f"{line_num}.0", f"{line_num}.end+1c")
            self.see(f"{line_num}.0")


class OutputPanel(tk.Text):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(state="disabled")
        self.tag_configure("error",        foreground=COLORS["error"])
        self.tag_configure("success",      foreground=COLORS["success"])
        self.tag_configure("info",         foreground=COLORS["subtext"])
        self.tag_configure("output",       foreground=COLORS["text"])
        self.tag_configure("input_prompt", foreground=COLORS["yellow"])

    def append(self, text, tag="output"):
        self.configure(state="normal")
        self.insert("end", text, tag)
        self.see("end")
        self.configure(state="disabled")

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class InputDialog(tk.Toplevel):
    def __init__(self, parent, prompt):
        super().__init__(parent)
        self.title("Input Required")
        self.result = None
        self.configure(bg=COLORS["bg"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 400, 160
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        frame = tk.Frame(self, bg=COLORS["bg"], padx=20, pady=15)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text=prompt, font=("Segoe UI", 11),
                 fg=COLORS["yellow"], bg=COLORS["bg"], anchor="w").pack(fill="x", pady=(0, 8))

        self.entry = tk.Entry(frame, font=("Cascadia Code", 12),
                              bg=COLORS["surface"], fg=COLORS["text"],
                              insertbackground=COLORS["cursor"], relief="flat", bd=0)
        self.entry.pack(fill="x", ipady=6)
        self.entry.focus_set()
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Escape>", lambda e: self._cancel())

        btn_frame = tk.Frame(frame, bg=COLORS["bg"])
        btn_frame.pack(fill="x", pady=(12, 0))

        tk.Button(btn_frame, text="OK", font=("Segoe UI", 10),
                  bg=COLORS["accent"], fg=COLORS["bg_tertiary"],
                  activebackground=COLORS["lavender"],
                  relief="flat", bd=0, padx=20, pady=4,
                  command=self._submit).pack(side="right")
        tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 10),
                  bg=COLORS["surface"], fg=COLORS["text"],
                  activebackground=COLORS["overlay"],
                  relief="flat", bd=0, padx=16, pady=4,
                  command=self._cancel).pack(side="right", padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _submit(self, _event=None):
        self.result = self.entry.get()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class RedirectOutput:
    def __init__(self, panel, tag="output"):
        self.panel = panel
        self.tag = tag

    def write(self, text):
        if text:
            self.panel.after(0, self.panel.append, text, self.tag)

    def flush(self):
        pass


# ──────────────── Dry Run Setup Dialog ────────────────

class DryRunSetupDialog(tk.Toplevel):
    """
    A-Level exam-style dry run setup.
    Shows detected INPUT statements, lets user supply values upfront,
    and choose which variables to include in the trace table.
    """

    def __init__(self, parent, input_info, declare_info):
        super().__init__(parent)
        self.title("Dry Run Setup")
        self.result = None  # Will be {'inputs': [...], 'traced_vars': set or None}
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 520, 560
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        self.minsize(400, 400)

        self.input_info = input_info    # [{'line': int, 'variable': str}, ...]
        self.declare_info = declare_info  # [{'name': str, 'type': str, 'is_array': bool}, ...]
        self.input_entries = []         # tk.Entry widgets for each input
        self.var_checkboxes = {}        # {name: tk.BooleanVar}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _build_ui(self):
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # ── Title ──
        tk.Label(main, text="Dry Run Setup",
                 font=("Segoe UI", 14, "bold"), fg=COLORS["accent"],
                 bg=COLORS["bg"]).pack(anchor="w", pady=(0, 4))
        tk.Label(main, text="Supply input values and select variables to trace,\n"
                            "just like a Cambridge 9618 exam trace table.",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"], justify="left").pack(anchor="w", pady=(0, 12))

        # ── Input Values Section ──
        input_frame = tk.LabelFrame(main, text=" Input Values ",
                                     font=("Segoe UI", 10, "bold"),
                                     fg=COLORS["yellow"], bg=COLORS["bg"],
                                     bd=1, relief="solid",
                                     highlightbackground=COLORS["surface"])
        input_frame.pack(fill="x", pady=(0, 12))

        if self.input_info:
            tk.Label(input_frame,
                     text="Enter values in order (as given on an exam paper):",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

            for i, info in enumerate(self.input_info):
                row = tk.Frame(input_frame, bg=COLORS["bg"])
                row.pack(fill="x", padx=10, pady=2)

                label_text = f"Input #{i+1}  (Line {info['line']}: {info['variable']})"
                tk.Label(row, text=label_text, font=("Segoe UI", 9),
                         fg=COLORS["text"], bg=COLORS["bg"], width=35,
                         anchor="w").pack(side="left")

                entry = tk.Entry(row, font=("Cascadia Code", 11),
                                 bg=COLORS["surface"], fg=COLORS["text"],
                                 insertbackground=COLORS["cursor"],
                                 relief="flat", bd=0, width=20)
                entry.pack(side="left", padx=(4, 0), ipady=3)
                self.input_entries.append(entry)

            # Also allow a bulk entry box
            tk.Label(input_frame,
                     text="Or enter all values comma-separated:",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(8, 2))

            self.bulk_entry = tk.Entry(input_frame, font=("Cascadia Code", 11),
                                        bg=COLORS["surface"], fg=COLORS["text"],
                                        insertbackground=COLORS["cursor"],
                                        relief="flat", bd=0)
            self.bulk_entry.pack(fill="x", padx=10, pady=(0, 8), ipady=4)
            self.bulk_entry.insert(0, "e.g. 5, 3, 8")
            self.bulk_entry.bind("<FocusIn>", self._clear_placeholder)
        else:
            tk.Label(input_frame,
                     text="No INPUT statements detected in this program.",
                     font=("Segoe UI", 9), fg=COLORS["subtext"],
                     bg=COLORS["bg"]).pack(padx=10, pady=8)
            self.bulk_entry = None

        # ── Variables to Trace Section ──
        var_frame = tk.LabelFrame(main, text=" Trace Table Columns ",
                                   font=("Segoe UI", 10, "bold"),
                                   fg=COLORS["teal"], bg=COLORS["bg"],
                                   bd=1, relief="solid",
                                   highlightbackground=COLORS["surface"])
        var_frame.pack(fill="both", expand=True, pady=(0, 12))

        tk.Label(var_frame,
                 text="Select which variables to show in the trace table:",
                 font=("Segoe UI", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(6, 4))

        # Scrollable checkbox area
        canvas_frame = tk.Frame(var_frame, bg=COLORS["bg"])
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        canvas = tk.Canvas(canvas_frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        cb
```

