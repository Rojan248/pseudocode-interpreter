"""
9618 Pseudocode IDE — Modern Desktop Interface
CustomTkinter + Tkinter hybrid GUI with dark Catppuccin theme,
syntax highlighting, line numbers, current-line highlight,
integrated output, and dry-run trace-table support.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import customtkinter as ctk
import sys
import io
import os
import threading
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexer import Lexer, LexerError
from parser import Parser, ParserError
from interpreter import Interpreter, InterpreterError
from dry_run_interpreter import DryRunInterpreter
from symbol_table import SymbolTable

# ── CustomTkinter global setup ──
ctk.set_appearance_mode("dark")

# ── Theme Colors  (Catppuccin Mocha) ──
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
    "current_line": "#232336",
}

# ── Keyword Lists for Highlighting ──
KEYWORDS_CONTROL = {
    'IF', 'THEN', 'ELSE', 'ENDIF', 'CASE', 'OF', 'OTHERWISE', 'ENDCASE',
    'FOR', 'TO', 'STEP', 'NEXT', 'WHILE', 'DO', 'ENDWHILE',
    'REPEAT', 'UNTIL', 'RETURN',
}
KEYWORDS_DECL = {
    'DECLARE', 'CONSTANT', 'TYPE', 'ENDTYPE', 'ARRAY',
    'PROCEDURE', 'ENDPROCEDURE', 'FUNCTION', 'ENDFUNCTION',
    'RETURNS', 'BYREF', 'BYVAL', 'CALL',
    'CLASS', 'ENDCLASS', 'INHERITS', 'NEW', 'SUPER',
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


# ═══════════════════════════════════════════════════════
#  Core Editor Widgets  (pure Tk — needed for tag-based
#  syntax highlighting and Canvas line numbers)
# ═══════════════════════════════════════════════════════

class LineNumbers(tk.Canvas):
    """Line-number gutter drawn on a Canvas."""

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
    """Text widget with syntax highlighting and current-line highlight."""

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
        "current_line":  {"background": COLORS["current_line"]},
    }

    # Tags that must NOT be cleared during syntax re-highlight
    _PERSISTENT_TAGS = {"current_line", "error_line"}

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._setup_tags()
        self.bind("<<Modified>>", self._on_modify)
        self.bind("<KeyRelease>", self._on_cursor_move)
        self.bind("<ButtonRelease-1>", self._on_cursor_move)
        self._highlight_job = None

    def _setup_tags(self):
        base_font = self.cget("font")
        for tag, cfg in self.TAG_CONFIG.items():
            opts = self._build_tag_options(base_font, cfg)
            self.tag_configure(tag, **opts)
        # current_line must sit BELOW all syntax tags so colours show through
        self.tag_lower("current_line")
        self.tag_raise("error_line")

    def _build_tag_options(self, base_font, cfg):
        """Build a dict of tag_configure options from a TAG_CONFIG entry."""
        opts = {"foreground": cfg.get("foreground", COLORS["text"])}
        if "background" in cfg:
            opts["background"] = cfg["background"]
        style = cfg.get("font_style", "")
        if style:
            actual = tkfont.Font(font=base_font).actual()
            weight = "bold" if "bold" in style else "normal"
            slant = "italic" if "italic" in style else "roman"
            opts["font"] = tkfont.Font(
                family=actual["family"], size=actual["size"],
                weight=weight, slant=slant,
            )
        return opts

    # ── Current-line highlight ──

    def _on_cursor_move(self, _event=None):
        self._highlight_current_line()

    def _highlight_current_line(self):
        self.tag_remove("current_line", "1.0", "end")
        self.tag_add("current_line", "insert linestart", "insert lineend+1c")

    # ── Syntax highlighting (debounced) ──

    def _on_modify(self, _event=None):
        if self.edit_modified():
            if self._highlight_job:
                self.after_cancel(self._highlight_job)
            self._highlight_job = self.after(80, self._do_highlight)
            self.edit_modified(False)
            self.event_generate("<<ContentChanged>>")

    def _do_highlight(self):
        self.highlight_syntax()
        self._highlight_current_line()

    def highlight_syntax(self):
        """Apply syntax highlighting to all text."""
        for tag in self.TAG_CONFIG:
            if tag not in self._PERSISTENT_TAGS:
                self.tag_remove(tag, "1.0", "end")

        code = self.get("1.0", "end-1c")
        self._apply_regex_patterns(code)
        self._apply_keyword_tags(code)

    # ── Syntax-highlighting helpers ──

    _REGEX_PATTERNS = [
        ("comment",      r'//[^\n]*'),
        ("string",       r'"[^"]*"'),
        ("char",         r"'[^']*'"),
        ("assign_arrow", r'<-'),
        ("number",       r'\b\d+\.\d+\b|\b\d+\b'),
        ("operator",     r'<>|<=|>=|[<>=+\-*/&]'),
    ]

    _WORD_GROUPS = [
        ("keyword_ctrl", KEYWORDS_CONTROL),
        ("keyword_decl", KEYWORDS_DECL),
        ("keyword_type", KEYWORDS_TYPE),
        ("keyword_io",   KEYWORDS_IO),
        ("keyword_op",   KEYWORDS_OP),
        ("builtin",      BUILTINS),
    ]

    def _apply_regex_patterns(self, code):
        """Apply regex-based syntax tags (comments, strings, numbers, etc.)."""
        for tag, pat in self._REGEX_PATTERNS:
            for m in re.finditer(pat, code):
                self.tag_add(tag, f"1.0+{m.start()}c", f"1.0+{m.end()}c")

    def _apply_keyword_tags(self, code):
        """Apply keyword/builtin tags, skipping positions inside literals."""
        for tag, words in self._WORD_GROUPS:
            # Combine words into a single regex pattern to reduce loop depth
            pattern = rf"\b({'|'.join(re.escape(w) for w in words)})\b"
            for m in re.finditer(pattern, code):
                pos = f"1.0+{m.start()}c"
                if not self._is_inside_literal(pos):
                    self.tag_add(tag, pos, f"1.0+{m.end()}c")

    def _is_inside_literal(self, pos):
        """Return True if the given text position is inside a string, comment, or char literal."""
        tags_at = self.tag_names(pos)
        return "string" in tags_at or "comment" in tags_at or "char" in tags_at

    def mark_error_line(self, line_num):
        """Highlight a specific line as an error."""
        self.tag_remove("error_line", "1.0", "end")
        if line_num and line_num > 0:
            self.tag_add("error_line", f"{line_num}.0", f"{line_num}.end+1c")
            self.see(f"{line_num}.0")


class OutputPanel(tk.Text):
    """Read-only output display panel."""

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


# ═══════════════════════════════════════════════════════
#  Helper Managers
# ═══════════════════════════════════════════════════════

class IDEFileManager:
    """Handles file operations (new, open, save) for the IDE."""
    def __init__(self, ide):
        self.ide = ide
        self.current_file = None

    def new_file(self):
        self.ide.editor.delete("1.0", "end")
        self.current_file = None
        self.ide._update_tab()
        self.ide._set_status("New file")
        self.ide.editor.edit_modified(False)

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open Pseudocode File",
            filetypes=[("Pseudocode", "*.pse *.pseudo *.txt"), ("All files", "*.*")],
        )
        if path:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.ide.editor.delete("1.0", "end")
            self.ide.editor.insert("1.0", content)
            self.current_file = path
            self.ide._update_tab()
            self.ide._set_status(f"Opened {os.path.basename(path)}")
            self.ide.editor.edit_modified(False)
            self.ide.editor.highlight_syntax()
            self.ide.editor._highlight_current_line()

    def save_file(self):
        if self.current_file:
            self.write_file(self.current_file)
        else:
            self.save_as()

    def save_as(self):
        path = filedialog.asksaveasfilename(
            title="Save Pseudocode File", defaultextension=".pse",
            filetypes=[("Pseudocode", "*.pse"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.write_file(path)
            self.current_file = path
            self.ide._update_tab()

    def write_file(self, path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.ide.editor.get("1.0", "end-1c"))
            self.ide._set_status(f"Saved {os.path.basename(path)}", COLORS["success"])
            self.ide.editor.edit_modified(False)
        except Exception as e:
            self.ide._show_error("File Error", f"Cannot save file: {e}")

class IDEExecutionEngine:
    """Handles execution of pseudocode (Normal Run and Dry Run)."""
    def __init__(self, ide):
        self.ide = ide
        self.is_running = False
        self.run_thread = None

    def run_code(self):
        if self.is_running:
            return
        source = self.ide.editor.get("1.0", "end-1c")
        if not source.strip():
            return
        self.ide.output.clear()
        self.ide.editor.tag_remove("error_line", "1.0", "end")
        self.is_running = True
        self.ide.run_btn.configure(fg_color=COLORS["overlay"])
        self.ide._set_status("Running...", COLORS["warning"])
        self.run_thread = threading.Thread(target=self._execute, args=(source,), daemon=True)
        self.run_thread.start()
        self._check_thread()

    def dry_run(self):
        """Launch A-Level exam-style dry run: setup → execute → trace table."""
        if self.is_running:
            return
        source = self.ide.editor.get("1.0", "end-1c")
        if not source.strip():
            return

        filename = self.ide.file_manager.current_file or "<editor>"
        try:
            lexer = Lexer(source, filename)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            statements = parser.parse()
        except Exception as e:
            self.ide._show_error("Error", f"Cannot dry-run: {e}")
            return

        input_info = DryRunInterpreter.scan_inputs(statements)
        declare_info = DryRunInterpreter.scan_declares(statements)

        dlg = DryRunSetupDialog(self.ide.root, input_info, declare_info)
        if dlg.result is None:
            return

        setup = dlg.result
        self.ide.output.clear()
        self.ide.editor.tag_remove("error_line", "1.0", "end")
        self.is_running = True
        self.ide.dryrun_btn.configure(fg_color=COLORS["overlay"])
        self.ide._set_status("Dry-running...", COLORS["warning"])

        self.run_thread = threading.Thread(
            target=self._execute_dryrun,
            args=(statements, setup), daemon=True,
        )
        self.run_thread.start()
        self._check_dryrun_thread()

    def _execute(self, source):
        tokens = self._lex(source)
        if tokens is None:
            return
        statements = self._parse(tokens)
        if statements is None:
            return

        symbol_table = SymbolTable()
        interpreter = Interpreter(symbol_table)

        import builtins
        original_input = builtins.input

        def gui_input(prompt=""):
            result = [None]
            event = threading.Event()
            def show_dialog():
                dlg = InputDialog(self.ide.root, prompt)
                result[0] = dlg.result
                event.set()
            self.ide.root.after(0, show_dialog)
            event.wait()
            if result[0] is None:
                raise InterpreterError("Input cancelled by user")
            self.ide.output.after(0, self.ide.output.append, f"{prompt}{result[0]}\n", "input_prompt")
            return result[0]

        builtins.input = gui_input
        old_stdout = sys.stdout
        sys.stdout = RedirectOutput(self.ide.output)

        try:
            interpreter.interpret(statements)
            self.ide._show_success()
        except InterpreterError as e:
            self.ide._show_error("Runtime Error", f"Line {interpreter.current_line}: {e}",
                             interpreter.current_line)
        except Exception as e:
            self.ide._show_error("Runtime Error", f"Line {interpreter.current_line}: {e}",
                             interpreter.current_line)
        finally:
            sys.stdout = old_stdout
            builtins.input = original_input

    def _execute_dryrun(self, statements, setup):
        symbol_table = SymbolTable()
        interpreter = DryRunInterpreter(
            symbol_table,
            input_queue=setup['inputs'],
            traced_vars=setup['traced_vars'],
            trace_columns=setup.get('trace_columns'),
        )

        old_stdout = sys.stdout
        sys.stdout = RedirectOutput(self.ide.output)

        try:
            interpreter.interpret(statements)
            self.ide._show_dryrun_success(interpreter)
        except (InterpreterError, Exception) as e:
            self.ide._show_error("Runtime Error",
                             f"Line {interpreter.current_line}: {e}",
                             interpreter.current_line)
            if interpreter.trace:
                self.ide._open_trace_window(interpreter)
        finally:
            sys.stdout = old_stdout

    def _check_thread(self):
        if self.run_thread and self.run_thread.is_alive():
            self.ide.root.after(100, self._check_thread)
        else:
            self.is_running = False
            self.ide.run_btn.configure(fg_color=COLORS["green"])

    def _check_dryrun_thread(self):
        if self.run_thread and self.run_thread.is_alive():
            self.ide.root.after(100, self._check_dryrun_thread)
        else:
            self.is_running = False
            self.ide.dryrun_btn.configure(fg_color=COLORS["yellow"])

    def _lex(self, source):
        """Tokenize source code, returning tokens or None on error."""
        filename = self.ide.file_manager.current_file or "<editor>"
        try:
            return Lexer(source, filename).tokenize()
        except LexerError as e:
            self.ide._show_error("Lexer Error", str(e))
        except Exception as e:
            self.ide._show_error("Lexer Error", str(e))
        return None

    def _parse(self, tokens):
        """Parse tokens into statements, returning statements or None on error."""
        try:
            return Parser(tokens).parse()
        except ParserError as e:
            self.ide._show_error("Parser Error", str(e))
        except Exception as e:
            self.ide._show_error("Parser Error", str(e))
        return None

# ═══════════════════════════════════════════════════════
#  Dialogs  (CTkToplevel → dark title-bar on Windows)
# ═══════════════════════════════════════════════════════

class InputDialog(ctk.CTkToplevel):
    """Modal dialog for INPUT statements during execution."""

    def __init__(self, parent, prompt):
        super().__init__(parent)
        self.title("Input Required")
        self.result = None
        self.configure(fg_color=COLORS["bg"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 440, 190
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        frame = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        frame.pack(fill="both", expand=True, padx=24, pady=18)

        ctk.CTkLabel(frame, text=prompt, font=("Segoe UI", 12),
                     text_color=COLORS["yellow"],
                     anchor="w").pack(fill="x", pady=(0, 10))

        self.entry = ctk.CTkEntry(
            frame, font=("Cascadia Code", 13),
            fg_color=COLORS["surface"], text_color=COLORS["text"],
            border_color=COLORS["overlay"], corner_radius=8, height=38,
        )
        self.entry.pack(fill="x")
        self.entry.focus()
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<Escape>", lambda e: self._cancel())

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(14, 0))

        ctk.CTkButton(
            btn_frame, text="OK", font=("Segoe UI", 11, "bold"),
            fg_color=COLORS["accent"], text_color=COLORS["bg_tertiary"],
            hover_color=COLORS["lavender"], corner_radius=8,
            width=90, height=34, command=self._submit,
        ).pack(side="right")

        ctk.CTkButton(
            btn_frame, text="Cancel", font=("Segoe UI", 11),
            fg_color=COLORS["surface"], text_color=COLORS["text"],
            hover_color=COLORS["overlay"], corner_radius=8,
            width=90, height=34, command=self._cancel,
        ).pack(side="right", padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _submit(self, _event=None):
        self.result = self.entry.get()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class RedirectOutput:
    """Redirect stdout to the output panel."""

    def __init__(self, panel, tag="output"):
        self.panel = panel
        self.tag = tag

    def write(self, text):
        if text:
            self.panel.after(0, self.panel.append, text, self.tag)

    def flush(self):
        pass


# ═══════════════════════════════════════════════════════
#  Dry-Run  Setup Dialog
# ═══════════════════════════════════════════════════════

class DryRunSetupDialog(ctk.CTkToplevel):
    """A-Level exam-style dry-run setup.
    Shows detected INPUT statements so user can supply values upfront,
    and lets user choose which variables to include in the trace table."""

    def __init__(self, parent, input_info, declare_info):
        super().__init__(parent)
        self.title("Dry Run Setup")
        self.result = None
        self.configure(fg_color=COLORS["bg"])
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = 540, 580
        self.geometry(f"{w}x{h}+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        self.minsize(420, 420)

        self.input_info = input_info
        self.declare_info = declare_info
        self.input_entries = []
        self.var_checkboxes = {}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    # ──────── UI ────────

    def _build_ui(self):
        main = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # Title
        ctk.CTkLabel(
            main, text="◇  Dry Run Setup",
            font=("Segoe UI", 16, "bold"), text_color=COLORS["accent"],
        ).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(
            main,
            text="Supply input values and select variables to trace,\n"
                 "just like a Cambridge 9618 exam trace table.",
            font=("Segoe UI", 10), text_color=COLORS["subtext"], justify="left",
        ).pack(anchor="w", pady=(0, 14))

        self._build_input_card(main)
        self._build_variable_card(main)
        self._build_action_buttons(main)

        if self.input_entries:
            self.input_entries[0].focus()

    def _build_input_card(self, parent):
        """Build the 'Input Values' card section."""
        input_card = ctk.CTkFrame(parent, fg_color=COLORS["surface"], corner_radius=10)
        input_card.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            input_card, text="  Input Values",
            font=("Segoe UI", 11, "bold"), text_color=COLORS["yellow"],
        ).pack(anchor="w", padx=12, pady=(10, 4))

        if self.input_info:
            ctk.CTkLabel(
                input_card,
                text="Enter values in order (as given on an exam paper):",
                font=("Segoe UI", 9), text_color=COLORS["subtext"],
            ).pack(anchor="w", padx=14, pady=(0, 4))

            for i, info in enumerate(self.input_info):
                row = ctk.CTkFrame(input_card, fg_color="transparent")
                row.pack(fill="x", padx=14, pady=2)
                label_text = f"Input #{i+1}  (Line {info['line']}: {info['variable']})"
                ctk.CTkLabel(
                    row, text=label_text, font=("Segoe UI", 9),
                    text_color=COLORS["text"], width=250, anchor="w",
                ).pack(side="left")
                entry = ctk.CTkEntry(
                    row, font=("Cascadia Code", 11),
                    fg_color=COLORS["bg"], text_color=COLORS["text"],
                    border_color=COLORS["overlay"], corner_radius=6,
                    width=180, height=30,
                )
                entry.pack(side="left", padx=(4, 0))
                self.input_entries.append(entry)

            ctk.CTkLabel(
                input_card,
                text="Or enter all values comma-separated:",
                font=("Segoe UI", 9), text_color=COLORS["subtext"],
            ).pack(anchor="w", padx=14, pady=(10, 2))

            self.bulk_entry = ctk.CTkEntry(
                input_card, font=("Cascadia Code", 11),
                fg_color=COLORS["bg"], text_color=COLORS["text"],
                border_color=COLORS["overlay"], corner_radius=6, height=32,
                placeholder_text="e.g. 5, 3, 8",
            )
            self.bulk_entry.pack(fill="x", padx=14, pady=(0, 10))
        else:
            ctk.CTkLabel(
                input_card,
                text="No INPUT statements detected in this program.",
                font=("Segoe UI", 9), text_color=COLORS["subtext"],
            ).pack(padx=14, pady=(0, 10))
            self.bulk_entry = None

    def _build_variable_card(self, parent):
        """Build the 'Trace Table Columns' card section."""
        var_card = ctk.CTkFrame(parent, fg_color=COLORS["surface"], corner_radius=10)
        var_card.pack(fill="both", expand=True, pady=(0, 12))

        ctk.CTkLabel(
            var_card, text="  Trace Table Columns",
            font=("Segoe UI", 11, "bold"), text_color=COLORS["teal"],
        ).pack(anchor="w", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            var_card,
            text="Define specific columns (comma-separated expressions) to match exam style:",
            font=("Segoe UI", 9), text_color=COLORS["subtext"],
        ).pack(anchor="w", padx=14, pady=(0, 6))

        self.custom_cols_entry = ctk.CTkEntry(
            var_card, font=("Cascadia Code", 11),
            fg_color=COLORS["bg"], text_color=COLORS["text"],
            border_color=COLORS["overlay"], corner_radius=6, height=32,
            placeholder_text="e.g. I, Key, J, Chars[J], Count + 1",
        )
        self.custom_cols_entry.pack(fill="x", padx=14, pady=(0, 10))

        ctk.CTkLabel(
            var_card,
            text="Or select from detected variables (if custom columns are empty):",
            font=("Segoe UI", 9), text_color=COLORS["subtext"],
        ).pack(anchor="w", padx=14, pady=(0, 6))

        self.select_all_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            var_card, text="All Variables (auto-detect)",
            variable=self.select_all_var,
            font=("Segoe UI", 11, "bold"),
            text_color=COLORS["accent"],
            fg_color=COLORS["accent"], hover_color=COLORS["lavender"],
            border_color=COLORS["overlay"], corner_radius=4,
            command=self._toggle_all,
        ).pack(anchor="w", padx=14, pady=(0, 6))

        # Scrollable variable list
        scroll_frame = ctk.CTkScrollableFrame(
            var_card, fg_color=COLORS["bg"], corner_radius=6,
            scrollbar_button_color=COLORS["surface"],
            scrollbar_button_hover_color=COLORS["overlay"],
        )
        scroll_frame.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        if self.declare_info:
            for info in self.declare_info:
                bv = tk.BooleanVar(value=True)
                lbl = f"{info['name']}  :  {info['type']}"
                if info['is_array']:
                    lbl = f"{info['name']}  :  ARRAY OF {info['type']}"
                ctk.CTkCheckBox(
                    scroll_frame, text=lbl, variable=bv,
                    font=("Cascadia Code", 10),
                    text_color=COLORS["text"],
                    fg_color=COLORS["teal"], hover_color=COLORS["sky"],
                    border_color=COLORS["overlay"], corner_radius=4,
                ).pack(anchor="w", padx=8, pady=2)
                self.var_checkboxes[info['name']] = bv
        else:
            ctk.CTkLabel(
                scroll_frame,
                text="No DECLARE statements found.\nAll variables will be auto-detected.",
                font=("Segoe UI", 9), text_color=COLORS["subtext"],
            ).pack(padx=8, pady=4)

    def _build_action_buttons(self, parent):
        """Build the bottom action buttons (Start / Cancel)."""
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.pack(fill="x")

        ctk.CTkButton(
            btn_frame, text="  ▶  Start Dry Run  ",
            font=("Segoe UI", 12, "bold"),
            fg_color=COLORS["yellow"], text_color=COLORS["bg_tertiary"],
            hover_color=COLORS["peach"], corner_radius=8,
            width=170, height=38, command=self._submit,
        ).pack(side="right")

        ctk.CTkButton(
            btn_frame, text="Cancel", font=("Segoe UI", 11),
            fg_color=COLORS["surface"], text_color=COLORS["text"],
            hover_color=COLORS["overlay"], corner_radius=8,
            width=90, height=38, command=self._cancel,
        ).pack(side="right", padx=(0, 8))

    # ──────── Helpers ────────

    def _toggle_all(self):
        state = self.select_all_var.get()
        for bv in self.var_checkboxes.values():
            bv.set(state)

    def _submit(self, _event=None):
        inputs = self._parse_bulk_inputs()
        if not inputs:
            inputs = [e.get().strip() for e in self.input_entries]
        trace_columns, traced_vars = self._resolve_trace_columns()
        self.result = {
            'inputs': inputs,
            'traced_vars': traced_vars,
            'trace_columns': trace_columns,
        }
        self.destroy()

    def _parse_bulk_inputs(self):
        """Parse comma-separated values from the bulk entry field."""
        if not self.bulk_entry:
            return []
        try:
            bulk = self.bulk_entry.get().strip()
            if bulk:
                return [v.strip() for v in bulk.split(",") if v.strip()]
        except Exception:
            pass
        return []

    def _resolve_trace_columns(self):
        """Determine trace columns from custom entry or checkbox selection.

        Returns (trace_columns, traced_vars) — one will be None.
        """
        custom_cols = self.custom_cols_entry.get().strip()
        if custom_cols:
            return [c.strip() for c in custom_cols.split(',') if c.strip()], None
        if not self.select_all_var.get():
            traced = {name for name, bv in self.var_checkboxes.items() if bv.get()}
            return None, traced
        return None, None

    def _cancel(self):
        self.result = None
        self.destroy()


# ═══════════════════════════════════════════════════════
#  Trace-Table Window
# ═══════════════════════════════════════════════════════

class TraceTableWindow(ctk.CTkToplevel):
    """Displays the dry-run trace table in Cambridge 9618 exam format.
    
    Key features matching exam paper style:
    - Array elements expanded into individual columns (e.g. Chars[1], Chars[2])
    - Values only shown when they change
    - DECLARE/definition steps filtered out
    - Clean column headers without Step/Statement/Note clutter
    """

    def __init__(self, parent, interpreter):
        super().__init__(parent)
        self.title("Trace Table — Cambridge 9618 Exam Format")
        self.configure(fg_color=COLORS["bg_tertiary"])
        self.geometry("1020x620")
        self.minsize(700, 400)
        self.transient(parent)
        self.interp = interpreter
        self._build_ui()
        self._populate()

    def _build_ui(self):
        # ── Top info bar ──
        top = ctk.CTkFrame(self, fg_color=COLORS["toolbar_bg"], corner_radius=0, height=44)
        top.pack(fill="x")
        top.pack_propagate(False)

        # Count execution steps (skip declares/definitions)
        exec_steps = len(self.interp.get_cambridge_trace())
        vc = len(self.interp.get_all_var_names())
        used = self.interp.input_index

        ctk.CTkLabel(
            top,
            text=f"  Trace Table  │  {exec_steps} steps  │  "
                 f"{vc} columns  │  {used} inputs consumed",
            font=("Segoe UI", 11, "bold"), text_color=COLORS["accent"],
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            top, text="Export", font=("Segoe UI", 10),
            fg_color=COLORS["button_bg"], text_color=COLORS["text"],
            hover_color=COLORS["button_hover"], corner_radius=6,
            width=70, height=28, command=self._export,
        ).pack(side="right", padx=10, pady=8)

        self._configure_treeview_style()
        self._build_tree()
        self._build_output_log_bar()

    def _configure_treeview_style(self):
        """Configure the ttk Treeview style to match the Catppuccin dark theme."""
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Trace.Treeview",
                        background=COLORS["bg"],
                        foreground=COLORS["text"],
                        fieldbackground=COLORS["bg"],
                        font=("Cascadia Code", 10),
                        rowheight=26)
        style.configure("Trace.Treeview.Heading",
                        background=COLORS["surface"],
                        foreground=COLORS["accent"],
                        font=("Segoe UI", 10, "bold"),
                        padding=(4, 4))
        style.map("Trace.Treeview",
                  background=[("selected", COLORS["selection"])],
                  foreground=[("selected", COLORS["text"])])
        style.map("Trace.Treeview.Heading",
                  background=[("active", COLORS["overlay"])])

    def _build_tree(self):
        """Build the Treeview widget with scrollbars and column headings."""
        tree_frame = tk.Frame(self, bg=COLORS["bg"])
        tree_frame.pack(fill="both", expand=True)

        xscroll = ctk.CTkScrollbar(tree_frame, orientation="horizontal",
                                   fg_color=COLORS["bg"],
                                   button_color=COLORS["surface"],
                                   button_hover_color=COLORS["overlay"])
        xscroll.pack(side="bottom", fill="x")

        yscroll = ctk.CTkScrollbar(tree_frame, orientation="vertical",
                                   fg_color=COLORS["bg"],
                                   button_color=COLORS["surface"],
                                   button_hover_color=COLORS["overlay"])
        yscroll.pack(side="right", fill="y")

        var_names = self.interp.get_all_var_names()
        columns = ['line'] + var_names

        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings",
                                 style="Trace.Treeview",
                                 xscrollcommand=xscroll.set,
                                 yscrollcommand=yscroll.set)
        self.tree.pack(fill="both", expand=True)
        xscroll.configure(command=self.tree.xview)
        yscroll.configure(command=self.tree.yview)

        # Line column
        self.tree.heading('line', text='Line')
        self.tree.column('line', width=55, minwidth=40, anchor='center')

        # Variable columns — Cambridge style
        for vn in var_names:
            self.tree.heading(vn, text=vn)
            width = 80 if '[' in vn else 100
            self.tree.column(vn, width=width, minwidth=50, anchor='center')

        self.tree.tag_configure('even', background=COLORS["bg"])
        self.tree.tag_configure('odd', background=COLORS["bg_secondary"])

    def _build_output_log_bar(self):
        """Build the optional output-log bar at the bottom."""
        if not self.interp.output_log:
            return
        out_bar = ctk.CTkFrame(self, fg_color=COLORS["bg_secondary"],
                               corner_radius=0, height=32)
        out_bar.pack(fill="x")
        out_bar.pack_propagate(False)
        ctk.CTkLabel(
            out_bar, text="  OUTPUT LOG:",
            font=("Segoe UI", 9, "bold"), text_color=COLORS["subtext"],
        ).pack(side="left", padx=4)
        out_text = " │ ".join(self.interp.output_log)
        ctk.CTkLabel(
            out_bar, text=out_text,
            font=("Cascadia Code", 9), text_color=COLORS["green"],
        ).pack(side="left", padx=8)

    # ──────── helpers ────────

    def _fmt(self, val):
        """Format a value for display in Cambridge trace-table style."""
        if isinstance(val, bool):
            return "TRUE" if val else "FALSE"
        if isinstance(val, str):
            return f"'{val}'" if len(val) == 1 else val
        return str(val)

    def _format_row_values(self, entry, var_names, prev_values):
        """Build a row of display values, showing only changed columns.

        Updates *prev_values* in-place and returns the list of cell strings.
        """
        # The 'line' column is handled separately in _populate and _export_csv
        values = []
        for vn in var_names:
            val = entry['variables'].get(vn)
            if val is None:
                values.append('')
                continue
            formatted = self._fmt(val)
            if prev_values.get(vn) != formatted:
                values.append(formatted)
                prev_values[vn] = formatted
            else:
                values.append('')
        return values

    def _populate(self):
        """Populate the trace table in Cambridge exam format."""
        var_names = self.interp.get_all_var_names()
        prev_values = {}  # Track previous formatted value per column
        row_idx = 0
        
        for entry in self.interp.get_cambridge_trace(): # Use get_cambridge_trace directly
            values = self._format_row_values(entry, var_names, prev_values)
            
            tag = 'even' if row_idx % 2 == 0 else 'odd'
            self.tree.insert('', 'end', values=[entry['line']] + values, tags=(tag,)) # Add line number here
            row_idx += 1

    def _export(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Export Trace Table",
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        if path.endswith('.csv'):
            self._export_csv(path)
        else:
            self._export_txt(path)

    def _export_txt(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.interp.format_trace_text())
        messagebox.showinfo("Export", f"Exported to:\n{path}", parent=self)

    def _export_csv(self, path):
        """Export the trace table as a CSV file."""
        var_names = self.interp.get_all_var_names()
        headers = ['Line'] + var_names
        prev_values = {}
        with open(path, 'w', encoding='utf-8') as f:
            f.write(','.join(headers) + '\n')
            for entry in self.interp.get_cambridge_trace():
                values = self._format_row_values(entry, var_names, prev_values)
                # Quote non-empty cells for CSV safety
                row = [f'"{v}"' if v != '' else '' for v in values]
                row.insert(0, str(entry['line'])) # Insert line number at the beginning, unquoted
                f.write(','.join(row) + '\n')
        messagebox.showinfo("Export", f"Exported to:\n{path}", parent=self)


# ═══════════════════════════════════════════════════════
#  Main IDE Application
# ═══════════════════════════════════════════════════════

class PseudocodeIDE:
    """Main IDE window built on CustomTkinter (dark title-bar, rounded buttons)."""

    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("9618 Pseudocode IDE")
        self.root.configure(fg_color=COLORS["bg_tertiary"])
        self.root.geometry("1100x750")
        self.root.minsize(800, 500)
        
        # Extracted Managers
        self.file_manager = IDEFileManager(self)
        self.execution_engine = IDEExecutionEngine(self)

        self._build_ui()
        self._bind_shortcuts()
        self._load_example()

    # ═══════ UI Construction ═══════

    def _build_ui(self):
        # Thin accent stripe across the very top
        ctk.CTkFrame(self.root, fg_color=COLORS["accent"],
                     corner_radius=0, height=2).pack(fill="x", side="top")

        self._build_toolbar()

        self.paned = tk.PanedWindow(
            self.root, orient="vertical",
            bg=COLORS["border"], sashwidth=4, sashrelief="flat",
        )
        self.paned.pack(fill="both", expand=True)

        self._build_editor()
        self._build_output()
        self._build_statusbar()

    def _build_toolbar(self):
        toolbar = ctk.CTkFrame(self.root, fg_color=COLORS["toolbar_bg"],
                               corner_radius=0, height=48)
        toolbar.pack(fill="x", side="top")
        toolbar.pack_propagate(False)

        # ── File buttons ──
        left = ctk.CTkFrame(toolbar, fg_color="transparent")
        left.pack(side="left", padx=8)

        for text, cmd in [("New", self.new_file), ("Open", self.open_file),
                          ("Save", self.save_file), ("Save As", self.save_as)]:
            ctk.CTkButton(
                left, text=text, command=cmd,
                fg_color=COLORS["button_bg"], text_color=COLORS["text"],
                hover_color=COLORS["button_hover"], corner_radius=6,
                font=("Segoe UI", 10), width=68, height=30,
            ).pack(side="left", padx=3, pady=8)

        # Vertical separator
        sep = ctk.CTkFrame(toolbar, fg_color=COLORS["overlay"],
                           width=2, height=26, corner_radius=1)
        sep.pack(side="left", padx=10)

        # ── Action buttons ──
        center = ctk.CTkFrame(toolbar, fg_color="transparent")
        center.pack(side="left", padx=4)

        self.run_btn = ctk.CTkButton(
            center, text="▶  Run", command=self.run_code,
            fg_color=COLORS["green"], text_color=COLORS["bg_tertiary"],
            hover_color="#8bd49a", corner_radius=8,
            font=("Segoe UI", 12, "bold"), width=90, height=32,
        )
        self.run_btn.pack(side="left", padx=4, pady=8)

        self.dryrun_btn = ctk.CTkButton(
            center, text="◇  Dry Run", command=self.dry_run,
            fg_color=COLORS["yellow"], text_color=COLORS["bg_tertiary"],
            hover_color="#f5d49a", corner_radius=8,
            font=("Segoe UI", 12, "bold"), width=110, height=32,
        )
        self.dryrun_btn.pack(side="left", padx=4, pady=8)

        ctk.CTkButton(
            center, text="✕  Clear", command=self.clear_output,
            fg_color=COLORS["button_bg"], text_color=COLORS["text"],
            hover_color=COLORS["button_hover"], corner_radius=8,
            font=("Segoe UI", 11), width=85, height=32,
        ).pack(side="left", padx=4, pady=8)

        # ── Branding ──
        ctk.CTkLabel(
            toolbar, text="9618 Pseudocode",
            font=("Segoe UI", 12, "bold"), text_color=COLORS["accent"],
        ).pack(side="right", padx=16)

    def _build_editor(self):
        editor_frame = tk.Frame(self.root, bg=COLORS["bg"])
        self.paned.add(editor_frame, stretch="always")

        # Tab bar
        tab_bar = ctk.CTkFrame(editor_frame, fg_color=COLORS["bg_tertiary"],
                               corner_radius=0, height=32)
        tab_bar.pack(fill="x")
        tab_bar.pack_propagate(False)
        self.tab_label = ctk.CTkLabel(
            tab_bar, text="  untitled.pse  ",
            font=("Segoe UI", 10), text_color=COLORS["text"],
            fg_color=COLORS["tab_active"], corner_radius=6,
        )
        self.tab_label.pack(side="left", padx=6, pady=3)

        # Editor container
        container = tk.Frame(editor_frame, bg=COLORS["bg"])
        container.pack(fill="both", expand=True)

        # Determine monospace font
        self.code_font = tkfont.Font(family="Cascadia Code", size=12)
        if "Cascadia Code" not in tkfont.families():
            for fam in ("JetBrains Mono", "Fira Code", "Consolas", "Courier New", "monospace"):
                if fam in tkfont.families():
                    self.code_font = tkfont.Font(family=fam, size=12)
                    break

        # Line numbers
        self.line_numbers = LineNumbers(
            container, None, width=50,
            bg=COLORS["gutter"], highlightthickness=0, bd=0,
        )
        self.line_numbers.pack(side="left", fill="y")

        # Modern scrollbar (vertical)
        scrollbar = ctk.CTkScrollbar(
            container, orientation="vertical",
            fg_color=COLORS["bg"],
            button_color=COLORS["surface"],
            button_hover_color=COLORS["overlay"],
        )
        scrollbar.pack(side="right", fill="y")

        # Code editor
        self.editor = CodeEditor(
            container, font=self.code_font,
            bg=COLORS["bg"], fg=COLORS["text"],
            insertbackground=COLORS["cursor"], insertwidth=2,
            selectbackground=COLORS["selection"], selectforeground=COLORS["text"],
            relief="flat", bd=0, padx=8, pady=8, wrap="none",
            undo=True, maxundo=-1,
            tabs=tkfont.Font(font=self.code_font).measure("    "),
            yscrollcommand=scrollbar.set,
        )
        self.editor.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=self._on_scroll)

        self.line_numbers.text_widget = self.editor
        self.line_numbers.font = self.code_font

        self.editor.bind("<<ContentChanged>>", self._on_editor_change)
        self.editor.bind("<Configure>", lambda e: self.line_numbers.redraw())
        self.editor.bind("<KeyRelease>", self._on_editor_change)
        self.editor.bind("<ButtonRelease-1>", self._on_editor_change)

    def _build_output(self):
        output_frame = tk.Frame(self.root, bg=COLORS["output_bg"])
        self.paned.add(output_frame, stretch="never", height=200)

        header = ctk.CTkFrame(output_frame, fg_color=COLORS["bg_secondary"],
                              corner_radius=0, height=28)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(
            header, text="  OUTPUT",
            font=("Segoe UI", 9, "bold"), text_color=COLORS["subtext"],
        ).pack(side="left", padx=4, pady=2)

        self.output = OutputPanel(
            output_frame,
            font=tkfont.Font(family=self.code_font.actual()["family"], size=11),
            bg=COLORS["output_bg"], fg=COLORS["text"],
            relief="flat", bd=0, padx=12, pady=8, wrap="word",
        )
        self.output.pack(fill="both", expand=True)

    def _build_statusbar(self):
        status = ctk.CTkFrame(self.root, fg_color=COLORS["status_bg"],
                              corner_radius=0, height=26)
        status.pack(fill="x", side="bottom")
        status.pack_propagate(False)

        self.status_pos = ctk.CTkLabel(
            status, text="Ln 1, Col 1",
            font=("Segoe UI", 9), text_color=COLORS["subtext"],
        )
        self.status_pos.pack(side="right", padx=12)

        self.status_msg = ctk.CTkLabel(
            status, text="Ready",
            font=("Segoe UI", 9), text_color=COLORS["subtext"],
        )
        self.status_msg.pack(side="left", padx=12)

        ctk.CTkLabel(
            status,
            text="Ctrl+S Save  │  F5 Run  │  F6 Dry Run  │  Ctrl+O Open  │  Ctrl+N New  │  Ctrl+±  Zoom",
            font=("Segoe UI", 8), text_color=COLORS["overlay"],
        ).pack(side="right", padx=20)

    # ═══════ Helpers ═══════

    def _on_scroll(self, *args):
        self.editor.yview(*args)
        self.line_numbers.redraw()

    def _on_editor_change(self, _event=None):
        self.file_manager.ide.editor.edit_modified(True)
        self._update_tab()
        pos = self.editor.index("insert")
        line, col = pos.split(".")
        self.status_pos.configure(text=f"Ln {line}, Col {int(col)+1}")

    def _bind_shortcuts(self):
        self.root.bind("<F5>", lambda e: self.run_code())
        self.root.bind("<F6>", lambda e: self.dry_run())
        self.root.bind("<Control-s>", lambda e: self.save_file())
        self.root.bind("<Control-S>", lambda e: self.save_file())
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-O>", lambda e: self.open_file())
        self.root.bind("<Control-n>", lambda e: self.new_file())
        self.root.bind("<Control-N>", lambda e: self.new_file())
        # Font-size zoom
        self.root.bind("<Control-equal>", lambda e: self._change_font_size(1))
        self.root.bind("<Control-plus>", lambda e: self._change_font_size(1))
        self.root.bind("<Control-minus>", lambda e: self._change_font_size(-1))
        self.root.bind("<Control-0>", lambda e: self._reset_font_size())

    def _change_font_size(self, delta):
        current = self.code_font.actual()["size"]
        new_size = max(8, min(28, current + delta))
        self.code_font.configure(size=new_size)
        self.editor._setup_tags()  # rebuild bold/italic fonts at new size
        self.editor.highlight_syntax()
        self.line_numbers.redraw()
        self._set_status(f"Font size: {new_size}", COLORS["subtext"])

    def _reset_font_size(self):
        self.code_font.configure(size=12)
        self.editor._setup_tags()
        self.editor.highlight_syntax()
        self.line_numbers.redraw()
        self._set_status("Font size: 12 (default)", COLORS["subtext"])

    def _load_example(self):
        example = (
            '// Welcome to 9618 Pseudocode IDE\n'
            '// Press F5 or click Run to execute\n\n'
            'DECLARE name : STRING\n'
            'DECLARE age : INTEGER\n\n'
            'name <- "Alice"\n'
            'age <- 20\n\n'
            'OUTPUT "Hello, " & name & "!"\n'
            'OUTPUT "You are", age, "years old."\n\n'
            '// Try changing the values and running again!\n\n'
            'FUNCTION Square(BYVAL n : INTEGER) RETURNS INTEGER\n'
            '    RETURN n * n\n'
            'ENDFUNCTION\n\n'
            'OUTPUT "Square of 7:", Square(7)\n'
        )
        self.editor.insert("1.0", example)
        self.editor.edit_modified(False)
        self.editor.highlight_syntax()
        self.editor._highlight_current_line()
        self.root.after(50, self.line_numbers.redraw)

    def _update_tab(self):
        title = "9618 Pseudocode IDE"
        if self.file_manager.current_file:
            title = f"{os.path.basename(self.file_manager.current_file)} - {title}"
        self.root.title(title)

    def _set_status(self, msg, color=None):
        self.status_msg.configure(text=msg, text_color=color or COLORS["subtext"])

    # ═══════ File Operations ═══════

    def new_file(self):
        self.file_manager.new_file()

    def open_file(self):
        self.file_manager.open_file()

    def save_file(self):
        self.file_manager.save_file()

    def save_as(self):
        self.file_manager.save_as()

    def clear_output(self):
        self.output.clear()
        self.editor.tag_remove("error_line", "1.0", "end")

    # ═══════ Run Code ═══════

    def run_code(self):
        self.execution_engine.run_code()

    def dry_run(self):
        self.execution_engine.dry_run()

    def _show_dryrun_success(self, interpreter):
        def _update():
            steps = len(interpreter.trace)
            vc = len(interpreter.get_all_var_names())
            used = interpreter.input_index
            total = len(interpreter.input_queue)
            self.output.append(
                f"\nDry run complete: {steps} steps, {vc} variables, "
                f"{used}/{total} inputs used.\n", "success")
            self._set_status("Dry Run Complete", COLORS["success"])
            self._open_trace_window(interpreter)
        self.root.after(0, _update)

    def _open_trace_window(self, interpreter):
        def _open():
            TraceTableWindow(self.root, interpreter)
        self.root.after(50, _open)

    # ═══════ Status helpers ═══════

    def _show_error(self, kind, message, line=None):
        def _update():
            self.output.append(f"\n{kind}: {message}\n", "error")
            self._set_status(kind, COLORS["error"])
            if line:
                self.editor.mark_error_line(line)
        self.root.after(0, _update)

    def _show_success(self):
        def _update():
            self.output.append("\nProgram finished successfully.\n", "success")
            self._set_status("Finished", COLORS["success"])
        self.root.after(0, _update)

    def start(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PseudocodeIDE()
    app.start()
