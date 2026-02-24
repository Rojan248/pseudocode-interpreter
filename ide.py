"""
9618 Pseudocode IDE — Modern Desktop Interface
CustomTkinter + Tkinter hybrid GUI with dark Catppuccin theme,
syntax highlighting, line numbers, current-line highlight,
integrated output, and dry-run trace-table support.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import customtkinter as ctk
import os
import re

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ide_theme import (COLORS, KEYWORDS_CONTROL, KEYWORDS_DECL,
                       KEYWORDS_TYPE, KEYWORDS_IO, KEYWORDS_OP, BUILTINS)
from ide_managers import IDEFileManager, IDEExecutionEngine
from ide_trace import TraceTableWindow

# ── CustomTkinter global setup ──
ctk.set_appearance_mode("dark")


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
