"""
IDE Trace Table Window — Cambridge 9618 exam-format trace display.
Extracted from ide.py to reduce module size and function count.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk

from ide_theme import COLORS


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
        prev_values = {}
        row_idx = 0
        
        for entry in self.interp.get_cambridge_trace():
            values = self._format_row_values(entry, var_names, prev_values)
            
            tag = 'even' if row_idx % 2 == 0 else 'odd'
            self.tree.insert('', 'end', values=[entry['line']] + values, tags=(tag,))
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
                row = [f'"{v}"' if v != '' else '' for v in values]
                row.insert(0, str(entry['line']))
                f.write(','.join(row) + '\n')
        messagebox.showinfo("Export", f"Exported to:\n{path}", parent=self)
