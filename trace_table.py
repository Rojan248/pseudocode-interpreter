import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
from theme import COLORS

class TraceTableWindow(ctk.CTkToplevel):
    """Displays the dry-run trace table in A-Level exam format."""

    def __init__(self, parent, interpreter):
        super().__init__(parent)
        self.title("Trace Table — Dry Run Result")
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

        steps = len(self.interp.trace)
        vc = len(self.interp.get_all_var_names())
        used = self.interp.input_index

        ctk.CTkLabel(
            top,
            text=f"  Trace Table  │  {steps} steps  │  "
                 f"{vc} variables  │  {used} inputs consumed",
            font=("Segoe UI", 11, "bold"), text_color=COLORS["accent"],
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            top, text="Export", font=("Segoe UI", 10),
            fg_color=COLORS["button_bg"], text_color=COLORS["text"],
            hover_color=COLORS["button_hover"], corner_radius=6,
            width=70, height=28, command=self._export,
        ).pack(side="right", padx=10, pady=8)

        # ── Treeview styling  (ttk — no CTk equivalent) ──
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

        # ── Tree + scrollbars ──
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
        columns = ['step', 'line', 'statement', 'note'] + var_names

        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings",
                                 style="Trace.Treeview",
                                 xscrollcommand=xscroll.set,
                                 yscrollcommand=yscroll.set)
        self.tree.pack(fill="both", expand=True)
        xscroll.configure(command=self.tree.xview)
        yscroll.configure(command=self.tree.yview)

        self.tree.heading('step', text='Step')
        self.tree.column('step', width=55, minwidth=40, anchor='center')
        self.tree.heading('line', text='Line')
        self.tree.column('line', width=55, minwidth=40, anchor='center')
        self.tree.heading('statement', text='Statement')
        self.tree.column('statement', width=200, minwidth=100)
        self.tree.heading('note', text='Note')
        self.tree.column('note', width=200, minwidth=80)
        for vn in var_names:
            self.tree.heading(vn, text=vn)
            self.tree.column(vn, width=120, minwidth=60, anchor='center')

        self.tree.tag_configure('even', background=COLORS["bg"])
        self.tree.tag_configure('odd', background=COLORS["bg_secondary"])

        # ── Output log bar ──
        if self.interp.output_log:
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
        if isinstance(val, bool):
            return "TRUE" if val else "FALSE"
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            return str(val)
        return str(val)

    def _populate(self):
        var_names = self.interp.get_all_var_names()
        for i, entry in enumerate(self.interp.trace):
            values = [entry['step'], entry['line'],
                      entry['statement'], entry['note']]
            for vn in var_names:
                val = entry['variables'].get(vn, '')
                if isinstance(val, dict):
                    val = str(val)
                elif val == '':
                    val = ''
                else:
                    val = self._fmt(val)
                values.append(val)
            tag = 'even' if i % 2 == 0 else 'odd'
            self.tree.insert('', 'end', values=values, tags=(tag,))

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
        var_names = self.interp.get_all_var_names()
        headers = ['Step', 'Line', 'Statement', 'Note'] + var_names
        with open(path, 'w', encoding='utf-8') as f:
            f.write(','.join(headers) + '\n')
            for entry in self.interp.trace:
                row = [str(entry['step']), str(entry['line']),
                       f'"{ entry["statement"] }"', f'"{ entry["note"] }"']
                for vn in var_names:
                    val = entry['variables'].get(vn, '')
                    row.append(f'"{self._fmt(val)}"')
                f.write(','.join(row) + '\n')
        messagebox.showinfo("Export", f"Exported to:\n{path}", parent=self)
