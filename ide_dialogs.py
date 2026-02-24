"""
IDE Dialogs — InputDialog, RedirectOutput, DryRunSetupDialog.
Extracted from ide.py to reduce module size and function count.
"""
import tkinter as tk
import customtkinter as ctk

from ide_theme import COLORS


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
