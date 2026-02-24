"""
IDE Helper Managers — File operations and execution engine.
Extracted from ide.py to reduce module size and function count.
"""
import os
import sys
import threading
from tkinter import filedialog

from lexer import Lexer, LexerError
from parser import Parser, ParserError
from interpreter import Interpreter, InterpreterError
from dry_run_interpreter import DryRunInterpreter
from symbol_table import SymbolTable
from ide_theme import COLORS


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
        from ide_dialogs import DryRunSetupDialog

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
        from ide_dialogs import InputDialog, RedirectOutput

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
        from ide_dialogs import RedirectOutput

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
