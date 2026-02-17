"""
File I/O operations for the 9618 Pseudocode Interpreter.

Extracted from interpreter.py to improve cohesion.
Uses a dispatch table to replace the deeply nested if-elif chain
in visit_FileStmt (cc=26, bumps=7).
"""
import os
import pickle
from symbol_table import DataType


class InterpreterFileError(Exception):
    """Raised when a file operation fails."""
    pass


def _require_open(filename, open_files):
    """Guard clause: raise if a file is not open."""
    if filename not in open_files:
        raise InterpreterFileError(f"File {filename} is not open")


# ── Individual file operation handlers ──

def _op_open(filename, stmt, open_files, evaluate_fn):
    """OPENFILE <filename> FOR <mode>"""
    mode_map = {"READ": "r", "WRITE": "w", "APPEND": "a"}
    try:
        if stmt.mode == "RANDOM":
            if not os.path.exists(filename):
                open(filename, 'wb').close()
            open_files[filename] = open(filename, "r+b")
            open_files[filename + "__mode"] = "RANDOM"
        else:
            open_files[filename] = open(filename, mode_map.get(stmt.mode, "r"))
    except IOError as e:
        raise InterpreterFileError(f"Failed to open file {filename}: {e}")


def _op_close(filename, stmt, open_files, evaluate_fn):
    """CLOSEFILE <filename>"""
    if filename in open_files:
        open_files[filename].close()
        del open_files[filename]


def _op_write(filename, stmt, open_files, evaluate_fn):
    """WRITEFILE <filename>, <data>"""
    _require_open(filename, open_files)
    open_files[filename].write(str(evaluate_fn(stmt.data)) + "\n")


def _op_read(filename, stmt, open_files, evaluate_fn):
    """READFILE <filename>, <variable>"""
    _require_open(filename, open_files)
    line = open_files[filename].readline()
    return ('assign', stmt.variable, line.strip() if line else "", DataType.STRING)


def _op_seek(filename, stmt, open_files, evaluate_fn):
    """SEEK <filename>, <position>"""
    _require_open(filename, open_files)
    open_files[filename + "__seek"] = int(evaluate_fn(stmt.data))


def _op_getrecord(filename, stmt, open_files, evaluate_fn):
    """GETRECORD <filename>, <variable>"""
    _require_open(filename, open_files)
    try:
        record_data = pickle.load(open_files[filename])
    except Exception:
        record_data = None
    if record_data is not None:
        return ('assign_infer', stmt.variable, record_data)
    return None


def _op_putrecord(filename, stmt, open_files, evaluate_fn):
    """PUTRECORD <filename>, <variable>"""
    _require_open(filename, open_files)
    return ('putrecord', stmt.variable, filename)


# ── Dispatch table ──

FILE_OP_DISPATCH = {
    "OPEN":      _op_open,
    "CLOSE":     _op_close,
    "WRITE":     _op_write,
    "READ":      _op_read,
    "SEEK":      _op_seek,
    "GETRECORD": _op_getrecord,
    "PUTRECORD": _op_putrecord,
}


def execute_file_operation(stmt, open_files, evaluate_fn):
    """
    Execute a file operation using the dispatch table.

    Args:
        stmt: FileStmt AST node with .operation, .filename, .mode, .data, .variable
        open_files: dict of currently open file handles
        evaluate_fn: callable to evaluate expressions (interpreter.evaluate)

    Returns:
        None, or a tuple describing a side-effect the interpreter must perform:
          ('assign', var_name, value, data_type)
          ('assign_infer', var_name, value)
          ('putrecord', var_name, filename)
    """
    filename = str(evaluate_fn(stmt.filename))
    handler = FILE_OP_DISPATCH.get(stmt.operation)
    if handler is None:
        raise InterpreterFileError(f"Unknown file operation: {stmt.operation}")
    return handler(filename, stmt, open_files, evaluate_fn)
