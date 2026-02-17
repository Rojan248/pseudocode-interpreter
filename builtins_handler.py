"""
Built-in function implementations for the 9618 Pseudocode Interpreter.

Extracted from interpreter.py to improve cohesion.
Each built-in is a standalone function that receives evaluated arguments
and returns a result. The dispatch table maps function names to handlers.
"""
import os
import math
import random


# ── Individual built-in implementations ──

def _builtin_length(args):
    return len(str(args[0]))

def _builtin_ucase(args):
    return str(args[0]).upper()

def _builtin_lcase(args):
    return str(args[0]).lower()

def _builtin_left(args):
    return str(args[0])[:int(args[1])]

def _builtin_right(args):
    n = int(args[1])
    return str(args[0])[-n:] if n > 0 else ""

def _builtin_mid(args):
    s = str(args[0])
    start = int(args[1]) - 1
    length = int(args[2])
    return s[start:start + length]

def _builtin_int(args):
    return int(float(args[0]))

def _builtin_num_to_str(args):
    return str(args[0])

def _builtin_str_to_num(args):
    s = str(args[0])
    return float(s) if '.' in s else int(s)

def _builtin_asc(args):
    return ord(str(args[0])[0])

def _builtin_chr(args):
    return chr(int(args[0]))

def _builtin_sqrt(args):
    return math.sqrt(float(args[0]))

def _builtin_rand(args):
    return random.random() * float(args[0])


# ── Dispatch table ──

BUILTIN_DISPATCH = {
    'LENGTH':     _builtin_length,
    'UCASE':      _builtin_ucase,
    'LCASE':      _builtin_lcase,
    'LEFT':       _builtin_left,
    'RIGHT':      _builtin_right,
    'MID':        _builtin_mid,
    'INT':        _builtin_int,
    'NUM_TO_STR': _builtin_num_to_str,
    'STR_TO_NUM': _builtin_str_to_num,
    'ASC':        _builtin_asc,
    'CHR':        _builtin_chr,
    'SQRT':       _builtin_sqrt,
    'RAND':       _builtin_rand,
}

# Set of all built-in function names (for quick membership checks)
BUILTIN_NAMES = frozenset(BUILTIN_DISPATCH.keys() | {'EOF'})


def call_builtin(name, args, open_files=None):
    """
    Dispatch a built-in function call.
    
    EOF is handled separately because it needs access to the file table.
    All other builtins use the dispatch table.
    
    Returns the result of the built-in function.
    Raises KeyError if the function name is not recognized.
    """
    if name == 'EOF':
        return _builtin_eof(args, open_files or {})

    handler = BUILTIN_DISPATCH.get(name)
    if handler is None:
        raise KeyError(f"Unknown built-in function: {name}")
    return handler(args)


def _builtin_eof(args, open_files):
    """EOF requires access to the interpreter's open_files dict."""
    filename = str(args[0])
    if filename not in open_files:
        return True
    f = open_files[filename]
    cur = f.tell()
    f.seek(0, os.SEEK_END)
    end = f.tell()
    f.seek(cur, os.SEEK_SET)
    return cur >= end
