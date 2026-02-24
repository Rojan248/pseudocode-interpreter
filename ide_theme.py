"""
IDE Theme — Shared color palette and keyword lists.
Extracted from ide.py so that sub-modules can import constants
without circular dependencies.
"""

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
