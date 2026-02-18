import re
from enum import Enum, auto

class TokenType(Enum):
    # Literals and identifiers
    INTEGER = auto()
    REAL = auto()
    STRING = auto()
    CHAR_LITERAL = auto()
    BOOLEAN = auto()
    IDENTIFIER = auto()
    
    # 9618-specific operators (CRITICAL: priority order matters for regex alternation)
    ASSIGN = auto()      # <-  MUST be recognized before LT
    NE = auto()          # <>  MUST be recognized before LT
    LE = auto()          # <=  MUST be recognized before LT
    GE = auto()          # >=  MUST be recognized before GT
    EQ = auto()          # =
    LT = auto()          # <   LOWEST priority among angle brackets
    GT = auto()          # >   LOWEST priority among angle brackets
    
    # Arithmetic operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    DIV = auto()         # Integer division keyword/operator
    MOD = auto()         # Modulo keyword/operator
    AMPER = auto()       # & for string concatenation
    
    # Keywords for declarations and types
    DECLARE = auto()
    CONSTANT = auto()
    INTEGER_KW = auto()
    REAL_KW = auto()
    STRING_KW = auto()
    BOOLEAN_KW = auto()
    CHAR_KW = auto()
    DATE_KW = auto()
    ARRAY = auto()
    OF = auto()
    TYPE = auto()
    ENDTYPE = auto()
    # File I/O
    OPENFILE = auto()
    READFILE = auto()
    WRITEFILE = auto()
    CLOSEFILE = auto()
    READ_MODE = auto()   # READ keyword for file mode
    WRITE_MODE = auto()  # WRITE keyword for file mode
    APPEND_MODE = auto() # APPEND keyword
    RANDOM_MODE = auto() # RANDOM keyword for file mode
    SEEK = auto()        # SEEK command for random files
    GETRECORD = auto()   # GETRECORD command for random files
    PUTRECORD = auto()   # PUTRECORD command for random files
    
    # Keywords for procedures and functions
    PROCEDURE = auto()
    ENDPROCEDURE = auto()
    FUNCTION = auto()
    ENDFUNCTION = auto()
    RETURNS = auto()
    BYREF = auto()
    BYVAL = auto()
    CALL = auto()
    RETURN = auto()
    
    # Control flow keywords
    IF = auto()
    THEN = auto()
    ELSE = auto()
    ENDIF = auto()
    CASE = auto()
    OTHERWISE = auto()
    ENDCASE = auto()
    FOR = auto()
    TO = auto()
    STEP = auto()
    NEXT = auto()
    REPEAT = auto()
    UNTIL = auto()
    WHILE = auto()
    DO = auto()
    ENDWHILE = auto()
    
    # Logic
    AND = auto()
    OR = auto()
    NOT = auto()

    # I/O keywords
    INPUT = auto()
    OUTPUT = auto()
    
    # OOP keywords
    CLASS = auto()
    ENDCLASS = auto()
    INHERITS = auto()
    PUBLIC = auto()
    PRIVATE = auto()
    NEW = auto()
    SUPER = auto()
    
    # Delimiters and punctuation
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    LBRACKET = auto()    # [  array index
    RBRACKET = auto()    # ]  array index
    COLON = auto()       # :  type separator in DECLARE
    SEMICOLON = auto()   # ;  statement separator (rare in 9618)
    COMMA = auto()       # ,  parameter/element separator
    DOT = auto()         # .  record access
    RANGE = auto()       # :  array bounds separator [1:10] (reused COLON usually, but maybe distinct in implementations)
    
    EOF = auto()

class Token:
    def __init__(self, type_, value, line, column):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.column})"

class LexerError(Exception):
    pass

class Lexer:
    # =====================================================================
    # ORDER IS SEMANTIC: earlier patterns have higher priority in alternation
    # =====================================================================
    TOKEN_SPECS = [
        # Whitespace and comments (discarded after position tracking)
        ('WHITESPACE', r'[ \t\r]+'),           # Horizontal whitespace
        ('NEWLINE', r'\n'),                     # Line breaks for line counting
        ('COMMENT', r'//[^\n]*'),               # Single-line comments
        
        # =====================================================================
        # CRITICAL SECTION: Multi-character operators BEFORE single-character
        # =====================================================================
        ('ASSIGN', r'←|<-'),                    # ← assignment (MUST precede LT)
        ('NE', r'<>'),                          # not equal (precedes LT, GT)
        ('LE', r'<='),                          # less or equal (precedes LT)
        ('GE', r'>='),                          # greater or equal (precedes GT)
        ('EQ', r'='),                           # equal (single char)
        
        # Single-character operators and delimiters
        ('LT', r'<'),                           
        ('GT', r'>'),                           
        ('PLUS', r'\+'),
        ('MINUS', r'-'),
        ('MULTIPLY', r'\*'),
        ('DIVIDE', r'/'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('LBRACKET', r'\['),
        ('RBRACKET', r'\]'),
        ('COLON', r':'),                        # type separator and range separator [1:10]
        ('SEMICOLON', r';'),
        ('COMMA', r','),
        ('DOT', r'\.'),
        ('AMPER', r'&'),
        
        # Literals 
        ('REAL', r'\d+\.\d+'),                  # Decimal number (before INTEGER)
        ('INTEGER', r'\d+'),                    # Integer number
        ('STRING', r'"[^"]*"'),                 # Double-quoted string
        ('CHAR', r"'[^']'"),                    # Single-quoted char
        
        # Keywords and identifiers
        ('IDENTIFIER', r'[A-Za-z_][A-Za-z0-9_]*'),
    ]

    KEYWORDS = {
        'DECLARE': TokenType.DECLARE,
        'CONSTANT': TokenType.CONSTANT,
        'INTEGER': TokenType.INTEGER_KW,
        'REAL': TokenType.REAL_KW,
        'STRING': TokenType.STRING_KW,
        'BOOLEAN': TokenType.BOOLEAN_KW,
        'CHAR': TokenType.CHAR_KW,
        'DATE': TokenType.DATE_KW,
        'ARRAY': TokenType.ARRAY,
        'OF': TokenType.OF,
        'TYPE': TokenType.TYPE,
        'ENDTYPE': TokenType.ENDTYPE,
        'PROCEDURE': TokenType.PROCEDURE,
        'ENDPROCEDURE': TokenType.ENDPROCEDURE,
        'FUNCTION': TokenType.FUNCTION,
        'ENDFUNCTION': TokenType.ENDFUNCTION,
        'RETURNS': TokenType.RETURNS,
        'BYREF': TokenType.BYREF,
        'BYVAL': TokenType.BYVAL,
        'IF': TokenType.IF,
        'THEN': TokenType.THEN,
        'ELSE': TokenType.ELSE,
        'ENDIF': TokenType.ENDIF,
        'CASE': TokenType.CASE,
        'OTHERWISE': TokenType.OTHERWISE,
        'ENDCASE': TokenType.ENDCASE,
        'FOR': TokenType.FOR,
        'TO': TokenType.TO,
        'STEP': TokenType.STEP,
        'NEXT': TokenType.NEXT,
        'REPEAT': TokenType.REPEAT,
        'UNTIL': TokenType.UNTIL,
        'WHILE': TokenType.WHILE,
        'DO': TokenType.DO,
        'ENDWHILE': TokenType.ENDWHILE,
        'RETURN': TokenType.RETURN,
        'CALL': TokenType.CALL,
        'INPUT': TokenType.INPUT,
        'OUTPUT': TokenType.OUTPUT,
        'DIV': TokenType.DIV,
        'MOD': TokenType.MOD,
        'AND': TokenType.AND,
        'OR': TokenType.OR,
        'NOT': TokenType.NOT,
        'TRUE': TokenType.BOOLEAN,
        'FALSE': TokenType.BOOLEAN,
        
        # OOP
        'CLASS': TokenType.CLASS,
        'ENDCLASS': TokenType.ENDCLASS,
        'INHERITS': TokenType.INHERITS,
        'PUBLIC': TokenType.PUBLIC,
        'PRIVATE': TokenType.PRIVATE,
        'NEW': TokenType.NEW,
        'SUPER': TokenType.SUPER,
        
        # File I/O
        'OPENFILE': TokenType.OPENFILE,
        'READFILE': TokenType.READFILE,
        'WRITEFILE': TokenType.WRITEFILE,
        'CLOSEFILE': TokenType.CLOSEFILE,
        'READ': TokenType.READ_MODE,    # Mode for OPENFILE
        'WRITE': TokenType.WRITE_MODE,  # Mode for OPENFILE (distinct from OUTPUT?)
        'APPEND': TokenType.APPEND_MODE,
        'RANDOM': TokenType.RANDOM_MODE,
        'SEEK': TokenType.SEEK,
        'GETRECORD': TokenType.GETRECORD,
        'PUTRECORD': TokenType.PUTRECORD,
    }
    
    def __init__(self, source, filename="<input>"):
        self.source = source
        self.filename = filename
        self.pos = 0                    
        self.line = 1                   
        self.column = 1                 
        self.tokens = []                
    
    @property
    def _regex(self):
        cls = self.__class__
        if '_MASTER_REGEX' not in cls.__dict__:
            pattern_parts = [f'(?P<{name}>{regex})' for name, regex in cls.TOKEN_SPECS]
            combined_pattern = '|'.join(pattern_parts)
            cls._MASTER_REGEX = re.compile(combined_pattern, re.MULTILINE)
        return cls._MASTER_REGEX

    def _update_position(self, text):
        newlines = text.count('\n')
        if newlines:
            self.line += newlines
            self.column = len(text) - text.rfind('\n')
        else:
            self.column += len(text)
    
    def _get_context(self) -> str:
        """Extract the current source line and add a ^ pointer for error display."""
        lines = self.source.split('\n')
        if self.line - 1 < len(lines):
            src_line = lines[self.line - 1]
            pointer = ' ' * (self.column - 1) + '^'
            return f"\n  {src_line}\n  {pointer}"
        return ""

    def error(self, message):
        context = self._get_context()
        raise LexerError(f"{self.filename}:{self.line}:{self.column}: {message}{context}")
    
    def tokenize(self):
        while self.pos < len(self.source):
            match = self._regex.match(self.source, self.pos)
            
            if not match:
                self.error(f"Unexpected character: {self.source[self.pos]!r}")
            
            kind = match.lastgroup
            value = match.group()
            start_line, start_col = self.line, self.column
            
            self.pos = match.end()
            self._update_position(value)
            
            if kind in ('WHITESPACE', 'COMMENT', 'NEWLINE'):
                continue
            
            token = self._create_token(kind, value, start_line, start_col)
            self.tokens.append(token)
        
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens
    
    # Class-level mapping: regex group name → TokenType (avoids rebuilding per token)
    _TOKEN_TYPE_MAP = {
        'ASSIGN': TokenType.ASSIGN, 'NE': TokenType.NE, 'LE': TokenType.LE,
        'GE': TokenType.GE, 'EQ': TokenType.EQ, 'LT': TokenType.LT,
        'GT': TokenType.GT, 'PLUS': TokenType.PLUS, 'MINUS': TokenType.MINUS,
        'MULTIPLY': TokenType.MULTIPLY, 'DIVIDE': TokenType.DIVIDE,
        'LPAREN': TokenType.LPAREN, 'RPAREN': TokenType.RPAREN,
        'LBRACKET': TokenType.LBRACKET, 'RBRACKET': TokenType.RBRACKET,
        'COLON': TokenType.COLON, 'SEMICOLON': TokenType.SEMICOLON,
        'COMMA': TokenType.COMMA, 'DOT': TokenType.DOT, 'AMPER': TokenType.AMPER,
    }

    def _create_token(self, kind, value, line, col):
        if kind == 'IDENTIFIER':
            upper_val = value.upper()
            token_type = self.KEYWORDS.get(upper_val)
            if token_type is not None:
                if token_type == TokenType.BOOLEAN:
                    return Token(token_type, upper_val == 'TRUE', line, col)
                return Token(token_type, upper_val, line, col)
            return Token(TokenType.IDENTIFIER, value, line, col)
        
        if kind == 'INTEGER':
            return Token(TokenType.INTEGER, int(value), line, col)
        if kind == 'REAL':
            return Token(TokenType.REAL, float(value), line, col)
        if kind == 'STRING':
            return Token(TokenType.STRING, value[1:-1], line, col)
        if kind == 'CHAR':
            return Token(TokenType.CHAR_LITERAL, value[1:-1], line, col)

        mapped = self._TOKEN_TYPE_MAP.get(kind)
        if mapped is not None:
            return Token(mapped, value, line, col)
            
        raise RuntimeError(f"Unhandled token kind: {kind}")

if __name__ == "__main__":
    # Test
    code = """
    DECLARE x : INTEGER
    x <- 10
    IF x <> 5 THEN
        OUTPUT "Not 5"
    ENDIF
    """
    lexer = Lexer(code)
    for t in lexer.tokenize():
        print(t)
