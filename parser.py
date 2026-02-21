from typing import List, Optional, Union
from lexer import TokenType, Token, Lexer
from ast_nodes import *

# Precedence table for binary operators (Higher value = higher precedence)
OPERATOR_PRECEDENCE = {
    TokenType.OR: 10,
    TokenType.AND: 20,
    TokenType.NE: 30, TokenType.EQ: 30,
    TokenType.LT: 30, TokenType.GT: 30,
    TokenType.LE: 30, TokenType.GE: 30,
    TokenType.PLUS: 40, TokenType.MINUS: 40,
    TokenType.AMPER: 40,
    TokenType.MULTIPLY: 50, TokenType.DIVIDE: 50,
    TokenType.DIV: 50, TokenType.MOD: 50,
}

# Token types that can start a return value expression
_RETURN_EXPR_TOKENS = frozenset({
    TokenType.INTEGER, TokenType.REAL, TokenType.STRING,
    TokenType.CHAR_LITERAL, TokenType.BOOLEAN, TokenType.IDENTIFIER,
    TokenType.LPAREN, TokenType.MINUS, TokenType.NOT,
})

# Token types that mark the end of a return context
_RETURN_END_TOKENS = frozenset({
    TokenType.EOF, TokenType.ENDPROCEDURE, TokenType.ENDFUNCTION,
})


class ParserError(Exception):
    pass

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current = tokens[0] if tokens else None
        
        # Statement dispatch table: TokenType → parse method
        self._stmt_dispatch = {
            TokenType.DECLARE: self.parse_declare,
            TokenType.CONSTANT: self.parse_constant,
            TokenType.INPUT: self.parse_input,
            TokenType.OUTPUT: self.parse_output,
            TokenType.IF: self.parse_if,
            TokenType.CASE: self.parse_case,
            TokenType.WHILE: self.parse_while,
            TokenType.REPEAT: self.parse_repeat,
            TokenType.FOR: self.parse_for,
            TokenType.PROCEDURE: self.parse_procedure_decl,
            TokenType.FUNCTION: self.parse_function_decl,
            TokenType.CALL: self.parse_call_stmt,
            TokenType.RETURN: self.parse_return,
            TokenType.TYPE: self.parse_type_decl,
            TokenType.CLASS: self.parse_class_decl,
            # File I/O tokens all route to the same method
            TokenType.OPENFILE: self.parse_file_stmt,
            TokenType.READFILE: self.parse_file_stmt,
            TokenType.WRITEFILE: self.parse_file_stmt,
            TokenType.CLOSEFILE: self.parse_file_stmt,
            TokenType.SEEK: self.parse_file_stmt,
            TokenType.GETRECORD: self.parse_file_stmt,
            TokenType.PUTRECORD: self.parse_file_stmt,
        }
        
        # File operation dispatch: TokenType → handler method
        self._file_dispatch = {
            TokenType.OPENFILE: self._parse_open_file,
            TokenType.READFILE: self._parse_read_file,
            TokenType.WRITEFILE: self._parse_write_file,
            TokenType.CLOSEFILE: self._parse_close_file,
            TokenType.SEEK: self._parse_seek_file,
            TokenType.GETRECORD: self._parse_getrecord_file,
            TokenType.PUTRECORD: self._parse_putrecord_file,
        }
        
        # Precedence table for binary operators
        self._precedence = OPERATOR_PRECEDENCE.copy()
    
    def peek(self, offset: int = 0) -> Optional[Token]:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return None
    
    def advance(self) -> Token:
        old = self.current
        self.pos += 1
        self.current = self.tokens[self.pos] if self.pos < len(self.tokens) else None
        return old
    
    def expect(self, token_type: TokenType, value: Optional[str] = None) -> Token:
        if self.current is None:
            raise ParserError(f"Unexpected EOF, expected {token_type.name}")
        
        if self.current.type != token_type:
            raise ParserError(f"Expected {token_type.name}, got {self.current.type.name} at line {self.current.line}")
        
        if value is not None and self.current.value != value:
            raise ParserError(f"Expected '{value}', got '{self.current.value}' at line {self.current.line}")
        
        return self.advance()
    
    def match(self, *token_types: TokenType) -> Optional[Token]:
        if self.current and self.current.type in token_types:
            return self.advance()
        return None
    
    def check(self, *token_types: TokenType) -> bool:
        return self.current is not None and self.current.type in token_types

    # ── Shared parsing helpers ──

    def _has_more_tokens(self) -> bool:
        """Check if there are more non-EOF tokens to parse."""
        return self.current is not None and self.current.type != TokenType.EOF

    def _parse_block_until(self, *end_tokens: TokenType) -> List[Stmt]:
        """Parse statements until one of the given token types is encountered."""
        stmts = []
        while self.current and not self.check(*end_tokens):
            new_stmts = self.parse_statement()
            if new_stmts:
                stmts.extend(new_stmts)
        return stmts

    def _parse_optional_params(self) -> List['Param']:
        """Parse an optional parenthesized parameter list."""
        if not self.match(TokenType.LPAREN):
            return []
        if self.check(TokenType.RPAREN):
            self.expect(TokenType.RPAREN)
            return []
        params = self.parse_params()
        self.expect(TokenType.RPAREN)
        return params

    # ── Top-level parsing ──

    def parse(self) -> List[Stmt]:
        statements = []
        while self._has_more_tokens():
            new_stmts = self.parse_statement()
            if new_stmts:
                statements.extend(new_stmts)
            elif self.current:
                raise ParserError(f"Unexpected token at top level: {self.current}")
        return statements

    def parse_statement(self) -> List[Stmt]:
        if not self.current:
            return []
            
        line = self.current.line
        stmts = []
        
        # Dispatch table lookup
        handler = self._stmt_dispatch.get(self.current.type)
        if handler:
            result = handler()
            if isinstance(result, list):
                stmts = result
            else:
                stmts = [result]
        elif self.check(TokenType.SUPER):
            # SUPER.Method(args) as a statement
            stmts = [self.parse_primary()]
        elif self.check(TokenType.IDENTIFIER):
            stmts = [self.parse_assignment_or_call()]
        
        for stmt in stmts:
            stmt.line = line
            
        return stmts

    def parse_declare(self) -> List[DeclareStmt]:
        self.expect(TokenType.DECLARE)
        names = [self.expect(TokenType.IDENTIFIER).value]
        while self.match(TokenType.COMMA):
            names.append(self.expect(TokenType.IDENTIFIER).value)
            
        self.expect(TokenType.COLON)
        
        is_array = False
        bounds = None
        
        if self.match(TokenType.ARRAY):
            is_array = True
            bounds = []
            self.expect(TokenType.LBRACKET)
            # Support multi-dim: [1:10, 1:10]
            while True:
                lower = int(self.expect(TokenType.INTEGER).value)
                self.expect(TokenType.COLON) # Range separator
                upper = int(self.expect(TokenType.INTEGER).value)
                bounds.append((lower, upper))
                if not self.match(TokenType.COMMA):
                    break
            self.expect(TokenType.RBRACKET)
            self.expect(TokenType.OF)
        
        type_token = self.advance() # DATE, INTEGER, STRING, etc.
        # Verify it's a type keyword or identifier (user defined)
        type_name = type_token.value
        
        return [DeclareStmt(name, type_name, is_array, bounds) for name in names]

    def parse_constant(self) -> 'ConstantDecl':
        """CONSTANT <name> = <value>"""
        self.expect(TokenType.CONSTANT)
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.EQ)
        value = self.parse_expression()
        return ConstantDecl(name, value)

    def parse_input(self) -> InputStmt:
        self.expect(TokenType.INPUT)
        # Could be variable or array access
        expr = self.parse_expression()
        # In 9618, INPUT takes an identifier (or array element).
        if isinstance(expr, (VariableExpr, ArrayAccessExpr)):
            return InputStmt(expr)
        else:
            raise ParserError(f"Invalid INPUT target: {expr}")

    def parse_output(self) -> OutputStmt:
        self.expect(TokenType.OUTPUT)
        values = []
        values.append(self.parse_expression())
        while self.match(TokenType.COMMA):
            values.append(self.parse_expression())
        return OutputStmt(values)

    def parse_assignment_or_call(self) -> Stmt:
        target = self.parse_lvalue() # Parses Identifier or Array access
        
        if self.match(TokenType.ASSIGN):
            value = self.parse_expression()
            return AssignStmt(target, value)
        
        raise ParserError(f"Expected assignment after identifier at line {self.current.line}")

    def parse_lvalue(self) -> Union[str, ArrayAccessExpr, MemberExpr]:
        name_tok = self.expect(TokenType.IDENTIFIER)
        expr = name_tok.value
        
        while self.current and self.check(TokenType.LBRACKET, TokenType.DOT):
            if self.match(TokenType.LBRACKET):
                expr = self._parse_lvalue_index(expr)
            elif self.match(TokenType.DOT):
                base = VariableExpr(expr) if isinstance(expr, str) else expr
                expr = MemberExpr(base, self.expect(TokenType.IDENTIFIER).value)
        
        return expr

    def _parse_lvalue_index(self, expr) -> ArrayAccessExpr:
        """Parse array index access in an l-value context (LBRACKET already consumed)."""
        indices = [self.parse_expression()]
        while self.match(TokenType.COMMA):
            indices.append(self.parse_expression())
        self.expect(TokenType.RBRACKET)
        if isinstance(expr, str):
            return ArrayAccessExpr(expr, indices)
        raise ParserError("Complex array indexing on expressions not fully supported yet")

    def parse_if(self) -> IfStmt:
        self.expect(TokenType.IF)
        condition = self.parse_expression()
        self.expect(TokenType.THEN)
        then_branch = self._parse_block_until(TokenType.ELSE, TokenType.ENDIF)
        else_branch = self._parse_else_branch()
        self.expect(TokenType.ENDIF)
        return IfStmt(condition, then_branch, else_branch)

    def _parse_else_branch(self) -> Optional[List[Stmt]]:
        """Parse the optional ELSE branch of an IF statement."""
        if not self.match(TokenType.ELSE):
            return None
        return self._parse_block_until(TokenType.ENDIF)

    def parse_case(self) -> CaseStmt:
        self.expect(TokenType.CASE)
        self.expect(TokenType.OF)
        selector = self.parse_expression()
        
        branches = []
        otherwise = None
        
        while self.current and not self.check(TokenType.ENDCASE):
            if self.check(TokenType.OTHERWISE):
                otherwise = self.parse_otherwise_branch()
                break
            
            branches.append(self.parse_case_branch())
            
        self.expect(TokenType.ENDCASE)
        return CaseStmt(selector, branches, otherwise)

    def parse_otherwise_branch(self) -> List[Stmt]:
        """Parses the OTHERWISE block: OTHERWISE : <statements>"""
        self.expect(TokenType.OTHERWISE)
        self.expect(TokenType.COLON)
        return self._parse_block_until(TokenType.ENDCASE)

    def parse_case_branch(self) -> CaseBranch:
        """Parses a single CASE branch: <labels> : <statements>"""
        values = self.parse_case_labels()
        self.expect(TokenType.COLON)
        stmts = []
        while self.current and not self._at_case_branch_end():
            new_stmts = self.parse_statement()
            if new_stmts: stmts.extend(new_stmts)
        return CaseBranch(values, stmts)

    def _at_case_branch_end(self) -> bool:
        """Check if we've reached the end of a CASE branch."""
        return self.check(TokenType.ENDCASE, TokenType.OTHERWISE) or self.is_case_label_start()

    def is_case_label_start(self) -> bool:
        if self.check(TokenType.INTEGER, TokenType.STRING, TokenType.CHAR_LITERAL, TokenType.BOOLEAN):
            return True
        if self._is_negative_literal_start():
            return True
        if self._is_identifier_label_start():
            return True
        return False

    def _is_negative_literal_start(self) -> bool:
        """Check for a negative literal like -3 at the start of a case label."""
        if not self.check(TokenType.MINUS):
            return False
        nxt = self.peek(1)
        return nxt is not None and nxt.type == TokenType.INTEGER

    def _is_identifier_label_start(self) -> bool:
        """Check for an identifier followed by a label separator (COLON, TO, COMMA)."""
        if not self.check(TokenType.IDENTIFIER):
            return False
        nxt = self.peek(1)
        return nxt is not None and nxt.type in (TokenType.COLON, TokenType.TO, TokenType.COMMA)

    def parse_case_labels(self) -> List[Expr]:
        labels = []
        while True:
            # Parse value
            start = self.parse_primary() # Simplification
            if self.match(TokenType.TO):
                end = self.parse_primary()
                labels.append(RangeExpr(start, end))
            else:
                labels.append(start)
            
            if not self.match(TokenType.COMMA):
                break
        return labels

    def parse_while(self) -> WhileStmt:
        self.expect(TokenType.WHILE)
        condition = self.parse_expression()
        self.match(TokenType.DO)  # Optional DO keyword
        body = self._parse_block_until(TokenType.ENDWHILE)
        self.expect(TokenType.ENDWHILE)
        return WhileStmt(condition, body)

    def parse_repeat(self) -> RepeatStmt:
        self.expect(TokenType.REPEAT)
        body = self._parse_block_until(TokenType.UNTIL)
        self.expect(TokenType.UNTIL)
        condition = self.parse_expression()
        return RepeatStmt(body, condition)

    def parse_for(self) -> ForStmt:
        self.expect(TokenType.FOR)
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        start = self.parse_expression()
        self.expect(TokenType.TO)
        end = self.parse_expression()
        step = self._parse_optional_step()
        body = self._parse_block_until(TokenType.NEXT)
        self.expect(TokenType.NEXT)
        self._validate_next_variable(name)
        return ForStmt(name, start, end, step, body)

    def _parse_optional_step(self) -> Optional[Expr]:
        """Parse an optional STEP clause in a FOR loop."""
        if self.match(TokenType.STEP):
            return self.parse_expression()
        return None

    def _validate_next_variable(self, loop_var: str):
        """Validate that the variable after NEXT matches the FOR loop variable."""
        if self.check(TokenType.IDENTIFIER):
            next_var = self.advance().value
            if next_var != loop_var:
                raise ParserError(
                    f"NEXT variable '{next_var}' does not match FOR variable '{loop_var}'"
                )

    def parse_call_stmt(self):
        self.expect(TokenType.CALL)
        name = self.expect(TokenType.IDENTIFIER).value
        
        if self.match(TokenType.DOT):
            method_name = self.expect(TokenType.IDENTIFIER).value
            args = self._parse_arglist() if self.match(TokenType.LPAREN) else []
            return ProcedureCallStmt(f"{name}.{method_name}", args)
        
        args = self._parse_arglist() if self.match(TokenType.LPAREN) else []
        return ProcedureCallStmt(name, args)

    # ... Expressions implementation ...
    def parse_expression(self, min_prec: int = 0) -> Expr:
        left = self.parse_primary()
        
        while self.current:
            prec = self._get_precedence(self.current.type)
            if prec < min_prec:
                break
            
            op_tok = self.advance()
            op = op_tok.value if op_tok.type != TokenType.AND and op_tok.type != TokenType.OR else op_tok.type.name
            
            right = self.parse_expression(prec + 1)
            left = BinaryExpr(left, op, right)
            
        return left

    def _parse_arglist(self) -> List[Expr]:
        """Parse comma-separated argument list (LPAREN already consumed)."""
        args = []
        if not self.check(TokenType.RPAREN):
            args.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                args.append(self.parse_expression())
        self.expect(TokenType.RPAREN)
        return args

    def parse_primary(self) -> Expr:
        if self.match(TokenType.LPAREN):
            return self._parse_grouped_expr()
        
        if self.check(TokenType.INTEGER, TokenType.REAL, TokenType.STRING, TokenType.CHAR_LITERAL, TokenType.BOOLEAN):
            return LiteralExpr(self.advance().value)
            
        if self.check(TokenType.IDENTIFIER):
            return self._parse_identifier_expr()
            
        if self.match(TokenType.NOT):
            return UnaryExpr("NOT", self.parse_primary())
             
        if self.match(TokenType.MINUS):
            return UnaryExpr("-", self.parse_primary())

        if self.match(TokenType.NEW):
            return self._parse_new_expr()

        if self.match(TokenType.SUPER):
            return self._parse_super_expr()

        raise ParserError(f"Unexpected token in expression: {self.current}")

    def _parse_grouped_expr(self) -> Expr:
        """Parse a parenthesized expression (LPAREN already consumed)."""
        expr = self.parse_expression()
        self.expect(TokenType.RPAREN)
        return expr

    def _parse_identifier_expr(self) -> Expr:
        """Parse an identifier and any chained access (array, dot, call)."""
        expr = VariableExpr(self.advance().value)
        return self._parse_chained_access(expr)

    def _parse_chained_access(self, expr: Expr) -> Expr:
        """Parse chained array access, member access, and call expressions."""
        while True:
            if self.match(TokenType.LBRACKET):
                expr = self._parse_array_access(expr)
            elif self.match(TokenType.DOT):
                expr = MemberExpr(expr, self.expect(TokenType.IDENTIFIER).value)
            elif self.match(TokenType.LPAREN):
                expr = self._parse_call_from_expr(expr)
            else:
                break
        return expr

    def _parse_array_access(self, expr: Expr) -> ArrayAccessExpr:
        """Parse array index access (LBRACKET already consumed)."""
        indices = [self.parse_expression()]
        while self.match(TokenType.COMMA):
            indices.append(self.parse_expression())
        self.expect(TokenType.RBRACKET)
        if isinstance(expr, VariableExpr):
            return ArrayAccessExpr(expr.name, indices)
        raise ParserError("Complex array indexing on expressions not supported yet")

    def _parse_call_from_expr(self, expr: Expr) -> Expr:
        """Parse a function/method call (LPAREN already consumed)."""
        args = self._parse_arglist()
        if isinstance(expr, VariableExpr):
            return CallExpr(expr.name, args)
        if isinstance(expr, MemberExpr):
            return MethodCallExpr(expr.record, expr.field, args)
        raise ParserError("Complex call not supported")

    def _parse_new_expr(self) -> NewExpr:
        """Parse a NEW class instantiation expression."""
        class_name = self.expect(TokenType.IDENTIFIER).value
        args = self._parse_arglist() if self.match(TokenType.LPAREN) else []
        return NewExpr(class_name, args)

    def _parse_super_expr(self) -> SuperExpr:
        """Parse a SUPER.method() call expression."""
        self.expect(TokenType.DOT)
        method_name = self.advance().value if self.check(TokenType.NEW) else self.expect(TokenType.IDENTIFIER).value
        args = self._parse_arglist() if self.match(TokenType.LPAREN) else []
        return SuperExpr(method_name, args)

    def _get_precedence(self, type_: TokenType) -> int:
        return self._precedence.get(type_, -1)
    
    def parse_procedure_decl(self) -> ProcedureDecl:
        self.expect(TokenType.PROCEDURE)
        name = self._parse_callable_name()
        params = self._parse_optional_params()
        body = self._parse_block_until(TokenType.ENDPROCEDURE)
        self.expect(TokenType.ENDPROCEDURE)
        return ProcedureDecl(name, params, body)

    def parse_function_decl(self) -> FunctionDecl:
        self.expect(TokenType.FUNCTION)
        name = self.expect(TokenType.IDENTIFIER).value
        params = self._parse_optional_params()
        self.expect(TokenType.RETURNS)
        return_type = self.advance().value
        body = self._parse_block_until(TokenType.ENDFUNCTION)
        self.expect(TokenType.ENDFUNCTION)
        return FunctionDecl(name, params, return_type, body)

    def _parse_callable_name(self) -> str:
        """Parse a procedure/function name, allowing NEW for constructors."""
        if self.check(TokenType.NEW):
            return self.advance().value
        return self.expect(TokenType.IDENTIFIER).value

    def parse_params(self) -> List[Param]:
        params = []
        # 9618 spec: BYREF/BYVAL keyword persists across subsequent params
        # e.g. PROCEDURE SWAP(BYREF X : INTEGER, Y : INTEGER) means both BYREF
        mode = "BYVAL"
        while True:
            if self.check(TokenType.BYREF):
                self.advance()
                mode = "BYREF"
            elif self.check(TokenType.BYVAL):
                self.advance()
                mode = "BYVAL"
            
            name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            type_name = self.advance().value # primitive type
            
            params.append(Param(name, type_name, mode))
            
            if not self.match(TokenType.COMMA):
                break
        return params

    def parse_return(self) -> Stmt:
        self.expect(TokenType.RETURN)
        if self._has_return_expression():
            return ReturnStmt(self.parse_expression())
        return ReturnStmt(None)

    def _has_return_expression(self) -> bool:
        """Check if a RETURN statement is followed by a return value expression."""
        if not self.current:
            return False
        if self.current.type in _RETURN_END_TOKENS:
            return False
        return self.current.type in _RETURN_EXPR_TOKENS

    def _parse_file_args(self, with_comma=True):
        """Parse filename and optional comma-separated second argument."""
        filename_expr = self.parse_primary()
        if with_comma:
            self.expect(TokenType.COMMA)
        return filename_expr

    def parse_file_stmt(self) -> FileStmt:
        handler = self._file_dispatch.get(self.current.type)
        if handler:
            self.advance()
            return handler()
        raise ParserError(f"Unexpected file operation at {self.current}")

    def _parse_open_file(self) -> FileStmt:
        """Parse OPENFILE <filename> FOR <mode>."""
        filename_expr = self.parse_primary()
        self.expect(TokenType.FOR)
        mode_map = {
            TokenType.READ_MODE: "READ", TokenType.WRITE_MODE: "WRITE",
            TokenType.APPEND_MODE: "APPEND", TokenType.RANDOM_MODE: "RANDOM",
        }
        for tok_type, mode_str in mode_map.items():
            if self.match(tok_type):
                return FileStmt("OPEN", filename_expr, mode=mode_str)
        raise ParserError("Expected READ, WRITE, APPEND, or RANDOM after FOR")

    def _parse_read_file(self) -> FileStmt:
        """Parse READFILE <filename>, <variable>."""
        fn = self._parse_file_args()
        return FileStmt("READ", fn, variable=self.expect(TokenType.IDENTIFIER).value)

    def _parse_write_file(self) -> FileStmt:
        """Parse WRITEFILE <filename>, <expression>."""
        fn = self._parse_file_args()
        return FileStmt("WRITE", fn, data=self.parse_expression())

    def _parse_close_file(self) -> FileStmt:
        """Parse CLOSEFILE <filename>."""
        return FileStmt("CLOSE", self.parse_primary())

    def _parse_seek_file(self) -> FileStmt:
        """Parse SEEK <filename>, <position>."""
        fn = self._parse_file_args()
        return FileStmt("SEEK", fn, data=self.parse_expression())

    def _parse_getrecord_file(self) -> FileStmt:
        """Parse GETRECORD <filename>, <variable>."""
        fn = self._parse_file_args()
        return FileStmt("GETRECORD", fn, variable=self.expect(TokenType.IDENTIFIER).value)

    def _parse_putrecord_file(self) -> FileStmt:
        """Parse PUTRECORD <filename>, <variable>."""
        fn = self._parse_file_args()
        return FileStmt("PUTRECORD", fn, variable=self.expect(TokenType.IDENTIFIER).value)

    def parse_class_decl(self) -> 'ClassDecl':
        """
        CLASS <name> [INHERITS <parent>]
            [PUBLIC|PRIVATE] DECLARE/PROCEDURE/FUNCTION ...
            or: [PUBLIC|PRIVATE] <field> : <type>
        ENDCLASS
        """
        self.expect(TokenType.CLASS)
        name = self.expect(TokenType.IDENTIFIER).value
        parent = self._parse_optional_inheritance()
        members = self._parse_class_body()
        self.expect(TokenType.ENDCLASS)
        return ClassDecl(name, parent, members)

    def _parse_optional_inheritance(self) -> Optional[str]:
        """Parse an optional INHERITS clause."""
        if self.match(TokenType.INHERITS):
            return self.expect(TokenType.IDENTIFIER).value
        return None

    def _parse_class_body(self) -> list:
        """Parse all class members until ENDCLASS."""
        members = []
        while self.current and not self.check(TokenType.ENDCLASS):
            new_members = self._parse_class_member()
            if new_members:
                members.extend(new_members)
        return members

    def _parse_class_member(self) -> List[Stmt]:
        """Parse a single class member (or list of members) with optional access modifier."""
        access = self._parse_access_modifier()
        
        members = []
        if self._is_bare_field_declaration():
            members = [self._parse_bare_field()]
        else:
            members = self.parse_statement()
        
        for member in members:
            member.access = access
        return members

    def _parse_access_modifier(self) -> str:
        """Parse an optional PUBLIC/PRIVATE access modifier, defaulting to PUBLIC."""
        if self.match(TokenType.PUBLIC):
            return "PUBLIC"
        if self.match(TokenType.PRIVATE):
            return "PRIVATE"
        return "PUBLIC"

    def _is_bare_field_declaration(self) -> bool:
        """Check if current position is a bare field: IDENTIFIER COLON TYPE (no DECLARE)."""
        if not self.check(TokenType.IDENTIFIER):
            return False
        nxt = self.peek(1)
        return nxt is not None and nxt.type == TokenType.COLON

    def _parse_bare_field(self) -> DeclareStmt:
        """Parse a bare field declaration: <name> : <type>."""
        field_name = self.advance().value
        self.expect(TokenType.COLON)
        type_name = self.advance().value
        return DeclareStmt(field_name, type_name)

    def parse_type_decl(self) -> TypeDecl:
        self.expect(TokenType.TYPE)
        name = self.expect(TokenType.IDENTIFIER).value

        if self.match(TokenType.EQ):
            return self._parse_enum_type(name)

        fields = self._parse_record_fields()
        self.expect(TokenType.ENDTYPE)
        return TypeDecl(name, fields)

    def _parse_enum_type(self, name: str) -> TypeDecl:
        """Parse an enumerated type: TYPE name = (Value1, Value2, ...)"""
        if not self.match(TokenType.LPAREN):
            raise ParserError(f"Expected '(' after '=' in TYPE declaration")
        values = [self.expect(TokenType.IDENTIFIER).value]
        while self.match(TokenType.COMMA):
            values.append(self.expect(TokenType.IDENTIFIER).value)
        self.expect(TokenType.RPAREN)
        return TypeDecl(name, [], enum_values=values)

    def _parse_record_fields(self) -> list:
        """Parse record type fields until ENDTYPE."""
        fields = []
        while self.current and not self.check(TokenType.ENDTYPE):
            self.match(TokenType.DECLARE)  # consume DECLARE if present
            
            names = [self.expect(TokenType.IDENTIFIER).value]
            while self.match(TokenType.COMMA):
                names.append(self.expect(TokenType.IDENTIFIER).value)
                
            self.expect(TokenType.COLON)
            type_tok = self.advance()
            
            for name in names:
                fields.append((name, type_tok.value))
        return fields

