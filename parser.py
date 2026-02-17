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

    def parse(self) -> List[Stmt]:
        statements = []
        while self.current and self.current.type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            else:
                 if self.current:
                     raise ParserError(f"Unexpected token at top level: {self.current}")
        return statements

    def parse_statement(self) -> Optional[Stmt]:
        if not self.current:
            return None
            
        line = self.current.line
        stmt = None
        
        # Dispatch table lookup
        handler = self._stmt_dispatch.get(self.current.type)
        if handler:
            stmt = handler()
        elif self.check(TokenType.SUPER):
            # SUPER.Method(args) as a statement
            stmt = self.parse_primary()
        elif self.check(TokenType.IDENTIFIER):
            stmt = self.parse_assignment_or_call()
        
        if stmt:
            stmt.line = line
            
        return stmt

    def parse_declare(self) -> DeclareStmt:
        self.expect(TokenType.DECLARE)
        name = self.expect(TokenType.IDENTIFIER).value
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
        # simplistic check:
        type_name = type_token.value
        
        return DeclareStmt(name, type_name, is_array, bounds)

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
                indices = [self.parse_expression()]
                while self.match(TokenType.COMMA):
                    indices.append(self.parse_expression())
                self.expect(TokenType.RBRACKET)
                if isinstance(expr, str):
                    expr = ArrayAccessExpr(expr, indices)
                else:
                    raise ParserError("Complex array indexing on expressions not fully supported yet")
            elif self.match(TokenType.DOT):
                field = self.expect(TokenType.IDENTIFIER).value
                base = VariableExpr(expr) if isinstance(expr, str) else expr
                expr = MemberExpr(base, field)
        
        return expr

    def parse_if(self) -> IfStmt:
        self.expect(TokenType.IF)
        condition = self.parse_expression()
        self.expect(TokenType.THEN)
        
        then_branch = []
        while self.current and not self.check(TokenType.ELSE, TokenType.ENDIF):
            stmt = self.parse_statement()
            if stmt: then_branch.append(stmt)
        
        else_branch = None
        if self.match(TokenType.ELSE):
            else_branch = []
            while self.current and not self.check(TokenType.ENDIF):
                stmt = self.parse_statement()
                if stmt: else_branch.append(stmt)
        
        self.expect(TokenType.ENDIF)
        return IfStmt(condition, then_branch, else_branch)

    def parse_case(self) -> CaseStmt:
        self.expect(TokenType.CASE)
        self.expect(TokenType.OF)
        selector = self.parse_expression()
        
        branches = []
        otherwise = None
        
        while self.current and not self.check(TokenType.ENDCASE):
            if self.match(TokenType.OTHERWISE):
                self.expect(TokenType.COLON)
                otherwise = []
                while self.current and not self.check(TokenType.ENDCASE):
                    stmt = self.parse_statement()
                    if stmt: otherwise.append(stmt)
                break
            
            # Case labels
            values = self.parse_case_labels()
            self.expect(TokenType.COLON)
            stmts = []
            # Parse statements until next start of case or ENDCASE/OTHERWISE
            # Use heuristic from blueprint
            while self.current and not self.check(TokenType.ENDCASE, TokenType.OTHERWISE) and not self.is_case_label_start():
                stmt = self.parse_statement()
                if stmt: stmts.append(stmt)
            
            branches.append(CaseBranch(values, stmts))
            
        self.expect(TokenType.ENDCASE)
        return CaseStmt(selector, branches, otherwise)

    def is_case_label_start(self) -> bool:
        if self.check(TokenType.INTEGER, TokenType.STRING, TokenType.CHAR_LITERAL, TokenType.BOOLEAN):
            return True
        # Negative literal: e.g. -3 : <statement>
        if self.check(TokenType.MINUS):
            nxt = self.peek(1)
            if nxt and nxt.type == TokenType.INTEGER:
                return True
        if self.check(TokenType.IDENTIFIER):
            # Lookahead for colon or TO
            nxt = self.peek(1)
            if nxt and nxt.type in (TokenType.COLON, TokenType.TO, TokenType.COMMA):
                return True
        return False

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
            
        body = []
        while self.current and not self.check(TokenType.ENDWHILE):
            stmt = self.parse_statement()
            if stmt: body.append(stmt)
        self.expect(TokenType.ENDWHILE)
        return WhileStmt(condition, body)

    def parse_repeat(self) -> RepeatStmt:
        self.expect(TokenType.REPEAT)
        body = []
        while self.current and not self.check(TokenType.UNTIL):
            stmt = self.parse_statement()
            if stmt: body.append(stmt)
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
        
        step = None
        if self.match(TokenType.STEP):
            step = self.parse_expression()
            
        body = []
        while self.current and not self.check(TokenType.NEXT):
            stmt = self.parse_statement()
            if stmt: body.append(stmt)
            
        self.expect(TokenType.NEXT)
        # Optional variable name after NEXT — must match the loop variable
        if self.check(TokenType.IDENTIFIER):
             next_var = self.advance().value
             if next_var != name:
                 raise ParserError(
                     f"NEXT variable '{next_var}' does not match FOR variable '{name}'"
                 )
             
        return ForStmt(name, start, end, step, body)

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
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        if self.check(TokenType.INTEGER, TokenType.REAL, TokenType.STRING, TokenType.CHAR_LITERAL, TokenType.BOOLEAN):
            return LiteralExpr(self.advance().value)
            
        if self.check(TokenType.IDENTIFIER):
            expr = VariableExpr(self.advance().value)
            
            while True:
                if self.match(TokenType.LBRACKET):
                    indices = [self.parse_expression()]
                    while self.match(TokenType.COMMA):
                        indices.append(self.parse_expression())
                    self.expect(TokenType.RBRACKET)
                    if isinstance(expr, VariableExpr):
                        expr = ArrayAccessExpr(expr.name, indices)
                    else:
                        raise ParserError("Complex array indexing on expressions not supported yet")
                elif self.match(TokenType.DOT):
                    expr = MemberExpr(expr, self.expect(TokenType.IDENTIFIER).value)
                elif self.match(TokenType.LPAREN):
                    args = self._parse_arglist()
                    if isinstance(expr, VariableExpr):
                        expr = CallExpr(expr.name, args)
                    elif isinstance(expr, MemberExpr):
                        expr = MethodCallExpr(expr.record, expr.field, args)
                    else:
                        raise ParserError("Complex call not supported")
                else:
                    break
            return expr
            
        if self.match(TokenType.NOT):
            return UnaryExpr("NOT", self.parse_primary())
             
        if self.match(TokenType.MINUS):
            return UnaryExpr("-", self.parse_primary())

        if self.match(TokenType.NEW):
            class_name = self.expect(TokenType.IDENTIFIER).value
            args = self._parse_arglist() if self.match(TokenType.LPAREN) else []
            if args or not self.check(TokenType.RPAREN):
                pass  # args already parsed or no parens
            return NewExpr(class_name, args)

        if self.match(TokenType.SUPER):
            self.expect(TokenType.DOT)
            method_name = self.advance().value if self.check(TokenType.NEW) else self.expect(TokenType.IDENTIFIER).value
            args = self._parse_arglist() if self.match(TokenType.LPAREN) else []
            return SuperExpr(method_name, args)

        raise ParserError(f"Unexpected token in expression: {self.current}")

    def _get_precedence(self, type_: TokenType) -> int:
        return self._precedence.get(type_, -1)
    
    def parse_procedure_decl(self) -> ProcedureDecl:
        self.expect(TokenType.PROCEDURE)
        # Allow NEW as procedure name (constructor in classes)
        if self.check(TokenType.NEW):
            name = self.advance().value
        else:
            name = self.expect(TokenType.IDENTIFIER).value
        params = []
        if self.match(TokenType.LPAREN):
            if not self.check(TokenType.RPAREN):
                params = self.parse_params()
            self.expect(TokenType.RPAREN)
            
        body = []
        while self.current and not self.check(TokenType.ENDPROCEDURE):
             stmt = self.parse_statement()
             if stmt: body.append(stmt)
        self.expect(TokenType.ENDPROCEDURE)
        return ProcedureDecl(name, params, body)

    def parse_function_decl(self) -> FunctionDecl:
        self.expect(TokenType.FUNCTION)
        name = self.expect(TokenType.IDENTIFIER).value
        params = []
        if self.match(TokenType.LPAREN):
            if not self.check(TokenType.RPAREN):
                params = self.parse_params()
            self.expect(TokenType.RPAREN)
        
        self.expect(TokenType.RETURNS)
        # Type
        type_tok = self.advance() # Should be type
        return_type = type_tok.value
        
        body = []
        while self.current and not self.check(TokenType.ENDFUNCTION):
             stmt = self.parse_statement()
             if stmt: body.append(stmt)
        self.expect(TokenType.ENDFUNCTION)
        return FunctionDecl(name, params, return_type, body)

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
        # Check if there's an expression following RETURN
        # Bare RETURN (no value) is valid in procedures
        if (self.current and self.current.type not in (
            TokenType.EOF, TokenType.ENDPROCEDURE, TokenType.ENDFUNCTION
        ) and self.current.type in (
            TokenType.INTEGER, TokenType.REAL, TokenType.STRING,
            TokenType.CHAR_LITERAL, TokenType.BOOLEAN, TokenType.IDENTIFIER,
            TokenType.LPAREN, TokenType.MINUS, TokenType.NOT
        )):
            expr = self.parse_expression()
            return ReturnStmt(expr)
        return ReturnStmt(None)

    def _parse_file_args(self, with_comma=True):
        """Parse filename and optional comma-separated second argument."""
        filename_expr = self.parse_primary()
        if with_comma:
            self.expect(TokenType.COMMA)
        return filename_expr

    def parse_file_stmt(self) -> FileStmt:
        if self.match(TokenType.OPENFILE):
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
        
        if self.match(TokenType.READFILE):
            fn = self._parse_file_args()
            return FileStmt("READ", fn, variable=self.expect(TokenType.IDENTIFIER).value)
        
        if self.match(TokenType.WRITEFILE):
            fn = self._parse_file_args()
            return FileStmt("WRITE", fn, data=self.parse_expression())
        
        if self.match(TokenType.CLOSEFILE):
            return FileStmt("CLOSE", self.parse_primary())
        
        if self.match(TokenType.SEEK):
            fn = self._parse_file_args()
            return FileStmt("SEEK", fn, data=self.parse_expression())
        
        if self.match(TokenType.GETRECORD):
            fn = self._parse_file_args()
            return FileStmt("GETRECORD", fn, variable=self.expect(TokenType.IDENTIFIER).value)
        
        if self.match(TokenType.PUTRECORD):
            fn = self._parse_file_args()
            return FileStmt("PUTRECORD", fn, variable=self.expect(TokenType.IDENTIFIER).value)
        
        raise ParserError(f"Unexpected file operation at {self.current}")

    def parse_class_decl(self) -> 'ClassDecl':
        """
        CLASS <name> [INHERITS <parent>]
            [PUBLIC|PRIVATE] DECLARE/PROCEDURE/FUNCTION ...
            or: [PUBLIC|PRIVATE] <field> : <type>
        ENDCLASS
        """
        self.expect(TokenType.CLASS)
        name = self.expect(TokenType.IDENTIFIER).value
        
        parent = None
        if self.match(TokenType.INHERITS):
            parent = self.expect(TokenType.IDENTIFIER).value
        
        members = []
        while self.current and not self.check(TokenType.ENDCLASS):
            # Optional access modifier
            access = None
            if self.match(TokenType.PUBLIC):
                access = "PUBLIC"
            elif self.match(TokenType.PRIVATE):
                access = "PRIVATE"
            
            # Check if this is a bare field declaration: <Name> : <Type>
            # (i.e. no DECLARE keyword, just IDENTIFIER followed by COLON)
            if self.check(TokenType.IDENTIFIER) and self.peek(1) and self.peek(1).type == TokenType.COLON:
                field_name = self.advance().value
                self.expect(TokenType.COLON)
                type_tok = self.advance()
                type_name = type_tok.value
                member = DeclareStmt(field_name, type_name)
            else:
                member = self.parse_statement()
            
            if member:
                # Attach access modifier as attribute
                member.access = access if access else "PUBLIC"
                members.append(member)
        
        self.expect(TokenType.ENDCLASS)
        return ClassDecl(name, parent, members)

    def parse_type_decl(self) -> TypeDecl:
        self.expect(TokenType.TYPE)
        name = self.expect(TokenType.IDENTIFIER).value

        # Check for enumerated type: TYPE Season = (Spring, Summer, Autumn, Winter)
        if self.match(TokenType.EQ):
            if self.match(TokenType.LPAREN):
                values = []
                values.append(self.expect(TokenType.IDENTIFIER).value)
                while self.match(TokenType.COMMA):
                    values.append(self.expect(TokenType.IDENTIFIER).value)
                self.expect(TokenType.RPAREN)
                return TypeDecl(name, [], enum_values=values)
            else:
                raise ParserError(f"Expected '(' after '=' in TYPE declaration")

        # Record type: fields use DECLARE keyword per 9618 spec
        fields = []
        while self.current and not self.check(TokenType.ENDTYPE):
            # 9618 spec requires DECLARE before each field
            self.match(TokenType.DECLARE)  # consume DECLARE if present (also accept without for backwards compat)
            field_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            type_tok = self.advance()
            fields.append((field_name, type_tok.value))
        self.expect(TokenType.ENDTYPE)
        return TypeDecl(name, fields)

