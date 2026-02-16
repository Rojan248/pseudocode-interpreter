from dataclasses import dataclass
from typing import List, Optional, Union, Any

@dataclass
class Expr:
    """Base class for all expressions."""
    pass

@dataclass
class BinaryExpr(Expr):
    left: Expr
    operator: str
    right: Expr

@dataclass
class UnaryExpr(Expr):
    operator: str
    operand: Expr

@dataclass
class LiteralExpr(Expr):
    value: Union[int, float, str, bool]

@dataclass
class VariableExpr(Expr):
    name: str

@dataclass
class ArrayAccessExpr(Expr):
    array: str
    indices: List[Expr]

@dataclass
class MemberExpr(Expr):
    record: Expr
    field: str

@dataclass
class RangeExpr(Expr):
    """Represents value1 TO value2 in CASE labels."""
    start: Expr
    end: Expr

@dataclass
class CallExpr(Expr):
    """Function call expression."""
    callee: str
    arguments: List[Expr]

@dataclass
class Stmt:
    """Base class for all statements."""
    pass

@dataclass
class AssignStmt(Stmt):
    target: Union[str, ArrayAccessExpr, MemberExpr]
    value: Expr

@dataclass
class CaseBranch:
    """
    Single branch of CASE statement.
    values: List of expressions (literals, constants, or RangeExpr)
    statements: Body executed on match
    """
    values: List[Expr]
    statements: List[Stmt]

@dataclass
class CaseStmt(Stmt):
    selector: Expr
    branches: List[CaseBranch]
    otherwise_branch: Optional[List[Stmt]] = None

@dataclass
class IfStmt(Stmt):
    condition: Expr
    then_branch: List[Stmt]
    else_branch: Optional[List[Stmt]] = None

@dataclass
class WhileStmt(Stmt):
    condition: Expr
    body: List[Stmt]

@dataclass
class RepeatStmt(Stmt):
    body: List[Stmt]
    condition: Expr

@dataclass
class ForStmt(Stmt):
    identifier: str
    start_value: Expr
    end_value: Expr
    step_value: Optional[Expr]
    body: List[Stmt]

@dataclass
class InputStmt(Stmt):
    target: Any  # VariableExpr, ArrayAccessExpr, or str (legacy)

@dataclass
class OutputStmt(Stmt):
    values: List[Expr]

@dataclass
class ProcedureCallStmt(Stmt):
    name: str
    arguments: List[Expr]

@dataclass
class ProcedureDecl(Stmt):
    name: str
    params: List['Param']
    body: List[Stmt]

@dataclass
class FunctionDecl(Stmt):
    name: str
    params: List['Param']
    return_type: str
    body: List[Stmt]

@dataclass
class Param:
    name: str
    type_name: str
    mode: str = "BYVAL"  # BYVAL or BYREF

@dataclass
class DeclareStmt(Stmt):
    name: str
    type_name: str
    is_array: bool = False
    array_bounds: Optional[List[tuple]] = None # List of (lower, upper)

@dataclass
class ConstantDecl(Stmt):
    name: str
    value: Expr

@dataclass
class ReturnStmt(Stmt):
    value: Expr

@dataclass
class FileStmt(Stmt):
    """
    OPENFILE <file> FOR <mode>
    READFILE <file>, <var>
    WRITEFILE <file>, <data>
    CLOSEFILE <file>
    SEEK <file>, <address>
    GETRECORD <file>, <var>
    PUTRECORD <file>, <var>
    """
    operation: str  # OPEN, READ, WRITE, CLOSE, SEEK, GETRECORD, PUTRECORD
    filename: Expr
    mode: Optional[str] = None # READ, WRITE, APPEND, RANDOM (for OPEN)
    variable: Optional[str] = None # For READ, GETRECORD, PUTRECORD
    data: Optional[Expr] = None # For WRITE, SEEK (address)

@dataclass
class TypeDecl(Stmt):
    """
    TYPE <name>
        DECLARE <field> : <type>
        ...
    ENDTYPE

    or Enumerated:
    TYPE <name> = (value1, value2, ...)
    """
    name: str
    fields: List[tuple] # (name, type) for records
    enum_values: Optional[List[str]] = None  # For enumerated types

@dataclass
class ClassDecl(Stmt):
    """
    CLASS <name> [INHERITS <parent>]
        <members>  (DECLARE, PROCEDURE, FUNCTION with optional PUBLIC/PRIVATE)
    ENDCLASS
    """
    name: str
    parent: Optional[str]
    members: List[Stmt]  # DeclareStmt, ProcedureDecl, FunctionDecl with access info

@dataclass
class MethodCallExpr(Expr):
    """Method call on an object: obj.Method(args)"""
    object_expr: Expr
    method_name: str
    arguments: List[Expr]

@dataclass
class NewExpr(Expr):
    """Object creation: <obj> ← NEW <class>(<params>)"""
    class_name: str
    arguments: List[Expr]

@dataclass
class SuperExpr(Expr):
    """SUPER.Method(...) call."""
    method: str
    arguments: List[Expr]
