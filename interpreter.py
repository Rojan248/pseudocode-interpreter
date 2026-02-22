"""
9618 Pseudocode Interpreter — Core Execution Engine.

Refactored for improved cohesion and reduced complexity:
  - Built-in functions extracted to builtins_handler.py
  - File I/O operations extracted to file_handler.py
  - DryRunInterpreter extracted to dry_run_interpreter.py
"""
import os
import pickle
from typing import Any, List, Union
from ast_nodes import *
from symbol_table import SymbolTable, Cell, DataType, ArrayBounds, SymbolInfo
from builtins_handler import BUILTIN_NAMES, call_builtin
from file_handler import execute_file_operation, InterpreterFileError


class InterpreterError(Exception):
    pass


class ReturnException(Exception):
    def __init__(self, value):
        self.value = value


class PseudocodeObject:
    """Runtime representation of a 9618 class instance."""
    def __init__(self, class_name, attributes, methods, parent_class=None):
        self.class_name = class_name
        self.attributes = attributes   # dict of {name: Cell}
        self.methods = methods         # dict of {name: ProcedureDecl/FunctionDecl}
        self.parent_class = parent_class


# ═══════════════════════════════════════════════════════
#  Core Interpreter
# ═══════════════════════════════════════════════════════

class Interpreter:
    _BUILTINS = BUILTIN_NAMES

    def __init__(self, symbol_table: SymbolTable):
        self.start_symbol_table = symbol_table
        self.symbol_table = symbol_table
        self.output_buffer = []
        self.open_files = {}
        self.user_types = {}
        self.procedures = {}
        self.functions = {}
        self.classes = {}
        self.current_object = None
        self.current_line = 0
        self._init_dispatch_tables()

    def _init_dispatch_tables(self):
        self._stmt_visitors = {
            DeclareStmt:        self.visit_DeclareStmt,
            ConstantDecl:       self.visit_ConstantDecl,
            AssignStmt:         self.visit_AssignStmt,
            InputStmt:          self.visit_InputStmt,
            OutputStmt:         self.visit_OutputStmt,
            IfStmt:             self.visit_IfStmt,
            CaseStmt:           self.visit_CaseStmt,
            WhileStmt:          self.visit_WhileStmt,
            RepeatStmt:         self.visit_RepeatStmt,
            ForStmt:            self.visit_ForStmt,
            ProcedureDecl:      self.visit_ProcedureDecl,
            FunctionDecl:       self.visit_FunctionDecl,
            ProcedureCallStmt:  self.visit_ProcedureCallStmt,
            ReturnStmt:         self.visit_ReturnStmt,
            TypeDecl:           self.visit_TypeDecl,
            ClassDecl:          self.visit_ClassDecl,
            FileStmt:           self.visit_FileStmt,
            CallExpr:           self.evaluate,
            MethodCallExpr:     self.evaluate,
            SuperExpr:          self.evaluate,
            BinaryExpr:         self.evaluate,
            UnaryExpr:          self.evaluate,
            LiteralExpr:        self.evaluate,
            VariableExpr:       self.evaluate,
            ArrayAccessExpr:    self.evaluate,
            MemberExpr:         self.evaluate,
            NewExpr:            self.evaluate,
        }
        self._expr_evaluators = {
            LiteralExpr:        self.evaluate_LiteralExpr,
            VariableExpr:       self.evaluate_VariableExpr,
            ArrayAccessExpr:    self.evaluate_ArrayAccessExpr,
            MemberExpr:         self.evaluate_MemberExpr,
            BinaryExpr:         self.evaluate_BinaryExpr,
            UnaryExpr:          self.evaluate_UnaryExpr,
            CallExpr:           self.evaluate_CallExpr,
            MethodCallExpr:     self.evaluate_MethodCallExpr,
            NewExpr:            self.evaluate_NewExpr,
            SuperExpr:          self.evaluate_SuperExpr,
        }

    # ── Core Execution ──

    def interpret(self, statements: List[Stmt]):
        for stmt in statements:
            self.execute(stmt)

    def execute(self, stmt: Stmt):
        if hasattr(stmt, 'line') and stmt.line > 0:
            self.current_line = stmt.line
        visitor = self._stmt_visitors.get(type(stmt))
        if visitor:
            return visitor(stmt)
        if isinstance(stmt, Expr):
            return self.evaluate(stmt)
        raise InterpreterError(f"No visit method for {type(stmt).__name__}")

    def evaluate(self, expr: Expr) -> Any:
        evaluator = self._expr_evaluators.get(type(expr))
        if evaluator:
            return evaluator(expr)
        raise InterpreterError(f"No evaluate method for {type(expr).__name__}")

    # ── Statement Visitors ──

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        dtype = self.symbol_table.resolve_type(stmt.type_name)
        if stmt.is_array:
            self._declare_array(stmt, dtype)
        else:
            self._declare_variable(stmt, dtype)

    def _declare_array(self, stmt, dtype):
        if not stmt.array_bounds:
            raise InterpreterError("Array declaration missing bounds")
        bounds = ArrayBounds(stmt.array_bounds, dtype)
        self.symbol_table.declare(stmt.name, DataType.ARRAY, is_array=True, array_bounds=bounds)

    def _declare_variable(self, stmt, dtype):
        initial_val = None
        if dtype == DataType.UNKNOWN and stmt.type_name in self.user_types:
            initial_val = self._init_record_instance(stmt.type_name)
            dtype = DataType.RECORD
        self.symbol_table.declare(stmt.name, dtype, initial_value=initial_val)

    def _init_record_instance(self, type_name):
        """Create a fresh record instance from a user-defined TYPE."""
        fields = self.user_types[type_name]
        record = {}
        for f_name, f_type_name in fields:
            f_dtype = self.symbol_table.resolve_type(f_type_name)
            f_default = Cell(None, f_dtype)._default_value(f_dtype)
            record[f_name] = Cell(f_default, f_dtype)
        return record

    def visit_ConstantDecl(self, stmt: ConstantDecl):
        value = self.evaluate(stmt.value)
        dtype = self._infer_type(value)
        self.symbol_table.declare(stmt.name, dtype, is_constant=True, constant_value=value)

    def visit_AssignStmt(self, stmt: AssignStmt):
        value = self.evaluate(stmt.value)
        val_type = self._infer_type(value)
        if isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, value, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            self._assign_to_array(stmt.target, value, val_type)
        elif isinstance(stmt.target, MemberExpr):
            self._assign_to_member(stmt.target, value, val_type)

    def _assign_to_array(self, target, value, val_type):
        """Assign a value to an array element."""
        indices = [self.evaluate(idx) for idx in target.indices]
        _validate_integer_indices(indices)
        self.symbol_table.array_assign(target.array, indices, value, val_type)

    def _assign_to_member(self, target, value, val_type):
        """Assign a value to a record field or object attribute."""
        record_val = self.evaluate(target.record)
        field = target.field
        if isinstance(record_val, PseudocodeObject):
            if field not in record_val.attributes:
                raise InterpreterError(f"Attribute {field} not found on object")
            record_val.attributes[field].set(value, val_type)
        elif isinstance(record_val, dict):
            if field not in record_val:
                raise InterpreterError(f"Field {field} not found")
            record_val[field].set(value, val_type)
        else:
            raise InterpreterError("Target is not a record or object")

    # ── Input / Output ──

    def visit_InputStmt(self, stmt: InputStmt):
        label = _extract_input_label(stmt)
        val_str = input(f"INPUT {label}: ")
        target_type = self._get_input_target_type(stmt)
        if target_type is not None:
            val, val_type = self._coerce_input(val_str, target_type)
        else:
            val, val_type = self._auto_parse_input(val_str)
        self._assign_input_to_target(stmt, val, val_type)

    def _get_input_target_type(self, stmt):
        """Look up the declared type of the INPUT target."""
        try:
            if isinstance(stmt.target, VariableExpr):
                return self.symbol_table.get_cell(stmt.target.name).type
            if isinstance(stmt.target, str):
                return self.symbol_table.get_cell(stmt.target).type
            if isinstance(stmt.target, ArrayAccessExpr):
                cell = self.symbol_table.get_cell(stmt.target.array)
                if cell.is_array and cell.array_bounds:
                    return cell.array_bounds.element_type
        except Exception:
            pass
        return None

    def _assign_input_to_target(self, stmt, val, val_type):
        """Route input value to the correct target (variable, array, etc.)."""
        if isinstance(stmt.target, VariableExpr):
            self.symbol_table.assign(stmt.target.name, val, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            self.symbol_table.array_assign(stmt.target.array, indices, val, val_type)
        elif isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, val, val_type)

    def _coerce_input(self, val_str: str, target_type: DataType):
        """Coerce an input string to the target variable's declared type."""
        coerce_fn = _INPUT_COERCERS.get(target_type)
        if coerce_fn:
            return coerce_fn(val_str, target_type)
        return self._auto_parse_input(val_str)

    def _auto_parse_input(self, val_str: str):
        """Auto-infer type from input string (fallback)."""
        if val_str.lower() == 'true':
            return True, DataType.BOOLEAN
        if val_str.lower() == 'false':
            return False, DataType.BOOLEAN
        try:
            return int(val_str), DataType.INTEGER
        except ValueError:
            try:
                return float(val_str), DataType.REAL
            except ValueError:
                return val_str, DataType.STRING

    def _format_output(self, val) -> str:
        """Format a value for OUTPUT, converting booleans to 9618 uppercase."""
        if isinstance(val, bool):
            return "TRUE" if val else "FALSE"
        return str(val)

    def visit_OutputStmt(self, stmt: OutputStmt):
        values = [self._format_output(self.evaluate(arg)) for arg in stmt.values]
        print("".join(values))

    # ── Control Flow ──

    def visit_IfStmt(self, stmt: IfStmt):
        branch = stmt.then_branch if self.evaluate(stmt.condition) else stmt.else_branch
        if branch:
            for s in branch:
                self.execute(s)

    def visit_CaseStmt(self, stmt: CaseStmt):
        sel_val = self.evaluate(stmt.selector)
        for branch in stmt.branches:
            if _case_branch_matches(branch, sel_val, self.evaluate):
                for s in branch.statements:
                    self.execute(s)
                return
        if stmt.otherwise_branch:
            for s in stmt.otherwise_branch:
                self.execute(s)

    def visit_WhileStmt(self, stmt: WhileStmt):
        while self.evaluate(stmt.condition):
            for s in stmt.body:
                self.execute(s)

    def visit_RepeatStmt(self, stmt: RepeatStmt):
        while True:
            for s in stmt.body:
                self.execute(s)
            if self.evaluate(stmt.condition):
                break

    def visit_ForStmt(self, stmt: ForStmt):
        var = stmt.identifier
        start = self.evaluate(stmt.start_value)
        end = self.evaluate(stmt.end_value)
        step = self.evaluate(stmt.step_value) if stmt.step_value else 1
        if step == 0:
            raise InterpreterError("FOR loop STEP cannot be zero")
        loop_type = self._infer_type(start)
        self.symbol_table.assign(var, start, loop_type)
        while True:
            curr = self.symbol_table.get_cell(var).get()
            if step > 0 and curr > end:
                break
            if step < 0 and curr < end:
                break
            for s in stmt.body:
                self.execute(s)
            curr = self.symbol_table.get_cell(var).get()
            self.symbol_table.assign(var, curr + step, loop_type)

    # ── Procedure / Function Handling ──

    def visit_ProcedureDecl(self, stmt: ProcedureDecl):
        self.procedures[stmt.name] = stmt

    def visit_FunctionDecl(self, stmt: FunctionDecl):
        self.functions[stmt.name] = stmt

    def visit_ProcedureCallStmt(self, stmt: ProcedureCallStmt):
        if '.' in stmt.name:
            self._call_method_on_object(stmt)
            return
        if stmt.name not in self.procedures:
            raise InterpreterError(f"Undefined procedure: {stmt.name}")
        self._call_callable(self.procedures[stmt.name], stmt.arguments)

    def _call_method_on_object(self, stmt):
        """Handle 'obj.Method(args)' procedure calls."""
        obj_name, method_name = stmt.name.split('.', 1)
        obj = self.symbol_table.get_cell(obj_name).get()
        if not isinstance(obj, PseudocodeObject):
            raise InterpreterError(f"{obj_name} is not an object")
        if method_name not in obj.methods:
            raise InterpreterError(f"Method '{method_name}' not found on {obj.class_name}")
        self._call_method(obj, obj.methods[method_name], stmt.arguments)

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        val = self.evaluate(stmt.value) if stmt.value is not None else None
        raise ReturnException(val)

    def evaluate_CallExpr(self, expr: CallExpr):
        if expr.callee in self._BUILTINS:
            return self._call_builtin(expr)
        if expr.callee not in self.functions:
            raise InterpreterError(f"Undefined function: {expr.callee}")
        try:
            self._call_callable(self.functions[expr.callee], expr.arguments)
        except ReturnException as e:
            return e.value
        raise InterpreterError(f"Function {expr.callee} did not return a value")

    def _call_callable(self, decl: Union[ProcedureDecl, FunctionDecl], args: List[Expr]):
        if len(args) != len(decl.params):
            raise TypeError(f"Argument count mismatch for {decl.name}: "
                            f"expected {len(decl.params)}, got {len(args)}")
        evaluated_args = self._prepare_arguments(decl, args)
        self.symbol_table.enter_scope()
        try:
            self._bind_parameters(decl, evaluated_args)
            for s in decl.body:
                self.execute(s)
        finally:
            self.symbol_table.exit_scope()

    def _prepare_arguments(self, decl, args):
        """Evaluate arguments: BYVAL gets values, BYREF gets cell references."""
        evaluated = []
        for i, param in enumerate(decl.params):
            if param.mode == "BYREF":
                cell = self._resolve_byref_cell(args[i], param.name)
                evaluated.append({'mode': 'BYREF', 'cell': cell,
                                  'name': param.name, 'type': param.type_name})
            else:
                val = self.evaluate(args[i])
                evaluated.append({'mode': 'BYVAL', 'value': val,
                                  'name': param.name, 'type': param.type_name})
        return evaluated

    def _resolve_byref_cell(self, arg_expr, param_name):
        """Resolve the cell for a BYREF argument."""
        if isinstance(arg_expr, VariableExpr):
            return self.symbol_table.get_cell(arg_expr.name)
        if isinstance(arg_expr, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in arg_expr.indices]
            return self.symbol_table.array_access(arg_expr.array, indices)
        raise TypeError(f"BYREF argument for {param_name} must be a variable")

    def _bind_parameters(self, decl, evaluated_args):
        """Bind evaluated arguments into the current scope."""
        for arg in evaluated_args:
            dtype = self.symbol_table.resolve_type(arg['type'])
            if arg['mode'] == 'BYREF':
                self.symbol_table.declare_parameter(arg['name'], dtype, "BYREF",
                                                    caller_cell=arg['cell'])
            else:
                temp_cell = Cell(arg['value'], dtype)
                self.symbol_table.declare_parameter(arg['name'], dtype, "BYVAL",
                                                    caller_cell=temp_cell)

    # ── Type Declarations ──

    def visit_TypeDecl(self, stmt: TypeDecl):
        if stmt.enum_values:
            self._declare_enum(stmt)
        else:
            self.user_types[stmt.name] = stmt.fields

    def _declare_enum(self, stmt):
        """Handle: TYPE Season = (Spring, Summer, Autumn, Winter)"""
        self.user_types[stmt.name] = {'__enum__': True, 'values': stmt.enum_values}
        for i, val_name in enumerate(stmt.enum_values):
            self.symbol_table.declare(val_name, DataType.INTEGER,
                                     is_constant=True, constant_value=i)

    # ── Class Handling ──

    def visit_ClassDecl(self, stmt):
        """Store class definition for later instantiation."""
        class_def = {
            'name': stmt.name, 'parent': stmt.parent,
            'members': stmt.members, 'methods': {}, 'fields': []
        }
        for member in stmt.members:
            if isinstance(member, (ProcedureDecl, FunctionDecl)):
                class_def['methods'][member.name] = member
            elif isinstance(member, DeclareStmt):
                class_def['fields'].append(member)
        self.classes[stmt.name] = class_def

    def evaluate_NewExpr(self, expr):
        """Create a new object: obj ← NEW ClassName(args)"""
        if expr.class_name not in self.classes:
            raise InterpreterError(f"Undefined class: {expr.class_name}")
        class_def = self.classes[expr.class_name]
        attributes = {}
        self._init_class_fields(class_def, attributes)
        methods = {}
        self._collect_methods(class_def, methods)
        obj = PseudocodeObject(expr.class_name, attributes, methods,
                               class_def.get('parent'))
        if 'NEW' in methods:
            self._invoke_constructor(obj, methods['NEW'], expr.arguments)
        return obj

    def _invoke_constructor(self, obj, constructor, arguments):
        """Call the NEW constructor on a freshly created object."""
        old_obj = self.current_object
        self.current_object = obj
        try:
            self._call_method(obj, constructor, arguments)
        finally:
            self.current_object = old_obj

    def _init_class_fields(self, class_def, attributes):
        """Initialize fields for a class, including inherited fields."""
        if class_def.get('parent') and class_def['parent'] in self.classes:
            self._init_class_fields(self.classes[class_def['parent']], attributes)
        for field_decl in class_def.get('fields', []):
            dtype = self.symbol_table.resolve_type(field_decl.type_name)
            default = Cell(None, dtype)._default_value(dtype)
            attributes[field_decl.name] = Cell(default, dtype)

    def _collect_methods(self, class_def, methods):
        """Collect methods, with child overriding parent."""
        if class_def.get('parent') and class_def['parent'] in self.classes:
            self._collect_methods(self.classes[class_def['parent']], methods)
        for name, method in class_def.get('methods', {}).items():
            methods[name] = method

    def _call_method(self, obj, method_decl, arg_exprs):
        """Call a method on an object, binding 'self' attributes to scope."""
        if len(arg_exprs) != len(method_decl.params):
            raise TypeError(f"Argument count mismatch for {method_decl.name}")
        self.symbol_table.enter_scope()
        old_obj = self.current_object
        self.current_object = obj
        try:
            self._bind_object_attributes(obj)
            for i, param in enumerate(method_decl.params):
                val = self.evaluate(arg_exprs[i])
                dtype = self.symbol_table.resolve_type(param.type_name)
                temp_cell = Cell(val, dtype)
                self.symbol_table.declare_parameter(param.name, dtype,
                                                    param.mode, caller_cell=temp_cell)
            for s in method_decl.body:
                self.execute(s)
        except ReturnException as e:
            return e.value
        finally:
            self.symbol_table.exit_scope()
            self.current_object = old_obj
        return None

    def _bind_object_attributes(self, obj):
        """Bind object attributes into the current scope."""
        for attr_name, attr_cell in obj.attributes.items():
            sym = SymbolInfo(
                name=attr_name, cell=attr_cell,
                scope_level=self.symbol_table.scope_level,
                is_parameter=False
            )
            self.symbol_table.inject_symbol(sym)

    def evaluate_SuperExpr(self, expr):
        """SUPER.Method(args) — call parent class method."""
        if self.current_object is None:
            raise InterpreterError("SUPER used outside of class context")
        obj = self.current_object
        if not obj.parent_class or obj.parent_class not in self.classes:
            raise InterpreterError("No parent class for SUPER call")
        parent_def = self.classes[obj.parent_class]
        if expr.method not in parent_def.get('methods', {}):
            raise InterpreterError(
                f"Method {expr.method} not found in parent class {obj.parent_class}")
        return self._call_method(obj, parent_def['methods'][expr.method], expr.arguments)

    # ── File I/O (delegated to file_handler) ──

    def visit_FileStmt(self, stmt: FileStmt):
        try:
            result = execute_file_operation(stmt, self.open_files, self.evaluate)
        except InterpreterFileError as e:
            raise InterpreterError(str(e))
        if result is None:
            return
        self._apply_file_result(result)

    def _apply_file_result(self, result):
        """Apply the side-effects returned by the file handler."""
        op = result[0]
        if op == 'assign':
            _, var_name, value, data_type = result
            self.symbol_table.assign(var_name, value, data_type)
        elif op == 'assign_infer':
            _, var_name, value = result
            self.symbol_table.assign(var_name, value, self._infer_type(value))
        elif op == 'putrecord':
            _, var_name, filename = result
            pickle.dump(self.symbol_table.get_cell(var_name).get(),
                        self.open_files[filename])

    # ── Built-in Functions (delegated to builtins_handler) ──

    def _call_builtin(self, expr: CallExpr):
        args = [self.evaluate(a) for a in expr.arguments]
        return call_builtin(expr.callee, args, self.open_files)

    # ── Expression Evaluators ──

    def evaluate_LiteralExpr(self, expr: LiteralExpr):
        return expr.value

    def evaluate_VariableExpr(self, expr: VariableExpr):
        return self.symbol_table.get_cell(expr.name).get()

    def evaluate_ArrayAccessExpr(self, expr: ArrayAccessExpr):
        indices = [self.evaluate(idx) for idx in expr.indices]
        _validate_integer_indices(indices)
        return self.symbol_table.array_access(expr.array, indices).get()

    def evaluate_MemberExpr(self, expr: MemberExpr):
        record_val = self.evaluate(expr.record)
        if isinstance(record_val, PseudocodeObject):
            return _get_object_attribute(record_val, expr.field)
        if isinstance(record_val, dict):
            return _get_record_field(record_val, expr.field)
        raise InterpreterError(
            f"Accessing field '{expr.field}' on non-record type {type(record_val)}")

    def evaluate_BinaryExpr(self, expr: BinaryExpr):
        # Optimization: Inline operator logic to avoid function call overhead
        # in the hot path of binary expression evaluation.
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)
        op = expr.operator

        if op == '+':
            return left + right
        elif op == '-':
            return left - right
        elif op == '*':
            return left * right
        elif op == '/':
            if right == 0:
                raise InterpreterError(f"Division by zero ({op})")
            return left / right
        elif op == 'DIV':
            if right == 0:
                raise InterpreterError(f"Division by zero ({op})")
            return int(left / right)
        elif op == 'MOD':
            if right == 0:
                raise InterpreterError(f"Division by zero ({op})")
            return left - int(left / right) * right
        elif op == '&':
            return str(left) + str(right)
        elif op == '=':
            return left == right
        elif op == '<>':
            return left != right
        elif op == '<':
            return left < right
        elif op == '>':
            return left > right
        elif op == '<=':
            return left <= right
        elif op == '>=':
            return left >= right
        elif op == 'AND':
            return bool(left and right)
        elif op == 'OR':
            return bool(left or right)

        raise InterpreterError(f"Unknown operator {op}")

    def evaluate_UnaryExpr(self, expr: UnaryExpr):
        val = self.evaluate(expr.operand)
        if expr.operator == '-':
            return -val
        if expr.operator == 'NOT':
            return not val
        raise InterpreterError(f"Unknown unary operator {expr.operator}")

    def evaluate_MethodCallExpr(self, expr):
        """Evaluate obj.Method(args) — method call on object."""
        obj = self.evaluate(expr.object_expr)
        if not isinstance(obj, PseudocodeObject):
            raise InterpreterError("Cannot call method on non-object")
        if expr.method_name not in obj.methods:
            raise InterpreterError(
                f"Method '{expr.method_name}' not found on object of class {obj.class_name}")
        return self._call_method(obj, obj.methods[expr.method_name], expr.arguments)

    def _infer_type(self, val: Any) -> DataType:
        if isinstance(val, bool):
            return DataType.BOOLEAN
        if isinstance(val, int):
            return DataType.INTEGER
        if isinstance(val, float):
            return DataType.REAL
        if isinstance(val, str):
            return DataType.STRING
        return DataType.UNKNOWN


# ═══════════════════════════════════════════════════════
#  Module-level helper functions
# ═══════════════════════════════════════════════════════

def _validate_integer_indices(indices):
    """Guard: validate that all array indices are integers."""
    for i, val in enumerate(indices):
        if not isinstance(val, int):
            raise TypeError(f"Array index {i+1} must be an integer, got {type(val)}")


def _extract_input_label(stmt):
    """Extract a display label from an INPUT statement target."""
    if isinstance(stmt.target, VariableExpr):
        return stmt.target.name
    if isinstance(stmt.target, ArrayAccessExpr):
        return stmt.target.array
    if isinstance(stmt.target, str):
        return stmt.target
    return "?"


def _get_object_attribute(obj, field):
    """Access an attribute on a PseudocodeObject."""
    if field in obj.attributes:
        cell = obj.attributes[field]
        return cell.get() if isinstance(cell, Cell) else cell
    raise InterpreterError(
        f"Attribute '{field}' not found on object of class {obj.class_name}")


def _get_record_field(record_val, field):
    """Access a field on a record (dict of Cells)."""
    if field not in record_val:
        raise InterpreterError(f"Field '{field}' not found in record")
    cell = record_val[field]
    return cell.get() if isinstance(cell, Cell) else cell


def _case_branch_matches(branch, sel_val, evaluate_fn):
    """Check if a CASE branch matches the selector value."""
    for val_expr in branch.values:
        if isinstance(val_expr, RangeExpr):
            start = evaluate_fn(val_expr.start)
            end = evaluate_fn(val_expr.end)
            if start <= sel_val <= end:
                return True
        else:
            if sel_val == evaluate_fn(val_expr):
                return True
    return False




# ── Input coercion dispatch table ──

def _coerce_integer(val_str, _):
    return int(val_str), DataType.INTEGER

def _coerce_real(val_str, _):
    return float(val_str), DataType.REAL

def _coerce_boolean(val_str, _):
    upper = val_str.upper()
    if upper == 'TRUE':
        return True, DataType.BOOLEAN
    if upper == 'FALSE':
        return False, DataType.BOOLEAN
    raise InterpreterError(f"Cannot convert '{val_str}' to BOOLEAN")

def _coerce_char(val_str, _):
    if len(val_str) != 1:
        raise InterpreterError(f"CHAR requires exactly one character, got '{val_str}'")
    return val_str, DataType.CHAR

def _coerce_string(val_str, _):
    return val_str, DataType.STRING


_INPUT_COERCERS = {
    DataType.INTEGER: _coerce_integer,
    DataType.REAL:    _coerce_real,
    DataType.BOOLEAN: _coerce_boolean,
    DataType.CHAR:    _coerce_char,
    DataType.STRING:  _coerce_string,
}


# DryRunInterpreter has been extracted to dry_run_interpreter.py
# Import it directly: from dry_run_interpreter import DryRunInterpreter
