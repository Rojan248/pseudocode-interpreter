import os
import math
import random
import pickle
from typing import Any, List, Optional, Union, Callable
from ast_nodes import *
from symbol_table import SymbolTable, Cell, DataType, ArrayBounds, SymbolInfo
from lexer import TokenType # for operator mapping?

class InterpreterError(Exception):
    pass

class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

class PseudocodeObject:
    """Runtime representation of a 9618 class instance."""
    def __init__(self, class_name, attributes, methods, parent_class=None):
        self.class_name = class_name
        self.attributes = attributes  # dict of {name: Cell}
        self.methods = methods        # dict of {name: ProcedureDecl/FunctionDecl}
        self.parent_class = parent_class

class Interpreter:
    # Builtin function names (class-level constant)
    _BUILTINS = frozenset({
        'LENGTH', 'UCASE', 'LCASE', 'MID', 'RIGHT', 'LEFT',
        'INT', 'RAND', 'SQRT', 'NUM_TO_STR', 'STR_TO_NUM',
        'ASC', 'CHR', 'EOF'
    })

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
            DeclareStmt: self.visit_DeclareStmt,
            ConstantDecl: self.visit_ConstantDecl,
            AssignStmt: self.visit_AssignStmt,
            InputStmt: self.visit_InputStmt,
            OutputStmt: self.visit_OutputStmt,
            IfStmt: self.visit_IfStmt,
            CaseStmt: self.visit_CaseStmt,
            WhileStmt: self.visit_WhileStmt,
            RepeatStmt: self.visit_RepeatStmt,
            ForStmt: self.visit_ForStmt,
            ProcedureDecl: self.visit_ProcedureDecl,
            FunctionDecl: self.visit_FunctionDecl,
            ProcedureCallStmt: self.visit_ProcedureCallStmt,
            ReturnStmt: self.visit_ReturnStmt,
            TypeDecl: self.visit_TypeDecl,
            ClassDecl: self.visit_ClassDecl,
            FileStmt: self.visit_FileStmt,
            # Handle expression statements
            CallExpr: self.evaluate,
            MethodCallExpr: self.evaluate,
            SuperExpr: self.evaluate,
            BinaryExpr: self.evaluate,
            UnaryExpr: self.evaluate,
            LiteralExpr: self.evaluate,
            VariableExpr: self.evaluate,
            ArrayAccessExpr: self.evaluate,
            MemberExpr: self.evaluate,
            NewExpr: self.evaluate,
        }

        self._expr_evaluators = {
            LiteralExpr: self.evaluate_LiteralExpr,
            VariableExpr: self.evaluate_VariableExpr,
            ArrayAccessExpr: self.evaluate_ArrayAccessExpr,
            MemberExpr: self.evaluate_MemberExpr,
            BinaryExpr: self.evaluate_BinaryExpr,
            UnaryExpr: self.evaluate_UnaryExpr,
            CallExpr: self.evaluate_CallExpr,
            MethodCallExpr: self.evaluate_MethodCallExpr,
            NewExpr: self.evaluate_NewExpr,
            SuperExpr: self.evaluate_SuperExpr,
        }

    def interpret(self, statements: List[Stmt]):
        for stmt in statements:
            self.execute(stmt)

    def execute(self, stmt: Stmt):
        if hasattr(stmt, 'line') and stmt.line > 0:
            self.current_line = stmt.line
        
        visitor = self._stmt_visitors.get(type(stmt))
        if visitor:
            return visitor(stmt)

        # Fallback for expression statements not in map (if any)
        if isinstance(stmt, Expr):
            return self.evaluate(stmt)

        return self.no_visit_method(stmt)

    def no_visit_method(self, node):
        raise InterpreterError(f"No visit method for {type(node).__name__}")

    def evaluate(self, expr: Expr) -> Any:
        evaluator = self._expr_evaluators.get(type(expr))
        if evaluator:
            return evaluator(expr)
        return self.no_eval_method(expr)

    def no_eval_method(self, node):
        raise InterpreterError(f"No evaluate method for {type(node).__name__}")

    # --- Statement Visitors ---

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        dtype = self.symbol_table.resolve_type(stmt.type_name)
        if stmt.is_array:
            if not stmt.array_bounds:
                raise InterpreterError("Array declaration missing bounds")
            bounds = ArrayBounds(stmt.array_bounds, dtype)
            self.symbol_table.declare(stmt.name, DataType.ARRAY, is_array=True, array_bounds=bounds)
        else:
            initial_val = None
            if dtype == DataType.UNKNOWN and stmt.type_name in self.user_types:
                fields = self.user_types[stmt.type_name]
                record_inst = {}
                for f_name, f_type_name in fields:
                    f_dtype = self.symbol_table.resolve_type(f_type_name)
                    f_default = Cell(None, f_dtype)._default_value(f_dtype)
                    record_inst[f_name] = Cell(f_default, f_dtype)
                initial_val = record_inst
                dtype = DataType.RECORD
            self.symbol_table.declare(stmt.name, dtype, initial_value=initial_val)

    def visit_ConstantDecl(self, stmt: 'ConstantDecl'):
        value = self.evaluate(stmt.value)
        # Infer type from the literal value
        # IMPORTANT: check bool before int because bool is subclass of int in Python
        if isinstance(value, bool):
            dtype = DataType.BOOLEAN
        elif isinstance(value, int):
            dtype = DataType.INTEGER
        elif isinstance(value, float):
            dtype = DataType.REAL
        elif isinstance(value, str):
            dtype = DataType.STRING
        else:
            dtype = DataType.STRING
        self.symbol_table.declare(stmt.name, dtype, is_constant=True, constant_value=value)

    def visit_AssignStmt(self, stmt: AssignStmt):
        value = self.evaluate(stmt.value)
        # Determine type of value (heuristic or explicit?)
        # 9618 requires types. We should infer type from value or just pass None and let Set handle it (if same type)
        # But we need val_type for set_array_element.
        val_type = self._infer_type(value)
        
        if isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, value, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            # Evaluate indices
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            for i, val in enumerate(indices):
                if not isinstance(val, int):
                    raise TypeError(f"Array index {i+1} must be an integer")
            self.symbol_table.array_assign(stmt.target.array, indices, value, val_type)
        elif isinstance(stmt.target, MemberExpr):
            # Assign to record field or object attribute
            record_val = self.evaluate(stmt.target.record)
            field = stmt.target.field
            
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

    def _coerce_input(self, val_str: str, target_type: DataType):
        """Coerce an input string to the target variable's declared type."""
        try:
            if target_type == DataType.INTEGER:
                return int(val_str), DataType.INTEGER
            elif target_type == DataType.REAL:
                return float(val_str), DataType.REAL
            elif target_type == DataType.BOOLEAN:
                if val_str.upper() == 'TRUE':
                    return True, DataType.BOOLEAN
                elif val_str.upper() == 'FALSE':
                    return False, DataType.BOOLEAN
                raise InterpreterError(f"Cannot convert '{val_str}' to BOOLEAN")
            elif target_type == DataType.CHAR:
                if len(val_str) != 1:
                    raise InterpreterError(f"CHAR requires exactly one character, got '{val_str}'")
                return val_str, DataType.CHAR
            elif target_type == DataType.STRING:
                return val_str, DataType.STRING
        except ValueError:
            raise InterpreterError(
                f"Cannot convert input '{val_str}' to {target_type.name}"
            )
        # Fallback: auto-infer (for UNKNOWN or unresolvable types)
        return self._auto_parse_input(val_str)

    def _auto_parse_input(self, val_str: str):
        """Auto-infer type from input string (fallback)."""
        if val_str.lower() == 'true': return True, DataType.BOOLEAN
        if val_str.lower() == 'false': return False, DataType.BOOLEAN
        try:
            return int(val_str), DataType.INTEGER
        except ValueError:
            try:
                return float(val_str), DataType.REAL
            except ValueError:
                return val_str, DataType.STRING

    def visit_InputStmt(self, stmt: InputStmt):
        # Determine prompt label and target type
        if isinstance(stmt.target, VariableExpr):
            label = stmt.target.name
        elif isinstance(stmt.target, ArrayAccessExpr):
            label = stmt.target.array
        elif isinstance(stmt.target, str):
            label = stmt.target
        else:
            label = "?"
        
        val_str = input(f"INPUT {label}: ")
        
        # Try to look up the target's declared type for proper coercion
        target_type = None
        try:
            if isinstance(stmt.target, VariableExpr):
                target_type = self.symbol_table.get_cell(stmt.target.name).type
            elif isinstance(stmt.target, str):
                target_type = self.symbol_table.get_cell(stmt.target).type
            elif isinstance(stmt.target, ArrayAccessExpr):
                cell = self.symbol_table.get_cell(stmt.target.array)
                if cell.is_array and cell.array_bounds:
                    target_type = cell.array_bounds.element_type
        except Exception:
            pass
        
        if target_type is not None:
            val, val_type = self._coerce_input(val_str, target_type)
        else:
            val, val_type = self._auto_parse_input(val_str)
        
        if isinstance(stmt.target, VariableExpr):
            self.symbol_table.assign(stmt.target.name, val, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            self.symbol_table.array_assign(stmt.target.array, indices, val, val_type)
        elif isinstance(stmt.target, str):
            # Legacy string identifier
            self.symbol_table.assign(stmt.target, val, val_type)

    def _format_output(self, val) -> str:
        """Format a value for OUTPUT, converting booleans to 9618 uppercase."""
        if isinstance(val, bool):
            return "TRUE" if val else "FALSE"
        return str(val)

    def visit_OutputStmt(self, stmt: OutputStmt):
        values = [self._format_output(self.evaluate(arg)) for arg in stmt.values]
        print("".join(values))

    def visit_IfStmt(self, stmt: IfStmt):
        if self.evaluate(stmt.condition):
            for s in stmt.then_branch:
                self.execute(s)
        elif stmt.else_branch:
            for s in stmt.else_branch:
                self.execute(s)

    def visit_CaseStmt(self, stmt: CaseStmt):
        sel_val = self.evaluate(stmt.selector)
        matched = False
        for branch in stmt.branches:
            for val_expr in branch.values:
                if isinstance(val_expr, RangeExpr):
                    start = self.evaluate(val_expr.start)
                    end = self.evaluate(val_expr.end)
                    if start <= sel_val <= end:
                        matched = True
                else:
                    val = self.evaluate(val_expr)
                    if sel_val == val:
                        matched = True
                        
                if matched:
                    for s in branch.statements:
                        self.execute(s)
                    return # Exit case
        
        if not matched and stmt.otherwise_branch:
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
            # Loop condition
            if step > 0 and curr > end: break
            if step < 0 and curr < end: break
            
            for s in stmt.body:
                self.execute(s)
                
            # Increment
            curr = self.symbol_table.get_cell(var).get()
            self.symbol_table.assign(var, curr + step, loop_type)

    # --- Procedure/Function Handling ---

    def visit_ProcedureDecl(self, stmt: ProcedureDecl):
        self.procedures[stmt.name] = stmt

    def visit_FunctionDecl(self, stmt: FunctionDecl):
        self.functions[stmt.name] = stmt

    def visit_ProcedureCallStmt(self, stmt: ProcedureCallStmt):
        # Check for method call: "obj.Method" format from parser
        if '.' in stmt.name:
            obj_name, method_name = stmt.name.split('.', 1)
            obj = self.symbol_table.get_cell(obj_name).get()
            if not isinstance(obj, PseudocodeObject):
                raise InterpreterError(f"{obj_name} is not an object")
            if method_name not in obj.methods:
                raise InterpreterError(f"Method '{method_name}' not found on {obj.class_name}")
            self._call_method(obj, obj.methods[method_name], stmt.arguments)
            return
        
        if stmt.name not in self.procedures:
            raise InterpreterError(f"Undefined procedure: {stmt.name}")
        
        proc_decl = self.procedures[stmt.name]
        self._call_callable(proc_decl, stmt.arguments)

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        val = self.evaluate(stmt.value) if stmt.value is not None else None
        raise ReturnException(val)

    def evaluate_CallExpr(self, expr: CallExpr):
        if expr.callee in self._BUILTINS:
            return self._call_builtin(expr)

        if expr.callee not in self.functions:
            raise InterpreterError(f"Undefined function: {expr.callee}")
        
        func_decl = self.functions[expr.callee]
        try:
            self._call_callable(func_decl, expr.arguments)
        except ReturnException as e:
            return e.value
        
        raise InterpreterError(f"Function {expr.callee} did not return a value")

    def _call_callable(self, decl: Union[ProcedureDecl, FunctionDecl], args: List[Expr]):
        if len(args) != len(decl.params):
            raise TypeError(f"Argument count mismatch for {decl.name}: expected {len(decl.params)}, got {len(args)}")

        # 1. Evaluate arguments (for BYVAL) or resolve references (for BYREF)
        evaluated_args = []
        for i, param in enumerate(decl.params):
            arg_expr = args[i]
            if param.mode == "BYREF":
                if not isinstance(arg_expr, (VariableExpr, ArrayAccessExpr)):
                    raise TypeError(f"BYREF argument for {param.name} must be a variable")
                
                # Resolve the cell of the argument
                if isinstance(arg_expr, VariableExpr):
                    cell = self.symbol_table.get_cell(arg_expr.name)
                else: # ArrayAccess
                    indices = [self.evaluate(idx) for idx in arg_expr.indices]
                    cell = self.symbol_table.array_access(arg_expr.array, indices)
                
                evaluated_args.append({'mode': 'BYREF', 'cell': cell, 'name': param.name, 'type': param.type_name})
            else:
                # BYVAL
                val = self.evaluate(arg_expr)
                evaluated_args.append({'mode': 'BYVAL', 'value': val, 'name': param.name, 'type': param.type_name})

        # 2. Enter new scope
        self.symbol_table.enter_scope()
        
        try:
            # 3. Bind parameters
            for arg in evaluated_args:
                dtype = self.symbol_table.resolve_type(arg['type'])
                if arg['mode'] == 'BYREF':
                    self.symbol_table.declare_parameter(arg['name'], dtype, "BYREF", caller_cell=arg['cell'])
                else:
                    # BYVAL: Create new cell with value
                    # Hack: declare_parameter expecting caller_cell for BYREF, but for BYVAL it creates valid COPY.
                    # But we have raw value here. We need to create a temporary cell or modify logic.
                    # SymbolTable.declare_parameter logic: if BYVAL and caller_cell is None -> init default.
                    # But we WANT to init with value.
                    # Let's declare normally and assign? No, declare_parameter is for params.
                    # Let's modify SymbolTable to accept initial value for BYVAL? 
                    # OR just create a temp cell here.
                    temp_cell = Cell(arg['value'], dtype)
                    self.symbol_table.declare_parameter(arg['name'], dtype, "BYVAL", caller_cell=temp_cell)

            # 4. Execute body
            for s in decl.body:
                self.execute(s)
                
        finally:
            # 5. Exit scope
            self.symbol_table.exit_scope()

    def visit_TypeDecl(self, stmt: TypeDecl):
        if stmt.enum_values:
            # Enumerated type: TYPE Season = (Spring, Summer, Autumn, Winter)
            # Store enum values as constants (0-indexed integers)
            self.user_types[stmt.name] = {'__enum__': True, 'values': stmt.enum_values}
            for i, val_name in enumerate(stmt.enum_values):
                self.symbol_table.declare(val_name, DataType.INTEGER, is_constant=True, constant_value=i)
        else:
            self.user_types[stmt.name] = stmt.fields

    def visit_ClassDecl(self, stmt):
        """Store class definition for later instantiation."""
        class_def = {
            'name': stmt.name,
            'parent': stmt.parent,
            'members': stmt.members,
            'methods': {},
            'fields': []
        }
        for member in stmt.members:
            if isinstance(member, (ProcedureDecl, FunctionDecl)):
                class_def['methods'][member.name] = member
            elif isinstance(member, DeclareStmt):
                class_def['fields'].append(member)
        self.classes[stmt.name] = class_def

    def evaluate_NewExpr(self, expr):
        """Create a new object: obj ← NEW ClassName(args)"""
        class_name = expr.class_name
        if class_name not in self.classes:
            raise InterpreterError(f"Undefined class: {class_name}")
        
        class_def = self.classes[class_name]
        
        # Create object attributes from field declarations (including inherited)
        attributes = {}
        self._init_class_fields(class_def, attributes)
        
        # Collect all methods (including inherited)
        methods = {}
        self._collect_methods(class_def, methods)
        
        obj = PseudocodeObject(class_name, attributes, methods, class_def.get('parent'))
        
        # Call constructor (NEW) if it exists
        if 'NEW' in methods:
            constructor = methods['NEW']
            old_obj = self.current_object
            self.current_object = obj
            try:
                self._call_method(obj, constructor, expr.arguments)
            finally:
                self.current_object = old_obj
        
        return obj

    def _init_class_fields(self, class_def, attributes):
        """Initialize fields for a class, including inherited fields."""
        if class_def.get('parent') and class_def['parent'] in self.classes:
            parent_def = self.classes[class_def['parent']]
            self._init_class_fields(parent_def, attributes)
        
        for field_decl in class_def.get('fields', []):
            dtype = self.symbol_table.resolve_type(field_decl.type_name)
            default = Cell(None, dtype)._default_value(dtype)
            attributes[field_decl.name] = Cell(default, dtype)

    def _collect_methods(self, class_def, methods):
        """Collect methods, with child overriding parent."""
        if class_def.get('parent') and class_def['parent'] in self.classes:
            parent_def = self.classes[class_def['parent']]
            self._collect_methods(parent_def, methods)
        
        for name, method in class_def.get('methods', {}).items():
            methods[name] = method

    def _call_method(self, obj, method_decl, arg_exprs):
        """Call a method on an object, binding 'self' attributes to scope."""
        args = arg_exprs
        if len(args) != len(method_decl.params):
            raise TypeError(f"Argument count mismatch for {method_decl.name}")
        
        self.symbol_table.enter_scope()
        old_obj = self.current_object
        self.current_object = obj
        
        try:
            # Bind object attributes into scope
            for attr_name, attr_cell in obj.attributes.items():
                sym = SymbolInfo(
                    name=attr_name, cell=attr_cell,
                    scope_level=self.symbol_table.scope_level,
                    is_parameter=False
                )
                self.symbol_table.scopes[self.symbol_table.scope_level][attr_name] = sym
            
            # Bind parameters
            for i, param in enumerate(method_decl.params):
                val = self.evaluate(args[i])
                dtype = self.symbol_table.resolve_type(param.type_name)
                temp_cell = Cell(val, dtype)
                self.symbol_table.declare_parameter(param.name, dtype, param.mode, caller_cell=temp_cell)
            
            # Execute method body
            for s in method_decl.body:
                self.execute(s)
        except ReturnException as e:
            return e.value
        finally:
            self.symbol_table.exit_scope()
            self.current_object = old_obj
        return None

    def evaluate_SuperExpr(self, expr):
        """SUPER.Method(args) — call parent class method."""
        if self.current_object is None:
            raise InterpreterError("SUPER used outside of class context")
        
        obj = self.current_object
        parent_name = obj.parent_class
        if not parent_name or parent_name not in self.classes:
            raise InterpreterError(f"No parent class for SUPER call")
        
        parent_def = self.classes[parent_name]
        method_name = expr.method
        
        # Find method in parent class hierarchy
        if method_name not in parent_def.get('methods', {}):
            raise InterpreterError(f"Method {method_name} not found in parent class {parent_name}")
        
        method = parent_def['methods'][method_name]
        return self._call_method(obj, method, expr.arguments)

    def visit_FileStmt(self, stmt: FileStmt):
        filename = str(self.evaluate(stmt.filename))

        if stmt.operation == "OPEN":
            mode_map = {"READ": "r", "WRITE": "w", "APPEND": "a"}
            try:
                if stmt.mode == "RANDOM":
                    if not os.path.exists(filename):
                        open(filename, 'wb').close()
                    self.open_files[filename] = open(filename, "r+b")
                    self.open_files[filename + "__mode"] = "RANDOM"
                else:
                    self.open_files[filename] = open(filename, mode_map.get(stmt.mode, "r"))
            except IOError as e:
                raise InterpreterError(f"Failed to open file {filename}: {e}")

        elif stmt.operation == "CLOSE":
            if filename in self.open_files:
                self.open_files[filename].close()
                del self.open_files[filename]
        
        elif stmt.operation == "WRITE":
            if filename not in self.open_files:
                raise InterpreterError(f"File {filename} is not open")
            self.open_files[filename].write(str(self.evaluate(stmt.data)) + "\n")
            
        elif stmt.operation == "READ":
            if filename not in self.open_files:
                raise InterpreterError(f"File {filename} is not open")
            line = self.open_files[filename].readline()
            self.symbol_table.assign(stmt.variable, line.strip() if line else "", DataType.STRING)

        elif stmt.operation == "SEEK":
            if filename not in self.open_files:
                raise InterpreterError(f"File {filename} is not open")
            self.open_files[filename + "__seek"] = int(self.evaluate(stmt.data))

        elif stmt.operation == "GETRECORD":
            if filename not in self.open_files:
                raise InterpreterError(f"File {filename} is not open")
            try:
                record_data = pickle.load(self.open_files[filename])
            except Exception:
                record_data = None
            if record_data is not None:
                self.symbol_table.assign(stmt.variable, record_data, self._infer_type(record_data))

        elif stmt.operation == "PUTRECORD":
            if filename not in self.open_files:
                raise InterpreterError(f"File {filename} is not open")
            pickle.dump(self.symbol_table.get_cell(stmt.variable).get(), self.open_files[filename])

    def _call_builtin(self, expr: CallExpr):
        name = expr.callee
        args = [self.evaluate(a) for a in expr.arguments]
        
        if name == 'LENGTH': return len(str(args[0]))
        if name == 'UCASE': return str(args[0]).upper()
        if name == 'LCASE': return str(args[0]).lower()
        if name == 'LEFT': return str(args[0])[:int(args[1])]
        if name == 'RIGHT':
            n = int(args[1])
            return str(args[0])[-n:] if n > 0 else ""
        if name == 'MID':
            s, start, length = str(args[0]), int(args[1]) - 1, int(args[2])
            return s[start:start + length]
        if name == 'INT': return int(float(args[0]))
        if name == 'NUM_TO_STR': return str(args[0])
        if name == 'STR_TO_NUM':
            s = str(args[0])
            return float(s) if '.' in s else int(s)
        if name == 'ASC': return ord(str(args[0])[0])
        if name == 'CHR': return chr(int(args[0]))
        if name == 'SQRT': return math.sqrt(float(args[0]))
        if name == 'RAND': return random.random() * float(args[0])
        if name == 'EOF':
            filename = str(args[0])
            if filename not in self.open_files:
                return True
            f = self.open_files[filename]
            cur = f.tell()
            f.seek(0, os.SEEK_END)
            end = f.tell()
            f.seek(cur, os.SEEK_SET)
            return cur >= end
        return None

    def evaluate_LiteralExpr(self, expr: LiteralExpr):
        return expr.value

    def evaluate_VariableExpr(self, expr: VariableExpr):
        return self.symbol_table.get_cell(expr.name).get()

    def evaluate_ArrayAccessExpr(self, expr: ArrayAccessExpr):
        indices = [self.evaluate(idx) for idx in expr.indices]
        # Verify all are integers
        for i, val in enumerate(indices):
             if not isinstance(val, int):
                 raise TypeError(f"Array index {i+1} must be an integer, got {type(val)}")
        return self.symbol_table.array_access(expr.array, indices).get()

    def evaluate_MemberExpr(self, expr: MemberExpr):
        # Evaluate base expression
        record_val = self.evaluate(expr.record)
        
        # Handle PseudocodeObject attribute access
        if isinstance(record_val, PseudocodeObject):
            if expr.field in record_val.attributes:
                cell = record_val.attributes[expr.field]
                return cell.get() if isinstance(cell, Cell) else cell
            raise InterpreterError(f"Attribute '{expr.field}' not found on object of class {record_val.class_name}")
        
        if not isinstance(record_val, dict):
            raise InterpreterError(f"Accessing field '{expr.field}' on non-record type {type(record_val)}")
        
        if expr.field not in record_val:
            raise InterpreterError(f"Field '{expr.field}' not found in record")
        
        cell = record_val[expr.field]
        if isinstance(cell, Cell):
            return cell.get()
        return cell

    def evaluate_BinaryExpr(self, expr: BinaryExpr):
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)
        op = expr.operator
        
        if op in ('+', '-', '*', '&', '=', '<>', '<', '>', '<=', '>=', 'AND', 'OR'):
            return self._SIMPLE_OPS[op](left, right)
        
        if op in ('/', 'DIV', 'MOD'):
            if right == 0:
                raise InterpreterError(f"Division by zero ({op})")
            if op == '/': return left / right
            if op == 'DIV': return int(left / right)
            return left - int(left / right) * right
        
        raise InterpreterError(f"Unknown operator {op}")
    
    _SIMPLE_OPS = {
        '+': lambda l, r: l + r, '-': lambda l, r: l - r,
        '*': lambda l, r: l * r, '&': lambda l, r: str(l) + str(r),
        '=': lambda l, r: l == r, '<>': lambda l, r: l != r,
        '<': lambda l, r: l < r, '>': lambda l, r: l > r,
        '<=': lambda l, r: l <= r, '>=': lambda l, r: l >= r,
        'AND': lambda l, r: bool(l and r), 'OR': lambda l, r: bool(l or r),
    }

    def evaluate_UnaryExpr(self, expr: UnaryExpr):
        val = self.evaluate(expr.operand)
        if expr.operator == '-': return -val
        if expr.operator == 'NOT': return not val
        raise InterpreterError(f"Unknown unary operator {expr.operator}")

    def evaluate_MethodCallExpr(self, expr):
        """Evaluate obj.Method(args) — method call on object."""
        obj = self.evaluate(expr.object_expr)
        if not isinstance(obj, PseudocodeObject):
            raise InterpreterError(f"Cannot call method on non-object")
        
        method_name = expr.method_name
        if method_name not in obj.methods:
            raise InterpreterError(f"Method '{method_name}' not found on object of class {obj.class_name}")
        
        method = obj.methods[method_name]
        return self._call_method(obj, method, expr.arguments)

    def _infer_type(self, val: Any) -> DataType:
        if isinstance(val, bool): return DataType.BOOLEAN
        if isinstance(val, int): return DataType.INTEGER
        if isinstance(val, float): return DataType.REAL
        if isinstance(val, str): return DataType.STRING
        return DataType.UNKNOWN


# ──────────────── Dry-Run / Trace Mode ────────────────

class DryRunInterpreter(Interpreter):
    """
    Cambridge 9618 A-Level exam-style dry-run interpreter.
    - Accepts pre-supplied input values (as given on an exam paper)
    - Records a trace table of selected variables at each executed step
    - Feeds inputs from a queue instead of prompting interactively
    """

    def __init__(self, symbol_table: SymbolTable, input_queue=None,
                 traced_vars=None, max_steps=5000):
        super().__init__(symbol_table)
        self.trace = []
        self.step_count = 0
        self.max_steps = max_steps
        self.input_queue = list(input_queue) if input_queue else []
        self.input_index = 0
        self.traced_vars = set(traced_vars) if traced_vars else None
        self.output_log = []

    # ── AST scanning (static, no execution) ──

    @staticmethod
    def scan_inputs(statements):
        """Walk AST to find all INPUT statements.
        Returns list of dicts: {'line': int, 'variable': str}
        """
        results = []
        DryRunInterpreter._walk_for_inputs(statements, results)
        return results

    @staticmethod
    def _walk_for_inputs(stmts, results):
        for stmt in stmts:
            if isinstance(stmt, InputStmt):
                if isinstance(stmt.target, VariableExpr):
                    var = stmt.target.name
                elif isinstance(stmt.target, ArrayAccessExpr):
                    var = stmt.target.array + "[...]"
                elif isinstance(stmt.target, str):
                    var = stmt.target
                else:
                    var = "?"
                results.append({'line': getattr(stmt, 'line', 0), 'variable': var})
            # Recurse into compound statements
            if isinstance(stmt, IfStmt):
                DryRunInterpreter._walk_for_inputs(stmt.then_branch, results)
                if stmt.else_branch:
                    DryRunInterpreter._walk_for_inputs(stmt.else_branch, results)
            elif isinstance(stmt, WhileStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, RepeatStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, ForStmt):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, CaseStmt):
                for branch in stmt.branches:
                    DryRunInterpreter._walk_for_inputs(branch.statements, results)
                if stmt.otherwise_branch:
                    DryRunInterpreter._walk_for_inputs(stmt.otherwise_branch, results)
            elif isinstance(stmt, (ProcedureDecl, FunctionDecl)):
                DryRunInterpreter._walk_for_inputs(stmt.body, results)
            elif isinstance(stmt, ClassDecl):
                for member in stmt.members:
                    if isinstance(member, (ProcedureDecl, FunctionDecl)):
                        DryRunInterpreter._walk_for_inputs(member.body, results)

    @staticmethod
    def scan_declares(statements):
        """Walk AST to find all DECLARE / CONSTANT statements.
        Returns list of dicts: {'name': str, 'type': str, 'is_array': bool}
        """
        results = []
        for stmt in statements:
            if isinstance(stmt, DeclareStmt):
                results.append({
                    'name': stmt.name,
                    'type': stmt.type_name,
                    'is_array': stmt.is_array,
                })
            elif isinstance(stmt, ConstantDecl):
                results.append({
                    'name': stmt.name,
                    'type': 'CONSTANT',
                    'is_array': False,
                })
        return results

    # ── Snapshot & trace recording ──

    def _snapshot_vars(self):
        """Capture current values of traced variables."""
        snapshot = {}
        for scope_level in range(self.symbol_table.scope_level + 1):
            scope = self.symbol_table.scopes[scope_level]
            for name, sym in scope.items():
                if self.traced_vars is not None and name not in self.traced_vars:
                    continue
                cell = sym.cell
                if cell.is_array:
                    if cell.array_elements:
                        arr = {}
                        for key, ec in sorted(cell.array_elements.items()):
                            idx = ",".join(str(k) for k in key)
                            arr[f"[{idx}]"] = ec.get()
                        snapshot[name] = arr
                    else:
                        snapshot[name] = ""
                elif cell.type == DataType.RECORD and isinstance(cell.value, dict):
                    rec = {}
                    for fn, fc in cell.value.items():
                        rec[fn] = fc.get() if isinstance(fc, Cell) else fc
                    snapshot[name] = rec
                elif isinstance(cell.value, PseudocodeObject):
                    attrs = {}
                    for an, ac in cell.value.attributes.items():
                        attrs[an] = ac.get() if isinstance(ac, Cell) else ac
                    snapshot[name] = f"<{cell.value.class_name}>{attrs}"
                else:
                    try:
                        snapshot[name] = cell.get()
                    except Exception:
                        snapshot[name] = cell.value
        return snapshot

    def _fmt(self, val):
        """Format a value for trace display."""
        if isinstance(val, bool):
            return "TRUE" if val else "FALSE"
        if isinstance(val, str):
            return f'"{val}"'
        return str(val)

    def _describe(self, stmt):
        """Short human-readable description of a statement."""
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            if isinstance(t, str):
                return f"{t} \u2190 ..."
            if isinstance(t, ArrayAccessExpr):
                return f"{t.array}[...] \u2190 ..."
            if isinstance(t, MemberExpr):
                return f".{t.field} \u2190 ..."
            return "ASSIGN"
        if isinstance(stmt, DeclareStmt):
            arr = "ARRAY " if stmt.is_array else ""
            return f"DECLARE {stmt.name} : {arr}{stmt.type_name}"
        if isinstance(stmt, ConstantDecl):
            return f"CONSTANT {stmt.name}"
        if isinstance(stmt, OutputStmt):
            return "OUTPUT"
        if isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr):
                return f"INPUT {stmt.target.name}"
            if isinstance(stmt.target, ArrayAccessExpr):
                return f"INPUT {stmt.target.array}[...]"
            return "INPUT"
        if isinstance(stmt, IfStmt):
            return "IF"
        if isinstance(stmt, WhileStmt):
            return "WHILE"
        if isinstance(stmt, RepeatStmt):
            return "REPEAT"
        if isinstance(stmt, ForStmt):
            return f"FOR {stmt.identifier}"
        if isinstance(stmt, CaseStmt):
            return "CASE OF"
        if isinstance(stmt, ProcedureCallStmt):
            return f"CALL {stmt.name}"
        if isinstance(stmt, ReturnStmt):
            return "RETURN"
        if isinstance(stmt, ProcedureDecl):
            return f"PROCEDURE {stmt.name}"
        if isinstance(stmt, FunctionDecl):
            return f"FUNCTION {stmt.name}"
        if isinstance(stmt, TypeDecl):
            return f"TYPE {stmt.name}"
        if isinstance(stmt, ClassDecl):
            return f"CLASS {stmt.name}"
        if isinstance(stmt, FileStmt):
            return f"{stmt.operation}FILE"
        return type(stmt).__name__

    def _record(self, stmt, note=""):
        """Record one trace table row."""
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise InterpreterError(
                f"Dry-run stopped after {self.max_steps} steps "
                f"(possible infinite loop)")
        self.trace.append({
            'step': self.step_count,
            'line': getattr(stmt, 'line', self.current_line),
            'statement': self._describe(stmt),
            'note': note,
            'variables': self._snapshot_vars(),
        })

    # ── Override execute to record trace ──

    def execute(self, stmt: Stmt):
        if hasattr(stmt, 'line') and stmt.line > 0:
            self.current_line = stmt.line

        # Definitions: store but don't trace body execution
        if isinstance(stmt, (ProcedureDecl, FunctionDecl, TypeDecl, ClassDecl)):
            self._record(stmt, "defined")
            return super().execute(stmt)

        # Expression-statements (e.g. SUPER.Method())
        if isinstance(stmt, Expr):
            self._record(stmt, "expr")
            return super().execute(stmt)

        # Execute the statement
        result = super().execute(stmt)

        # Build note about what changed
        note = ""
        if isinstance(stmt, AssignStmt):
            t = stmt.target
            try:
                if isinstance(t, str):
                    v = self.symbol_table.get_cell(t).get()
                    note = f"= {self._fmt(v)}"
                elif isinstance(t, ArrayAccessExpr):
                    indices = [self.evaluate(idx) for idx in t.indices]
                    v = self.symbol_table.array_access(t.array, indices).get()
                    idx_s = ",".join(str(i) for i in indices)
                    note = f"[{idx_s}] = {self._fmt(v)}"
            except Exception:
                pass
        elif isinstance(stmt, DeclareStmt):
            note = stmt.type_name
        elif isinstance(stmt, OutputStmt):
            vals = [self._format_output(self.evaluate(a)) for a in stmt.values]
            out = "".join(vals)
            self.output_log.append(out)
            note = f"OUT: {out}"
        elif isinstance(stmt, InputStmt):
            if isinstance(stmt.target, VariableExpr):
                try:
                    v = self.symbol_table.get_cell(stmt.target.name).get()
                    note = f"= {self._fmt(v)}"
                except Exception:
                    pass
        elif isinstance(stmt, IfStmt):
            note = "condition"
        elif isinstance(stmt, ForStmt):
            note = f"loop {stmt.identifier}"

        self._record(stmt, note)
        return result

    # ── Override INPUT to use pre-supplied queue ──

    def visit_InputStmt(self, stmt: InputStmt):
        if self.input_index < len(self.input_queue):
            val_str = str(self.input_queue[self.input_index])
            self.input_index += 1
        else:
            raise InterpreterError(
                f"Dry-run ran out of pre-supplied input values. "
                f"Needed input #{self.input_index + 1} but only "
                f"{len(self.input_queue)} value(s) were provided.")

        target_type = None
        try:
            if isinstance(stmt.target, VariableExpr):
                target_type = self.symbol_table.get_cell(stmt.target.name).type
            elif isinstance(stmt.target, str):
                target_type = self.symbol_table.get_cell(stmt.target).type
            elif isinstance(stmt.target, ArrayAccessExpr):
                cell = self.symbol_table.get_cell(stmt.target.array)
                if cell.is_array and cell.array_bounds:
                    target_type = cell.array_bounds.element_type
        except Exception:
            pass

        if target_type is not None:
            val, val_type = self._coerce_input(val_str, target_type)
        else:
            val, val_type = self._auto_parse_input(val_str)

        if isinstance(stmt.target, VariableExpr):
            self.symbol_table.assign(stmt.target.name, val, val_type)
        elif isinstance(stmt.target, ArrayAccessExpr):
            indices = [self.evaluate(idx) for idx in stmt.target.indices]
            self.symbol_table.array_assign(stmt.target.array, indices, val, val_type)
        elif isinstance(stmt.target, str):
            self.symbol_table.assign(stmt.target, val, val_type)

    # ── Results ──

    def get_all_var_names(self):
        """Sorted list of all variable names that appeared in the trace."""
        names = set()
        for entry in self.trace:
            names.update(entry['variables'].keys())
        names -= set(self.procedures.keys())
        names -= set(self.functions.keys())
        return sorted(names)

    def format_trace_text(self):
        """Format the trace as an ASCII table string."""
        if not self.trace:
            return "No trace data recorded."

        var_names = self.get_all_var_names()
        headers = ['Step', 'Line', 'Statement', 'Note'] + var_names
        rows = []
        for entry in self.trace:
            row = [str(entry['step']), str(entry['line']),
                   entry['statement'], entry['note']]
            for vn in var_names:
                val = entry['variables'].get(vn, '')
                if isinstance(val, dict):
                    val = str(val)
                elif val == '':
                    val = ''
                else:
                    val = self._fmt(val) if not isinstance(val, str) else val
                row.append(val)
            rows.append(row)

        col_w = [len(h) for h in headers]
        for row in rows:
            for i, c in enumerate(row):
                col_w[i] = max(col_w[i], len(str(c)))
        col_w = [min(w, 30) for w in col_w]

        def pad(s, w):
            return str(s)[:w].ljust(w)

        sep = '+' + '+'.join('-' * (w + 2) for w in col_w) + '+'
        hdr = '|' + '|'.join(f" {pad(h, w)} " for h, w in zip(headers, col_w)) + '|'
        lines = [sep, hdr, sep]
        for row in rows:
            r = '|' + '|'.join(f" {pad(c, w)} " for c, w in zip(row, col_w)) + '|'
            lines.append(r)
        lines.append(sep)
        return '\n'.join(lines)
