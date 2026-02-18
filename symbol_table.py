from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum, auto
from copy import deepcopy

class DataType(Enum):
    """9618 primitive and composite types."""
    INTEGER = auto()
    REAL = auto()
    STRING = auto()
    BOOLEAN = auto()
    CHAR = auto()
    DATE = auto()
    ARRAY = auto()
    RECORD = auto()      # For TYPE definitions
    UNKNOWN = auto()

@dataclass
class ArrayBounds:
    """
    9618 array bounds specification.
    Supports multiple dimensions.
    """
    dims: List[tuple]    # List of (lower, upper) tuples
    element_type: DataType

@dataclass 
class Cell:
    """Mutable container for all 9618 values."""
    value: Any
    type: DataType
    is_constant: bool = False
    is_array: bool = False
    array_bounds: Optional[ArrayBounds] = None
    array_elements: Optional[Dict[tuple, 'Cell']] = None
    
    # Class-level default values for each type
    _DEFAULTS = {
        DataType.INTEGER: 0, DataType.REAL: 0.0,
        DataType.STRING: "", DataType.CHAR: "",
        DataType.BOOLEAN: False,
    }
    
    def __post_init__(self):
        if self.is_array and self.array_elements is None:
            self.array_elements = {}
    
    def get(self) -> Any:
        if self.is_array:
            raise TypeError("Cannot get array value directly; use index access")
        return self.value
    
    def set(self, new_value: Any, new_type: Optional[DataType] = None):
        if self.is_constant:
            raise ValueError(f"Cannot modify constant value (currently {self.value!r})")
        
        # Type compatibility checking
        if new_type is not None and new_type != self.type:
            # Allow INTEGER to REAL promotion (9618 rule)
            if not (self.type == DataType.REAL and new_type == DataType.INTEGER):
                raise TypeError(
                    f"Type mismatch: cannot assign {new_type.name} to {self.type.name}"
                )
            # If promoting INTEGER to REAL, ensure value is float
            if self.type == DataType.REAL and new_type == DataType.INTEGER:
                new_value = float(new_value)
        
        self.value = new_value
    
    def get_array_element(self, indices: List[int]) -> 'Cell':
        """Multi-dimensional 1-based array access."""
        if not self.is_array:
            raise TypeError("Not an array type")
        
        bounds = self.array_bounds
        if len(indices) != len(bounds.dims):
            raise IndexError(f"Incorrect number of indices: expected {len(bounds.dims)}, got {len(indices)}")

        for i, (idx, (lower, upper)) in enumerate(zip(indices, bounds.dims)):
            if idx < lower or idx > upper:
                raise IndexError(
                    f"Array index {idx} out of bounds [{lower}:{upper}] for dimension {i+1}"
                )
        
        # Convert list of indices to tuple for dict key
        key = tuple(indices)
        
        # Lazy initialization
        if key not in self.array_elements:
            default = self._default_value(bounds.element_type)
            self.array_elements[key] = Cell(default, bounds.element_type)
        
        return self.array_elements[key]
    
    def set_array_element(self, indices: List[int], value: Any, val_type: DataType):
        """Multi-dimensional 1-based array assignment."""
        if not self.is_array:
            raise TypeError("Not an array type")
        
        bounds = self.array_bounds
        if len(indices) != len(bounds.dims):
            raise IndexError(f"Incorrect number of indices: expected {len(bounds.dims)}, got {len(indices)}")

        for i, (idx, (lower, upper)) in enumerate(zip(indices, bounds.dims)):
            if idx < lower or idx > upper:
                raise IndexError(
                    f"Array index {idx} out of bounds [{lower}:{upper}] for dimension {i+1}"
                )
        
        # Element type compatibility
        if val_type != bounds.element_type:
            if not (bounds.element_type == DataType.REAL and val_type == DataType.INTEGER):
                raise TypeError(
                    f"Array element type mismatch: expected {bounds.element_type.name}, "
                    f"got {val_type.name}"
                )
            if bounds.element_type == DataType.REAL and val_type == DataType.INTEGER:
                value = float(value)
        
        key = tuple(indices)
        
        if key not in self.array_elements:
            self.array_elements[key] = Cell(None, bounds.element_type)
        
        self.array_elements[key].value = value

    def _default_value(self, dtype: DataType) -> Any:
        return self._DEFAULTS.get(dtype)
    
    def copy_value(self) -> 'Cell':
        if self.is_array:
            # Deep copy array structure
            new_elements = {}
            if self.array_elements:
                for key, elem_cell in self.array_elements.items():
                    new_elements[key] = Cell(
                        deepcopy(elem_cell.value),
                        elem_cell.type,
                        is_constant=False
                    )
            
            return Cell(
                value=None,
                type=DataType.ARRAY,
                is_constant=False,
                is_array=True,
                array_bounds=self.array_bounds,  # Shared bounds definition
                array_elements=new_elements
            )
        
        return Cell(deepcopy(self.value), self.type, is_constant=False)

@dataclass
class SymbolInfo:
    """
    Symbol table entry linking name to Cell with metadata.
    """
    name: str
    cell: Cell                          # Reference to mutable value container
    scope_level: int                    # Nesting depth of declaration
    is_parameter: bool = False          # True for procedure/function parameters
    param_mode: Optional[str] = None    # 'BYREF' or 'BYVAL' if parameter
    declared_line: int = 0              # Source location for debugging

class SymbolTable:
    """Hierarchical symbol table implementing 9618 semantics."""
    
    # Class-level built-in type resolution map
    _BUILTIN_TYPES = {
        'INTEGER': DataType.INTEGER, 'REAL': DataType.REAL,
        'STRING': DataType.STRING, 'BOOLEAN': DataType.BOOLEAN,
        'CHAR': DataType.CHAR, 'DATE': DataType.DATE,
    }
    
    def __init__(self):
        # Stack of scopes: global at [0], nested procedures push new dicts
        self.scopes: List[Dict[str, SymbolInfo]] = [{}]
        self.scope_level: int = 0
        self.type_aliases: Dict[str, DataType] = {}  # TYPE definitions
    
    def enter_scope(self):
        """Enter new scope for procedure/function body."""
        self.scope_level += 1
        self.scopes.append({})
    
    def exit_scope(self):
        """Exit current scope, cleaning up local variables."""
        if self.scope_level == 0:
            raise RuntimeError("Cannot exit global scope")
        self.scopes.pop()
        self.scope_level -= 1
    

    def declare(self, name: str, dtype: DataType,
                is_constant: bool = False,
                constant_value: Any = None,
                is_array: bool = False,
                array_bounds: Optional[ArrayBounds] = None,
                initial_value: Any = None,
                line: int = 0) -> Cell:
        current_scope = self.scopes[self.scope_level]
        
        if name in current_scope:
            raise NameError(
                f"Variable '{name}' already declared in current scope at line {line}"
            )
        
        # Create appropriate Cell based on declaration kind
        if is_constant:
            if constant_value is None:
                raise ValueError("Constant declaration requires initial value")
            cell = Cell(constant_value, dtype, is_constant=True)
        
        elif is_array:
            cell = Cell(
                value=None, type=DataType.ARRAY,
                is_array=True, array_bounds=array_bounds
            )
        
        else:  # Scalar variable (or Record)
            if initial_value is not None:
                initial = initial_value
            else:
                initial = Cell(None, dtype)._default_value(dtype)
            cell = Cell(initial, dtype)
        
        # Record symbol information
        sym = SymbolInfo(
            name=name,
            cell=cell,
            scope_level=self.scope_level,
            declared_line=line
        )
        current_scope[name] = sym
        return cell
    
    def lookup(self, name: str) -> Optional[SymbolInfo]:
        """Search for symbol from innermost to outermost scope."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    
    def get_cell(self, name: str) -> Cell:
        """Retrieve variable's Cell for access or modification."""
        sym = self.lookup(name)
        if sym is None:
            raise NameError(f"Variable '{name}' not declared")
        return sym.cell
    
    def declare_parameter(self, name: str, dtype: DataType,
                         param_mode: str,
                         caller_cell: Optional[Cell] = None,
                         initial_value: Any = None,
                         line: int = 0) -> Cell:
        """Parameter declaration with BYREF (shared Cell) or BYVAL (copied Cell)."""
        current_scope = self.scopes[self.scope_level]
        
        if name in current_scope:
            raise NameError(f"Parameter '{name}' already declared")
        
        if param_mode == 'BYREF':
            if caller_cell is None:
                raise ValueError("BYREF parameter requires caller's Cell")
            cell = caller_cell
        
        else:  # BYVAL
            if caller_cell is not None:
                cell = caller_cell.copy_value()
            elif initial_value is not None:
                cell = Cell(initial_value, dtype)
            else:
                initial = Cell(None, dtype)._default_value(dtype)
                cell = Cell(initial, dtype)
        
        sym = SymbolInfo(
            name=name,
            cell=cell,
            scope_level=self.scope_level,
            is_parameter=True,
            param_mode=param_mode,
            declared_line=line
        )
        current_scope[name] = sym
        return cell
    
    def assign(self, name: str, value: Any, val_type: DataType, line: int = 0):
        """Assignment: name <- value with type checking."""
        cell = self.get_cell(name)
        cell.set(value, val_type)
    
    def array_access(self, name: str, indices) -> Cell:
        """1-based array access: arr[indices] — indices is a list of ints or a single int."""
        cell = self.get_cell(name)
        if isinstance(indices, int):
            indices = [indices]
        return cell.get_array_element(indices)
    
    def array_assign(self, name: str, indices, value: Any, val_type: DataType):
        """1-based array assignment: arr[indices] <- value"""
        cell = self.get_cell(name)
        if isinstance(indices, int):
            indices = [indices]
        cell.set_array_element(indices, value, val_type)

    def resolve_type(self, type_name: str) -> DataType:
        upper = type_name.upper()
        if upper in self._BUILTIN_TYPES:
            return self._BUILTIN_TYPES[upper]
        if type_name in self.type_aliases:
            return self.type_aliases[type_name]
        return DataType.UNKNOWN

    def debug_dump(self):
        print("=" * 60)
        print("SYMBOL TABLE DUMP")
        print("=" * 60)
        for level, scope in enumerate(self.scopes):
            print(f"\n--- Scope Level {level} ---")
            for name, sym in sorted(scope.items()):
                cell = sym.cell
                parts = [f"{name}: {cell.type.name}"]
                
                if cell.is_constant:
                    parts.append("(CONSTANT)")
                if sym.is_parameter:
                    parts.append(f"[param {sym.param_mode}]")
                if cell.is_array:
                    b = cell.array_bounds
                    # Display dims
                    dim_str = ", ".join([f"{l}:{u}" for l, u in b.dims])
                    parts.append(f"ARRAY[{dim_str}] OF {b.element_type.name}")
                    parts.append(f"elements={len(cell.array_elements or {})}")
                else:
                    parts.append(f"value={cell.value!r}")
                
                print("  " + " ".join(parts))
        print("\n" + "=" * 60)
