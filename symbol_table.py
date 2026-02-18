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
        new_value = self._check_type_compatibility(new_value, new_type)
        self.value = new_value

    def _check_type_compatibility(self, value: Any, new_type: Optional[DataType]) -> Any:
        """Validate type assignment rules and apply INT→REAL promotion."""
        if new_type is None or new_type == self.type:
            return value
        if self._is_int_to_real_promotion(new_type):
            return float(value)
        raise TypeError(
            f"Type mismatch: cannot assign {new_type.name} to {self.type.name}"
        )

    def _is_int_to_real_promotion(self, source_type: DataType) -> bool:
        """Check if assignment is a valid INTEGER→REAL promotion."""
        return self.type == DataType.REAL and source_type == DataType.INTEGER
    
    def get_array_element(self, indices: List[int]) -> 'Cell':
        """Multi-dimensional 1-based array access."""
        self._validate_array_indices(indices)
        key = tuple(indices)
        if key not in self.array_elements:
            default = self._default_value(self.array_bounds.element_type)
            self.array_elements[key] = Cell(default, self.array_bounds.element_type)
        return self.array_elements[key]
    
    def set_array_element(self, indices: List[int], value: Any, val_type: DataType):
        """Multi-dimensional 1-based array assignment."""
        self._validate_array_indices(indices)
        value = self._coerce_element_type(value, val_type)
        key = tuple(indices)
        if key not in self.array_elements:
            self.array_elements[key] = Cell(None, self.array_bounds.element_type)
        self.array_elements[key].value = value

    def _validate_array_indices(self, indices: List[int]):
        """Validate that this is an array and indices are within bounds."""
        if not self.is_array:
            raise TypeError("Not an array type")
        bounds = self.array_bounds
        if len(indices) != len(bounds.dims):
            raise IndexError(
                f"Incorrect number of indices: expected {len(bounds.dims)}, got {len(indices)}"
            )
        for i, (idx, (lower, upper)) in enumerate(zip(indices, bounds.dims)):
            if idx < lower or idx > upper:
                raise IndexError(
                    f"Array index {idx} out of bounds [{lower}:{upper}] for dimension {i+1}"
                )

    def _coerce_element_type(self, value: Any, val_type: DataType) -> Any:
        """Check element type compatibility and apply INT→REAL promotion."""
        expected = self.array_bounds.element_type
        if val_type == expected:
            return value
        if expected == DataType.REAL and val_type == DataType.INTEGER:
            return float(value)
        raise TypeError(
            f"Array element type mismatch: expected {expected.name}, got {val_type.name}"
        )

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
class DeclarationSpec:
    """Encapsulates all parameters for a variable declaration."""
    name: str
    dtype: DataType
    is_constant: bool = False
    constant_value: Any = None
    is_array: bool = False
    array_bounds: Optional[ArrayBounds] = None
    initial_value: Any = None
    line: int = 0


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
        # Fast lookup cache: name -> stack of SymbolInfo (most recent at end)
        self._lookup_cache: Dict[str, List[SymbolInfo]] = {}

    def enter_scope(self):
        """Enter new scope for procedure/function body."""
        self.scope_level += 1
        self.scopes.append({})
    
    def exit_scope(self):
        """Exit current scope, cleaning up local variables."""
        if self.scope_level == 0:
            raise RuntimeError("Cannot exit global scope")

        # Cleanup lookup cache for variables in the scope being removed
        current_scope = self.scopes[self.scope_level]
        for name in current_scope:
            if name in self._lookup_cache:
                self._lookup_cache[name].pop()
                if not self._lookup_cache[name]:
                    del self._lookup_cache[name]

        self.scopes.pop()
        self.scope_level -= 1
    

    def declare(self, name: str, dtype: DataType,
                is_constant: bool = False,
                constant_value: Any = None,
                is_array: bool = False,
                array_bounds: Optional[ArrayBounds] = None,
                initial_value: Any = None,
                line: int = 0) -> Cell:
        """Declare a variable, constant, or array in the current scope."""
        spec = DeclarationSpec(
            name=name, dtype=dtype, is_constant=is_constant,
            constant_value=constant_value, is_array=is_array,
            array_bounds=array_bounds, initial_value=initial_value, line=line
        )
        return self._declare_from_spec(spec)

    def _declare_from_spec(self, spec: DeclarationSpec) -> Cell:
        """Internal declaration using a DeclarationSpec."""
        current_scope = self.scopes[self.scope_level]
        if spec.name in current_scope:
            raise NameError(
                f"Variable '{spec.name}' already declared in current scope at line {spec.line}"
            )
        cell = self._create_cell(spec)
        sym = SymbolInfo(
            name=spec.name, cell=cell,
            scope_level=self.scope_level, declared_line=spec.line
        )
        current_scope[spec.name] = sym

        # Update lookup cache
        if spec.name not in self._lookup_cache:
            self._lookup_cache[spec.name] = []
        self._lookup_cache[spec.name].append(sym)

        return cell

    def _create_cell(self, spec: DeclarationSpec) -> Cell:
        """Create the appropriate Cell based on declaration kind."""
        if spec.is_constant:
            return self._create_constant_cell(spec)
        if spec.is_array:
            return self._create_array_cell(spec)
        return self._create_scalar_cell(spec)

    @staticmethod
    def _create_constant_cell(spec: DeclarationSpec) -> Cell:
        """Create a Cell for a CONSTANT declaration."""
        if spec.constant_value is None:
            raise ValueError("Constant declaration requires initial value")
        return Cell(spec.constant_value, spec.dtype, is_constant=True)

    @staticmethod
    def _create_array_cell(spec: DeclarationSpec) -> Cell:
        """Create a Cell for an ARRAY declaration."""
        return Cell(
            value=None, type=DataType.ARRAY,
            is_array=True, array_bounds=spec.array_bounds
        )

    @staticmethod
    def _create_scalar_cell(spec: DeclarationSpec) -> Cell:
        """Create a Cell for a scalar variable or record."""
        if spec.initial_value is not None:
            initial = spec.initial_value
        else:
            initial = Cell(None, spec.dtype)._default_value(spec.dtype)
        return Cell(initial, spec.dtype)
    
    def lookup(self, name: str) -> Optional[SymbolInfo]:
        """Search for symbol from innermost to outermost scope."""
        # O(1) Lookup Cache
        if name in self._lookup_cache and self._lookup_cache[name]:
            return self._lookup_cache[name][-1]
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
                         line: int = 0) -> Cell:
        """Parameter declaration with BYREF (shared Cell) or BYVAL (copied Cell)."""
        current_scope = self.scopes[self.scope_level]
        if name in current_scope:
            raise NameError(f"Parameter '{name}' already declared")
        cell = self._create_param_cell(dtype, param_mode, caller_cell)
        sym = SymbolInfo(
            name=name, cell=cell,
            scope_level=self.scope_level,
            is_parameter=True, param_mode=param_mode,
            declared_line=line
        )
        current_scope[name] = sym

        # Update lookup cache
        if name not in self._lookup_cache:
            self._lookup_cache[name] = []
        self._lookup_cache[name].append(sym)

        return cell

    def inject_symbol(self, sym: SymbolInfo):
        """
        Manually inject a symbol into the current scope (e.g. for object attributes).
        Updates both the scope dictionary and the fast lookup cache.
        """
        self.scopes[self.scope_level][sym.name] = sym

        if sym.name not in self._lookup_cache:
            self._lookup_cache[sym.name] = []
        self._lookup_cache[sym.name].append(sym)

    @staticmethod
    def _create_param_cell(dtype: DataType, param_mode: str,
                           caller_cell: Optional[Cell]) -> Cell:
        """Create a Cell for a parameter based on passing mode."""
        if param_mode == 'BYREF':
            if caller_cell is None:
                raise ValueError("BYREF parameter requires caller's Cell")
            return caller_cell
        # BYVAL: copy or create fresh
        if caller_cell is not None:
            return caller_cell.copy_value()
        initial = Cell(None, dtype)._default_value(dtype)
        return Cell(initial, dtype)
    
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
