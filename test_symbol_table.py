import unittest
from symbol_table import SymbolTable, DataType, ArrayBounds, Cell, DeclarationSpec

class TestSymbolTable(unittest.TestCase):
    def setUp(self):
        self.st = SymbolTable()

    def test_declare_scalar_defaults(self):
        """Test declaring a scalar variable with default value."""
        # DataType.INTEGER default is 0
        cell = self.st.declare(DeclarationSpec(name="x", dtype=DataType.INTEGER, line=1))
        self.assertEqual(cell.value, 0)
        self.assertEqual(cell.type, DataType.INTEGER)
        self.assertFalse(cell.is_constant)
        self.assertFalse(cell.is_array)

        # DataType.STRING default is ""
        cell_str = self.st.declare(DeclarationSpec(name="s", dtype=DataType.STRING, line=2))
        self.assertEqual(cell_str.value, "")
        self.assertEqual(cell_str.type, DataType.STRING)

    def test_declare_scalar_initial_value(self):
        """Test declaring a scalar variable with an initial value."""
        cell = self.st.declare(DeclarationSpec(
            name="y", dtype=DataType.INTEGER, initial_value=10, line=3
        ))
        self.assertEqual(cell.value, 10)
        self.assertEqual(cell.type, DataType.INTEGER)

    def test_declare_constant(self):
        """Test declaring a constant."""
        cell = self.st.declare(DeclarationSpec(
            name="PI", dtype=DataType.REAL		,
            is_constant=True, constant_value=3.14159, line=4
        ))
        self.assertEqual(cell.value, 3.14159)
        self.assertEqual(cell.type, DataType.REAL)
        self.assertTrue(cell.is_constant)

        # Verify it cannot be modified
        with self.assertRaises(ValueError):
            cell.set(3.14, DataType.REAL)

    def test_declare_constant_missing_value(self):
        """Test declaring a constant without a value raises ValueError."""
        with self.assertRaises(ValueError):
            self.st.declare(DeclarationSpec(
                name="CONST_NO_VAL", dtype=DataType.INTEGER,
                is_constant=True, line=5
            ))

    def test_declare_array(self):
        """Test declaring an array."""
        bounds = ArrayBounds(dims=[(1, 10)], element_type=DataType.INTEGER)
        cell = self.st.declare(DeclarationSpec(
            name="arr", dtype=DataType.ARRAY,
            is_array=True, array_bounds=bounds, line=6
        ))

        self.assertTrue(cell.is_array)
        self.assertEqual(cell.array_bounds, bounds)
        self.assertEqual(cell.type, DataType.ARRAY)
        self.assertIsNone(cell.value) # Arrays don't hold a single value

    def test_redeclaration_error(self):
        """Test redeclaring a variable in the same scope raises NameError."""
        self.st.declare(DeclarationSpec(name="z", dtype=DataType.INTEGER, line=7))
        with self.assertRaises(NameError):
            self.st.declare(DeclarationSpec(name="z", dtype=DataType.REAL, line=8))

    def test_shadowing(self):
        """Test variable shadowing in nested scopes."""
        # Declare in global scope
        self.st.declare(DeclarationSpec(
            name="shadowed", dtype=DataType.INTEGER, initial_value=1, line=9
        ))

        # Enter new scope
        self.st.enter_scope()

        # Declare same name, different type/value
        cell_inner = self.st.declare(DeclarationSpec(
            name="shadowed", dtype=DataType.STRING, initial_value="inner", line=10
        ))
        self.assertEqual(cell_inner.value, "inner")
        self.assertEqual(cell_inner.type, DataType.STRING)

        # Verify lookup finds inner variable
        found = self.st.lookup("shadowed")
        self.assertIsNotNone(found)
        self.assertEqual(found.cell, cell_inner)

        # Exit scope
        self.st.exit_scope()

        # Verify lookup finds outer variable
        found_outer = self.st.lookup("shadowed")
        self.assertIsNotNone(found_outer)
        self.assertEqual(found_outer.cell.value, 1)
        self.assertEqual(found_outer.cell.type, DataType.INTEGER)

    def test_copy_value_immutable(self):
        """Test that copy_value correctly copies immutable types."""
        # Integer
        c1 = Cell(42, DataType.INTEGER)
        c2 = c1.copy_value()
        self.assertEqual(c2.value, 42)
        self.assertEqual(c2.type, DataType.INTEGER)

        # String
        c1 = Cell("hello", DataType.STRING)
        c2 = c1.copy_value()
        self.assertEqual(c2.value, "hello")

        # Boolean
        c1 = Cell(True, DataType.BOOLEAN)
        c2 = c1.copy_value()
        self.assertEqual(c2.value, True)

    def test_copy_value_mutable_array(self):
        """Test that arrays are deep copied."""
        bounds = ArrayBounds(dims=[(1, 2)], element_type=DataType.INTEGER)
        c1 = Cell(None, DataType.ARRAY, is_array=True, array_bounds=bounds)
        c1.set_array_element([1], 10, DataType.INTEGER)

        c2 = c1.copy_value()
        self.assertEqual(c2.get_array_element([1]).value, 10)

        # Modify c2, c1 should not change
        c2.set_array_element([1], 20, DataType.INTEGER)
        self.assertEqual(c1.get_array_element([1]).value, 10)
        self.assertEqual(c2.get_array_element([1]).value, 20)

if __name__ == '__main__':
    unittest.main()
