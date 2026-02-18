import pytest
from lexer import Lexer, LexerError

def test_unexpected_char_at_start():
    source = "@DECLARE x : INTEGER"
    lexer = Lexer(source, "test.pse")
    with pytest.raises(LexerError) as excinfo:
        lexer.tokenize()
    assert "test.pse:1:1: Unexpected character: '@'" in str(excinfo.value)
    assert "\n  @DECLARE x : INTEGER\n  ^" in str(excinfo.value)

def test_unexpected_char_in_middle():
    source = "DECLARE x @ : INTEGER"
    lexer = Lexer(source, "test.pse")
    with pytest.raises(LexerError) as excinfo:
        lexer.tokenize()
    # "DECLARE x " is 10 chars, so @ is at column 11
    assert "test.pse:1:11: Unexpected character: '@'" in str(excinfo.value)
    assert "\n  DECLARE x @ : INTEGER\n            ^" in str(excinfo.value)

def test_unexpected_char_at_end():
    source = "DECLARE x : INTEGER !"
    lexer = Lexer(source, "test.pse")
    with pytest.raises(LexerError) as excinfo:
        lexer.tokenize()
    # "DECLARE x : INTEGER " is 20 chars, so ! is at column 21
    assert "test.pse:1:21: Unexpected character: '!'" in str(excinfo.value)
    assert "\n  DECLARE x : INTEGER !\n                      ^" in str(excinfo.value)

def test_unexpected_char_line_tracking():
    source = "DECLARE x : INTEGER\n\n  x <- @ 10"
    lexer = Lexer(source, "test.pse")
    with pytest.raises(LexerError) as excinfo:
        lexer.tokenize()
    assert "test.pse:3:8: Unexpected character: '@'" in str(excinfo.value)
    assert "\n    x <- @ 10\n         ^" in str(excinfo.value)

def test_unexpected_char_column_tracking():
    source = "OUTPUT \"Hello\" # \"World\""
    lexer = Lexer(source, "test.pse")
    with pytest.raises(LexerError) as excinfo:
        lexer.tokenize()
    # OUTPUT "Hello"  is 6+1+7 = 14 chars. 15th is space, 16th is #
    assert "test.pse:1:16: Unexpected character: '#'" in str(excinfo.value)
    assert "\n  OUTPUT \"Hello\" # \"World\"\n                 ^" in str(excinfo.value)

def test_error_context_formatting():
    source = "x <- 1\n  ?\ny <- 2"
    lexer = Lexer(source, "test.pse")
    with pytest.raises(LexerError) as excinfo:
        lexer.tokenize()
    assert "test.pse:2:3: Unexpected character: '?'" in str(excinfo.value)
    # The context should show only the line with the error
    # Implementation adds 2 spaces of indentation, and the source line already had 2 spaces.
    assert "\n    ?\n    ^" in str(excinfo.value)
    assert "x <- 1" not in str(excinfo.value)
    assert "y <- 2" not in str(excinfo.value)

def test_lexer_error_exception_type():
    source = "$"
    lexer = Lexer(source)
    with pytest.raises(LexerError):
        lexer.tokenize()

def test_manual_error_no_context():
    lexer = Lexer("short")
    lexer.line = 10  # Out of range
    with pytest.raises(LexerError) as excinfo:
        lexer.error("Manual error")
    # Should not have the context part (no ^ and no source line)
    assert "Manual error" in str(excinfo.value)
    assert "^" not in str(excinfo.value)
    assert "short" not in str(excinfo.value)

def test_unclosed_string_error():
    # As discovered, unclosed string currently reports unexpected character '"'
    source = '"abc'
    lexer = Lexer(source)
    with pytest.raises(LexerError) as excinfo:
        lexer.tokenize()
    assert "Unexpected character: '\"'" in str(excinfo.value)
    assert "1:1" in str(excinfo.value)

def test_unclosed_char_error():
    source = "'a"
    lexer = Lexer(source)
    with pytest.raises(LexerError) as excinfo:
        lexer.tokenize()
    assert "Unexpected character: \"'\"" in str(excinfo.value)
    assert "1:1" in str(excinfo.value)
