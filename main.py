import sys
import os
from lexer import Lexer, TokenType
from parser import Parser, ParserError
from interpreter import Interpreter, InterpreterError, DryRunInterpreter
from symbol_table import SymbolTable

def run_file(filename, dry_run=False, inputs=None):
    with open(filename, 'r') as f:
        source = f.read()
    run(source, filename, dry_run, inputs)

def run(source, filename="<stdin>", dry_run=False, inputs=None):
    # Lexing
    lexer = Lexer(source, filename)
    try:
        tokens = lexer.tokenize()
    except Exception as e:
        print(f"Lexer Error: {e}")
        return

    # Parsing
    parser = Parser(tokens)
    try:
        statements = parser.parse()
    except Exception as e:
        print(f"Parser Error: {e}")
        return

    # Interpreting
    symbol_table = SymbolTable()

    if dry_run:
        input_info = DryRunInterpreter.scan_inputs(statements)
        declare_info = DryRunInterpreter.scan_declares(statements)

        if inputs is None and input_info:
            print(f"\nFound {len(input_info)} INPUT statement(s):")
            for i, info in enumerate(input_info):
                print(f"  #{i+1}: Line {info['line']} -> {info['variable']}")
            raw = input("\nEnter all input values (comma-separated): ")
            inputs = [v.strip() for v in raw.split(",") if v.strip()]

        if declare_info:
            print(f"\nVariables: {', '.join(d['name'] for d in declare_info)}")

        interpreter = DryRunInterpreter(symbol_table, input_queue=inputs or [])
        try:
            interpreter.interpret(statements)
        except InterpreterError as e:
            print(f"Runtime Error at line {interpreter.current_line}: {e}")
        except Exception as e:
            print(f"Runtime Error at line {interpreter.current_line}: {e}")

        print("\n" + "=" * 60)
        print("TRACE TABLE")
        print("=" * 60)
        print(interpreter.format_trace_text())
    else:
        interpreter = Interpreter(symbol_table)
        try:
            interpreter.interpret(statements)
        except InterpreterError as e:
            print(f"Runtime Error at line {interpreter.current_line}: {e}")
        except Exception as e:
            print(f"Runtime Error at line {interpreter.current_line}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = None
        dry_run = False
        inputs = None
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == '--dryrun':
                dry_run = True
            elif sys.argv[i] == '--inputs' and i + 1 < len(sys.argv):
                i += 1
                inputs = [v.strip() for v in sys.argv[i].split(",") if v.strip()]
            else:
                filename = sys.argv[i]
            i += 1
        if filename:
            run_file(filename, dry_run, inputs)
        else:
            print("Usage: python main.py <file.pse> [--dryrun] [--inputs \"5,3,8\"]")
    else:
        # REPL
        print("9618 Pseudocode Interpreter (Type 'exit' to quit)")
        while True:
            try:
                line = input(">>> ")
                if line.lower() == 'exit': break
                run(line)
            except EOFError:
                break
