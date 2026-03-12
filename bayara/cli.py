from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .errors import BayaraError
from .lexer import Lexer
from .parser import Parser
from .transpiler import transpile
from .validator import validate
from .version import __version__


def load_program(input_path: str):
    source = Path(input_path).read_text(encoding='utf-8')
    tokens = Lexer(source).lex()
    program = Parser(tokens).parse()
    return program


def cmd_check(args: argparse.Namespace) -> int:
    try:
        program = load_program(args.input)
        validate(program)
        print(f'Check passed: {args.input}')
        return 0
    except BayaraError as exc:
        print(exc, file=sys.stderr)
        return 1


def cmd_compile(args: argparse.Namespace) -> int:
    try:
        program = load_program(args.input)
        output = transpile(program)
        Path(args.output).write_text(output, encoding='utf-8')
        print(f'Compiled successfully: {args.output}')
        return 0
    except BayaraError as exc:
        print(exc, file=sys.stderr)
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    try:
        program = load_program(args.input)
        output = transpile(program)
        Path(args.output).write_text(output, encoding='utf-8')
        print(f'Running Bayara script: {args.input}')
        completed = subprocess.run([sys.executable, args.output], check=False)
        return completed.returncode
    except BayaraError as exc:
        print(exc, file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='bayara', description='Bayara CLI')
    sub = parser.add_subparsers(dest='command', required=True)

    p_check = sub.add_parser('check', help='validate a .bay script')
    p_check.add_argument('input')
    p_check.set_defaults(func=cmd_check)

    p_compile = sub.add_parser('compile', help='compile a .bay script into Python')
    p_compile.add_argument('input')
    p_compile.add_argument('output', nargs='?', default='output.py')
    p_compile.set_defaults(func=cmd_compile)

    p_run = sub.add_parser('run', help='compile and run a .bay script')
    p_run.add_argument('input')
    p_run.add_argument('output', nargs='?', default='output.py')
    p_run.set_defaults(func=cmd_run)

    p_version = sub.add_parser('version', help='show Bayara version')
    p_version.set_defaults(func=lambda _args: print(__version__) or 0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
