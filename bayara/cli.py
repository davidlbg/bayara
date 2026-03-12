from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import __version__
from .errors import BayaraError
from .parser import parse_source
from .transpiler import transpile_program


def _read_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f'file not found: {path}')
    return path.read_text(encoding='utf-8')


def cmd_compile(input_path: Path, output_path: Path) -> int:
    source = _read_file(input_path)
    program = parse_source(source)
    python_code = transpile_program(program)
    output_path.write_text(python_code, encoding='utf-8')
    print(f'compiled {input_path} -> {output_path}')
    return 0


def cmd_check(input_path: Path) -> int:
    source = _read_file(input_path)
    program = parse_source(source)
    transpile_program(program)
    print(f'{input_path}: OK')
    return 0


def cmd_run(input_path: Path, output_path: Path | None) -> int:
    generated = output_path or input_path.with_suffix('.generated.py')
    cmd_compile(input_path, generated)
    print(f'running {generated}...')
    result = subprocess.run([sys.executable, str(generated)], check=False)
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='bayara', description='Bayara 1.0 CLI')
    sub = parser.add_subparsers(dest='command', required=True)

    p_compile = sub.add_parser('compile', help='Compile a .bay file to Python')
    p_compile.add_argument('input', type=Path)
    p_compile.add_argument('output', type=Path, nargs='?', default=Path('output.py'))

    p_run = sub.add_parser('run', help='Compile and run a .bay file')
    p_run.add_argument('input', type=Path)
    p_run.add_argument('output', type=Path, nargs='?')

    p_check = sub.add_parser('check', help='Validate a .bay file')
    p_check.add_argument('input', type=Path)

    sub.add_parser('version', help='Show the Bayara version')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == 'compile':
            return cmd_compile(args.input, args.output)
        if args.command == 'run':
            return cmd_run(args.input, args.output)
        if args.command == 'check':
            return cmd_check(args.input)
        if args.command == 'version':
            print(__version__)
            return 0
        parser.print_help()
        return 1
    except BayaraError as exc:
        print(exc, file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
