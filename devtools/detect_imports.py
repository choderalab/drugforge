import ast
import os
from typing import Dict, Set
import click
from pathlib import Path


def get_imports_from_file(file_path: str) -> Set[str]:
    """Extract import names from a Python file."""
    imports = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])

    except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
        print(f"Could not parse {file_path}: {str(e)}")

    return imports


def find_all_imports_with_files(directory: str) -> Dict[str, Set[str]]:
    """
    Find all unique imports in Python files and track which files use them.

    Parameters
    ----------
    directory : str
        Directory to search recursively

    Returns
    -------
    Dict[str, Set[str]]
        Dictionary mapping package names to sets of file paths that import them
    """
    import_to_files = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory)
                file_imports = get_imports_from_file(file_path)

                for imp in file_imports:
                    if imp not in import_to_files:
                        import_to_files[imp] = set()
                    import_to_files[imp].add(rel_path)

    # Remove standard library modules
    std_libs = set()
    for pkg in import_to_files:
        try:
            if not hasattr(__import__(pkg), '__file__'):
                std_libs.add(pkg)
        except ImportError:
            continue

    for pkg in std_libs:
        import_to_files.pop(pkg)

    return import_to_files


@click.command()
@click.argument('directory',
                type=click.Path(exists=True, file_okay=False, dir_okay=True),
                default='.')
@click.option('--output', '-o',
              type=click.Path(dir_okay=False),
              default='import_report.txt',
              help='Output file path')
def main(directory: str, output: str):
    """Find all unique package imports and their importing files within the specified directory."""
    directory = str(Path(directory).resolve())
    imports_with_files = find_all_imports_with_files(directory)

    with open(output, 'w', encoding='utf-8') as f:
        f.write("Unique third-party packages and their importing files:\n")
        for pkg in sorted(imports_with_files):
            f.write(f"\n{pkg}:\n")
            for file in sorted(imports_with_files[pkg]):
                f.write(f"  - {file}\n")

    print(f"Report written to: {output}")


if __name__ == "__main__":
    main()