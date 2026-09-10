import ast
import glob
import builtins

BUILTIN_NAMES = set(dir(builtins))

class ScopeVisitor(ast.NodeVisitor):
    def __init__(self, filepath):
        self.filepath = filepath
        self.scopes = [set(BUILTIN_NAMES)]
        self.missing = []

    def current_scope(self):
        return self.scopes[-1]

    def add_to_scope(self, name):
        self.scopes[-1].add(name)

    def is_known(self, name):
        return any(name in scope for scope in reversed(self.scopes))

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name.split('.')[0]
            self.add_to_scope(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self.add_to_scope(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.add_to_scope(node.name)
        func_scope = set()
        for arg in node.args.args:
            func_scope.add(arg.arg)
        if node.args.vararg:
            func_scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            func_scope.add(node.args.kwarg.arg)
        
        self.scopes.append(func_scope)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_ClassDef(self, node):
        self.add_to_scope(node.name)
        self.scopes.append(set())
        self.generic_visit(node)
        self.scopes.pop()

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                self.add_to_scope(item.optional_vars.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.add_to_scope(node.name)
        self.generic_visit(node)

    def visit_For(self, node):
        if isinstance(node.target, ast.Name):
            self.add_to_scope(node.target.id)
        elif isinstance(node.target, (ast.Tuple, ast.List)):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self.add_to_scope(elt.id)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.add_to_scope(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.add_to_scope(elt.id)
        self.generic_visit(node)

    def visit_ListComp(self, node):
        comp_scope = set()
        for gen in node.generators:
            if isinstance(gen.target, ast.Name):
                comp_scope.add(gen.target.id)
            elif isinstance(gen.target, (ast.Tuple, ast.List)):
                for elt in gen.target.elts:
                    if isinstance(elt, ast.Name):
                        comp_scope.add(elt.id)
        self.scopes.append(comp_scope)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_DictComp(self, node):
        comp_scope = set()
        for gen in node.generators:
            if isinstance(gen.target, ast.Name):
                comp_scope.add(gen.target.id)
            elif isinstance(gen.target, (ast.Tuple, ast.List)):
                for elt in gen.target.elts:
                    if isinstance(elt, ast.Name):
                        comp_scope.add(elt.id)
        self.scopes.append(comp_scope)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if not self.is_known(node.id) and not node.id.startswith("_"):
                self.missing.append((node.id, node.lineno))
        self.generic_visit(node)

def main():
    py_files = glob.glob("**/*.py", recursive=True)
    found_any = False
    for filepath in py_files:
        if "scratch" in filepath or "venv" in filepath or ".venv" in filepath:
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
        v = ScopeVisitor(filepath)
        v.visit(tree)
        if v.missing:
            found_any = True
            print(f"File: {filepath}")
            for name, line in sorted(set(v.missing), key=lambda x: x[1]):
                print(f"  Line {line}: missing import / undefined symbol '{name}'")

    if not found_any:
        print("✅ NO MISSING IMPORTS OR UNDEFINED SYMBOLS FOUND IN ANY PROJECT MODULE!")

if __name__ == "__main__":
    main()
