import ast
import os
import builtins

class ScopeChecker(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.scopes = [set(dir(builtins))]
        self.undefined = []

    def push_scope(self):
        self.scopes.append(set())

    def pop_scope(self):
        self.scopes.pop()

    def add_name(self, name):
        self.scopes[-1].add(name)

    def is_defined(self, name):
        return any(name in scope for scope in reversed(self.scopes))

    def visit_Import(self, node):
        for alias in node.names:
            self.add_name(alias.asname if alias.asname else alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.add_name(alias.asname if alias.asname else alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.add_name(node.name)
        self.push_scope()
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            self.add_name(arg.arg)
        if node.args.vararg:
            self.add_name(node.args.vararg.arg)
        if node.args.kwarg:
            self.add_name(node.args.kwarg.arg)
        for stmt in node.body:
            self.visit(stmt)
        self.pop_scope()

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node):
        self.push_scope()
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            self.add_name(arg.arg)
        if node.args.vararg:
            self.add_name(node.args.vararg.arg)
        if node.args.kwarg:
            self.add_name(node.args.kwarg.arg)
        self.visit(node.body)
        self.pop_scope()

    def visit_ClassDef(self, node):
        self.add_name(node.name)
        self.push_scope()
        for stmt in node.body:
            self.visit(stmt)
        self.pop_scope()

    def visit_Assign(self, node):
        self.visit(node.value)
        for target in node.targets:
            self._add_targets(target)

    def visit_AugAssign(self, node):
        self.visit(node.value)
        self.visit(node.target)

    def visit_NamedExpr(self, node): # :=
        self.visit(node.value)
        self._add_targets(node.target)

    def visit_For(self, node):
        self.visit(node.iter)
        self._add_targets(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_ExceptHandler(self, node):
        self.push_scope()
        if node.name:
            self.add_name(node.name)
        if node.type:
            self.visit(node.type)
        for stmt in node.body:
            self.visit(stmt)
        self.pop_scope()

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars:
                self._add_targets(item.optional_vars)
            self.visit(item.context_expr)
        for stmt in node.body:
            self.visit(stmt)

    def visit_ListComp(self, node):
        self.push_scope()
        for gen in node.generators:
            self.visit(gen.iter)
            self._add_targets(gen.target)
            for if_expr in gen.ifs:
                self.visit(if_expr)
        self.visit(node.elt)
        self.pop_scope()

    def visit_SetComp(self, node):
        self.push_scope()
        for gen in node.generators:
            self.visit(gen.iter)
            self._add_targets(gen.target)
            for if_expr in gen.ifs:
                self.visit(if_expr)
        self.visit(node.elt)
        self.pop_scope()

    def visit_DictComp(self, node):
        self.push_scope()
        for gen in node.generators:
            self.visit(gen.iter)
            self._add_targets(gen.target)
            for if_expr in gen.ifs:
                self.visit(if_expr)
        self.visit(node.key)
        self.visit(node.value)
        self.pop_scope()

    def visit_GeneratorExp(self, node):
        self.push_scope()
        for gen in node.generators:
            self.visit(gen.iter)
            self._add_targets(gen.target)
            for if_expr in gen.ifs:
                self.visit(if_expr)
        self.visit(node.elt)
        self.pop_scope()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if not self.is_defined(node.id):
                self.undefined.append((node.lineno, node.id))
        elif isinstance(node.ctx, ast.Store):
            self.add_name(node.id)

    def _add_targets(self, target):
        if isinstance(target, ast.Name):
            self.add_name(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._add_targets(elt)
        elif isinstance(target, ast.Starred):
            self._add_targets(target.value)

def check_all():
    files = ['app.py'] + [os.path.join('core', f) for f in os.listdir('core') if f.endswith('.py')]
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filepath)
        checker = ScopeChecker(filepath)
        checker.visit(tree)
        if checker.undefined:
            print(f'Undefined in {filepath}:')
            for line, name in checker.undefined:
                print(f'  Line {line}: {name}')
        else:
            print(f'Clean: {filepath}')

if __name__ == '__main__':
    check_all()
