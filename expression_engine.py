from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple


class TokenType(Enum):
    NUMBER = auto()
    IDENTIFIER = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    position: int


class Tokenizer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def tokenize(self) -> List[Token]:
        tokens = []
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char.isspace():
                self.pos += 1
                continue
            if char.isdigit() or char == '.':
                tokens.append(self._read_number())
            elif char.isalpha() or char == '_':
                tokens.append(self._read_identifier())
            else:
                tokens.append(self._read_operator())
        tokens.append(Token(TokenType.EOF, None, self.pos))
        return tokens

    def _read_number(self) -> Token:
        start = self.pos
        has_dot = False
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c.isdigit():
                self.pos += 1
            elif c == '.' and not has_dot:
                has_dot = True
                self.pos += 1
            else:
                break
        value = float(self.text[start:self.pos])
        return Token(TokenType.NUMBER, value, start)

    def _read_identifier(self) -> Token:
        start = self.pos
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c.isalnum() or c == '_':
                self.pos += 1
            else:
                break
        name = self.text[start:self.pos]
        return Token(TokenType.IDENTIFIER, name, start)

    def _read_operator(self) -> Token:
        c = self.text[self.pos]
        start = self.pos
        self.pos += 1
        if c == '+':
            return Token(TokenType.PLUS, '+', start)
        elif c == '-':
            return Token(TokenType.MINUS, '-', start)
        elif c == '*':
            return Token(TokenType.MUL, '*', start)
        elif c == '/':
            return Token(TokenType.DIV, '/', start)
        elif c == '^':
            return Token(TokenType.POW, '^', start)
        elif c == '(':
            return Token(TokenType.LPAREN, '(', start)
        elif c == ')':
            return Token(TokenType.RPAREN, ')', start)
        elif c == ',':
            return Token(TokenType.COMMA, ',', start)
        else:
            raise ValueError(f"Unexpected character '{c}' at position {start}")


class NodeType(Enum):
    CONSTANT = auto()
    VARIABLE = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    FUNCTION_CALL = auto()


class BinaryOp(Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    POW = '^'


class UnaryOp(Enum):
    NEGATE = '-'


@dataclass
class ASTNode:
    node_type: NodeType
    value: Any = None
    left: Optional['ASTNode'] = None
    right: Optional['ASTNode'] = None
    op: Any = None
    func_name: Optional[str] = None
    args: List['ASTNode'] = field(default_factory=list)
    _hash: Optional[int] = None

    def structural_hash(self) -> int:
        if self._hash is not None:
            return self._hash
        if self.node_type == NodeType.CONSTANT:
            h = hash((NodeType.CONSTANT, self.value))
        elif self.node_type == NodeType.VARIABLE:
            h = hash((NodeType.VARIABLE, self.value))
        elif self.node_type == NodeType.BINARY_OP:
            h = hash((NodeType.BINARY_OP, self.op, self.left.structural_hash(), self.right.structural_hash()))
        elif self.node_type == NodeType.UNARY_OP:
            h = hash((NodeType.UNARY_OP, self.op, self.operand.structural_hash()))
        elif self.node_type == NodeType.FUNCTION_CALL:
            arg_hashes = tuple(a.structural_hash() for a in self.args)
            h = hash((NodeType.FUNCTION_CALL, self.func_name, arg_hashes))
        else:
            h = hash(self.node_type)
        self._hash = h
        return h

    def structural_equal(self, other: 'ASTNode') -> bool:
        if self.node_type != other.node_type:
            return False
        if self.node_type == NodeType.CONSTANT:
            return self.value == other.value
        if self.node_type == NodeType.VARIABLE:
            return self.value == other.value
        if self.node_type == NodeType.BINARY_OP:
            return (self.op == other.op and
                    self.left.structural_equal(other.left) and
                    self.right.structural_equal(other.right))
        if self.node_type == NodeType.UNARY_OP:
            return self.op == other.op and self.operand.structural_equal(other.operand)
        if self.node_type == NodeType.FUNCTION_CALL:
            return (self.func_name == other.func_name and
                    len(self.args) == len(other.args) and
                    all(a.structural_equal(b) for a, b in zip(self.args, other.args)))
        return False

    @property
    def operand(self) -> 'ASTNode':
        return self.left

    @operand.setter
    def operand(self, val: 'ASTNode'):
        self.left = val

    def format(self) -> str:
        if self.node_type == NodeType.CONSTANT:
            v = self.value
            if float(v).is_integer():
                return str(int(v))
            return repr(v)
        if self.node_type == NodeType.VARIABLE:
            return str(self.value)
        if self.node_type == NodeType.UNARY_OP:
            inner = self.operand.format()
            if self.operand.node_type in (NodeType.BINARY_OP, NodeType.UNARY_OP):
                inner = f"({inner})"
            if self.op == UnaryOp.NEGATE:
                return f"-{inner}"
            return f"{self.op}{inner}"
        if self.node_type == NodeType.BINARY_OP:
            left = self.left.format()
            right = self.right.format()
            if self.left.node_type == NodeType.BINARY_OP:
                left = f"({left})"
            if self.right.node_type in (NodeType.BINARY_OP, NodeType.UNARY_OP):
                right = f"({right})"
            return f"{left} {self.op.value} {right}"
        if self.node_type == NodeType.FUNCTION_CALL:
            args = ", ".join(a.format() for a in self.args)
            return f"{self.func_name}({args})"
        return f"<{self.node_type.name}>"

    def to_tree_lines(self, prefix: str = "", is_last: bool = True) -> List[str]:
        lines: List[str] = []
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + self.format())
        child_prefix = prefix + ("    " if is_last else "│   ")
        children: List[ASTNode] = []
        if self.node_type == NodeType.UNARY_OP:
            children = [self.operand]
        elif self.node_type == NodeType.BINARY_OP:
            children = [self.left, self.right]
        elif self.node_type == NodeType.FUNCTION_CALL:
            children = list(self.args)
        for i, child in enumerate(children):
            lines.extend(child.to_tree_lines(child_prefix, i == len(children) - 1))
        return lines

    def to_tree_str(self) -> str:
        return "\n".join(self.to_tree_lines("", True))


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> ASTNode:
        result = self._parse_expression()
        if self._current().type != TokenType.EOF:
            raise ValueError(f"Unexpected token {self._current()}")
        return result

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _parse_expression(self) -> ASTNode:
        return self._parse_add_sub()

    def _parse_add_sub(self) -> ASTNode:
        node = self._parse_mul_div()
        while self._current().type in (TokenType.PLUS, TokenType.MINUS):
            token = self._advance()
            op = BinaryOp.ADD if token.type == TokenType.PLUS else BinaryOp.SUB
            right = self._parse_mul_div()
            node = ASTNode(node_type=NodeType.BINARY_OP, op=op, left=node, right=right)
        return node

    def _parse_mul_div(self) -> ASTNode:
        node = self._parse_unary()
        while self._current().type in (TokenType.MUL, TokenType.DIV):
            token = self._advance()
            op = BinaryOp.MUL if token.type == TokenType.MUL else BinaryOp.DIV
            right = self._parse_unary()
            node = ASTNode(node_type=NodeType.BINARY_OP, op=op, left=node, right=right)
        return node

    def _parse_unary(self) -> ASTNode:
        if self._current().type == TokenType.MINUS:
            self._advance()
            operand = self._parse_unary()
            return ASTNode(node_type=NodeType.UNARY_OP, op=UnaryOp.NEGATE, left=operand)
        return self._parse_power()

    def _parse_power(self) -> ASTNode:
        node = self._parse_primary()
        if self._current().type == TokenType.POW:
            self._advance()
            right = self._parse_unary()
            return ASTNode(node_type=NodeType.BINARY_OP, op=BinaryOp.POW, left=node, right=right)
        return node

    def _parse_primary(self) -> ASTNode:
        token = self._current()
        if token.type == TokenType.NUMBER:
            self._advance()
            return ASTNode(node_type=NodeType.CONSTANT, value=token.value)
        elif token.type == TokenType.IDENTIFIER:
            self._advance()
            if self._current().type == TokenType.LPAREN:
                return self._parse_function_call(token.value)
            return ASTNode(node_type=NodeType.VARIABLE, value=token.value)
        elif token.type == TokenType.LPAREN:
            self._advance()
            node = self._parse_expression()
            if self._current().type != TokenType.RPAREN:
                raise ValueError("Expected closing parenthesis")
            self._advance()
            return node
        else:
            raise ValueError(f"Unexpected token {token}")

    def _parse_function_call(self, func_name: str) -> ASTNode:
        self._advance()
        args = []
        if self._current().type != TokenType.RPAREN:
            args.append(self._parse_expression())
            while self._current().type == TokenType.COMMA:
                self._advance()
                args.append(self._parse_expression())
        if self._current().type != TokenType.RPAREN:
            raise ValueError("Expected closing parenthesis in function call")
        self._advance()
        return ASTNode(node_type=NodeType.FUNCTION_CALL, func_name=func_name, args=args)


class ConstantFolder:
    _math = __import__('math')
    BUILTIN_FUNCS: Dict[str, Callable] = {
        'sin': _math.sin,
        'cos': _math.cos,
        'tan': _math.tan,
        'sqrt': _math.sqrt,
        'exp': _math.exp,
        'log': _math.log,
        'abs': abs,
        'min': min,
        'max': max,
        'floor': _math.floor,
        'ceil': _math.ceil,
        'if': lambda c, t, e: t if c > 0 else e,
        'clamp': lambda x, lo, hi: max(lo, min(x, hi)),
        'sign': lambda x: (1.0 if x > 0 else (-1.0 if x < 0 else 0.0)),
    }

    @classmethod
    def fold(cls, node: ASTNode) -> ASTNode:
        if node.node_type == NodeType.CONSTANT:
            return node
        if node.node_type == NodeType.VARIABLE:
            return node
        if node.node_type == NodeType.UNARY_OP:
            folded_operand = cls.fold(node.operand)
            node.operand = folded_operand
            if folded_operand.node_type == NodeType.CONSTANT:
                if node.op == UnaryOp.NEGATE:
                    return ASTNode(node_type=NodeType.CONSTANT, value=-folded_operand.value)
            return node
        if node.node_type == NodeType.BINARY_OP:
            folded_left = cls.fold(node.left)
            folded_right = cls.fold(node.right)
            node.left = folded_left
            node.right = folded_right
            if folded_left.node_type == NodeType.CONSTANT and folded_right.node_type == NodeType.CONSTANT:
                return cls._fold_binary(node.op, folded_left.value, folded_right.value)
            return cls._simplify_binary(node, folded_left, folded_right)
        if node.node_type == NodeType.FUNCTION_CALL:
            folded_args = [cls.fold(arg) for arg in node.args]
            node.args = folded_args
            if all(a.node_type == NodeType.CONSTANT for a in folded_args):
                func = cls.BUILTIN_FUNCS.get(node.func_name)
                if func:
                    values = [a.value for a in folded_args]
                    try:
                        result = func(*values)
                        return ASTNode(node_type=NodeType.CONSTANT, value=result)
                    except:
                        pass
            return node
        return node

    @classmethod
    def _fold_binary(cls, op: BinaryOp, left_val: float, right_val: float) -> ASTNode:
        if op == BinaryOp.ADD:
            value = left_val + right_val
        elif op == BinaryOp.SUB:
            value = left_val - right_val
        elif op == BinaryOp.MUL:
            value = left_val * right_val
        elif op == BinaryOp.DIV:
            if right_val == 0:
                raise ZeroDivisionError("Division by zero in constant folding")
            value = left_val / right_val
        elif op == BinaryOp.POW:
            value = left_val ** right_val
        else:
            raise ValueError(f"Unknown binary operator: {op}")
        return ASTNode(node_type=NodeType.CONSTANT, value=value)

    @classmethod
    def _simplify_binary(cls, node: ASTNode, left: ASTNode, right: ASTNode) -> ASTNode:
        if node.op == BinaryOp.ADD:
            if left.node_type == NodeType.CONSTANT and left.value == 0:
                return right
            if right.node_type == NodeType.CONSTANT and right.value == 0:
                return left
        elif node.op == BinaryOp.SUB:
            if right.node_type == NodeType.CONSTANT and right.value == 0:
                return left
            if left.node_type == NodeType.CONSTANT and left.value == 0:
                return ASTNode(node_type=NodeType.UNARY_OP, op=UnaryOp.NEGATE, left=right)
        elif node.op == BinaryOp.MUL:
            if left.node_type == NodeType.CONSTANT:
                if left.value == 0:
                    return ASTNode(node_type=NodeType.CONSTANT, value=0.0)
                if left.value == 1:
                    return right
            if right.node_type == NodeType.CONSTANT:
                if right.value == 0:
                    return ASTNode(node_type=NodeType.CONSTANT, value=0.0)
                if right.value == 1:
                    return left
        elif node.op == BinaryOp.DIV:
            if left.node_type == NodeType.CONSTANT and left.value == 0:
                return ASTNode(node_type=NodeType.CONSTANT, value=0.0)
            if right.node_type == NodeType.CONSTANT and right.value == 1:
                return left
        elif node.op == BinaryOp.POW:
            if left.node_type == NodeType.CONSTANT and left.value == 0:
                return ASTNode(node_type=NodeType.CONSTANT, value=0.0)
            if right.node_type == NodeType.CONSTANT:
                if right.value == 0:
                    return ASTNode(node_type=NodeType.CONSTANT, value=1.0)
                if right.value == 1:
                    return left
        return node


@dataclass
class CSEResult:
    root: ASTNode
    temp_bindings: List[ASTNode] = field(default_factory=list)


class CSE:
    def __init__(self):
        self.expr_map: Dict[int, int] = {}
        self.temp_bindings: List[ASTNode] = []

    def eliminate(self, node: ASTNode) -> CSEResult:
        self.expr_map = {}
        self.temp_bindings = []
        new_root = self._process(node)
        return CSEResult(root=new_root, temp_bindings=self.temp_bindings)

    def _process(self, node: ASTNode) -> ASTNode:
        if node.node_type in (NodeType.CONSTANT, NodeType.VARIABLE):
            return node

        if node.node_type == NodeType.UNARY_OP:
            node.operand = self._process(node.operand)
            return self._intern(node)

        if node.node_type == NodeType.BINARY_OP:
            node.left = self._process(node.left)
            node.right = self._process(node.right)
            return self._intern(node)

        if node.node_type == NodeType.FUNCTION_CALL:
            node.args = [self._process(arg) for arg in node.args]
            return self._intern(node)

        return node

    def _intern(self, node: ASTNode) -> ASTNode:
        node._hash = None
        h = node.structural_hash()

        if h in self.expr_map:
            temp_id = self.expr_map[h]
            return ASTNode(node_type=NodeType.VARIABLE, value=f'__temp_{temp_id}')

        if not self._contains_variable(node):
            self.expr_map[h] = -1
            return node

        temp_id = len(self.temp_bindings)
        self.temp_bindings.append(node)
        self.expr_map[h] = temp_id
        return ASTNode(node_type=NodeType.VARIABLE, value=f'__temp_{temp_id}')

    def _contains_variable(self, node: ASTNode) -> bool:
        if node.node_type == NodeType.VARIABLE:
            if str(node.value).startswith('__temp_'):
                return True
            return True
        if node.node_type == NodeType.CONSTANT:
            return False
        if node.node_type == NodeType.UNARY_OP:
            return self._contains_variable(node.operand)
        if node.node_type == NodeType.BINARY_OP:
            return self._contains_variable(node.left) or self._contains_variable(node.right)
        if node.node_type == NodeType.FUNCTION_CALL:
            return any(self._contains_variable(a) for a in node.args)
        return False


class Opcode(Enum):
    LOAD_CONST = auto()
    LOAD_VAR = auto()
    LOAD_TEMP = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    NEGATE = auto()
    CALL_FUNC = auto()
    STORE_TEMP = auto()


@dataclass
class Instruction:
    opcode: Opcode
    operand: Any = None


class CodeGenerator:
    def __init__(self):
        self.instructions: List[Instruction] = []
        self.variable_map: Dict[str, int] = {}
        self.constants: List[float] = []
        self.temp_count = 0

    def generate(self, cse_result: CSEResult) -> 'CompiledExpression':
        self.instructions = []
        self.variable_map = {}
        self.constants = []
        self.temp_count = 0

        root = cse_result.root
        temp_bindings = cse_result.temp_bindings

        self._collect_variables(root)
        for binding in temp_bindings:
            self._collect_variables(binding)

        self.temp_count = len(temp_bindings)
        for i, binding in enumerate(temp_bindings):
            self._emit_expr(binding)
            self.instructions.append(Instruction(Opcode.STORE_TEMP, i))

        self._emit_expr(root)
        return CompiledExpression(
            instructions=self.instructions,
            variable_map=self.variable_map,
            constants=self.constants,
            temp_count=self.temp_count,
            required_variables=list(self.variable_map.keys())
        )

    def _collect_variables(self, node: ASTNode):
        if node.node_type == NodeType.VARIABLE:
            name = node.value
            if not name.startswith('__temp_') and name not in self.variable_map:
                self.variable_map[name] = len(self.variable_map)
        elif node.node_type == NodeType.UNARY_OP:
            self._collect_variables(node.operand)
        elif node.node_type == NodeType.BINARY_OP:
            self._collect_variables(node.left)
            self._collect_variables(node.right)
        elif node.node_type == NodeType.FUNCTION_CALL:
            for arg in node.args:
                self._collect_variables(arg)

    def _temp_index(self, var_name: str) -> int:
        prefix = '__temp_'
        return int(var_name[len(prefix):])

    def _emit_expr(self, node: ASTNode):
        if node.node_type == NodeType.CONSTANT:
            const_idx = len(self.constants)
            self.constants.append(node.value)
            self.instructions.append(Instruction(Opcode.LOAD_CONST, const_idx))
        elif node.node_type == NodeType.VARIABLE:
            name = node.value
            if name.startswith('__temp_'):
                self.instructions.append(Instruction(Opcode.LOAD_TEMP, self._temp_index(name)))
            else:
                if name not in self.variable_map:
                    self.variable_map[name] = len(self.variable_map)
                self.instructions.append(Instruction(Opcode.LOAD_VAR, self.variable_map[name]))
        elif node.node_type == NodeType.UNARY_OP:
            self._emit_expr(node.operand)
            if node.op == UnaryOp.NEGATE:
                self.instructions.append(Instruction(Opcode.NEGATE))
        elif node.node_type == NodeType.BINARY_OP:
            self._emit_expr(node.left)
            self._emit_expr(node.right)
            if node.op == BinaryOp.ADD:
                self.instructions.append(Instruction(Opcode.ADD))
            elif node.op == BinaryOp.SUB:
                self.instructions.append(Instruction(Opcode.SUB))
            elif node.op == BinaryOp.MUL:
                self.instructions.append(Instruction(Opcode.MUL))
            elif node.op == BinaryOp.DIV:
                self.instructions.append(Instruction(Opcode.DIV))
            elif node.op == BinaryOp.POW:
                self.instructions.append(Instruction(Opcode.POW))
        elif node.node_type == NodeType.FUNCTION_CALL:
            for arg in node.args:
                self._emit_expr(arg)
            self.instructions.append(Instruction(Opcode.CALL_FUNC, (node.func_name, len(node.args))))


@dataclass
class CompiledExpression:
    instructions: List[Instruction]
    variable_map: Dict[str, int]
    constants: List[float]
    temp_count: int
    required_variables: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._fast_ops: List[int] = []
        self._fast_args: List[Any] = []
        self._func_refs = ConstantFolder.BUILTIN_FUNCS
        op_map = {
            Opcode.LOAD_CONST: 0,
            Opcode.LOAD_VAR: 1,
            Opcode.LOAD_TEMP: 2,
            Opcode.STORE_TEMP: 3,
            Opcode.ADD: 4,
            Opcode.SUB: 5,
            Opcode.MUL: 6,
            Opcode.DIV: 7,
            Opcode.POW: 8,
            Opcode.NEGATE: 9,
            Opcode.CALL_FUNC: 10,
        }
        for instr in self.instructions:
            self._fast_ops.append(op_map[instr.opcode])
            if instr.opcode == Opcode.CALL_FUNC:
                fname, arity = instr.operand
                self._fast_args.append((self._func_refs[fname], arity))
            else:
                self._fast_args.append(instr.operand)

    def evaluate(self, variables: Optional[Dict[str, float]] = None) -> float:
        var_values = [0.0] * len(self.variable_map)
        if variables:
            missing = []
            for name, idx in self.variable_map.items():
                if name in variables:
                    var_values[idx] = variables[name]
                else:
                    missing.append(name)
            if missing:
                if len(missing) == 1:
                    raise ValueError(f"Missing required variable: '{missing[0]}'")
                raise ValueError(f"Missing required variables: {missing}")
        elif self.variable_map:
            missing = list(self.variable_map.keys())
            raise ValueError(f"Missing required variables: {missing}")
        return self._eval_arr(var_values)

    def evaluate_arr(self, var_values: List[float]) -> float:
        return self._eval_arr(var_values)

    def _eval_arr(self, var_values: List[float]) -> float:
        temps = [0.0] * self.temp_count
        stack_append = list.append
        stack_pop = list.pop
        stack = []
        consts = self.constants
        ops = self._fast_ops
        args = self._fast_args
        n = len(ops)
        i = 0

        while i < n:
            op = ops[i]
            arg = args[i]
            if op == 0:
                stack_append(stack, consts[arg])
            elif op == 1:
                stack_append(stack, var_values[arg])
            elif op == 2:
                stack_append(stack, temps[arg])
            elif op == 3:
                temps[arg] = stack[-1]
            elif op == 4:
                b = stack_pop(stack)
                a = stack_pop(stack)
                stack_append(stack, a + b)
            elif op == 5:
                b = stack_pop(stack)
                a = stack_pop(stack)
                stack_append(stack, a - b)
            elif op == 6:
                b = stack_pop(stack)
                a = stack_pop(stack)
                stack_append(stack, a * b)
            elif op == 7:
                b = stack_pop(stack)
                a = stack_pop(stack)
                stack_append(stack, a / b)
            elif op == 8:
                b = stack_pop(stack)
                a = stack_pop(stack)
                stack_append(stack, a ** b)
            elif op == 9:
                a = stack_pop(stack)
                stack_append(stack, -a)
            elif op == 10:
                func, arity = arg
                if arity == 1:
                    a = stack_pop(stack)
                    stack_append(stack, func(a))
                elif arity == 2:
                    b = stack_pop(stack)
                    a = stack_pop(stack)
                    stack_append(stack, func(a, b))
                else:
                    popped = [stack_pop(stack) for _ in range(arity)]
                    popped.reverse()
                    stack_append(stack, func(*popped))
            i += 1

        return stack[-1]


class TreeEvaluator:
    BUILTIN_FUNCS = ConstantFolder.BUILTIN_FUNCS

    @staticmethod
    def evaluate(node: ASTNode, variables: Optional[Dict[str, float]] = None) -> float:
        if variables is None:
            variables = {}
        return TreeEvaluator._eval(node, variables)

    @staticmethod
    def _eval(node: ASTNode, variables: Dict[str, float]) -> float:
        if node.node_type == NodeType.CONSTANT:
            return node.value
        if node.node_type == NodeType.VARIABLE:
            name = node.value
            if name not in variables:
                raise ValueError(f"Undefined variable: {name}")
            return variables[name]
        if node.node_type == NodeType.UNARY_OP:
            val = TreeEvaluator._eval(node.operand, variables)
            if node.op == UnaryOp.NEGATE:
                return -val
        if node.node_type == NodeType.BINARY_OP:
            left_val = TreeEvaluator._eval(node.left, variables)
            right_val = TreeEvaluator._eval(node.right, variables)
            if node.op == BinaryOp.ADD:
                return left_val + right_val
            if node.op == BinaryOp.SUB:
                return left_val - right_val
            if node.op == BinaryOp.MUL:
                return left_val * right_val
            if node.op == BinaryOp.DIV:
                return left_val / right_val
            if node.op == BinaryOp.POW:
                return left_val ** right_val
        if node.node_type == NodeType.FUNCTION_CALL:
            if node.func_name == '__let__':
                return TreeEvaluator._eval(node.args[2], variables)
            func = TreeEvaluator.BUILTIN_FUNCS.get(node.func_name)
            if not func:
                raise ValueError(f"Unknown function: {node.func_name}")
            args = [TreeEvaluator._eval(a, variables) for a in node.args]
            return func(*args)
        raise ValueError(f"Unknown node type: {node.node_type}")


@dataclass
class CompilationStages:
    expression: str
    required_variables: List[str]
    defaults: Dict[str, float]
    raw_ast: ASTNode
    folded_ast: ASTNode
    cse_temp_bindings: List[Tuple[str, ASTNode]]
    cse_root: ASTNode
    instructions: List[Instruction]
    constants: List[float]
    variable_map: Dict[str, int]
    temp_count: int


class ExpressionEngine:
    def __init__(self, expression: str, **default_variables):
        self.original_expression = expression
        tokens = Tokenizer(expression).tokenize()
        self.raw_ast = Parser(tokens).parse()
        self.folded_ast = ConstantFolder.fold(self._clone_ast(self.raw_ast))
        self.cse_result = CSE().eliminate(self._clone_ast(self.folded_ast))
        self.optimized_root = self.cse_result.root
        self.temp_bindings = self.cse_result.temp_bindings
        self.compiled = CodeGenerator().generate(self.cse_result)
        self._native_func = None
        self._sorted_vars = self.variables()
        self.defaults: Dict[str, float] = {}
        self.set_defaults(**default_variables)

    def compilation_report(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("编译报告")
        lines.append("=" * 70)
        lines.append(f"原始表达式: {self.original_expression}")
        lines.append(f"变量列表: {self._sorted_vars}")
        if self.defaults:
            lines.append(f"默认值: {self.defaults}")

        lines.append("")
        lines.append("--- [阶段 1] 原始 AST 树 ---")
        lines.append("")
        lines.append(self.raw_ast.to_tree_str())

        lines.append("")
        lines.append("--- [阶段 2] 常量折叠后 AST 树 ---")
        lines.append("")
        folded_str = self.folded_ast.to_tree_str()
        if folded_str == self.raw_ast.to_tree_str():
            lines.append("（无变化，表达式中没有可折叠的全常量子树）")
        else:
            lines.append(folded_str)

        lines.append("")
        lines.append("--- [阶段 3] 公共子表达式（CSE）临时变量绑定 ---")
        lines.append("")
        if not self.temp_bindings:
            lines.append("（无公共子表达式）")
        else:
            for i, binding in enumerate(self.temp_bindings):
                lines.append(f"  __temp_{i} = {binding.format()}")

        lines.append("")
        lines.append("--- [阶段 4] CSE 后主表达式 ---")
        lines.append("")
        lines.append(f"  => {self.optimized_root.format()}")

        lines.append("")
        lines.append("--- [阶段 5] 最终栈式指令序列 ---")
        lines.append("")
        lines.append(f"  常量池: {self.compiled.constants}")
        lines.append(f"  变量槽位: {dict(self.compiled.variable_map)}")
        lines.append(f"  临时变量数: {self.compiled.temp_count}")
        lines.append("")
        for i, instr in enumerate(self.instructions()):
            lines.append(f"  {i:3d}: {instr}")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def compilation_stages(self) -> CompilationStages:
        return CompilationStages(
            expression=self.original_expression,
            required_variables=list(self._sorted_vars),
            defaults=dict(self.defaults),
            raw_ast=self.raw_ast,
            folded_ast=self.folded_ast,
            cse_temp_bindings=[(f'__temp_{i}', b) for i, b in enumerate(self.temp_bindings)],
            cse_root=self.optimized_root,
            instructions=list(self.compiled.instructions),
            constants=list(self.compiled.constants),
            variable_map=dict(self.compiled.variable_map),
            temp_count=self.compiled.temp_count,
        )

    def set_defaults(self, **kwargs) -> 'ExpressionEngine':
        for k, v in kwargs.items():
            if k not in self.compiled.variable_map:
                raise ValueError(f"Unknown variable '{k}' for expression '{self.original_expression}'")
            self.defaults[k] = float(v)
        self._native_func = None
        return self

    def _resolve_variables(self, variables: Optional[Dict[str, float]]) -> Dict[str, float]:
        required = self._sorted_vars
        if variables is None:
            variables = {}
        resolved: Dict[str, float] = {}
        missing = []
        for name in required:
            if name in variables:
                resolved[name] = float(variables[name])
            elif name in self.defaults:
                resolved[name] = float(self.defaults[name])
            else:
                missing.append(name)
        if missing:
            if len(missing) == 1:
                raise ValueError(f"Missing required variable: '{missing[0]}'")
            raise ValueError(f"Missing required variables: {missing}")
        return resolved

    def _clone_ast(self, node: ASTNode) -> ASTNode:
        new_node = ASTNode(node_type=node.node_type, value=node.value, op=node.op, func_name=node.func_name)
        if node.left is not None:
            new_node.left = self._clone_ast(node.left)
        if node.right is not None:
            new_node.right = self._clone_ast(node.right)
        new_node.args = [self._clone_ast(a) for a in node.args]
        return new_node

    def _ast_to_python(self, node: ASTNode) -> str:
        if node.node_type == NodeType.CONSTANT:
            return repr(node.value)
        if node.node_type == NodeType.VARIABLE:
            return str(node.value)
        if node.node_type == NodeType.UNARY_OP:
            operand = self._ast_to_python(node.operand)
            if node.op == UnaryOp.NEGATE:
                return f"(-{operand})"
        if node.node_type == NodeType.BINARY_OP:
            left = self._ast_to_python(node.left)
            right = self._ast_to_python(node.right)
            if node.op == BinaryOp.ADD:
                return f"({left} + {right})"
            if node.op == BinaryOp.SUB:
                return f"({left} - {right})"
            if node.op == BinaryOp.MUL:
                return f"({left} * {right})"
            if node.op == BinaryOp.DIV:
                return f"({left} / {right})"
            if node.op == BinaryOp.POW:
                return f"({left} ** {right})"
        if node.node_type == NodeType.FUNCTION_CALL:
            if node.func_name == 'if':
                c, t, e = [self._ast_to_python(a) for a in node.args]
                return f"(({t}) if ({c}) > 0 else ({e}))"
            if node.func_name == 'clamp':
                x, lo, hi = [self._ast_to_python(a) for a in node.args]
                return f"max({lo}, min({x}, {hi}))"
            if node.func_name == 'sign':
                x = self._ast_to_python(node.args[0])
                return f"(1 if ({x}) > 0 else (-1 if ({x}) < 0 else 0))"
            args = ", ".join(self._ast_to_python(a) for a in node.args)
            return f"{node.func_name}({args})"
        raise ValueError(f"Cannot convert node: {node.node_type}")

    def compile_to_native(self) -> Callable:
        if self._native_func is not None:
            return self._native_func

        var_names = self._sorted_vars
        func_params = ", ".join(var_names)

        lines = ["import math as _m", "import builtins as _b"]
        for fname in ConstantFolder.BUILTIN_FUNCS:
            if fname in ('sin', 'cos', 'tan', 'sqrt', 'exp', 'log', 'floor', 'ceil'):
                lines.append(f"{fname} = _m.{fname}")
            elif fname in ('abs', 'min', 'max'):
                lines.append(f"{fname} = _b.{fname}")
            elif fname == 'sign':
                lines.append("def _sign(x): return (1 if x > 0 else (-1 if x < 0 else 0))")
                lines.append("sign = _sign")

        for i, binding in enumerate(self.temp_bindings):
            expr_str = self._ast_to_python(binding)
            lines.append(f"__temp_{i} = {expr_str}")

        result_expr = self._ast_to_python(self.optimized_root)
        lines.append(f"return {result_expr}")

        body = "\n    ".join(lines)
        func_src = f"def _native_func({func_params}):\n    {body}"

        namespace = {}
        exec(func_src, namespace)
        self._native_func = namespace['_native_func']
        return self._native_func

    def evaluate_native(self, variables: Optional[Dict[str, float]] = None) -> float:
        resolved = self._resolve_variables(variables)
        func = self.compile_to_native()
        return func(*(resolved[n] for n in self._sorted_vars))

    def evaluate_tree(self, variables: Optional[Dict[str, float]] = None) -> float:
        resolved = self._resolve_variables(variables)
        return TreeEvaluator.evaluate(self.raw_ast, resolved)

    def evaluate_optimized_tree(self, variables: Optional[Dict[str, float]] = None) -> float:
        resolved = self._resolve_variables(variables)
        temp_values = {}
        for i, binding in enumerate(self.temp_bindings):
            temp_values[f'__temp_{i}'] = TreeEvaluator._eval(binding, {**resolved, **temp_values})
        return TreeEvaluator._eval(self.optimized_root, {**resolved, **temp_values})

    def evaluate(self, variables: Optional[Dict[str, float]] = None) -> float:
        resolved = self._resolve_variables(variables)
        var_map = self.compiled.variable_map
        arr = [0.0] * len(var_map)
        for name, idx in var_map.items():
            arr[idx] = resolved[name]
        return self.compiled.evaluate_arr(arr)

    def _build_var_array(self, resolved: Dict[str, float]) -> List[float]:
        var_map = self.compiled.variable_map
        arr = [0.0] * len(var_map)
        for name, idx in var_map.items():
            arr[idx] = resolved[name]
        return arr

    def evaluate_batch(self, variable_list: List[Dict[str, float]]) -> List[float]:
        var_names = self._sorted_vars
        var_map = self.compiled.variable_map
        num_vars = len(var_names)
        indices = [var_map[n] for n in var_names]
        defaults = self.defaults
        compiled = self.compiled
        results = []
        append = results.append
        for vars_dict in variable_list:
            missing = []
            arr = [0.0] * num_vars
            for name, idx in zip(var_names, indices):
                if name in vars_dict:
                    arr[idx] = float(vars_dict[name])
                elif name in defaults:
                    arr[idx] = float(defaults[name])
                else:
                    missing.append(name)
            if missing:
                if len(missing) == 1:
                    raise ValueError(f"Missing required variable: '{missing[0]}'")
                raise ValueError(f"Missing required variables: {missing}")
            append(compiled.evaluate_arr(arr))
        return results

    def evaluate_columns(self, columns: Dict[str, List[float]]) -> List[float]:
        var_names = self._sorted_vars
        defaults = self.defaults
        missing = [n for n in var_names if n not in columns and n not in defaults]
        if missing:
            if len(missing) == 1:
                raise ValueError(f"Missing required column: '{missing[0]}'")
            raise ValueError(f"Missing required columns: {missing}")
        lengths = set()
        for n in var_names:
            if n in columns:
                lengths.add(len(columns[n]))
        if len(lengths) != 1:
            info = {n: len(columns[n]) for n in var_names if n in columns}
            raise ValueError(f"All column arrays must have the same length. Got: {info}")
        N = lengths.pop() if lengths else 0
        col_data = {}
        for n in var_names:
            if n in columns:
                col_data[n] = columns[n]
            else:
                col_data[n] = [defaults[n]] * N
        var_map = self.compiled.variable_map
        num_vars = len(var_names)
        indices = [var_map[n] for n in var_names]
        col_arrays = [col_data[n] for n in var_names]
        compiled = self.compiled
        results = [0.0] * N
        for i in range(N):
            arr = [0.0] * num_vars
            for j in range(num_vars):
                arr[indices[j]] = float(col_arrays[j][i])
            results[i] = compiled.evaluate_arr(arr)
        return results

    def evaluate_batch_native(self, variable_list: List[Dict[str, float]]) -> List[float]:
        func = self.compile_to_native()
        var_names = self._sorted_vars
        defaults = self.defaults
        results = []
        append = results.append
        for v in variable_list:
            missing = []
            args = []
            for n in var_names:
                if n in v:
                    args.append(float(v[n]))
                elif n in defaults:
                    args.append(float(defaults[n]))
                else:
                    missing.append(n)
            if missing:
                if len(missing) == 1:
                    raise ValueError(f"Missing required variable: '{missing[0]}'")
                raise ValueError(f"Missing required variables: {missing}")
            append(func(*args))
        return results

    def variables(self) -> List[str]:
        return sorted(self.compiled.variable_map.keys())

    def instructions(self) -> List[str]:
        result = []
        for instr in self.compiled.instructions:
            if instr.operand is not None:
                result.append(f"{instr.opcode.name} {instr.operand}")
            else:
                result.append(instr.opcode.name)
        return result

    def evaluator(self) -> BatchEvaluator:
        return BatchEvaluator(self.compiled, self._sorted_vars, self.defaults, self.compile_to_native())


class BatchEvaluator:
    __slots__ = ('_compiled', '_var_names', '_defaults', '_indices', '_num_vars', '_native_func', '_eval_arr')

    def __init__(self, compiled: CompiledExpression, var_names: List[str], defaults: Dict[str, float], native_func: Optional[Callable] = None):
        self._compiled = compiled
        self._var_names = var_names
        self._defaults = defaults
        self._indices = [compiled.variable_map[n] for n in var_names]
        self._num_vars = len(var_names)
        self._native_func = native_func
        self._eval_arr = compiled._eval_arr

    def __call__(self, **variables) -> float:
        missing = []
        arr = [0.0] * self._num_vars
        for name, idx in zip(self._var_names, self._indices):
            if name in variables:
                arr[idx] = float(variables[name])
            elif name in self._defaults:
                arr[idx] = float(self._defaults[name])
            else:
                missing.append(name)
        if missing:
            if len(missing) == 1:
                raise ValueError(f"Missing required variable: '{missing[0]}'")
            raise ValueError(f"Missing required variables: {missing}")
        return self._eval_arr(arr)

    def eval_dict_list(self, variable_list: List[Dict[str, float]]) -> List[float]:
        var_names = self._var_names
        defaults = self._defaults
        indices = self._indices
        num_vars = self._num_vars
        eval_arr = self._eval_arr
        results = []
        append = results.append
        for vd in variable_list:
            missing = []
            arr = [0.0] * num_vars
            for name, idx in zip(var_names, indices):
                if name in vd:
                    arr[idx] = float(vd[name])
                elif name in defaults:
                    arr[idx] = float(defaults[name])
                else:
                    missing.append(name)
            if missing:
                if len(missing) == 1:
                    raise ValueError(f"Missing required variable: '{missing[0]}'")
                raise ValueError(f"Missing required variables: {missing}")
            append(eval_arr(arr))
        return results

    def eval_columns(self, columns: Dict[str, List[float]]) -> List[float]:
        var_names = self._var_names
        defaults = self._defaults
        missing = [n for n in var_names if n not in columns and n not in defaults]
        if missing:
            if len(missing) == 1:
                raise ValueError(f"Missing required column: '{missing[0]}'")
            raise ValueError(f"Missing required columns: {missing}")
        lengths = set()
        for n in var_names:
            if n in columns:
                lengths.add(len(columns[n]))
        if len(lengths) != 1:
            info = {n: len(columns[n]) for n in var_names if n in columns}
            raise ValueError(f"All column arrays must have the same length. Got: {info}")
        N = lengths.pop() if lengths else 0
        col_data = {}
        for n in var_names:
            if n in columns:
                col_data[n] = columns[n]
            else:
                col_data[n] = [defaults[n]] * N
        indices = self._indices
        num_vars = self._num_vars
        eval_arr = self._eval_arr
        col_arrays = [col_data[n] for n in var_names]
        results = [0.0] * N
        for i in range(N):
            arr = [0.0] * num_vars
            for j in range(num_vars):
                arr[indices[j]] = float(col_arrays[j][i])
            results[i] = eval_arr(arr)
        return results

    def eval_arrays(self, *arrays: List[float]) -> List[float]:
        num_vars = self._num_vars
        if len(arrays) != num_vars:
            raise ValueError(f"Expected {num_vars} arrays (one per variable), got {len(arrays)}")
        N = len(arrays[0])
        for i, a in enumerate(arrays):
            if len(a) != N:
                raise ValueError(f"Array {i} has length {len(a)}, expected {N}")
        indices = self._indices
        eval_arr = self._eval_arr
        results = [0.0] * N
        for i in range(N):
            arr = [0.0] * num_vars
            for j in range(num_vars):
                arr[indices[j]] = float(arrays[j][i])
            results[i] = eval_arr(arr)
        return results

    def eval_native_dict_list(self, variable_list: List[Dict[str, float]]) -> List[float]:
        if self._native_func is None:
            raise ValueError("Native function not available")
        var_names = self._var_names
        defaults = self._defaults
        func = self._native_func
        results = []
        append = results.append
        for vd in variable_list:
            missing = []
            args = []
            for n in var_names:
                if n in vd:
                    args.append(float(vd[n]))
                elif n in defaults:
                    args.append(float(defaults[n]))
                else:
                    missing.append(n)
            if missing:
                if len(missing) == 1:
                    raise ValueError(f"Missing required variable: '{missing[0]}'")
                raise ValueError(f"Missing required variables: {missing}")
            append(func(*args))
        return results


def run_tests():
    import math as _m
    print("=" * 70)
    print("表达式树编译与求值优化引擎 - 测试套件")
    print("=" * 70)

    print("\n--- 测试 1: 纯常量表达式求值 ---")
    engine = ExpressionEngine("2 + 3 * 4")
    assert abs(engine.evaluate() - 14.0) < 1e-10
    print("✓ 2 + 3 * 4 = 14")

    print("\n--- 测试 2: 变量求值 ---")
    engine = ExpressionEngine("x + y * 2")
    result = engine.evaluate({'x': 5, 'y': 3})
    assert abs(result - 11.0) < 1e-10
    print(f"✓ x + y * 2  (x=5, y=3) = {result}")

    print("\n--- 测试 3: 严格变量检查（缺变量报错） ---")
    engine = ExpressionEngine("x + y")
    try:
        engine.evaluate({'x': 1})
        raise AssertionError("应该报错缺变量y")
    except ValueError as e:
        assert "'y'" in str(e)
        print(f"✓ 缺变量时报错正确: {e}")

    print("\n--- 测试 4: 用户主动设置默认值 ---")
    engine = ExpressionEngine("x + y", y=10)
    r1 = engine.evaluate({'x': 5})
    assert abs(r1 - 15.0) < 1e-10
    r2 = engine.evaluate({'x': 5, 'y': 20})
    assert abs(r2 - 25.0) < 1e-10
    engine.set_defaults(x=100)
    r3 = engine.evaluate()
    assert abs(r3 - 110.0) < 1e-10
    print(f"✓ 默认值工作: x+y(x=5,默认y=10)={r1}; x+y(x=5,y=20)={r2}; 全默认(x=100,y=10)={r3}")

    print("\n--- 测试 5: 设置不存在变量的默认值时报错 ---")
    try:
        ExpressionEngine("x + y", z=1)
        raise AssertionError("应该报错z不存在")
    except ValueError as e:
        assert "'z'" in str(e)
        print(f"✓ 设置不存在变量默认值时报错正确: {e}")

    print("\n--- 测试 6: 新增函数 min/max/floor/ceil ---")
    engine = ExpressionEngine("min(a, b) + max(a, b) + floor(x) + ceil(y)")
    v = {'a': 2.3, 'b': 5.7, 'x': 3.7, 'y': 3.1}
    r_tree = engine.evaluate_tree(v)
    r_vm = engine.evaluate(v)
    r_native = engine.evaluate_native(v)
    expected = min(2.3, 5.7) + max(2.3, 5.7) + _m.floor(3.7) + _m.ceil(3.1)
    assert abs(r_tree - expected) < 1e-10
    assert abs(r_vm - expected) < 1e-10
    assert abs(r_native - expected) < 1e-10
    print(f"  min(2.3,5.7)={min(2.3,5.7)}, max(2.3,5.7)={max(2.3,5.7)}, floor(3.7)={_m.floor(3.7)}, ceil(3.1)={_m.ceil(3.1)}")
    print(f"  树遍历={r_tree}, VM={r_vm}, 原生={r_native}, 期望={expected}")
    print("✓ min/max/floor/ceil 在三种求值路径结果一致")

    print("\n--- 测试 7: 新函数的常量折叠 ---")
    engine = ExpressionEngine("floor(3.9) + ceil(2.1) + min(10, 20)")
    r = engine.evaluate()
    expected = 3 + 3 + 10
    assert abs(r - expected) < 1e-10
    print(f"  floor(3.9) + ceil(2.1) + min(10,20) = {r} (期望 {expected})")
    print("✓ 新函数参与常量折叠正确")

    print("\n--- 测试 8: 条件函数 if ---")
    engine = ExpressionEngine("if(x - 5, x, 0)")
    v1 = {'x': 10.0}
    v2 = {'x': 3.0}
    r1_tree = engine.evaluate_tree(v1)
    r1_vm = engine.evaluate(v1)
    r1_native = engine.evaluate_native(v1)
    r2_tree = engine.evaluate_tree(v2)
    r2_vm = engine.evaluate(v2)
    r2_native = engine.evaluate_native(v2)
    assert abs(r1_tree - 10.0) < 1e-10
    assert abs(r1_vm - 10.0) < 1e-10
    assert abs(r1_native - 10.0) < 1e-10
    assert abs(r2_tree - 0.0) < 1e-10
    assert abs(r2_vm - 0.0) < 1e-10
    assert abs(r2_native - 0.0) < 1e-10
    print(f"  if(10-5, 10, 0) = {r1_vm} (x=10, cond>0, 应得10)")
    print(f"  if(3-5, 3, 0) = {r2_vm} (x=3, cond<=0, 应得0)")
    print("✓ if 函数在三种求值路径结果一致")

    print("\n--- 测试 9: if 常量折叠 ---")
    engine = ExpressionEngine("if(1, 100, 200) + if(-1, 300, 400)")
    r = engine.evaluate()
    assert abs(r - 500.0) < 1e-10
    print(f"  if(1,100,200) + if(-1,300,400) = {r} (期望 500)")
    print("✓ if 函数常量折叠正确")

    print("\n--- 测试 10: clamp 函数 ---")
    engine = ExpressionEngine("clamp(x, 0, 10)")
    cases = [(-5, 0), (5, 5), (15, 10)]
    for val, exp in cases:
        v = {'x': float(val)}
        r_tree = engine.evaluate_tree(v)
        r_vm = engine.evaluate(v)
        r_native = engine.evaluate_native(v)
        assert abs(r_tree - exp) < 1e-10
        assert abs(r_vm - exp) < 1e-10
        assert abs(r_native - exp) < 1e-10
    print(f"  clamp(-5,0,10)={engine.evaluate({'x':-5})}, clamp(5,0,10)={engine.evaluate({'x':5})}, clamp(15,0,10)={engine.evaluate({'x':15})}")
    print("✓ clamp 函数在三种求值路径结果一致")

    print("\n--- 测试 11: sign 函数 ---")
    engine = ExpressionEngine("sign(x)")
    cases = [(-7, -1), (0, 0), (3, 1)]
    for val, exp in cases:
        v = {'x': float(val)}
        r_tree = engine.evaluate_tree(v)
        r_vm = engine.evaluate(v)
        r_native = engine.evaluate_native(v)
        assert abs(r_tree - exp) < 1e-10
        assert abs(r_vm - exp) < 1e-10
        assert abs(r_native - exp) < 1e-10
    print(f"  sign(-7)={engine.evaluate({'x':-7})}, sign(0)={engine.evaluate({'x':0})}, sign(3)={engine.evaluate({'x':3})}")
    print("✓ sign 函数在三种求值路径结果一致")

    print("\n--- 测试 12: 常量折叠 ---")
    engine = ExpressionEngine("(2 + 3) * (4 + 5) + x")
    raw_result = engine.evaluate_tree({'x': 1})
    opt_result = engine.evaluate({'x': 1})
    print(f"  原始表达式: (2 + 3) * (4 + 5) + x")
    print(f"  常量折叠后: 5 * 9 + x = 45 + x")
    print(f"  原始树求值 (x=1): {raw_result}")
    print(f"  优化后编译求值 (x=1): {opt_result}")
    assert abs(raw_result - opt_result) < 1e-10
    print("✓ 常量折叠正确")

    print("\n--- 测试 13: 公共子表达式消除 ---")
    expr = "sin(x) * sin(x) + cos(x) * cos(x) + sin(x) * cos(x)"
    engine = ExpressionEngine(expr)
    result = engine.evaluate({'x': 0.5})
    expected = _m.sin(0.5)**2 + _m.cos(0.5)**2 + _m.sin(0.5) * _m.cos(0.5)
    assert abs(result - expected) < 1e-10
    print(f"✓ CSE 正确, 结果 = {result}")

    print("\n--- 测试 14: 一元负号和优先级 ---")
    engine = ExpressionEngine("-x^2 + (-y)^2")
    result = engine.evaluate({'x': 3, 'y': 4})
    expected = -9 + 16
    print(f"  -3^2 + (-4)^2 = {result} (期望 {expected})")
    assert abs(result - expected) < 1e-10
    print("✓ 一元运算符正确")

    print("\n--- 测试 15: 函数调用 ---")
    engine = ExpressionEngine("sqrt(x^2 + y^2)")
    result = engine.evaluate({'x': 3, 'y': 4})
    print(f"  sqrt(3^2 + 4^2) = {result} (期望 5)")
    assert abs(result - 5.0) < 1e-10
    print("✓ 函数调用正确")

    print("\n--- 测试 16: 变量列表 ---")
    engine = ExpressionEngine("a + b * c - d / e")
    vars_list = engine.variables()
    print(f"  变量: {vars_list}")
    assert vars_list == ['a', 'b', 'c', 'd', 'e']
    print("✓ 变量列表正确")

    print("\n--- 测试 17: 原生函数编译正确性 ---")
    engine = ExpressionEngine("sin(x) * cos(x) + sqrt(x^2 + y^2) + min(x, y)")
    v = {'x': 1.5, 'y': 2.5}
    r_tree = engine.evaluate_tree(v)
    r_compiled = engine.evaluate(v)
    r_native = engine.evaluate_native(v)
    print(f"  树遍历:     {r_tree}")
    print(f"  线性编译:   {r_compiled}")
    print(f"  原生Python: {r_native}")
    assert abs(r_tree - r_compiled) < 1e-10
    assert abs(r_tree - r_native) < 1e-10
    print("✓ 三种求值方式结果一致")

    print("\n--- 测试 18: 按列批量求值 ---")
    engine = ExpressionEngine("x + y * 2")
    cols = {'x': [1.0, 2.0, 3.0, 4.0], 'y': [10.0, 20.0, 30.0, 40.0]}
    results = engine.evaluate_columns(cols)
    expected = [1 + 10*2, 2 + 20*2, 3 + 30*2, 4 + 40*2]
    assert all(abs(a - b) < 1e-10 for a, b in zip(results, expected))
    print(f"  x列={cols['x']}, y列={cols['y']}")
    print(f"  结果={results}")
    print("✓ 按列批量求值顺序正确、结果正确")

    print("\n--- 测试 19: 按列批量求值缺列时报错 ---")
    try:
        engine.evaluate_columns({'x': [1.0, 2.0]})
        raise AssertionError("应该报错缺y列")
    except ValueError as e:
        print(f"✓ 缺列时报错正确: {e}")

    print("\n--- 测试 20: evaluate_columns 默认值生效 ---")
    engine = ExpressionEngine("x + y * 2", y=5)
    results = engine.evaluate_columns({'x': [1.0, 2.0, 3.0]})
    expected = [1 + 5*2, 2 + 5*2, 3 + 5*2]
    assert all(abs(a - b) < 1e-10 for a, b in zip(results, expected))
    print(f"  x列=[1,2,3], y默认=5, 结果={results}")
    print("✓ evaluate_columns 默认值正确生效")

    print("\n--- 测试 21: CompiledExpression 缺变量报错 ---")
    engine = ExpressionEngine("a + b")
    compiled = engine.compiled
    try:
        compiled.evaluate({'a': 1.0})
        raise AssertionError("应该报错缺变量b")
    except ValueError as e:
        assert "'b'" in str(e)
        print(f"✓ CompiledExpression 缺变量报错正确: {e}")

    print("\n--- 测试 22: 编译报告可视化 ---")
    engine = ExpressionEngine("(1 + 2) * x + sin(x) * sin(x) + min(x, y)")
    report = engine.compilation_report()
    assert "原始 AST 树" in report
    assert "常量折叠后 AST 树" in report
    assert "公共子表达式" in report
    assert "最终栈式指令序列" in report
    assert "__temp_" in report
    print("✓ 编译报告包含全部 5 个阶段")

    print("\n--- 测试 23: 编译结果结构化导出 ---")
    engine = ExpressionEngine("(1 + 2) * x + sin(x)", x=0)
    stages = engine.compilation_stages()
    assert stages.expression == "(1 + 2) * x + sin(x)"
    assert 'x' in stages.required_variables
    assert 'x' in stages.defaults
    assert stages.raw_ast is not None
    assert stages.folded_ast is not None
    assert len(stages.instructions) > 0
    assert len(stages.constants) > 0
    assert isinstance(stages.variable_map, dict)
    assert isinstance(stages.cse_temp_bindings, list)
    assert stages.temp_count >= 0
    print(f"  表达式: {stages.expression}")
    print(f"  变量: {stages.required_variables}")
    print(f"  默认值: {stages.defaults}")
    print(f"  指令数: {len(stages.instructions)}")
    print(f"  常量池: {stages.constants}")
    print("✓ CompilationStages 结构化导出正确")

    print("\n--- 测试 24: BatchEvaluator 单值调用 ---")
    engine = ExpressionEngine("x + y", y=10)
    ev = engine.evaluator()
    r = ev(x=5)
    assert abs(r - 15.0) < 1e-10
    r2 = ev(x=5, y=20)
    assert abs(r2 - 25.0) < 1e-10
    print(f"  ev(x=5)={r}, ev(x=5,y=20)={r2}")
    print("✓ BatchEvaluator __call__ 正确")

    print("\n--- 测试 25: BatchEvaluator 缺变量报错 ---")
    try:
        ev(x=5)
    except ValueError:
        pass
    engine2 = ExpressionEngine("x + y")
    ev2 = engine2.evaluator()
    try:
        ev2(x=5)
        raise AssertionError("应该报错缺变量y")
    except ValueError as e:
        assert "'y'" in str(e)
        print(f"✓ BatchEvaluator 缺变量报错正确: {e}")

    print("\n--- 测试 26: BatchEvaluator eval_dict_list ---")
    engine = ExpressionEngine("x + y")
    ev = engine.evaluator()
    data = [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}, {'x': 5, 'y': 6}]
    results = ev.eval_dict_list(data)
    assert results == [3.0, 7.0, 11.0]
    print(f"  结果={results}")
    print("✓ BatchEvaluator eval_dict_list 正确")

    print("\n--- 测试 27: BatchEvaluator eval_columns ---")
    engine = ExpressionEngine("x + y", y=5)
    ev = engine.evaluator()
    results = ev.eval_columns({'x': [1.0, 2.0, 3.0]})
    assert results == [6.0, 7.0, 8.0]
    print(f"  x列=[1,2,3], y默认=5, 结果={results}")
    print("✓ BatchEvaluator eval_columns 正确（含默认值）")

    print("\n--- 测试 28: BatchEvaluator eval_arrays ---")
    engine = ExpressionEngine("x + y")
    ev = engine.evaluator()
    results = ev.eval_arrays([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert results == [11.0, 22.0, 33.0]
    print(f"  x=[1,2,3], y=[10,20,30], 结果={results}")
    print("✓ BatchEvaluator eval_arrays 正确")

    print("\n--- 测试 29: BatchEvaluator eval_native_dict_list ---")
    engine = ExpressionEngine("x * y + sign(x)")
    ev = engine.evaluator()
    data = [{'x': 2, 'y': 3}, {'x': -1, 'y': 5}, {'x': 0, 'y': 7}]
    results_vm = ev.eval_dict_list(data)
    results_native = ev.eval_native_dict_list(data)
    assert all(abs(a - b) < 1e-10 for a, b in zip(results_vm, results_native))
    print(f"  VM结果={results_vm}, 原生结果={results_native}")
    print("✓ BatchEvaluator eval_native_dict_list 与 VM 结果一致")

    print("\n--- 测试 30: 复杂条件表达式五路径一致 ---")
    engine = ExpressionEngine("if(x - 3, clamp(x, 0, 10), -1) + sign(y)")
    test_vals = [
        {'x': 5, 'y': 2},
        {'x': 1, 'y': -4},
        {'x': 15, 'y': 0},
        {'x': -2, 'y': 0.5},
    ]
    for v in test_vals:
        r_tree = engine.evaluate_tree(v)
        r_vm = engine.evaluate(v)
        r_native = engine.evaluate_native(v)
        assert abs(r_tree - r_vm) < 1e-10, f"tree={r_tree} vm={r_vm} vars={v}"
        assert abs(r_tree - r_native) < 1e-10, f"tree={r_tree} native={r_native} vars={v}"
    print(f"  {len(test_vals)} 组变量全部通过")
    print("✓ if+clamp+sign 复合表达式在三条路径结果一致")

    print("\n--- 测试 31: 性能对比 ---")
    expr_complex = ("(x + y) * (x + y) + sin(x + y) * cos(x + y) + "
                   "sqrt(x + y + 1) * log(x + y + 2) + exp(x + y) + min(x, y) + floor(x)")
    engine = ExpressionEngine(expr_complex)

    N = 100000
    import random
    random.seed(42)
    test_data = [{'x': random.random() * 2, 'y': random.random() * 2} for _ in range(N)]

    print(f"  测试表达式: {expr_complex}")
    print(f"  数据点数: {N}")

    t0 = time.time()
    tree_results = [engine.evaluate_tree(v) for v in test_data]
    t_tree = time.time() - t0

    t0 = time.time()
    compiled_results = engine.evaluate_batch(test_data)
    t_compiled = time.time() - t0

    t0 = time.time()
    native_results = engine.evaluate_batch_native(test_data)
    t_native = time.time() - t0

    t0 = time.time()
    ev = engine.evaluator()
    evaluator_results = ev.eval_dict_list(test_data)
    t_evaluator = time.time() - t0

    match1 = all(abs(a - b) < 1e-10 for a, b in zip(tree_results, compiled_results))
    match2 = all(abs(a - b) < 1e-10 for a, b in zip(tree_results, native_results))
    match3 = all(abs(a - b) < 1e-10 for a, b in zip(tree_results, evaluator_results))
    print(f"  原始树遍历:          {t_tree:.4f} 秒")
    print(f"  栈式VM线性求值:     {t_compiled:.4f} 秒  (vs树遍历 {t_tree/t_compiled:.2f}x)")
    print(f"  原生Python函数编译: {t_native:.4f} 秒  (vs树遍历 {t_tree/t_native:.2f}x)")
    print(f"  BatchEvaluator:     {t_evaluator:.4f} 秒  (vs树遍历 {t_tree/t_evaluator:.2f}x)")
    print(f"  结果一致: {'✓' if match1 and match2 and match3 else '✗'}")
    assert match1 and match2 and match3

    print("\n" + "=" * 70)
    print("所有测试通过!")
    print("=" * 70)


def explain_design():
    print("\n" + "=" * 70)
    print("设计说明")
    print("=" * 70)

    print("""
1. 常量折叠 (Constant Folding)
   原理：在编译期遍历AST，当发现某个子树的所有节点都是常量时，
   直接计算出结果并用一个常量节点替换该子树。
   
   例子：表达式 "(2 + 3) * x" 中的 "(2 + 3)" 子树全是常量，
   编译时直接计算为 5，表达式简化为 "5 * x"。
   
   好处：
   - 减少运行时计算量
   - 可能触发后续的代数化简（如乘0、乘1、加0等）
   - 在 [ConstantFolder.fold](file:///d:/trae-bz/TraeProjects/8372/expression_engine.py) 中实现
""")

    print("""
2. 公共子表达式消除 (Common Subexpression Elimination, CSE)
   原理：给每个子树计算"结构哈希"——包含节点类型、运算符、以及
   递归的子节点哈希值。如果两个子树结构完全相同（类型、操作符、
   子节点都相同），它们的哈希值相同，可判定为等价计算。
   
   实现：
   - 第一次出现某个表达式时，为它分配一个临时变量 __temp_N
   - 以后再遇到相同结构的表达式时，直接引用该临时变量
   - 利用 [ASTNode.structural_hash](file:///d:/trae-bz/TraeProjects/8372/expression_engine.py) 
     和 [CSE._intern](file:///d:/trae-bz/TraeProjects/8372/expression_engine.py) 实现
   
   例子："sin(x)*sin(x) + cos(x)*cos(x)" 中 sin(x) 和 cos(x) 
   各出现两次，优化后只各计算一次，存入临时变量复用。
""")

    print("""
3. 表达式树线性化为栈式指令
   将树形结构的AST后序遍历（Post-order Traversal），
   转成一串线性的栈式虚拟机指令：
   
   AST:        +
             /   \\
            *     *
           / \\   / \\
          a   b c   d
   
   线性指令:
     LOAD_VAR 0   # 将 a 压栈
     LOAD_VAR 1   # 将 b 压栈
     MUL          # 弹出两个，计算 a*b，结果压栈
     LOAD_VAR 2   # 将 c 压栈
     LOAD_VAR 3   # 将 d 压栈
     MUL          # 弹出两个，计算 c*d，结果压栈
     ADD          # 弹出两个，计算 (a*b)+(c*d)
   
   在 [CodeGenerator.generate](file:///d:/trae-bz/TraeProjects/8372/expression_engine.py) 中实现
""")

    print("""
4. 为什么编译后反复求值更快？
   
   树遍历求值的开销:
   - 每个节点都是一次 Python 函数调用（递归或迭代）
   - 需要检查节点类型（if/elif 链）
   - 需要属性访问（node.left, node.right 等）
   - Python 解释器的函数调用开销大
   
   线性指令求值的开销:
   - 遍历一个扁平列表，内存局部性好（缓存友好）
   - 用栈直接操作数值，避免对象访问
   - 循环开销远小于递归函数调用
   - 常量和变量都映射到数组下标，O(1) 访问
   
   当需要对成百上千万个数据点代入同一表达式时
   （如数值积分、参数扫描、批量评估等），
   编译的收益会被 N 倍放大。
""")

    print("""
5. 变量如何映射到指令里的输入槽位
   
   在代码生成阶段，[CodeGenerator._collect_variables](file:///d:/trae-bz/TraeProjects/8372/expression_engine.py) 
   会遍历整个AST，收集所有变量名，并按出现顺序分配从0开始的整数索引：
   
     表达式: x + y * z - x
     variable_map = {'x': 0, 'y': 1, 'z': 2}
   
   生成指令时:
     x → LOAD_VAR 0
     y → LOAD_VAR 1
     z → LOAD_VAR 2
   
   求值时，用户传入的字典会被转换成一个数组:
     var_values[0] = variables['x']
     var_values[1] = variables['y']
     var_values[2] = variables['z']
   
   这样 LOAD_VAR k 就是简单的数组索引访问 var_values[k]，
   避免了字典查找的哈希开销。
""")


if __name__ == '__main__':
    run_tests()
    explain_design()
