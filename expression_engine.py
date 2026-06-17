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
    BUILTIN_FUNCS: Dict[str, Callable] = {
        'sin': __import__('math').sin,
        'cos': __import__('math').cos,
        'tan': __import__('math').tan,
        'sqrt': __import__('math').sqrt,
        'exp': __import__('math').exp,
        'log': __import__('math').log,
        'abs': abs,
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
            temp_count=self.temp_count
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
            for name, idx in self.variable_map.items():
                if name in variables:
                    var_values[idx] = variables[name]
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


class ExpressionEngine:
    def __init__(self, expression: str):
        self.original_expression = expression
        tokens = Tokenizer(expression).tokenize()
        self.raw_ast = Parser(tokens).parse()
        folded_ast = ConstantFolder.fold(self._clone_ast(self.raw_ast))
        self.cse_result = CSE().eliminate(folded_ast)
        self.optimized_root = self.cse_result.root
        self.temp_bindings = self.cse_result.temp_bindings
        self.compiled = CodeGenerator().generate(self.cse_result)
        self._native_func = None
        self._sorted_vars = self.variables()

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
            if fname in ('sin', 'cos', 'tan', 'sqrt', 'exp', 'log'):
                lines.append(f"{fname} = _m.{fname}")
            elif fname == 'abs':
                lines.append("abs = _b.abs")

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

    def evaluate_native(self, variables: Dict[str, float]) -> float:
        func = self.compile_to_native()
        args = [variables[name] for name in self._sorted_vars]
        return func(*args)

    def evaluate_tree(self, variables: Optional[Dict[str, float]] = None) -> float:
        return TreeEvaluator.evaluate(self.raw_ast, variables)

    def evaluate_optimized_tree(self, variables: Optional[Dict[str, float]] = None) -> float:
        if variables is None:
            variables = {}
        temp_values = {}
        for i, binding in enumerate(self.temp_bindings):
            temp_values[f'__temp_{i}'] = TreeEvaluator._eval(binding, {**variables, **temp_values})
        return TreeEvaluator._eval(self.optimized_root, {**variables, **temp_values})

    def evaluate(self, variables: Optional[Dict[str, float]] = None) -> float:
        return self.compiled.evaluate(variables)

    def evaluate_batch(self, variable_list: List[Dict[str, float]]) -> List[float]:
        var_names = self._sorted_vars
        var_map = self.compiled.variable_map
        indices = [var_map[n] for n in var_names]
        num_vars = len(var_names)
        compiled = self.compiled
        results = []
        append = results.append
        for vars_dict in variable_list:
            arr = [0.0] * num_vars
            for name, idx in zip(var_names, indices):
                arr[idx] = vars_dict[name]
            append(compiled.evaluate_arr(arr))
        return results

    def evaluate_batch_native(self, variable_list: List[Dict[str, float]]) -> List[float]:
        func = self.compile_to_native()
        var_names = self._sorted_vars
        return [func(*(v[n] for n in var_names)) for v in variable_list]

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


def run_tests():
    print("=" * 70)
    print("表达式树编译与求值优化引擎 - 测试套件")
    print("=" * 70)

    print("\n--- 测试 1: 基本表达式求值 ---")
    engine = ExpressionEngine("2 + 3 * 4")
    assert abs(engine.evaluate() - 14.0) < 1e-10
    print("✓ 2 + 3 * 4 = 14")

    print("\n--- 测试 2: 变量求值 ---")
    engine = ExpressionEngine("x + y * 2")
    result = engine.evaluate({'x': 5, 'y': 3})
    assert abs(result - 11.0) < 1e-10
    print(f"✓ x + y * 2  (x=5, y=3) = {result}")

    print("\n--- 测试 3: 常量折叠 ---")
    engine = ExpressionEngine("(2 + 3) * (4 + 5) + x")
    raw_result = engine.evaluate_tree({'x': 1})
    opt_result = engine.evaluate({'x': 1})
    print(f"  原始表达式: (2 + 3) * (4 + 5) + x")
    print(f"  常量折叠后: 5 * 9 + x = 45 + x")
    print(f"  原始树求值 (x=1): {raw_result}")
    print(f"  优化后编译求值 (x=1): {opt_result}")
    assert abs(raw_result - opt_result) < 1e-10
    print("✓ 常量折叠正确")

    print("\n--- 测试 4: 公共子表达式消除 ---")
    expr = "sin(x) * sin(x) + cos(x) * cos(x) + sin(x) * cos(x)"
    engine = ExpressionEngine(expr)
    result = engine.evaluate({'x': 0.5})
    import math
    expected = math.sin(0.5)**2 + math.cos(0.5)**2 + math.sin(0.5) * math.cos(0.5)
    print(f"  表达式: {expr}")
    print(f"  指令序列:")
    for i, instr in enumerate(engine.instructions()):
        print(f"    {i:3d}: {instr}")
    assert abs(result - expected) < 1e-10
    print(f"✓ CSE 正确, 结果 = {result}")

    print("\n--- 测试 5: 一元负号和优先级 ---")
    engine = ExpressionEngine("-x^2 + (-y)^2")
    result = engine.evaluate({'x': 3, 'y': 4})
    expected = -9 + 16
    print(f"  -3^2 + (-4)^2 = {result} (期望 {expected})")
    assert abs(result - expected) < 1e-10
    print("✓ 一元运算符正确")

    print("\n--- 测试 6: 函数调用 ---")
    engine = ExpressionEngine("sqrt(x^2 + y^2)")
    result = engine.evaluate({'x': 3, 'y': 4})
    print(f"  sqrt(3^2 + 4^2) = {result} (期望 5)")
    assert abs(result - 5.0) < 1e-10
    print("✓ 函数调用正确")

    print("\n--- 测试 7: 变量列表 ---")
    engine = ExpressionEngine("a + b * c - d / e")
    vars_list = engine.variables()
    print(f"  变量: {vars_list}")
    assert vars_list == ['a', 'b', 'c', 'd', 'e']
    print("✓ 变量列表正确")

    print("\n--- 测试 8: 原生函数编译正确性 ---")
    engine = ExpressionEngine("sin(x) * cos(x) + sqrt(x^2 + y^2)")
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

    print("\n--- 测试 9: 性能对比 ---")
    expr_complex = ("(x + y) * (x + y) + sin(x + y) * cos(x + y) + "
                   "sqrt(x + y + 1) * log(x + y + 2) + exp(x + y)")
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

    match1 = all(abs(a - b) < 1e-10 for a, b in zip(tree_results, compiled_results))
    match2 = all(abs(a - b) < 1e-10 for a, b in zip(tree_results, native_results))
    print(f"  原始树遍历:          {t_tree:.4f} 秒")
    print(f"  栈式VM线性求值:     {t_compiled:.4f} 秒  (vs树遍历 {t_tree/t_compiled:.2f}x)")
    print(f"  原生Python函数编译: {t_native:.4f} 秒  (vs树遍历 {t_tree/t_native:.2f}x)")
    print(f"  结果一致: {'✓' if match1 and match2 else '✗'}")
    assert match1 and match2

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
