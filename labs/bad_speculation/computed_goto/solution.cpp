#include "solution.h"

#include <memory>
#include <stdexcept>
	
enum TokenType { NUMBER, VARIABLE, OPERATOR, LPAREN, RPAREN, SENTINEL };

struct Token {
  TokenType type;
  std::string value;
};

std::vector<Token> tokenize(const std::string& expr) {
  std::vector<Token> tokens;
  for (size_t i = 0; i < expr.size(); ++i) {
    if (std::isdigit(expr[i])) {
      std::string num;
      while (i < expr.size() && std::isdigit(expr[i])) {
        num += expr[i++];
      }
      tokens.push_back({NUMBER, num});
      --i;
    } else if (expr[i] >= 'a' && expr[i] <= 'z') {
      tokens.push_back({VARIABLE, std::string(1, expr[i])});
    } else if (expr[i] == '+' || expr[i] == '-' || expr[i] == '*' || expr[i] == '/') {
      tokens.push_back({OPERATOR, std::string(1, expr[i])});
    } else if (expr[i] == '(') {
      tokens.push_back({LPAREN, "("});
    } else if (expr[i] == ')') {
      tokens.push_back({RPAREN, ")"});
    } else if (!std::isspace(expr[i])) {
      throw std::runtime_error("Unexpected token: '" + expr.substr(i) + "'");
    }
  }
  return tokens;
}

struct ASTNode;
using ASTNodePtr = std::unique_ptr<ASTNode>;

struct ASTNode {
  Token token;
  ASTNodePtr left;
  ASTNodePtr right;

  ASTNode(Token token, ASTNodePtr left, ASTNodePtr right) :
    token(token), left(std::move(left)), right(std::move(right))
  {}
  ASTNode(Token token) : token(token) {}
};

class Parser {
public:
  explicit Parser(const std::vector<Token>& tokens) : tokens(tokens), pos(0) {}
  ASTNodePtr parse() {
    return parse_expression();
  }

private:
  const Token& token() const {
    return pos < tokens.size() ? tokens[pos] : sentinel;
  }
  void next_token() {
    ++pos;
  }
  ASTNodePtr parse_expression() {
    auto left = parse_term();
    while (token().type == OPERATOR && (token().value == "+" || token().value == "-")) {
      const auto& op = token();
      next_token();
      auto right = parse_term();
      left = std::make_unique<ASTNode>(op, std::move(left), std::move(right));
    }
    return left;
  }
  ASTNodePtr parse_term() {
    auto left = parse_factor();
    while (token().type == OPERATOR && (token().value == "*" || token().value == "/")) {
      const auto& op = token();
      next_token();
      auto right = parse_factor();
      left = std::make_unique<ASTNode>(op, std::move(left), std::move(right));
    }
    return left;
  }
  ASTNodePtr parse_factor() {
    if (token().type == NUMBER || token().type == VARIABLE) {
      auto node = std::make_unique<ASTNode>(token());
      next_token();
      return node;
    }
    if (token().type == LPAREN) {
      next_token();
      auto node = parse_expression();
      if (token().type != RPAREN) {
        throw std::runtime_error("Unbalanced parentheses");
      }
      next_token();
      return node;
    }
    throw std::runtime_error("Invalid expression");
  }

  const std::vector<Token>& tokens;
  size_t pos;
  static const Token sentinel;
};

const Token Parser::sentinel = {SENTINEL};

std::vector<Bytecode> generate_bytecode(const ASTNode* node) {
  std::vector<Bytecode> bytecode;
  if (node->token.type == NUMBER) {
    bytecode.push_back({PUSH, stoi(node->token.value)});
  } else if (node->token.type == VARIABLE) {
    bytecode.push_back({LOAD_VAR, node->token.value[0] - 'a'});
  } else if (node->token.type == OPERATOR) {
    auto left = generate_bytecode(node->left.get());
    auto right = generate_bytecode(node->right.get());
    bytecode.insert(bytecode.end(), left.begin(), left.end());
    bytecode.insert(bytecode.end(), right.begin(), right.end());
    if (node->token.value == "+") bytecode.push_back({ADD});
    else if (node->token.value == "-") bytecode.push_back({SUB});
    else if (node->token.value == "*") bytecode.push_back({MUL});
    else if (node->token.value == "/") bytecode.push_back({DIV});
  }
  return bytecode;
}

std::vector<Bytecode> parse_expression(const std::string& expression) {
  auto tokens = tokenize(expression);
  auto ast = Parser(tokens).parse();
  return generate_bytecode(ast.get());
}

std::vector<double> stack;

int interpret_bytecode(const std::vector<Bytecode>& bytecode, const std::vector<int>& variable_values) {
  for (const auto& instr : bytecode) {
    if (instr.op == PUSH) {
      stack.push_back(instr.operand);
    } else if (instr.op == LOAD_VAR) {
      stack.push_back(variable_values[instr.operand]);
    } else if (instr.op == ADD) {
      const auto right = stack.back(); stack.pop_back();
      const auto left = stack.back(); stack.pop_back();
      stack.push_back(left + right);
    } else if (instr.op == SUB) {
      const auto right = stack.back(); stack.pop_back();
      const auto left = stack.back(); stack.pop_back();
      stack.push_back(left - right);
    } else if (instr.op == MUL) {
      const auto right = stack.back(); stack.pop_back();
      const auto left = stack.back(); stack.pop_back();
      stack.push_back(left * right);
    } else if (instr.op == DIV) {
      const auto right = stack.back(); stack.pop_back();
      const auto left = stack.back(); stack.pop_back();
      stack.push_back(left / right);
    }
  }
  return stack.back();
}

int interpret_bytecode_opt(const std::vector<Bytecode>& bytecode, const std::vector<int>& variable_values) {
  // enum BytecodeOp { PUSH, ADD, SUB, MUL, DIV, LOAD_VAR };
  static void* dispatch_table[] = {&&op_push, &&op_add, &&op_sub, &&op_mul, &&op_div, &&op_load_var};
  for (const auto& instr : bytecode) {
    goto *dispatch_table[instr.op];
    op_push: {
      stack.push_back(instr.operand);
      continue;
    }
    op_add: {
      const auto right = stack.back(); stack.pop_back();
      const auto left = stack.back(); stack.pop_back();
      stack.push_back(left + right);
      continue;
    }
    op_sub: {
      const auto right = stack.back(); stack.pop_back();
      const auto left = stack.back(); stack.pop_back();
      stack.push_back(left - right);
      continue;
    }
    op_mul: {
      const auto right = stack.back(); stack.pop_back();
      const auto left = stack.back(); stack.pop_back();
      stack.push_back(left * right);
      continue;
    }
    op_div: {
      const auto right = stack.back(); stack.pop_back();
      const auto left = stack.back(); stack.pop_back();
      stack.push_back(left / right);
      continue;
    }
    op_load_var: {
      stack.push_back(variable_values[instr.operand]);
      continue;
    }
  }
  return stack.back();
}
