#include <string>
#include <vector>

enum BytecodeOp { PUSH, ADD, SUB, MUL, DIV, LOAD_VAR };

struct Bytecode {
  BytecodeOp op;
  int operand = 0;
};

constexpr int kVarCnt = 'z' - 'a' + 1;

std::vector<Bytecode> parse_expression(const std::string& expression);
int interpret_bytecode(const std::vector<Bytecode>& bytecode, const std::vector<int>& variable_values);
int interpret_bytecode_opt(const std::vector<Bytecode>& bytecode, const std::vector<int>& variable_values);
