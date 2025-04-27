#include "solution.h"

#include <iostream>
#include <vector>
#include <map>
#include <cctype>
#include <string>

#include <cstdlib>
#include <ctime>


//std::vector<std::string> variables = {"a", "b", "c", "x", "y", "z"};
//std::vector<std::string> operators = {"+", "-", "*", "/"};
//
//// Generates a random number or variable
//char getRandomOperand() {
//    return (rand() % ('z' - 'a' + 1)) + 'a';
//}
//
//// Generates a random operator
//std::string getRandomOperator() {
//    return operators[rand() % operators.size()];
//}
//
//// Recursively generates a random expression with a given number of operations
//std::string generateExpression(int ops) {
//    if (ops == 0) return std::string(1, getRandomOperand());
//
//    std::string left = generateExpression(ops / 2);
//    std::string right = generateExpression(ops / 2);
//    std::string oper = getRandomOperator();
//
//    return "(" + left + " " + oper + " " + right + ")";
//}

int main() {
    //srand(time(0));
    //int numOps = 10;  // Number of operations in the expression
    //std::string expression = generateExpression(numOps);
    //std::cout << "Random Expression: " << expression << std::endl;

  const std::string expression = "((((z + u) + (r + q)) + ((a * t) - (l / z))) * (((e - k) / (g + n)) + ((r - n) - (x * k))))";
  auto bytecode = parse_expression(expression );
  std::vector<int> variable_values(kVarCnt);
  std::fill(variable_values.begin(), variable_values.end(), 1);
  std::cout << "Expression Value: " << interpret_bytecode(bytecode, variable_values) << std::endl;
  std::cout << "Opt Expression Value: " << interpret_bytecode_opt(bytecode, variable_values) << std::endl;
  return 0;
}
