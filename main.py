import re
import math  # 添加math模块以使用log等函数
import random
from time import sleep
from typing import Any, Union, List, Tuple
from collections import deque
from sympy import symbols, sympify
from DynamicExecutor import *

def calculate_weight(weight_expr, parent_dif, calculated_weights=None):
    """计算weight表达式的值"""
    if isinstance(weight_expr, (int, float)):
        return float(weight_expr)
    elif isinstance(weight_expr, str) and weight_expr.startswith('lambda'):
        try:
            # 使用executor的namespace和global_context
            global_dict = {}
            global_dict.update(executor.namespace)
            global_dict.update(executor.global_context)
            
            # 创建lambda函数
            lambda_func = eval(weight_expr, global_dict)
            
            if calculated_weights:
                # 解析lambda表达式中的参数
                param_str = weight_expr[weight_expr.index('lambda')+6:weight_expr.index(':')].strip()
                params = [p.strip() for p in param_str.split(',')]
                
                # 构建参数列表
                args = [parent_dif]  # 第一个参数总是parent_dif
                for param in params[1:]:  # 跳过parent_dif
                    # 检查参数是否在calculated_weights中
                    param_value = calculated_weights.get(param.strip(), None)
                    if param_value is None:
                        print(f"Warning: Parameter {param} not found in calculated weights")
                        print(f"Available weights: {calculated_weights}")
                    args.append(param_value)
                
                # print(f"Debug - Expression: {weight_expr}")
                # print(f"Debug - Parameters: {params}")
                # print(f"Debug - Arguments: {args}")
                
                return lambda_func(*args)
            return lambda_func(parent_dif)
                
        except Exception as e:
            # print(f"Error in weight calculation:")
            # print(f"Expression: {weight_expr}")
            # print(f"Parent difficulty: {parent_dif}")
            # print(f"Calculated weights: {calculated_weights}")
            # print(f"Error: {str(e)}")
            return 0
    return 0

def generate_example_dfs(start_symbol, rule_map, nonterminals):
    stack = [(start_symbol, None)]
    threshold = executor.get_constant('threshold')

    while stack:
        current_symbol, parent_dif = stack.pop()

        if current_symbol in nonterminals:
            if current_symbol in rule_map:
                available_rules = [rule for rule in rule_map[current_symbol] 
                                 if not rule.get('marked', False)]
                
                while available_rules:
                    chosen_rule = random.choice(available_rules)
                    weights = chosen_rule.get('weight', {})
                    calculated_weights = {}
                    is_valid = True

                    try:
                        # 计算并验证每个子节点的权重
                        for child, weight_expr in weights.items():
                            weight = calculate_weight(weight_expr, parent_dif, calculated_weights)
                            
                            if weight <= threshold:
                                is_valid = False
                                break
                            calculated_weights[child] = weight
                            # 设置难度值到变量中
                            executor._set_obj_attr(child, 'dif', weight)

                    except Exception as e:
                        print(f"Error in rule '{chosen_rule.get('rule', 'unknown rule')}': {e}")
                        chosen_rule['marked'] = True
                        available_rules.remove(chosen_rule)
                        continue

                    if is_valid:
                        right_side = chosen_rule['rules'][::-1]
                        for symbol in right_side:
                            symbol_dif = calculated_weights.get(symbol, parent_dif)
                            stack.append((symbol, symbol_dif))

                        # 处理actions（如果存在）
                        actions = chosen_rule.get('actions', [])
                        if actions:
                            for action in actions:
                                execute_action(action)
                        break
                    else:
                        chosen_rule['marked'] = True
                        available_rules.remove(chosen_rule)

                if not available_rules:
                    raise ValueError(f"No valid rules available for {current_symbol} with difficulty {parent_dif}")
        else:
            executor.result.append(current_symbol)
    
    # 处理替换
    for index, symbol in enumerate(executor.result):
        if symbol in executor.variables and 'target' in executor.variables[symbol]:
            replacement = executor.variables[symbol]['target']
            if replacement:
                executor.result[index] = replacement

    return ' '.join(executor.result)

def execute_action(stmt):
    try:
        if ':=' in stmt:
            left, right = stmt.split(':=')
            obj_name, attr = left.strip().split('.')
            try:
                result = executor.execute(right.strip())
                executor._set_obj_attr(obj_name, attr, result)
                # print(f"✓ {obj_name}.{attr} := {right}")
                # print()
                print((f"✓ executed sucessfully: {obj_name}.{attr} := {right}"))
                # sleep(2)
            except StatementExecutionError as e:
                print(e)
        else:
            try:
                result = executor.execute(stmt)
                # print(f"✓ 执行成功: {stmt}")
                print((f"✓ executed sucessfully: {stmt}"))
            except StatementExecutionError as e:
                print(e)
                
    except Exception as e:
        print(f"\n语法错误:\n语句: {stmt}\n错误: {str(e)}")

if __name__ == "__main__":
    executor = DynamicExecutor(grammar_file_address = "input/grammar2.yml")
    globals().update(executor.namespace)

    nonterminals, terminals, rule_map, left_symbols, right_symbols = analyze_syntax(executor.grammar['syntax'])
    start_symbol = get_start_symbol(left_symbols, right_symbols)
    
    try:
        result = generate_example_dfs(start_symbol, rule_map, nonterminals)
        print("Final result:", result)
        # print("\nVariables state:")
        # for var_name, var_attrs in executor.variables.items():
        #     print(f"{var_name}: {var_attrs}")
    except ValueError as e:
        print(f"\nGeneration failed: {e}")