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
    if calculated_weights is None:
        calculated_weights = {}
        
    # 创建一个基础符号到权重的映射
    base_weights = {}
    for symbol, weight in calculated_weights.items():
        base_symbol = symbol.split('_')[0]
        base_weights[base_symbol] = weight

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
                    # 使用基础符号名查找权重
                    param_value = base_weights.get(param.strip(), None)
                    if param_value is None:
                        print(f"Warning: Parameter {param} not found in calculated weights")
                        print(f"Available weights: {calculated_weights}")
                    args.append(param_value)
                
                return lambda_func(*args)
            return lambda_func(parent_dif)
                
        except Exception as e:
            print(f"Error in weight calculation: {str(e)}")
            return 0
    return 0

def generate_example_dfs(start_symbol, rule_map, nonterminals):
    symbol_counters = {}  # 用于追踪每个符号的索引
    weight_map = {}  # 用于存储每个符号实例的权重
    
    def get_indexed_symbol(symbol):
        base_symbol = symbol.split('_')[0]
        if base_symbol not in symbol_counters:
            symbol_counters[base_symbol] = 0
        current_index = symbol_counters[base_symbol]
        symbol_counters[base_symbol] += 1
        # 更新 executor 中的索引信息
        executor.symbol_indices[base_symbol] = current_index
        return f"{base_symbol}_{current_index}"


    # 为起始符号添加索引
    indexed_start = get_indexed_symbol(start_symbol)
    stack = [(indexed_start, None)]
    threshold = executor.get_constant('threshold')
    weight_map = {}
    
    # print(f"\n=== Starting Generation with {start_symbol} ===\n")

    while stack:
        current_symbol, parent_dif = stack.pop()
        # print(f"\nProcessing symbol: {current_symbol}")
        # print(f"Current variables: {executor.variables}")
        
        if current_symbol.split('_')[0] in nonterminals:
            matching_rules = []
            for rule_key, rules in rule_map.items():
                if rule_key.split('_')[0] == current_symbol.split('_')[0]:
                    matching_rules.extend([r for r in rules if not r.get('marked', False)])
            
            if matching_rules:
                while matching_rules:
                    chosen_rule = random.choice(matching_rules)
                    # print(f"Chosen rule: {chosen_rule['rule']}")
                    indexed_rules = [get_indexed_symbol(symbol) for symbol in chosen_rule['rules']]
                    # print(f"Indexed rules: {indexed_rules}")
                    weights = chosen_rule.get('weight', {})
                    calculated_weights = {}
                    is_valid = True

                    try:
                        # 计算并验证每个子节点的权重
                        for symbol in indexed_rules:
                            base = symbol.split('_')[0]
                            
                            if base in weights:
                                weight_expr = weights[base]
                                weight = calculate_weight(weight_expr, parent_dif, weight_map)  # 使用 weight_map
                                
                                if weight < threshold:
                                    is_valid = False
                                    break
                                    
                                calculated_weights[symbol] = weight
                                weight_map[symbol] = weight  # 更新全局 weight_map

                    except Exception as e:
                        print(f"\nError in rule: {e}")
                        chosen_rule['marked'] = True
                        matching_rules.remove(chosen_rule)
                        continue

                    if is_valid:
                        for symbol in indexed_rules[::-1]:
                            symbol_weight = weight_map.get(symbol)
                            stack.append((symbol, symbol_weight))

                        # 处理 actions 时使用 weight_map
                        actions = chosen_rule.get('actions', [])
                        if actions:
                            for action in actions:
                                execute_action(action, weight_map)  # 传递 weight_map
                        break
                    else:
                        chosen_rule['marked'] = True
                        matching_rules.remove(chosen_rule)

                if not matching_rules:
                    raise ValueError(f"No valid rules available for {current_symbol} with difficulty {parent_dif}")
        else:
            executor.result.append(current_symbol)
    
    # 处理替换
    # print("\n=== Final Replacements ===")
    for index, symbol in enumerate(executor.result):
        base_symbol = symbol.split('_')[0]
        symbol_index = symbol  # 保存完整的带索引的符号名
        
        # 首先检查带索引的版本
        if symbol_index in executor.variables and 'target' in executor.variables[symbol_index]:
            replacement = executor.variables[symbol_index]['target']
            if replacement:
                executor.result[index] = replacement
            else:
                executor.result[index] = base_symbol
        # 如果找不到带索引的版本，检查基础符号
        elif base_symbol in executor.variables and 'target' in executor.variables[base_symbol]:
            replacement = executor.variables[base_symbol]['target']
            if replacement:
                executor.result[index] = replacement
            else:
                executor.result[index] = base_symbol
        else:
            executor.result[index] = base_symbol

    # print("\n=== Final Weight Map ===")
    print(weight_map)
    
    return ' '.join(executor.result)

def execute_action(stmt, weight_map):  # 添加 weight_map 参数
    try:
        if ':=' in stmt:
            left, right = stmt.split(':=')
            obj_name, attr = left.strip().split('.')
            try:
                result = executor.execute(right.strip(), weight_map)  # 传递 weight_map
                executor._set_obj_attr(obj_name, attr, result)
                # print(f"✓ {obj_name}.{attr} := {right}")
                # print((f"✓ executed sucessfully: {obj_name}.{attr} := {right}"))
            except StatementExecutionError as e:
                print(e)
        else:
            try:
                result = executor.execute(stmt, weight_map)  # 传递 weight_map
                # print(f"✓ 执行成功: {stmt}")
                # print((f"✓ executed sucessfully: {stmt}"))
            except StatementExecutionError as e:
                print(e)
                
    except Exception as e:
        print(f"\n语法错误:\n语句: {stmt}\n错误: {str(e)}")

if __name__ == "__main__":
    executor = DynamicExecutor(grammar_file_address="input/grammar1.yml")
    globals().update(executor.namespace)

    nonterminals, terminals, rule_map, left_symbols, right_symbols = analyze_syntax(executor.grammar['syntax'])
    # 注册所有有效符号
    executor.register_symbols(nonterminals | terminals)
    
    start_symbol = get_start_symbol(left_symbols, right_symbols)
    
    try:
        result = generate_example_dfs(start_symbol, rule_map, nonterminals)
        print("Final result:", result)
    except ValueError as e:
        print(f"\nGeneration failed: {e}")

    # if 'T_1' in executor.variables != None and 'T_1' in executor.variables:
    #     print("\n",executor.variables['T_0'])
    #     print("\n",executor.variables['T_1']) 
    # print("\n",executor.variables) 