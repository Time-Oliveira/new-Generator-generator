import re
import math  # 添加math模块以使用log等函数
import random
from time import sleep
from typing import Any, Union, List, Tuple
from collections import deque
import numpy as np
from sympy import symbols, sympify
from DynamicExecutor import *

def format_matrix(matrix):
    """格式化矩阵输出"""
    if isinstance(matrix, np.matrix):
        rows, cols = matrix.shape
        return f"[{rows}x{cols} Matrix]"
    return str(matrix)

def process_final_result(executor, weight_map):
    """处理最终生成结果，显示具体的矩阵值"""
    result = []
    
    for symbol in executor.result:
        base_symbol = symbol.split('_')[0]
        symbol_index = symbol
        
        # 处理操作符
        if base_symbol in {'*', '+', '(', ')'}:
            result.append('\n\n')
            result.append(base_symbol)
            result.append('\n\n')
            continue
            
        # 处理矩阵
        if symbol_index in executor.variables and 'target' in executor.variables[symbol_index]:
            matrix = executor.variables[symbol_index]['target']
            if isinstance(matrix, np.matrix):
                result.append(str(matrix))
            else:
                result.append(str(matrix))
        else:
            result.append(base_symbol)
    
    # 后处理：清理格式
    final_str = ' '.join(result)
    # 清理矩阵字符串中的多余空格
    final_str = re.sub(r'\(\s+\[\[', '([[', final_str)
    final_str = re.sub(r'\]\]\s+\)', ']])', final_str)
    
    return final_str

def parse_condition(condition_str: str) -> tuple:
    """解析条件字符串，返回(左边界, 右边界, 左闭合, 右闭合)"""
    if condition_str is None:
        return None  # 表示无条件限制
        
    # 移除空格
    condition_str = condition_str.strip()
    
    # 获取区间类型
    left_closed = condition_str[0] == '['
    right_closed = condition_str[-1] == ']'
    
    # 提取数值
    values = condition_str[1:-1].split(',')
    left_val = values[0].strip()
    right_val = values[1].strip()
    
    # 处理无穷
    left_bound = float('-inf') if left_val == '~' else float(left_val)
    right_bound = float('inf') if right_val == '~' else float(right_val)
    
    return (left_bound, right_bound, left_closed, right_closed)

def check_condition(value: float, condition: tuple) -> bool:
    """检查值是否满足条件"""
    if condition is None:
        return True
        
    left_bound, right_bound, left_closed, right_closed = condition
    
    # print(f"Checking condition: value={value}, bounds=[{left_bound}, {right_bound}], closed=[{left_closed}, {right_closed}]")
    
    # 检查左边界
    if left_closed:
        if value < left_bound:
            # print(f"Failed left closed boundary check: {value} < {left_bound}")
            return False
    else:
        if value <= left_bound:
            # print(f"Failed left open boundary check: {value} <= {left_bound}")
            return False
            
    # 检查右边界
    if right_closed:
        if value > right_bound:
            # print(f"Failed right closed boundary check: {value} > {right_bound}")
            return False
    else:
        if value >= right_bound:
            # print(f"Failed right open boundary check: {value} >= {right_bound}")
            return False
            
    # print(f"Condition satisfied")
    return True

def calculate_weight(weight_expr, parent_dif, calculated_weights=None):
    """计算weight表达式的值"""
    # print(f"\n=== Weight Calculation Debug ===")
    # print(f"Expression: {weight_expr}")
    # print(f"Parent difficulty: {parent_dif}")
    # print(f"Calculated weights: {calculated_weights}")
    
    if calculated_weights is None:
        calculated_weights = {}
        
    # 创建一个基础符号到权重的映射
    base_weights = {}
    for symbol, weight in calculated_weights.items():
        base_symbol = symbol.split('_')[0]
        if base_symbol not in base_weights:
            base_weights[base_symbol] = weight
            
    # print(f"Base weights: {base_weights}")

    if isinstance(weight_expr, (int, float)):
        # print(f"Returning numeric value: {float(weight_expr)}")
        return float(weight_expr)
    elif isinstance(weight_expr, str) and weight_expr.startswith('lambda'):
        try:
            # 使用executor的namespace和global_context
            global_dict = {}
            global_dict.update(executor.namespace)
            global_dict.update(executor.global_context)
            
            # 创建lambda函数
            lambda_func = eval(weight_expr, global_dict)
            
            # 解析lambda表达式中的参数
            param_str = weight_expr[weight_expr.index('lambda')+6:weight_expr.index(':')].strip()
            params = [p.strip() for p in param_str.split(',')]
            # print(f"Lambda parameters: {params}")
            
            # 构建参数列表
            args = [parent_dif]  # 第一个参数总是parent_dif
            
            # 处理其他参数
            for param in params[1:]:
                param_value = base_weights.get(param.strip(), None)
                # print(f"Looking for parameter {param}: {'found' if param_value is not None else 'not found'}")
                if param_value is None:
                    # print(f"Warning: Parameter {param} not found in base_weights")
                    # print(f"Available base_weights: {base_weights}")
                    return 0
                args.append(param_value)
            
            # print(f"Final arguments for lambda: {args}")
            result = lambda_func(*args)
            # print(f"Lambda result: {result}")
            return result
                
        except Exception as e:
            print(f"Error in weight calculation: {str(e)}")
            print(f"Lambda expression: {weight_expr}")
            print(f"Arguments: {args if 'args' in locals() else 'not created'}")
            return 0
    return 0

def generate_example_dfs(start_symbol, rule_map, nonterminals):
    """
    使用DFS生成示例，包含权重计算和条件判断
    """
    symbol_counters = {}  # 用于追踪每个符号的索引
    weight_map = {}  # 用于存储每个符号实例的权重
    
    def get_indexed_symbol(symbol):
        """为符号生成带索引的版本"""
        base_symbol = symbol.split('_')[0]
        if base_symbol not in symbol_counters:
            symbol_counters[base_symbol] = 0
        current_index = symbol_counters[base_symbol]
        symbol_counters[base_symbol] += 1
        executor.symbol_indices[base_symbol] = current_index
        return f"{base_symbol}_{current_index}"

    # 初始化
    indexed_start = get_indexed_symbol(start_symbol)
    stack = [(indexed_start, executor.get_constant('general_difficult'))]  # 使用初始难度
    threshold = executor.get_constant('threshold')
    
    # print(f"\n=== Starting Generation with {start_symbol} ===")
    
    while stack:
        current_symbol, parent_dif = stack.pop()
        # print(f"\nProcessing symbol: {current_symbol} with difficulty: {parent_dif}")
        
        if current_symbol.split('_')[0] in nonterminals:
            # 收集所有可能的规则
            valid_rules = []
            
            # 检查所有可能的规则
            for rule_key, rules in rule_map.items():
                if rule_key.split('_')[0] == current_symbol.split('_')[0]:
                    # print(f"\nChecking rules for {rule_key}")
                    for rule in rules:
                        if rule.get('marked', False):
                            continue
                            
                        # print(f"\nEvaluating rule: {rule['rule']}")
                        # print(rule_key, rules)
                        # 检查条件
                        condition_str = rule.get('condition')
                        # print(f"Checking condition: {condition_str}")
                        if condition_str:
                            condition = parse_condition(condition_str)
                            # print(f"Checking condition: {condition_str}")
                            # print(f"Checking condition: {condition}")
                            if parent_dif is not None:
                                if check_condition(parent_dif, condition):
                                    # print(f"Rule {rule['rule']} condition met")
                                    valid_rules.append(rule)
                                # else:
                                    # print(f"Rule {rule['rule']} condition not met")
                        else:
                            # print(f"No condition for rule {rule['rule']}, adding to valid rules")
                            valid_rules.append(rule)
            
            # print(f"\nValid rules: {[rule['rule'] for rule in valid_rules]}")
            
            if valid_rules:
                # 从有效规则中随机选择一个
                chosen_rule = random.choice(valid_rules)
                # print(f"Chosen rule: {chosen_rule['rule']}")
                
                # 计算权重并验证
                indexed_rules = [get_indexed_symbol(symbol) for symbol in chosen_rule['rules']]
                weights = chosen_rule.get('weight', {})
                calculated_weights = {}
                is_valid = True
                
                try:
                    # 计算并验证每个子节点的权重
                    for symbol in indexed_rules:
                        base = symbol.split('_')[0]
                        if base in weights:
                            weight_expr = weights[base]
                            weight = calculate_weight(weight_expr, parent_dif, weight_map)
                            # print(f"Calculated weight for {symbol}: {weight}")
                            
                            if weight < threshold:
                                # print(f"Weight {weight} is below threshold {threshold}")
                                is_valid = False
                                break
                                
                            calculated_weights[symbol] = weight
                            weight_map[symbol] = weight
                    
                    if is_valid:
                        # 只有在确认规则有效后才将符号压入栈
                        for symbol in indexed_rules[::-1]:  # 逆序压栈
                            symbol_weight = weight_map.get(symbol)
                            stack.append((symbol, symbol_weight))
                        
                        # 处理动作
                        actions = chosen_rule.get('actions', [])
                        if actions:
                            for action in actions:
                                execute_action(action, weight_map)
                    else:
                        raise ValueError(f"Weight below threshold for rule {chosen_rule['rule']}")
                        
                except Exception as e:
                    # print(f"Error processing rule: {e}")
                    chosen_rule['marked'] = True
                    continue
            
            if not valid_rules:
                raise ValueError(f"No valid rules available for {current_symbol} with difficulty {parent_dif}")
        else:
            executor.result.append(current_symbol)
    
    # 处理最终结果的替换
    print("\n=== Final Weight Map ===")
    print(weight_map)
    
    return process_final_result(executor, weight_map)

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
    executor = DynamicExecutor(grammar_file_address="input/grammar2.yml")
    globals().update(executor.namespace)

    nonterminals, terminals, rule_map, left_symbols, right_symbols = analyze_syntax(executor.grammar['syntax'])
    # 注册所有有效符号
    # print(rule_map)
    executor.register_symbols(nonterminals | terminals)
    
    start_symbol = get_start_symbol(left_symbols, right_symbols)
    
    try:
        result = generate_example_dfs(start_symbol, rule_map, nonterminals)
        print("Final result:\n", result)
    except ValueError as e:
        print(f"\nGeneration failed: {e}")

    # if 'T_1' in executor.variables != None and 'T_1' in executor.variables:
    #     print("\n",executor.variables['T_0'])
    #     print("\n",executor.variables['T_1']) 
    # print("\n",executor.variables) 