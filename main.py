
import re
import math
import random
from time import sleep
from typing import Any, Union, List, Tuple
from collections import deque
from sympy import symbols, sympify
from DynamicExecutor import *

'''用dfs来派生example'''
def generate_example_dfs(start_symbol, rule_map, nonterminals):
    stack = [start_symbol]  # 栈用于处理非终结符

    while stack:
        current_symbol = stack.pop()

        if current_symbol in nonterminals:
            # 随机选择适用规则进行展开
            if current_symbol in rule_map:
                chosen_rule = random.choice(rule_map[current_symbol])
                right_side = chosen_rule['rules'][::-1]  # 逆序加入栈以保持顺序
                stack.extend(right_side)

                # 处理语义动作部分
                actions = chosen_rule.get('actions', [])
                for action in actions:
                    execute_action(action)
        else:
            # 如果是terminal，直接添加到结果
            executor.result.append(current_symbol)
        
    # 对生成的结果进行替换, 将symbol替换为symbol.target中的值(semantics[symbol]['target'])
    for index, symbol in enumerate(executor.result):
        # 判断是否存在symbol.target
        if symbol in executor.variables and 'target' in executor.variables[symbol]:
            replacement = executor.variables[symbol]['target']
            # 判断是否存在symbol.target是否为空
            if replacement:
                executor.result[index] = replacement

    return ' '.join(executor.result)


"""执行actions语句"""
def execute_action(stmt):
    try:
        if ':=' in stmt:
            left, right = stmt.split(':=')
            obj_name, attr = left.strip().split('.')
            try:
                result = executor.execute(right.strip())
                executor._set_obj_attr(obj_name, attr, result)
                print(f"✓ {obj_name}.{attr} := {right}")
                # sleep(2)
            except StatementExecutionError as e:
                print(e)
        else:
            try:
                result = executor.execute(stmt)
                print(f"✓ 执行成功: {stmt}")
            except StatementExecutionError as e:
                print(e)
                
    except Exception as e:
        print(f"\n语法错误:\n语句: {stmt}\n错误: {str(e)}")

if __name__ == "__main__":
    
    executor = DynamicExecutor(grammar_file_address = "input/grammar2.yml")
    # 将命名空间中的类添加到全局命名空间
    
    globals().update(executor.namespace)

    nonterminals, terminals, rule_map, left_symbols, right_symbols = analyze_syntax(executor.grammar['syntax'])

    # 自动智能判断起点符号
    start_symbol = get_start_symbol(left_symbols, right_symbols)
    
    # 从起点符号开始生成派生例子
    generate_example_dfs(start_symbol, rule_map, nonterminals)

    # 打印最终的example
    print("result:", ' '.join(executor.result))