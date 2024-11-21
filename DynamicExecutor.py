import importlib
import inspect
import yaml
import re
import sys
import math
import numpy
import statistics
from typing import Any, Dict, Callable
from typing import Any, Dict
from setup.readin import *
from CustomClass import CustomClass
from ConstantTable import ConstantTable
from symboltable.symboltable import *
from ConstantTable.ConstantTable import *

def create_function_with_context(func_name: str, code: str, namespace: Dict[str, Any] = None) -> Callable:
    """
    Create a function with the given namespace context
    """
    namespace = namespace or {}
    
    # Execute the function definition
    exec(code.strip(), namespace)
    
    # Get and return the function object
    func = namespace[func_name]
    func._source = code
    
    return func

def extract_functions_from_yaml(yaml_content: str, constants: Dict[str, Any] = None) -> Dict[str, Callable]:
    try:
        # Create GrammarLoader instance and process imports first
        loader = GrammarLoader()
        yaml_data = yaml.safe_load(yaml_content)
        
        # Process imports and get the namespace
        imports = yaml_data.get('imports', [])
        namespace = loader.dynamic_import(imports)
        
        # Add the imported namespace to globals
        globals().update(namespace)
        
        # Add constants to namespace
        if constants:
            namespace.update(constants)
        
        functions_data = yaml_data.get('functions', {})
        
        if not functions_data:
            return {}
        
        external_functions = {}
        
        for func_name, func_info in functions_data.items():
            implementation = func_info.get('implementation', '').strip()
            if not implementation:
                continue
                
            func_match = re.search(r'def\s+(\w+)\s*\([^)]*\):', implementation)
            if not func_match:
                continue
                
            actual_func_name = func_match.group(1)
            
            try:
                func_obj = create_function_with_context(
                    actual_func_name,
                    implementation,
                    namespace
                )
                
                external_functions[func_name] = func_obj
                
            except Exception as e:
                raise ValueError(f"Error processing function {func_name}: {e}")
        
        return external_functions, namespace  # Return both functions and namespace
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML content: {e}")

def load_functions_from_yaml_file(file_path: str, constants: Dict[str, Any] = None) -> Dict[str, Callable]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
        functions, namespace = extract_functions_from_yaml(yaml_content, constants)
        return functions, namespace
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading functions from YAML: {e}")

class StatementExecutionError(Exception):
    def __init__(self, statement, error_type, error_message, modified_expr=None, variables=None):
        self.statement = statement
        self.error_type = error_type
        self.error_message = error_message
        self.modified_expr = modified_expr
        self.variables = variables
        self.analysis = self._analyze_error()
        
        msg = self._format_error_message()
        super().__init__(msg)
    
    def _analyze_error(self):
        analysis = {
            'cause': None,
            'suggestion': None,
            'variable_state': {}
        }
        
        if isinstance(self.error_message, str):
            # 类型错误分析
            if "can only concatenate str" in self.error_message:
                analysis['cause'] = "类型不匹配：试图将字符串与数字进行连接"
                analysis['suggestion'] = "请确保操作数类型一致，可能需要进行类型转换"
            
            # 名称错误分析
            elif "name" in self.error_message and "is not defined" in self.error_message:
                undefined_var = self.error_message.split("'")[1]
                analysis['cause'] = f"未定义的变量或函数：{undefined_var}"
                analysis['suggestion'] = f"请检查 '{undefined_var}' 是否已正确定义或导入"
            
            # 数学运算错误分析
            elif "math domain error" in self.error_message:
                analysis['cause'] = "数学运算域错误"
                analysis['suggestion'] = "请检查数学函数的输入值是否在有效范围内"
            
            # 索引错误分析
            elif "index" in self.error_message and "out of range" in self.error_message:
                analysis['cause'] = "索引越界"
                analysis['suggestion'] = "请检查数组索引是否在有效范围内"
                
            # 类型转换错误
            elif "Python int too large to convert" in self.error_message:
                analysis['cause'] = "数值转换溢出"
                analysis['suggestion'] = "数值超出了允许的范围，请检查计算过程或使用适当的数据类型"
                
            # 属性错误分析
            elif "has no attribute" in self.error_message:
                analysis['cause'] = "访问了不存在的属性"
                analysis['suggestion'] = "请检查对象属性名称是否正确"

        # 添加变量状态分析
        if self.variables:
            for obj_name, obj_dict in self.variables.items():
                for attr_name, value in obj_dict.items():
                    analysis['variable_state'][f"{obj_name}.{attr_name}"] = {
                        'value': value,
                        'type': type(value).__name__
                    }

        return analysis
    
    def _format_error_message(self):
        msg = "\n" + "="*50 + "\n"
        msg += f"错误发生在语句: {self.statement}\n"
        msg += f"错误类型: {self.error_type}\n"
        msg += f"错误信息: {self.error_message}\n"
        
        if self.analysis['cause']:
            msg += f"\n错误原因: {self.analysis['cause']}\n"
        if self.analysis['suggestion']:
            msg += f"建议解决方案: {self.analysis['suggestion']}\n"
            
        if self.modified_expr:
            msg += f"\n执行表达式: {self.modified_expr}\n"
            
        if self.analysis['variable_state']:
            msg += "\n相关变量状态:\n"
            for var_name, state in self.analysis['variable_state'].items():
                msg += f"  {var_name} = {state['value']} (类型: {state['type']})\n"
                
        msg += "="*50
        return msg

class DynamicExecutor:
    def __init__(self, grammar_file_address: Dict[str, Any] = None):
        self.grammar_file_address = grammar_file_address
        self.grammar = {}
        self.variables = {}
        self.constants = {}
        self.functions = {}
        self.global_context = {}
        self.module_names = set()
        self.symbol_table = symbol_table
        self.custom_types = {}  # 添加custom_types字典来存储自定义类型
        self.setup()
        self._init_context()
        self.error_count = 0
        self.success_count = 0
        self.result = []

    def _init_context(self):
        self.global_context.update(globals())
        self.global_context.update(self.constants)
        
        # 记录所有已导入模块的名称
        for module_name, module in sys.modules.items():
            if module and not module_name.startswith('_'):
                self.global_context[module_name] = module
                self.module_names.add(module_name.split('.')[0])  # 只取主模块名
                if inspect.ismodule(module):
                    for attr_name, attr_value in module.__dict__.items():
                        if not attr_name.startswith('_'):
                            self.global_context[attr_name] = attr_value
                            if inspect.ismodule(attr_value):
                                self.module_names.add(attr_name)

    def _load_custom_types(self):
        """从YAML文件中加载自定义类型定义"""
        if 'custom_types' not in self.grammar:
            return

        # 首先加载所有导入
        imports = self.grammar.get('imports', [])
        import_namespace = {}
        for import_stmt in imports:
            try:
                exec(import_stmt, import_namespace)
            except Exception as e:
                print(f"Error processing import {import_stmt}: {e}")

        # 更新global_context和globals()
        self.global_context.update(import_namespace)
        globals().update(import_namespace)

        # 加载自定义类型
        for custom_type in self.grammar.get('custom_types', []):
            if 'name' not in custom_type or 'code' not in custom_type:
                continue

            try:
                # 在包含imports的上下文中执行类定义
                exec_context = {**self.global_context, **import_namespace}
                exec(custom_type['code'], exec_context)

                # 获取新定义的类
                class_name = custom_type['name']
                if class_name in exec_context:
                    # 存储到custom_types字典中
                    self.custom_types[class_name] = exec_context[class_name]
                    # 更新global_context和globals
                    self.global_context[class_name] = exec_context[class_name]
                    globals()[class_name] = exec_context[class_name]

            except Exception as e:
                print(f"Error loading custom type {custom_type.get('name', 'unknown')}: {e}")

    def setup(self):
        grammar_loader = GrammarLoader()
        self.grammar = grammar_loader.parse_grammar_yml(self.grammar_file_address)
        self.namespace = grammar_loader.get_namespace()
        
        # 添加symbol_table到global context
        self.global_context['symbol_table'] = self.symbol_table
        
        # 先加载custom_types
        self._load_custom_types()
        
        if 'constants' in self.grammar and self.grammar['constants']:
            self.constants = {k:v for d in self.grammar['constants'] for k, v in d.items()}
            load_constants_into_ConstantTable(self.grammar['constants'])
        if 'columns' in self.grammar and self.grammar['columns']:
            load_columns_into_symboltable(self.grammar['columns'])
        if 'tables' in self.grammar and self.grammar['tables']: 
            load_tables_into_symboltable(self.grammar['tables'])
            
        external_functions, namespace = load_functions_from_yaml_file(
            self.grammar_file_address, 
            constants=self.constants
        )
        
        self.global_context.update(namespace)
        self.add_functions(external_functions)

    def get_custom_type(self, type_name: str):
        """获取自定义类型"""
        return self.custom_types.get(type_name)
    
    def add_functions(self, new_functions: Dict[str, Callable]):
        """添加新的函数到执行环境"""
        # 验证输入都是可调用的函数
        for name, func in new_functions.items():
            if not callable(func):
                raise ValueError(f"{name} 不是一个有效的函数")
        
        # 更新函数字典
        self.functions.update(new_functions)
        
        # 更新全局上下文
        self.global_context.update(self.functions)

    def _get_obj_attr(self, obj_str: str, attr: str) -> Any:
        if obj_str in self.variables:
            return self.variables[obj_str].get(attr)
        return None

    def _set_obj_attr(self, obj_str: str, attr: str, value: Any):
        if obj_str not in self.variables:
            self.variables[obj_str] = {}
        self.variables[obj_str][attr] = value

    def add_functions(self, new_functions: Dict[str, Callable]):
        """添加新的函数到执行环境"""
        # 验证输入都是可调用的函数
        for name, func in new_functions.items():
            if not callable(func):
                raise ValueError(f"{name} 不是一个有效的函数")
        
        # Create a namespace that includes the current context and constants
        namespace = {}
        namespace.update(self.global_context)
        namespace.update(self.constants)
        
        # Add each function to the namespace
        for name, func in new_functions.items():
            try:
                # Try to get source from _source attribute (for YAML-loaded functions)
                if hasattr(func, '_source'):
                    source = func._source
                    exec(source, namespace)
                    func_name = re.search(r'def\s+(\w+)', source).group(1)
                    self.functions[name] = namespace[func_name]
                else:
                    # For regular Python functions, just add them directly
                    self.functions[name] = func
            except Exception as e:
                print(f"Warning: Error processing function {name}: {e}")
                # If there's an error, use the original function
                self.functions[name] = func
        
        # Update global context with all functions
        self.global_context.update(self.functions)
        
        # 更新module_names以包含新添加的模块
        for name, value in namespace.items():
            if inspect.ismodule(value):
                self.module_names.add(name.split('.')[0])

    def _prepare_expression(self, expr: str) -> str:
        # 先分离出字符串字面量，避免处理字符串内的点号
        string_literals = []
        def replace_string(match):
            string_literals.append(match.group(0))
            return f"__STR{len(string_literals)-1}__"
        
        # 保存字符串字面量
        modified_expr = re.sub(r'"[^"]*"|\'[^\']*\'', replace_string, expr)
        
        # 查找所有的 xxx.yyy 模式
        dot_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.finditer(dot_pattern, modified_expr)
        
        # 从后向前替换，避免干扰位置
        replacements = []
        for match in matches:
            obj_name, attr_name = match.groups()
            # 跳过已知模块的方法调用
            if (obj_name not in self.module_names and  # 不是已知模块
                obj_name in self.variables):  # 是我们的动态对象
                replacements.append((
                    match.span(),
                    f"self._get_obj_attr('{obj_name}', '{attr_name}')"
                ))
        
        # 从后向前替换
        for (start, end), replacement in reversed(replacements):
            modified_expr = modified_expr[:start] + replacement + modified_expr[end:]
        
        # 恢复字符串字面量
        for i, literal in enumerate(string_literals):
            modified_expr = modified_expr.replace(f"__STR{i}__", literal)
            
        return modified_expr

    def execute(self, expr: str):
        try:
            exec_locals = {}
            exec_locals.update(self.variables)
            # 获取当前规则的权重值
            calculated_weights = {
                name: attrs.get('dif', 0) 
                for name, attrs in self.variables.items()
                if 'dif' in attrs
            }
            # 在执行环境中包含常量、函数和动态变量
            exec_globals = {
                'self': self,
                'weight': calculated_weights,  # 添加weight字典
                **self.global_context,
                **self.constants,
                **self.functions
            }
            
            modified_expr = self._prepare_expression(expr)
            result = eval(modified_expr, exec_globals, exec_locals)
            self.success_count += 1
            return result
                
        except Exception as e:
            self.error_count += 1
            # 在错误信息中包含所有上下文信息
            context = {
                **self.variables,
                'constants': self.constants,
                'functions': {name: f"<function {name}>" for name in self.functions}
            }
            raise StatementExecutionError(
                statement=expr,
                error_type=type(e).__name__,
                error_message=str(e),
                modified_expr=modified_expr,
                variables=context
            )
    
    def get_execution_stats(self):
        total = self.success_count + self.error_count
        success_rate = (self.success_count / total) * 100 if total > 0 else 0
        return {
            'total_statements': total,
            'successful': self.success_count,
            'failed': self.error_count,
            'success_rate': f"{success_rate:.1f}%"
        }

    def get_variable(self, obj: str, attr: str) -> Any:
        return self._get_obj_attr(obj, attr)

    def get_constant(self, name: str) -> Any:
        return self.constants.get(name)

    def get_function(self, name: str) -> Callable:
        return self.functions.get(name)