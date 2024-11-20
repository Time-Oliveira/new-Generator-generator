import importlib
import inspect
import yaml
import re
import sys
import math
import statistics
from typing import Any, Dict, Callable
from setup.readin import *
from CustomClass.CustomClass import *

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
        # 创建 GrammarLoader 实例
        loader = GrammarLoader()
        yaml_data = yaml.safe_load(yaml_content)
        
        # 获取完整的命名空间（包含自定义类和导入）
        namespace = {}
        
        # 处理自定义类型和导入
        loader.parse_grammar_yml('input/test.yml')  # 或者直接传入 yaml_content
        namespace.update(loader.get_namespace())
        
        # 添加常量到命名空间
        if constants:
            namespace.update(constants)
            
        functions_data = yaml_data.get('functions', {})
        if not functions_data:
            return {}, namespace
            
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
        
        return external_functions, namespace
        
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
    def __init__(self, 
                constants: Dict[str, Any] = None,
                functions: Dict[str, Callable] = None,
                imports: list = None):
        self.variables = {}
        self.constants = constants or {}
        self.functions = {}
        self.global_context = {}
        self.module_names = set()
        self._init_context()
        
        # Process imports if provided
        if imports:
            self._process_imports(imports)
        
        # Load functions after context is initialized
        if functions:
            self.add_functions(functions)
            
        self.error_count = 0
        self.success_count = 0  

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

    def _process_imports(self, import_statements):
        """Process import statements and add them to the global context"""
        namespace = {}
        for statement in import_statements:
            try:
                exec(statement, namespace)
                # Update global context with imported modules
                for name, value in namespace.items():
                    if not name.startswith('_'):
                        self.global_context[name] = value
                        if inspect.ismodule(value):
                            self.module_names.add(name.split('.')[0])
            except Exception as e:
                print(f"Warning: Failed to process import: {statement}")
                print(f"Error: {str(e)}")

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
    
    def execute_statement(self, statement: str):
        """执行单个语句，支持赋值和函数调用"""
        try:
            if ':=' in statement:
                # 处理赋值语句
                left, right = statement.split(':=')
                left = left.strip()
                right = right.strip()
                
                if '.' in left:
                    # 处理对象属性赋值
                    obj_name, attr = left.split('.')
                    result = self.execute(right)
                    self._set_obj_attr(obj_name.strip(), attr.strip(), result)
                    print(f"✓ {obj_name.strip()}.{attr.strip()} = {result}")
                else:
                    raise ValueError("Invalid assignment format. Must be 'object.attribute := value'")
            else:
                # 直接执行表达式
                result = self.execute(statement)
                print(f"✓ {statement} = {result}")
                return result
                
        except Exception as e:
            if isinstance(e, StatementExecutionError):
                print(e)
            else:
                print(f"\n语法错误:\n语句: {statement}\n错误: {str(e)}")
            return None

    def execute(self, expr: str):
        try:
            exec_locals = {}
            exec_locals.update(self.variables)
            # 在执行环境中包含常量、函数和动态变量
            exec_globals = {
                'self': self,
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

if __name__ == "__main__":
    # 示例：外部常量
    external_constants = {
        'PI': 3.14159,
        'GRAVITY': 9.81,
    }

    executor = DynamicExecutor(constants=external_constants)

    # Load functions and namespace from YAML
    external_functions, namespace = load_functions_from_yaml_file('input/test.yml', constants=external_constants)
    
    # Update executor's global context with the namespace from YAML imports
    executor.global_context.update(namespace)
    
    # Add functions to executor
    executor.add_functions(external_functions)
    
    # 后续添加函数
    def temperature_convert(celsius):
        return celsius * 9/5 + 32
    
    executor.add_functions({
        'to_fahrenheit': temperature_convert
    })
    
    # 测试语句
    statements = [
    # 1. 基础数学运算和函数调用
    "data.x := 5",
    "data.y := 3",
    "result.sum := data.x + data.y",
    "result.product := data.x * data.y",
    "result.area := calc_area(data.x)",
    
    # 2. 字符串操作
    "person.name := 'Alice'",
    "person.age := 25",
    "person.greeting := format_str('Hello {}, you are {} years old', person.name, person.age)",
    
    # 3. 数学库函数使用
    "math_results.sqrt := math.sqrt(16)",
    "math_results.sin := math.sin(math.pi/2)",
    "math_results.log := math.log(10)",
    
    # 4. NumPy操作
    "array.data := numpy.array([1, 2, 3, 4, 5])",
    "array.mean := numpy.mean(array.data)",
    "array.std := numpy.std(array.data)",
    
    # 5. 统计计算
    "stats.numbers := [1, 2, 2, 3, 3, 3, 4, 4, 5]",
    "stats.mode := statistics.mode(stats.numbers)",
    "stats.median := statistics.median(stats.numbers)",
    
    # 6. 自定义函数测试
    "calc.result1 := custom_mul(2, 3)",
    "calc.result2 := custom_mul(2, 3, factor=2)",
    
    # 7. 错误测试
    "error.div := 1/0",  # 除零错误
    "error.undefined := undefined_var",  # 未定义变量
    "error.type := 'hello' + 5",  # 类型错误
    
    # 8. 复合运算
    "comp.base := 10",
    "comp.power := 2",
    "comp.result := math.pow(comp.base, comp.power) + calc_area(comp.base)",
    
    # 9. 字符串格式化复杂测试
    "user.first_name := 'John'",
    "user.last_name := 'Doe'",
    "user.points := 95.5",
    "user.message := format_str('User: {} {}, Score: {:.1f}', user.first_name, user.last_name, user.points)",
    
    # 10. 数组和列表操作
    "list.data := [1, 2, 3, 4, 5]",
    "list.reversed := list.data[::-1]",
    "list.sum := sum(list.data)",
    "list.max := max(list.data)",
    
    # 11. 条件表达式
    "flag.x := 10",
    "flag.y := 20",
    "flag.result := 'x is bigger' if flag.x > flag.y else 'y is bigger'",

    "test.test1 := generate_custom_int(1, 5)",
    "test.test2 := generate_CustomFloat(2, 0, 4)",
    'test.test3 := generate_CustomID("T{1-6}|T29")'
    ]

    for stmt in statements:
        try:
            if ':=' in stmt:
                left, right = stmt.split(':=')
                obj_name, attr = left.strip().split('.')
                try:
                    result = executor.execute(right.strip())
                    executor._set_obj_attr(obj_name, attr, result)
                    print(f"✓ {obj_name}.{attr} = {result}")
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

    stats = executor.get_execution_stats()
    print("\n执行统计:")
    print(f"总语句数: {stats['total_statements']}")
    print(f"成功: {stats['successful']}")
    print(f"失败: {stats['failed']}")
    print(f"成功率: {stats['success_rate']}")

    print("\n最终变量状态:")
    print(executor.variables)
    
    print("\n可用的函数:")
    for func_name in executor.functions:
        print(f"- {func_name}")