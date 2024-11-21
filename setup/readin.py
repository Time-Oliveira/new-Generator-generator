import yaml
import importlib
import inspect
import importlib.util
from pathlib import Path
import sys
from typing import Dict, Any

"""分析syntax中的terminal和noterminal,通过left_symbols和right_symbols"""
def analyze_syntax(syntax_rules):
    left_symbols, right_symbols = set(), set()
    rule_map = {}

    for rule in syntax_rules:
        rule_str = rule['rule']
        left_side, right_side = rule_str.split("->")
        left_side = left_side.strip()
        right_side = right_side.strip().split()

        left_symbols.add(left_side)
        right_symbols.update(right_side)

        # 添加规则和对应的动作到映射表
        rule_entry = {
            "rule": rule_str,  # 保存完整的规则字符串
            "rules": right_side,
            "actions": rule.get("actions", []),
            "weight": rule.get("weight", {})
        }
        rule_map.setdefault(left_side, []).append(rule_entry)

    terminals = right_symbols - left_symbols
    nonterminals = left_symbols
    return nonterminals, terminals, rule_map, left_symbols, right_symbols

"""自动判断start_symbol: 选择出现在左侧但从未出现在右侧的符号"""
def get_start_symbol(left_symbols, right_symbols):

    # 起点符号应是没有在右侧出现的左侧符号
    start_candidates = left_symbols - right_symbols

    if start_candidates:
        # 如果start_symbol有多个，返回任意一个候选符号
        return next(iter(start_candidates))
    else:
        raise ValueError("无法找到有效的起点符号，所有左侧符号都出现在右侧。")

"""处理语法文件加载和动态导入的类"""
class GrammarLoader:
    def __init__(self):
        self.namespace: Dict[str, Any] = {}
        self.custom_classes = {}
        self._load_initial_classes()

    def _load_initial_classes(self):
        """Load built-in custom classes during initialization"""
        custom_class_path = Path(__file__).parent.parent / "CustomClass" / "CustomClass.py"
        self.load_custom_classes(str(custom_class_path))
        
    def load_custom_classes(self, module_path: str):
        """加载 CustomClass.py 中的自定义类"""
        try:
            # 获取绝对路径
            abs_path = str(Path(module_path).resolve())
            module_name = "custom_classes"
            
            # 动态导入模块
            spec = importlib.util.spec_from_file_location(module_name, abs_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # 获取所有自定义类
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and not name.startswith('_'):
                    self.custom_classes[name] = obj
                    self.namespace[name] = obj
                    globals()[name] = obj  # 确保全局可访问
                    
            return True
        except Exception as e:
            print(f"Error loading custom classes: {e}")
            return False
        
    """加载并解析 grammar.yml 文件"""
    def parse_grammar_yml(self, file_name: str) -> dict:
        with open(file_name, 'r', encoding="utf-8") as file:
            data = yaml.safe_load(file)

        # Process imports
        imports = data.get('imports', [])
        self.namespace.update(self.dynamic_import(imports))
        
        # Process custom types
        custom_types = data.get('custom_types', [])
        for custom_type in custom_types:
            code = custom_type.get('code', '')
            try:
                # Execute in both namespace and globals
                exec(code, self.namespace, self.namespace)
                
                # Register class if name is provided
                if 'name' in custom_type:
                    class_name = custom_type['name']
                    if class_name in self.namespace:
                        self.custom_classes[class_name] = self.namespace[class_name]
                        globals()[class_name] = self.namespace[class_name]
                
            except Exception as e:
                print(f"Error processing custom type {custom_type.get('name', 'unknown')}: {e}")

        return data

    """根据 grammar.yml 中的 imports 部分动态导入库
    现在支持的格式：                                        
    import xxx
    import xxx, xxx
    from xxx import xxx
    from xxx import *
    from xxx import xxx, xxx, xxx
    """                   
    def dynamic_import(self, imports: list) -> dict:
        namespace = self.namespace.copy()
        
        for item in imports:
            try:
                if isinstance(item, str):
                    if item.startswith('import '):
                        # 去掉'import'前缀
                        import_statement = item[len('import '):].strip()
                        
                        # Handle multiple imports (e.g., 'import xxx, yyy, zzz')
                        for module_spec in import_statement.split(','):
                            module_spec = module_spec.strip()
                            
                            # Handle 'import xxx as yyy'
                            parts = module_spec.split(' as ')
                            module_name = parts[0].strip()
                            alias = parts[1].strip() if len(parts) > 1 else module_name
                            
                            module = importlib.import_module(module_name)
                            namespace[alias] = module
                            globals()[alias] = module
                            
                    elif item.startswith('from '):
                        # Remove 'from ' prefix and split into module path and imports
                        _, rest = item.split('from ', 1)
                        module_path, imports_part = rest.strip().split(' import ')
                        module = importlib.import_module(module_path)
                        
                        if imports_part.strip() == '*':
                            # Handle 'from xxx import *'
                            # Get all public attributes (not starting with '_')
                            for name in dir(module):
                                if not name.startswith('_'):
                                    obj = getattr(module, name)
                                    namespace[name] = obj
                                    globals()[name] = obj
                        else:
                            # Handle regular imports and multiple imports
                            for name in imports_part.split(','):
                                name = name.strip()
                                if ' as ' in name:
                                    orig_name, alias = name.split(' as ')
                                    orig_name = orig_name.strip()
                                    alias = alias.strip()
                                else:
                                    orig_name = alias = name
                                    
                                obj = getattr(module, orig_name)
                                namespace[alias] = obj
                                globals()[alias] = obj
                                
            except ImportError as e:
                print(f"Error importing {item}: {e}")
            except Exception as e:
                print(f"Unexpected error while importing {item}: {e}")
        
        return namespace

    def get_namespace(self) -> dict:
        """返回当前命名空间"""
        return self.namespace