import yaml
import re
from typing import Dict, Callable, Any

def extract_functions_from_yaml(yaml_content: str) -> Dict[str, Callable]:
    try:
        yaml_data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML content: {e}")
    
    # Extract functions section
    functions_data = yaml_data.get('functions', {})
    if not functions_data:
        return {}
    
    external_functions = {}
    
    for func_name, func_info in functions_data.items():
        # print(func_name, func_info)
        # Get implementation code
        implementation = func_info.get('implementation', '').strip()
        if not implementation:
            continue
            
        # Extract function definition
        func_match = re.search(r'def\s+(\w+)\s*\([^)]*\):', implementation)
        if not func_match:
            continue

        # Create function namespace
        namespace = {}
        
        # Add necessary imports from the YAML imports section
        imports = yaml_data.get('imports', [])
        import_code = '\n'.join(imports)
        
        try:
            # Execute imports in the namespace
            exec(import_code, namespace)
            
            # Execute function implementation in the namespace
            exec(implementation, namespace)
            
            # Get the function object from namespace
            func_obj = namespace[func_match.group(1)]
            
            # Add to external functions dictionary
            external_functions[func_name] = func_obj
            
        except Exception as e:
            raise ValueError(f"Error processing function {func_name}: {e}")
    
    return external_functions

def load_functions_from_yaml_file(file_path: str) -> Dict[str, Callable]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
        
        # print(yaml_content)

        return extract_functions_from_yaml(yaml_content)
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading functions from YAML: {e}")
    
# Using file path
functions = load_functions_from_yaml_file('input/test.yml')

print(functions)

# # Or using YAML content directly
# with open('input/grammar1.yml', 'r', encoding='utf-8') as f:
#     yaml_content = f.read()
# functions = extract_functions_from_yaml(yaml_content)

# The functions can then be accessed from the dictionary
# result = functions['select_table_from_symt'](symbol_table, difficulty)