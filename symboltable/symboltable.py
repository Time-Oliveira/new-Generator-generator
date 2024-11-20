import random
import yaml
import pandas as pd
from CustomClass.CustomClass import *

class SymbolTable:
    def __init__(self):
        self.symbols = {}

    def add_symbol(self, name, value, sym_type, dif=0):
        # 添加symbol到Symbol Table
        symbol = {
            'name': name,
            'value': value,
            'type': sym_type,
            'dif': dif
        }
        # 通过name作为key插入或更新字典
        self.symbols[name] = symbol

    def get_symbol(self, name):

        # 查询Symbol Table中的Symbol
        return self.symbols.get(name, None)

    def repr(self):
        return f"SymbolTable({self.symbols})"

def load_columns_into_symboltable(columns):
    for column in columns:
        name = column['name']
        attr_type = column['type']
        params = column['params']
        dif = column['dif']

        # 将参数格式化为字符串，并保持类型信息
        # 保证params中的每个元素都转换为适当的类型，避免参数丢失
        '''例如，为了防止 
        - name: Major
          type: RandomSelector
          params: ['CST', 'DS', 'AI', 'STAT', 'APSY', 'ACCT', 'PRA', 'AM', 'LSE', 'EPIN', 'FIN', 'MKT', 'AE', 'BA', 'FM', 'GAD', 'BA']
          dif: 2

          params被翻译为CST而不是'CST'
        '''
        params_str = ', '.join([repr(param) for param in params])  # 使用repr保留类型信息
        value = f"{attr_type}({params_str})"
        # print(params_str, value)

        # 将symbol添加到Symbol Table
        symbol_table.add_symbol(name, value, "columns", dif)

# 将Table加载到Symbol Table中，表格中的属性通过符号表查询
def load_tables_into_symboltable(tables):
    for table in tables:
        table_name = table['name']
        columns = table['columns']
        dif = table['dif']

        # 从符号表中查询表的每个属性的值，并组合成表格的value
        table_value = ', '.join([symbol_table.get_symbol(attr)['name'] for attr in columns])

        # 将Table作为symbol添加到symbol Table
        symbol_table.add_symbol(table_name, table_value, "table", dif)

# Initialize SymbolTable
symbol_table = SymbolTable()