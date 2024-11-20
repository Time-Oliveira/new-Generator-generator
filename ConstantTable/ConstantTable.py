class ConstantTable:
    def __init__(self):
        self.values = {}

    '''添加值到ConstantTable'''
    def add_value(self, name, value):

        Constant = {
            'name': name,
            'value': value,
        }

        # 用name作为key插入或更新字典
        self.values[name] = Constant

    '''从ConstantTable中查询值'''
    def get_value(self, name):
        # return self.values.get(name, None)

        if self.has_value(name):
            return self.values.get(name)['value']
        
    '''从ConstantTable中查询是否有该constant'''
    def has_value(self, name):
        return name in self.values
    
    def __repr__(self):
        return f"ConstantTable({self.values})"
    
def load_constants_into_ConstantTable(constants):
    for constant in constants:
        for name, value in constant.items():
            ConstantTable.add_value(name, value)

ConstantTable = ConstantTable()