import random
import string
import re

class CustomID:
    def __init__(self, pattern):
        self.pattern = pattern

    def generate(self):
        # 通过解析模板生成 ID
        id_string = self._parse_pattern(self.pattern)
        return id_string

    def _parse_pattern(self, pattern):
        """
        解析 ID 模板，根据模板生成最终的 ID 字符串。
        """
        result = []
        i = 0
        while i < len(pattern):
            char = pattern[i]
            
            # 处理大写字母部分
            if char == 'L' and i + 1 < len(pattern) and pattern[i + 1].isdigit():
                count = int(pattern[i + 1])
                result.extend(random.choice(string.ascii_uppercase) for _ in range(count))
                i += 2
                
            # 处理小写字母部分
            elif char == 'l' and i + 1 < len(pattern) and pattern[i + 1].isdigit():
                count = int(pattern[i + 1])
                result.extend(random.choice(string.ascii_lowercase) for _ in range(count))
                i += 2
                
            # 处理数字部分
            elif char == 'N' and i + 1 < len(pattern) and pattern[i + 1].isdigit():
                count = int(pattern[i + 1])
                result.extend(random.choice(string.digits) for _ in range(count))
                i += 2
                
            # 处理数字范围部分
            elif char == '{':
                end = pattern.find('}', i)
                if end != -1:
                    range_part = pattern[i+1:end]
                    if '-' in range_part:
                        start, end_num = map(int, range_part.split('-'))
                        number = str(random.randint(start, end_num))
                        result.append(number)
                    i = end + 1
                else:
                    result.append(char)
                    i += 1
                    
            # 处理选择符
            elif char == '|':
                options = pattern[:i].split('|') + [pattern[i+1:]]
                choice = random.choice(options)
                return self._parse_pattern(choice)
                
            else:
                result.append(char)
                i += 1
                
        return ''.join(result)

class CustomInt:
    def __init__(self, *params):
        self.data_type = int
        self.min_value = params[0]
        self.max_value = params[1]

    def generate(self):
        return random.randint(self.min_value, self.max_value)

class CustomString:
    def __init__(self, *params):
        self.data_type = str
        self.max_length = params[0]

    def generate(self):
        length = random.randint(1, self.max_length)
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

class CustomFloat:
    def __init__(self, *params):
        self.data_type = float
        self.precision = params[0]
        self.min_value = params[1]
        self.max_value = params[2]

    def generate(self):
        return round(random.uniform(self.min_value, self.max_value), self.precision)

class RandomSelector:
    def __init__(self, *params):
        self.data_type = list
        self.values = params

    def generate(self):
        return random.choice(self.values)
    
if __name__ == "__main__":
    # 测试大写字母
    upper_id_generator = CustomID(pattern="L3N2")
    upper_id = upper_id_generator.generate()
    print("Upper case ID:", upper_id)  # 例如：ABC12

    # 测试小写字母
    lower_id_generator = CustomID(pattern="l3N2")
    lower_id = lower_id_generator.generate()
    print("Lower case ID:", lower_id)  # 例如：abc12

    # 测试混合大小写
    mixed_id_generator = CustomID(pattern="L2l2N3")
    mixed_id = mixed_id_generator.generate()
    print("Mixed case ID:", mixed_id)  # 例如：ABcd123

    # 测试教学楼编号（保持大写T）
    building_id_generator = CustomID(pattern="T{1-6}|T29")
    building_id = building_id_generator.generate()
    print("Building ID:", building_id)  # 例如：T4 或 T29

    building_id_generator = CustomID(pattern="T{1-6}-{1-6}N2|T29-{1-6}N2")
    building_id = building_id_generator.generate()
    print("Building ID:", building_id)  # 例如：T4 或 T29

    # 测试更复杂的混合模式
    complex_id_generator = CustomID(pattern="L1l2N3-L2")
    complex_id = complex_id_generator.generate()
    print("Complex ID:", complex_id)  # 例如：Abc123-XY