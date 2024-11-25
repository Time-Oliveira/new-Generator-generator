import random
import string
import re
import names
from typing import Union, Tuple, Optional
class RandomName:
    def __init__(self, min_length: int = 6, max_length: Optional[int] = None, 
                 include_middle: bool = False, gender: Optional[str] = None):
        self.min_length = min_length
        self.max_length = max_length if max_length and max_length >= min_length else min_length + 20
        self.include_middle = include_middle
        self.gender = gender
        
        # 验证参数
        if self.min_length < 6:
            raise ValueError("Minimum length should be at least 6 characters (3 for first name + space + 2 for last name)")
        
        if gender and gender.lower() not in ['male', 'female']:
            raise ValueError("Gender must be either 'male', 'female' or None")

    def generate(self, format_type: str = 'full') -> str:
        attempt_count = 0
        max_attempts = 100  # 防止无限循环
        
        while attempt_count < max_attempts:
            # 生成基本名字部分
            gender = self.gender.lower() if self.gender else random.choice(['male', 'female'])
            first_name = names.get_first_name(gender=gender)
            middle_name = names.get_first_name(gender=gender) if self.include_middle else ''
            last_name = names.get_last_name()
            
            # 构建完整名字
            if self.include_middle:
                full_name = f"{first_name} {middle_name} {last_name}"
            else:
                full_name = f"{first_name} {last_name}"
                
            # 检查长度是否符合要求
            if self.min_length <= len(full_name) <= self.max_length:
                # 根据format_type返回不同格式
                if format_type == 'full':
                    return full_name
                elif format_type == 'initials':
                    initials = ''.join(name[0] for name in full_name.split())
                    return initials
                elif format_type == 'first_last':
                    return f"{first_name} {last_name}"
                elif format_type == 'formal':
                    return f"{last_name}, {first_name}"
                else:
                    raise ValueError("Invalid format_type")
                    
            attempt_count += 1
            
        raise RuntimeError(f"Could not generate name matching criteria after {max_attempts} attempts")

    def generate_batch(self, count: int, unique: bool = True) -> list:
        if unique and count > 1000:  # 设置一个合理的上限
            raise ValueError("For unique names, count should be less than 1000")
            
        names_list = []
        attempts = 0
        max_attempts = count * 2  # 给予足够的尝试次数
        
        while len(names_list) < count and attempts < max_attempts:
            new_name = self.generate()
            if not unique or new_name not in names_list:
                names_list.append(new_name)
            attempts += 1
            
        if len(names_list) < count:
            raise RuntimeError(f"Could not generate {count} unique names")
            
        return names_list

    @staticmethod
    def get_name_stats(name: str) -> dict:
        parts = name.split()
        return {
            'total_length': len(name),
            'parts_count': len(parts),
            'each_part_length': [len(part) for part in parts],
            'has_middle_name': len(parts) > 2,
            'initials': ''.join(part[0] for part in parts)
        }        
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