import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass  # 注意这里是 dataclasses 而不是 dataclass

@dataclass
class WeightAnalysis:
    min_weight: float
    decay_rate: Optional[float]
    conditions: List[str]
    rule_type: str

class NestedQueryAnalyzer:
    def __init__(self, grammar_rules):
        self.grammar_rules = grammar_rules
        self.analyzed_weights = self._analyze_weight_rules()
        
    def _analyze_weight_expr(self, expr: str) -> dict:
        """分析weight表达式的特征"""
        analysis = {
            'constants': set(),
            'has_multiplication': False,
            'decay_rate': None,
            'conditions': []
        }
        
        # 提取条件语句中的常量
        if_conditions = re.findall(r'if\s+(.+?)\s+else', expr)
        for cond in if_conditions:
            analysis['conditions'].append(cond)
            numbers = re.findall(r'[-+]?\d*\.?\d+', cond)
            analysis['constants'].update(float(n) for n in numbers)
        
        # 检测乘法衰减模式
        mult_match = re.search(r'weight\[[\'"]?\w+[\'"]?\]\s*\*\s*(\d*\.?\d+)', expr)
        if mult_match:
            analysis['has_multiplication'] = True
            analysis['decay_rate'] = float(mult_match.group(1))
            
        return analysis

    def _analyze_weight_rules(self) -> Dict[str, WeightAnalysis]:
        """分析所有与weight相关的规则"""
        analyses = {}
        
        for rule in self.grammar_rules:
            if 'weight' not in rule or 'rule' not in rule:
                continue
                
            rule_name = rule['rule'].split('->')[0].strip()
            weights = rule['weight']
            
            # 确定规则类型
            rule_type = 'basic'
            if '( E )' in rule['rule']:
                rule_type = 'nesting'
                
            min_weight = float('inf')
            decay_rate = None
            all_conditions = []
            
            for weight_expr in weights.values():
                analysis = self._analyze_weight_expr(weight_expr)
                
                if analysis['constants']:
                    min_weight = min(min_weight, min(analysis['constants']))
                    
                if analysis['decay_rate']:
                    decay_rate = analysis['decay_rate']
                    
                all_conditions.extend(analysis['conditions'])
            
            analyses[rule_name] = WeightAnalysis(
                min_weight=min_weight if min_weight != float('inf') else 0,
                decay_rate=decay_rate,
                conditions=all_conditions,
                rule_type=rule_type
            )
            
        return analyses

    def calculate_safe_nesting(self, initial_weight: float) -> dict:
        """计算安全的嵌套深度"""
        current_weight = initial_weight
        depths = [current_weight]
        max_depth = 0
        
        # 获取嵌套规则的分析结果
        nesting_rules = {
            name: analysis 
            for name, analysis in self.analyzed_weights.items() 
            if analysis.rule_type == 'nesting'
        }
        
        if not nesting_rules:
            return {
                'max_safe_depth': 0,
                'weight_sequence': depths,
                'min_valid_weight': 0,
                'decay_rate': 1.0,
                'error': 'No nesting rules found'
            }
        
        # 找出最小的有效权重
        min_valid_weight = max(
            analysis.min_weight 
            for analysis in self.analyzed_weights.values()
        )
        
        # 获取衰减率，如果没有明确的衰减率则使用默认值
        decay_rates = [
            analysis.decay_rate 
            for analysis in nesting_rules.values() 
            if analysis.decay_rate is not None
        ]
        decay_rate = min(decay_rates) if decay_rates else 0.8  # 使用默认衰减率
        
        # 计算安全嵌套深度
        while current_weight > min_valid_weight:
            current_weight *= decay_rate
            if current_weight <= min_valid_weight:
                break
            depths.append(current_weight)
            max_depth += 1
                
        return {
            'max_safe_depth': max_depth,
            'weight_sequence': depths,
            'min_valid_weight': min_valid_weight,
            'decay_rate': decay_rate
        }
    def should_continue_nesting(self, current_depth: int, current_weight: float) -> bool:
        """判断是否应该继续嵌套"""
        safety_info = self.calculate_safe_nesting(current_weight)
        
        if current_depth >= safety_info['max_safe_depth']:
            return False
            
        next_weight = current_weight * safety_info['decay_rate']
        return next_weight > safety_info['min_valid_weight']

    def get_safety_check(self, initial_weight: float) -> dict:
        """获取完整的安全性检查报告"""
        safety_info = self.calculate_safe_nesting(initial_weight)
        
        return {
            'is_safe': safety_info['max_safe_depth'] > 0,
            'max_safe_depth': safety_info['max_safe_depth'],
            'weight_thresholds': {
                depth: weight 
                for depth, weight in enumerate(safety_info['weight_sequence'])
            },
            'min_valid_weight': safety_info['min_valid_weight'],
            'decay_pattern': safety_info['decay_rate']
        }