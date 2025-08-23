"""
Activation Function Manager
管理NEAT网络中的激活函数多样性，提供智能的激活函数选择策略。
"""

import random
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np

try:
    from .neat_core import ActivationFunction
except ImportError:
    from neat_core import ActivationFunction


class ActivationFunctionManager:
    """激活函数管理器 - 提供多样化的激活函数选择策略"""
    
    def __init__(self, 
                 diversity_weight: float = 0.3,
                 performance_weight: float = 0.4,
                 task_specific_weight: float = 0.3):
        """
        初始化激活函数管理器
        
        Args:
            diversity_weight: 多样性权重
            performance_weight: 性能权重  
            task_specific_weight: 任务特定权重
        """
        self.diversity_weight = diversity_weight
        self.performance_weight = performance_weight
        self.task_specific_weight = task_specific_weight
        
        # 激活函数统计
        self.usage_stats = {func: 0 for func in ActivationFunction}
        self.performance_stats = {func: [] for func in ActivationFunction}
        
        # 任务特定的激活函数偏好
        self.task_preferences = {
            'classification': [ActivationFunction.SIGMOID, ActivationFunction.TANH],
            'regression': [ActivationFunction.LINEAR, ActivationFunction.RELU],
            'reinforcement_learning': [ActivationFunction.RELU, ActivationFunction.TANH],
            'pattern_recognition': [ActivationFunction.SIGMOID, ActivationFunction.TANH]
        }
        
        # 激活函数特性
        self.activation_properties = {
            ActivationFunction.SIGMOID: {
                'range': (0, 1),
                'saturation': 'high',
                'gradient': 'vanishing',
                'suitable_for': ['classification', 'probability_output'],
                'complexity': 'medium'
            },
            ActivationFunction.TANH: {
                'range': (-1, 1),
                'saturation': 'medium',
                'gradient': 'vanishing',
                'suitable_for': ['classification', 'bounded_output'],
                'complexity': 'medium'
            },
            ActivationFunction.RELU: {
                'range': (0, float('inf')),
                'saturation': 'none',
                'gradient': 'stable',
                'suitable_for': ['regression', 'deep_networks'],
                'complexity': 'low'
            },
            ActivationFunction.LINEAR: {
                'range': (-float('inf'), float('inf')),
                'saturation': 'none',
                'gradient': 'constant',
                'suitable_for': ['regression', 'linear_mapping'],
                'complexity': 'low'
            }
        }
    
    def get_random_activation(self, 
                            exclude_functions: Optional[List[ActivationFunction]] = None,
                            bias_towards_diversity: bool = True) -> ActivationFunction:
        """
        获取随机激活函数
        
        Args:
            exclude_functions: 要排除的激活函数列表
            bias_towards_diversity: 是否偏向多样性选择
            
        Returns:
            选择的激活函数
        """
        available_functions = list(ActivationFunction)
        
        if exclude_functions:
            available_functions = [f for f in available_functions if f not in exclude_functions]
        
        if not available_functions:
            return ActivationFunction.SIGMOID  # 默认回退
        
        if bias_towards_diversity:
            # 偏向选择使用较少的激活函数
            weights = []
            for func in available_functions:
                usage_count = self.usage_stats.get(func, 0)
                # 使用越少，权重越高
                weight = 1.0 / (usage_count + 1)
                weights.append(weight)
            
            # 归一化权重
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            return random.choices(available_functions, weights=weights)[0]
        else:
            return random.choice(available_functions)
    
    def get_task_appropriate_activation(self, 
                                     task_type: str,
                                     node_type: str = 'hidden') -> ActivationFunction:
        """
        根据任务类型获取合适的激活函数
        
        Args:
            task_type: 任务类型 ('classification', 'regression', 'reinforcement_learning', 'pattern_recognition')
            node_type: 节点类型 ('input', 'hidden', 'output')
            
        Returns:
            合适的激活函数
        """
        if task_type in self.task_preferences:
            preferred_functions = self.task_preferences[task_type]
            
            # 对于输出节点，优先选择任务相关的激活函数
            if node_type == 'output':
                return random.choice(preferred_functions)
            
            # 对于隐藏节点，混合任务相关和多样性
            if random.random() < 0.7:  # 70%概率选择任务相关的
                return random.choice(preferred_functions)
            else:  # 30%概率选择其他激活函数增加多样性
                other_functions = [f for f in ActivationFunction if f not in preferred_functions]
                return random.choice(other_functions)
        
        # 如果没有特定任务偏好，返回随机选择
        return self.get_random_activation(bias_towards_diversity=True)
    
    def get_balanced_activation(self, 
                              current_population: List,
                              target_diversity: float = 0.4) -> ActivationFunction:
        """
        基于当前种群多样性获取平衡的激活函数
        
        Args:
            current_population: 当前种群
            target_diversity: 目标多样性比例
            
        Returns:
            平衡的激活函数
        """
        # 统计当前种群中激活函数的使用情况
        current_usage = {func: 0 for func in ActivationFunction}
        total_nodes = 0
        
        for genome in current_population:
            for node in genome.nodes.values():
                if hasattr(node, 'activation'):
                    current_usage[node.activation] += 1
                    total_nodes += 1
        
        if total_nodes == 0:
            return self.get_random_activation()
        
        # 计算当前多样性
        current_diversity = len([f for f, count in current_usage.items() if count > 0]) / len(ActivationFunction)
        
        if current_diversity < target_diversity:
            # 多样性不足，选择使用最少的激活函数
            min_usage_func = min(current_usage.items(), key=lambda x: x[1])[0]
            return min_usage_func
        else:
            # 多样性足够，随机选择
            return self.get_random_activation()
    
    def get_adaptive_activation(self, 
                              genome_complexity: int,
                              generation: int,
                              best_fitness: float) -> ActivationFunction:
        """
        基于基因组复杂度和演化阶段获取自适应激活函数
        
        Args:
            genome_complexity: 基因组复杂度
            generation: 当前代数
            best_fitness: 最佳适应度
            
        Returns:
            自适应的激活函数
        """
        # 早期阶段：使用简单激活函数
        if generation < 10:
            if random.random() < 0.8:
                return random.choice([ActivationFunction.SIGMOID, ActivationFunction.LINEAR])
            else:
                return random.choice([ActivationFunction.TANH, ActivationFunction.RELU])
        
        # 中期阶段：增加多样性
        elif generation < 50:
            if genome_complexity < 20:
                # 简单网络：偏向SIGMOID和TANH
                return random.choice([ActivationFunction.SIGMOID, ActivationFunction.TANH])
            else:
                # 复杂网络：增加RELU
                return random.choice([ActivationFunction.SIGMOID, ActivationFunction.TANH, ActivationFunction.RELU])
        
        # 后期阶段：基于性能选择
        else:
            if best_fitness > 0.8:
                # 高性能：保持多样性
                return self.get_random_activation(bias_towards_diversity=True)
            else:
                # 低性能：偏向表现好的激活函数
                return self.get_performance_based_activation()
    
    def get_performance_based_activation(self) -> ActivationFunction:
        """
        基于历史性能选择激活函数
        
        Returns:
            性能最佳的激活函数
        """
        if not any(self.performance_stats.values()):
            return self.get_random_activation()
        
        # 计算每个激活函数的平均性能
        avg_performance = {}
        for func, performances in self.performance_stats.items():
            if performances:
                avg_performance[func] = np.mean(performances)
            else:
                avg_performance[func] = 0.0
        
        # 选择性能最好的激活函数，但有一定随机性
        best_func = max(avg_performance.items(), key=lambda x: x[1])[0]
        
        if random.random() < 0.7:  # 70%概率选择最佳的
            return best_func
        else:  # 30%概率随机选择，保持探索
            return self.get_random_activation()
    
    def update_usage_stats(self, activation_func: ActivationFunction):
        """更新激活函数使用统计"""
        self.usage_stats[activation_func] += 1
    
    def update_performance_stats(self, activation_func: ActivationFunction, performance: float):
        """更新激活函数性能统计"""
        self.performance_stats[activation_func].append(performance)
        
        # 保持最近100个性能记录
        if len(self.performance_stats[activation_func]) > 100:
            self.performance_stats[activation_func] = self.performance_stats[activation_func][-100:]
    
    def get_diversity_report(self) -> Dict:
        """获取激活函数多样性报告"""
        total_usage = sum(self.usage_stats.values())
        
        if total_usage == 0:
            return {'diversity_score': 0.0, 'usage_distribution': {}}
        
        # 计算多样性分数（使用Shannon熵）
        usage_probs = [count / total_usage for count in self.usage_stats.values()]
        diversity_score = -sum(p * np.log2(p) for p in usage_probs if p > 0)
        
        # 归一化到0-1范围
        max_entropy = np.log2(len(ActivationFunction))
        normalized_diversity = diversity_score / max_entropy if max_entropy > 0 else 0
        
        return {
            'diversity_score': normalized_diversity,
            'usage_distribution': {func.value: count for func, count in self.usage_stats.items()},
            'total_usage': total_usage,
            'performance_stats': {
                func.value: {
                    'count': len(perfs),
                    'avg_performance': np.mean(perfs) if perfs else 0.0,
                    'std_performance': np.std(perfs) if perfs else 0.0
                }
                for func, perfs in self.performance_stats.items()
            }
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.usage_stats = {func: 0 for func in ActivationFunction}
        self.performance_stats = {func: [] for func in ActivationFunction}
    
    def recommend_activation_functions(self, 
                                    task_type: str,
                                    network_depth: int,
                                    input_size: int,
                                    output_size: int) -> List[ActivationFunction]:
        """
        为特定任务推荐激活函数组合
        
        Args:
            task_type: 任务类型
            network_depth: 网络深度
            input_size: 输入大小
            output_size: 输出大小
            
        Returns:
            推荐的激活函数列表
        """
        recommendations = []
        
        if task_type == 'classification':
            if output_size == 1:
                # 二分类：输出层使用SIGMOID
                recommendations.extend([ActivationFunction.SIGMOID] * 3)
            else:
                # 多分类：输出层使用SIGMOID或TANH
                recommendations.extend([ActivationFunction.SIGMOID, ActivationFunction.TANH] * 2)
            
            # 隐藏层：混合使用
            recommendations.extend([ActivationFunction.SIGMOID, ActivationFunction.TANH, ActivationFunction.RELU])
            
        elif task_type == 'regression':
            # 回归任务：输出层使用LINEAR，隐藏层使用RELU
            recommendations.extend([ActivationFunction.LINEAR] * 2)
            recommendations.extend([ActivationFunction.RELU, ActivationFunction.TANH])
            
        elif task_type == 'reinforcement_learning':
            # 强化学习：输出层使用LINEAR，隐藏层使用RELU
            recommendations.extend([ActivationFunction.LINEAR] * 2)
            recommendations.extend([ActivationFunction.RELU, ActivationFunction.TANH])
            
        else:
            # 默认：平衡选择
            recommendations = list(ActivationFunction)
        
        # 根据网络深度调整
        if network_depth > 5:
            # 深度网络：增加RELU比例
            recommendations.extend([ActivationFunction.RELU] * 2)
        
        return recommendations[:10]  # 返回前10个推荐


def test_activation_function_manager():
    """测试激活函数管理器"""
    print("🧪 测试激活函数管理器...")
    
    # 创建管理器
    manager = ActivationFunctionManager()
    print("✅ 激活函数管理器创建成功")
    
    # 测试随机选择
    print("\n🔀 测试随机激活函数选择:")
    for i in range(10):
        activation = manager.get_random_activation()
        print(f"   随机选择 {i+1}: {activation.value}")
        manager.update_usage_stats(activation)
    
    # 测试任务特定选择
    print("\n🎯 测试任务特定激活函数选择:")
    task_types = ['classification', 'regression', 'reinforcement_learning', 'pattern_recognition']
    for task in task_types:
        activation = manager.get_task_appropriate_activation(task)
        print(f"   {task}: {activation.value}")
    
    # 测试平衡选择
    print("\n⚖️ 测试平衡激活函数选择:")
    # 模拟一个种群
    class MockGenome:
        def __init__(self):
            self.nodes = {0: MockNode(ActivationFunction.SIGMOID)}
    
    class MockNode:
        def __init__(self, activation):
            self.activation = activation
    
    mock_population = [MockGenome() for _ in range(5)]
    balanced_activation = manager.get_balanced_activation(mock_population)
    print(f"   平衡选择: {balanced_activation.value}")
    
    # 测试自适应选择
    print("\n🔄 测试自适应激活函数选择:")
    adaptive_activation = manager.get_adaptive_activation(genome_complexity=15, generation=25, best_fitness=0.6)
    print(f"   自适应选择: {adaptive_activation.value}")
    
    # 测试性能更新
    print("\n📊 测试性能统计更新:")
    for func in ActivationFunction:
        performance = random.uniform(0.1, 0.9)
        manager.update_performance_stats(func, performance)
        print(f"   {func.value}: {performance:.3f}")
    
    # 获取多样性报告
    print("\n📈 激活函数多样性报告:")
    diversity_report = manager.get_diversity_report()
    print(f"   多样性分数: {diversity_report['diversity_score']:.3f}")
    print(f"   使用分布: {diversity_report['usage_distribution']}")
    
    # 测试推荐功能
    print("\n💡 测试激活函数推荐:")
    recommendations = manager.recommend_activation_functions(
        task_type='classification',
        network_depth=3,
        input_size=10,
        output_size=3
    )
    print(f"   分类任务推荐: {[f.value for f in recommendations]}")
    
    print("\n🎉 激活函数管理器测试完成!")


if __name__ == "__main__":
    test_activation_function_manager()
