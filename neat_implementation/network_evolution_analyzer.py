"""
Network Evolution Analyzer
分析NEAT网络在演化过程中的复杂度变化、结构创新和演化趋势。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import json

try:
    from .neat_core import NEATGenome, NodeType, ConnectionGene
    from .neat_network import NetworkVisualizer
except ImportError:
    from neat_core import NEATGenome, NodeType, ConnectionGene
    from neat_network import NetworkVisualizer


@dataclass
class ComplexityMetrics:
    """网络复杂度指标"""
    mean: float
    std: float
    max: float
    min: float
    median: float
    q25: float  # 第25百分位数
    q75: float  # 第75百分位数


@dataclass
class StructuralInnovation:
    """结构创新统计"""
    new_nodes: int
    new_connections: int
    unique_topologies: int
    innovation_rate: float
    structural_diversity: float


@dataclass
class EvolutionSnapshot:
    """单代演化快照"""
    generation: int
    complexity: ComplexityMetrics
    innovation: StructuralInnovation
    fitness_stats: Dict
    population_size: int
    best_genome_id: Optional[int] = None


class NetworkEvolutionAnalyzer:
    """网络演化分析器 - 分析NEAT网络在演化过程中的变化"""
    
    def __init__(self):
        self.population_history: List[List[NEATGenome]] = []
        self.evolution_snapshots: List[EvolutionSnapshot] = []
        self.innovation_tracker: Dict[int, Set[Tuple[int, int]]] = {}  # 每代的创新连接
        self.topology_signatures: Dict[int, Set[str]] = {}  # 每代的拓扑签名
        
    def add_generation(self, population: List[NEATGenome], generation: int):
        """添加一代种群进行分析"""
        self.population_history.append(population.copy())
        
        # 分析当前代的复杂度
        complexity = self._analyze_generation_complexity(population)
        
        # 分析结构创新
        innovation = self._analyze_structural_innovation(population, generation)
        
        # 分析适应度统计
        fitness_stats = self._analyze_fitness_stats(population)
        
        # 找到最佳基因组
        best_genome = max(population, key=lambda g: g.fitness)
        best_genome_id = id(best_genome)
        
        # 创建演化快照
        snapshot = EvolutionSnapshot(
            generation=generation,
            complexity=complexity,
            innovation=innovation,
            fitness_stats=fitness_stats,
            population_size=len(population),
            best_genome_id=best_genome_id
        )
        
        self.evolution_snapshots.append(snapshot)
        
    def _analyze_generation_complexity(self, population: List[NEATGenome]) -> ComplexityMetrics:
        """分析单代种群的复杂度"""
        complexities = [genome.get_network_complexity() for genome in population]
        
        return ComplexityMetrics(
            mean=np.mean(complexities),
            std=np.std(complexities),
            max=max(complexities),
            min=min(complexities),
            median=np.median(complexities),
            q25=np.percentile(complexities, 25),
            q75=np.percentile(complexities, 75)
        )
    
    def _analyze_structural_innovation(self, population: List[NEATGenome], 
                                     generation: int) -> StructuralInnovation:
        """分析结构创新"""
        # 收集当前代的所有连接
        current_connections = set()
        current_topologies = set()
        
        total_nodes = 0
        total_connections = 0
        
        for genome in population:
            # 统计节点和连接
            total_nodes += len(genome.nodes)
            total_connections += len([c for c in genome.connections.values() if c.enabled])
            
            # 收集连接创新
            for conn in genome.connections.values():
                if conn.enabled:
                    current_connections.add((conn.input_node, conn.output_node))
            
            # 生成拓扑签名
            topology_sig = self._generate_topology_signature(genome)
            current_topologies.add(topology_sig)
        
        # 计算新创新
        previous_connections = self.innovation_tracker.get(generation - 1, set())
        new_connections = len(current_connections - previous_connections)
        
        previous_topologies = self.topology_signatures.get(generation - 1, set())
        unique_topologies = len(current_topologies)
        
        # 存储当前代的创新
        self.innovation_tracker[generation] = current_connections
        self.topology_signatures[generation] = current_topologies
        
        # 计算创新率
        innovation_rate = new_connections / len(current_connections) if current_connections else 0
        
        # 计算结构多样性
        structural_diversity = len(current_topologies) / len(population) if population else 0
        
        return StructuralInnovation(
            new_nodes=total_nodes // len(population) if population else 0,
            new_connections=new_connections,
            unique_topologies=unique_topologies,
            innovation_rate=innovation_rate,
            structural_diversity=structural_diversity
        )
    
    def _generate_topology_signature(self, genome: NEATGenome) -> str:
        """生成基因组的拓扑签名"""
        # 创建基于连接的拓扑签名
        connections = []
        for conn in genome.connections.values():
            if conn.enabled:
                connections.append(f"{conn.input_node}->{conn.output_node}")
        
        # 按字母顺序排序确保一致性
        connections.sort()
        return "|".join(connections)
    
    def _analyze_fitness_stats(self, population: List[NEATGenome]) -> Dict:
        """分析适应度统计"""
        fitnesses = [genome.fitness for genome in population]
        
        return {
            'mean_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'max_fitness': max(fitnesses),
            'min_fitness': min(fitnesses),
            'median_fitness': np.median(fitnesses)
        }
    
    def analyze_complexity_over_time(self) -> List[ComplexityMetrics]:
        """分析网络复杂度随时间的变化"""
        return [snapshot.complexity for snapshot in self.evolution_snapshots]
    
    def analyze_innovation_trends(self) -> List[StructuralInnovation]:
        """分析结构创新趋势"""
        return [snapshot.innovation for snapshot in self.evolution_snapshots]
    
    def analyze_fitness_evolution(self) -> List[Dict]:
        """分析适应度演化"""
        return [snapshot.fitness_stats for snapshot in self.evolution_snapshots]
    
    def get_complexity_growth_rate(self) -> float:
        """计算复杂度增长率"""
        if len(self.evolution_snapshots) < 2:
            return 0.0
        
        complexities = [s.complexity.mean for s in self.evolution_snapshots]
        
        # 计算线性回归斜率
        x = np.arange(len(complexities))
        slope = np.polyfit(x, complexities, 1)[0]
        
        return slope
    
    def get_innovation_efficiency(self) -> float:
        """计算创新效率（创新与适应度提升的关系）"""
        if len(self.evolution_snapshots) < 2:
            return 0.0
        
        innovations = [s.innovation.innovation_rate for s in self.evolution_snapshots]
        fitness_improvements = []
        
        for i in range(1, len(self.evolution_snapshots)):
            current_fitness = self.evolution_snapshots[i].fitness_stats['max_fitness']
            previous_fitness = self.evolution_snapshots[i-1].fitness_stats['max_fitness']
            improvement = current_fitness - previous_fitness
            fitness_improvements.append(improvement)
        
        if not fitness_improvements or not innovations[1:]:
            return 0.0
        
        # 计算创新率与适应度提升的相关性
        correlation = np.corrcoef(innovations[1:], fitness_improvements)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def identify_evolutionary_phases(self) -> List[Dict]:
        """识别演化阶段"""
        phases = []
        
        if len(self.evolution_snapshots) < 3:
            return phases
        
        # 分析复杂度变化率
        complexity_changes = []
        for i in range(1, len(self.evolution_snapshots)):
            prev_complexity = self.evolution_snapshots[i-1].complexity.mean
            curr_complexity = self.evolution_snapshots[i].complexity.mean
            change_rate = (curr_complexity - prev_complexity) / prev_complexity if prev_complexity > 0 else 0
            complexity_changes.append(change_rate)
        
        # 识别不同阶段
        current_phase = None
        phase_start = 0
        
        for i, change_rate in enumerate(complexity_changes):
            if change_rate > 0.1:  # 高增长阶段
                if current_phase != 'growth':
                    if current_phase:
                        phases.append({
                            'phase': current_phase,
                            'start_generation': phase_start,
                            'end_generation': i,
                            'duration': i - phase_start
                        })
                    current_phase = 'growth'
                    phase_start = i
            elif abs(change_rate) <= 0.05:  # 稳定阶段
                if current_phase != 'stable':
                    if current_phase:
                        phases.append({
                            'phase': current_phase,
                            'start_generation': phase_start,
                            'end_generation': i,
                            'duration': i - phase_start
                        })
                    current_phase = 'stable'
                    phase_start = i
            else:  # 缓慢变化阶段
                if current_phase != 'slow_change':
                    if current_phase:
                        phases.append({
                            'phase': current_phase,
                            'start_generation': phase_start,
                            'end_generation': i,
                            'duration': i - phase_start
                        })
                    current_phase = 'slow_change'
                    phase_start = i
        
        # 添加最后一个阶段
        if current_phase:
            phases.append({
                'phase': current_phase,
                'start_generation': phase_start,
                'end_generation': len(complexity_changes),
                'duration': len(complexity_changes) - phase_start
            })
        
        return phases
    
    def generate_evolution_report(self) -> str:
        """生成演化报告"""
        if not self.evolution_snapshots:
            return "没有可用的演化数据"
        
        report = f"""
🧬 NEAT网络演化分析报告
{'='*60}

📊 基础统计信息:
   • 总代数: {len(self.evolution_snapshots)}
   • 种群大小: {self.evolution_snapshots[0].population_size}
   • 分析时间跨度: 第0代 - 第{len(self.evolution_snapshots)-1}代

🔢 复杂度演化:
   • 初始平均复杂度: {self.evolution_snapshots[0].complexity.mean:.2f}
   • 最终平均复杂度: {self.evolution_snapshots[-1].complexity.mean:.2f}
   • 复杂度增长率: {self.get_complexity_growth_rate():.4f}/代
   • 最大复杂度: {max(s.complexity.max for s in self.evolution_snapshots):.2f}

🚀 创新分析:
   • 平均创新率: {np.mean([s.innovation.innovation_rate for s in self.evolution_snapshots]):.4f}
   • 结构多样性: {np.mean([s.innovation.structural_diversity for s in self.evolution_snapshots]):.4f}
   • 创新效率: {self.get_innovation_efficiency():.4f}
   • 独特拓扑数量: {max(s.innovation.unique_topologies for s in self.evolution_snapshots)}

📈 适应度演化:
   • 初始最佳适应度: {self.evolution_snapshots[0].fitness_stats['max_fitness']:.4f}
   • 最终最佳适应度: {self.evolution_snapshots[-1].fitness_stats['max_fitness']:.4f}
   • 适应度提升: {self.evolution_snapshots[-1].fitness_stats['max_fitness'] - self.evolution_snapshots[0].fitness_stats['max_fitness']:.4f}

🔄 演化阶段:
"""
        
        phases = self.identify_evolutionary_phases()
        for phase in phases:
            report += f"   • {phase['phase'].upper()}: 第{phase['start_generation']}-{phase['end_generation']}代 (持续{phase['duration']}代)\n"
        
        report += f"""
📊 最新代统计:
   • 平均复杂度: {self.evolution_snapshots[-1].complexity.mean:.2f} ± {self.evolution_snapshots[-1].complexity.std:.2f}
   • 复杂度范围: {self.evolution_snapshots[-1].complexity.min:.0f} - {self.evolution_snapshots[-1].complexity.max:.0f}
   • 平均适应度: {self.evolution_snapshots[-1].fitness_stats['mean_fitness']:.4f}
   • 适应度标准差: {self.evolution_snapshots[-1].fitness_stats['std_fitness']:.4f}
"""
        
        return report
    
    def visualize_evolution(self, save_path: str = None, figsize: tuple = (16, 12)):
        """可视化演化过程"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            print("❌ 缺少matplotlib库，无法生成可视化图表")
            print("请安装: pip install matplotlib")
            return
        
        if not self.evolution_snapshots:
            print("❌ 没有可用的演化数据")
            return
        
        # 创建多子图布局
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig)
        
        generations = list(range(len(self.evolution_snapshots)))
        
        # 1. 复杂度演化
        ax1 = fig.add_subplot(gs[0, :2])
        complexities = [s.complexity.mean for s in self.evolution_snapshots]
        complexity_stds = [s.complexity.std for s in self.evolution_snapshots]
        
        ax1.plot(generations, complexities, 'b-', linewidth=2, label='平均复杂度')
        ax1.fill_between(generations, 
                        [c - s for c, s in zip(complexities, complexity_stds)],
                        [c + s for c, s in zip(complexities, complexity_stds)],
                        alpha=0.3, color='blue')
        ax1.set_title('网络复杂度演化', fontsize=14, fontweight='bold')
        ax1.set_xlabel('代数')
        ax1.set_ylabel('复杂度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 适应度演化
        ax2 = fig.add_subplot(gs[0, 2])
        max_fitnesses = [s.fitness_stats['max_fitness'] for s in self.evolution_snapshots]
        mean_fitnesses = [s.fitness_stats['mean_fitness'] for s in self.evolution_snapshots]
        
        ax2.plot(generations, max_fitnesses, 'r-', linewidth=2, label='最佳适应度')
        ax2.plot(generations, mean_fitnesses, 'g-', linewidth=2, label='平均适应度')
        ax2.set_title('适应度演化', fontsize=12, fontweight='bold')
        ax2.set_xlabel('代数')
        ax2.set_ylabel('适应度')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 创新率
        ax3 = fig.add_subplot(gs[1, 0])
        innovation_rates = [s.innovation.innovation_rate for s in self.evolution_snapshots]
        ax3.plot(generations, innovation_rates, 'purple', linewidth=2)
        ax3.set_title('创新率', fontsize=12, fontweight='bold')
        ax3.set_xlabel('代数')
        ax3.set_ylabel('创新率')
        ax3.grid(True, alpha=0.3)
        
        # 4. 结构多样性
        ax4 = fig.add_subplot(gs[1, 1])
        structural_diversity = [s.innovation.structural_diversity for s in self.evolution_snapshots]
        ax4.plot(generations, structural_diversity, 'orange', linewidth=2)
        ax4.set_title('结构多样性', fontsize=12, fontweight='bold')
        ax4.set_xlabel('代数')
        ax4.set_ylabel('多样性')
        ax4.grid(True, alpha=0.3)
        
        # 5. 复杂度分布（最后几代）
        ax5 = fig.add_subplot(gs[1, 2])
        recent_snapshots = self.evolution_snapshots[-5:] if len(self.evolution_snapshots) >= 5 else self.evolution_snapshots
        recent_complexities = []
        for snapshot in recent_snapshots:
            if len(self.population_history) > snapshot.generation:
                gen_complexities = [g.get_network_complexity() for g in self.population_history[snapshot.generation]]
                recent_complexities.extend(gen_complexities)
        
        if recent_complexities:
            ax5.hist(recent_complexities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax5.set_title('复杂度分布(最近5代)', fontsize=12, fontweight='bold')
            ax5.set_xlabel('复杂度')
            ax5.set_ylabel('频次')
            ax5.grid(True, alpha=0.3)
        
        # 6. 演化阶段
        ax6 = fig.add_subplot(gs[2, :])
        phases = self.identify_evolutionary_phases()
        
        # 绘制阶段背景
        colors = {'growth': 'lightgreen', 'stable': 'lightblue', 'slow_change': 'lightyellow'}
        for phase in phases:
            ax6.axvspan(phase['start_generation'], phase['end_generation'], 
                       alpha=0.3, color=colors.get(phase['phase'], 'lightgray'),
                       label=f"{phase['phase']} ({phase['duration']}代)")
        
        # 重新绘制复杂度曲线
        ax6.plot(generations, complexities, 'b-', linewidth=3)
        ax6.set_title('演化阶段分析', fontsize=14, fontweight='bold')
        ax6.set_xlabel('代数')
        ax6.set_ylabel('平均复杂度')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 演化分析图已保存到: {save_path}")
        
        plt.show()
    
    def export_evolution_data(self, file_path: str):
        """导出演化数据为JSON格式"""
        data = {
            'metadata': {
                'total_generations': len(self.evolution_snapshots),
                'analysis_timestamp': str(np.datetime64('now')),
                'analyzer_version': '1.0'
            },
            'evolution_snapshots': [],
            'summary_statistics': {
                'complexity_growth_rate': self.get_complexity_growth_rate(),
                'innovation_efficiency': self.get_innovation_efficiency(),
                'evolutionary_phases': self.identify_evolutionary_phases()
            }
        }
        
        # 转换快照数据
        for snapshot in self.evolution_snapshots:
            snapshot_data = {
                'generation': snapshot.generation,
                'complexity': {
                    'mean': snapshot.complexity.mean,
                    'std': snapshot.complexity.std,
                    'max': snapshot.complexity.max,
                    'min': snapshot.complexity.min,
                    'median': snapshot.complexity.median,
                    'q25': snapshot.complexity.q25,
                    'q75': snapshot.complexity.q75
                },
                'innovation': {
                    'new_nodes': snapshot.innovation.new_nodes,
                    'new_connections': snapshot.innovation.new_connections,
                    'unique_topologies': snapshot.innovation.unique_topologies,
                    'innovation_rate': snapshot.innovation.innovation_rate,
                    'structural_diversity': snapshot.innovation.structural_diversity
                },
                'fitness_stats': snapshot.fitness_stats,
                'population_size': snapshot.population_size
            }
            data['evolution_snapshots'].append(snapshot_data)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 演化数据已导出到: {file_path}")


def test_network_evolution_analyzer():
    """测试网络演化分析器"""
    print("🧪 测试网络演化分析器...")
    
    try:
        from neat_core import NEATGenome, InnovationTracker, NodeGene, ConnectionGene, NodeType, ActivationFunction
        import random
        
        # 创建分析器
        analyzer = NetworkEvolutionAnalyzer()
        print("✅ 创建分析器成功")
        
        # 创建创新追踪器
        innovation_tracker = InnovationTracker()
        
        # 模拟几代演化
        for generation in range(10):
            population = []
            
            # 创建种群
            for i in range(20):
                genome = NEATGenome(3, 2)  # 3输入，2输出
                
                # 随机添加一些隐藏节点和连接
                if generation > 2:  # 从第3代开始增加复杂度
                    num_hidden = random.randint(1, generation)
                    for h in range(num_hidden):
                        hidden_id = len(genome.nodes)
                        # 随机选择激活函数，增加多样性
                        activation_functions = [
                            ActivationFunction.SIGMOID,
                            ActivationFunction.TANH,
                            ActivationFunction.RELU,
                            ActivationFunction.LINEAR
                        ]
                        selected_activation = random.choice(activation_functions)
                        
                        hidden_node = NodeGene(
                            node_id=hidden_id,
                            node_type=NodeType.HIDDEN,
                            activation=selected_activation,  # 随机激活函数
                            x_position=0.5,
                            y_position=random.random()
                        )
                        genome.add_node(hidden_node)
                        
                        # 添加连接
                        if random.random() < 0.7:
                            input_node = random.choice(list(range(3)))  # 输入节点
                            conn = ConnectionGene(
                                innovation_number=innovation_tracker.get_innovation_number(input_node, hidden_id),
                                input_node=input_node,
                                output_node=hidden_id,
                                weight=random.gauss(0, 1),
                                enabled=True
                            )
                            genome.add_connection(conn)
                        
                        if random.random() < 0.7:
                            output_node = random.choice([3, 4])  # 输出节点
                            conn = ConnectionGene(
                                innovation_number=innovation_tracker.get_innovation_number(hidden_id, output_node),
                                input_node=hidden_id,
                                output_node=output_node,
                                weight=random.gauss(0, 1),
                                enabled=True
                            )
                            genome.add_connection(conn)
                
                # 设置随机适应度（模拟演化改进）
                base_fitness = generation * 0.1
                genome.fitness = base_fitness + random.gauss(0, 0.2)
                
                population.append(genome)
            
            # 添加到分析器
            analyzer.add_generation(population, generation)
        
        print("✅ 模拟演化数据生成成功")
        
        # 测试分析功能
        complexity_evolution = analyzer.analyze_complexity_over_time()
        print(f"✅ 复杂度分析: {len(complexity_evolution)}代数据")
        
        innovation_trends = analyzer.analyze_innovation_trends()
        print(f"✅ 创新趋势分析: {len(innovation_trends)}代数据")
        
        growth_rate = analyzer.get_complexity_growth_rate()
        print(f"✅ 复杂度增长率: {growth_rate:.4f}")
        
        innovation_efficiency = analyzer.get_innovation_efficiency()
        print(f"✅ 创新效率: {innovation_efficiency:.4f}")
        
        phases = analyzer.identify_evolutionary_phases()
        print(f"✅ 演化阶段识别: {len(phases)}个阶段")
        
        # 生成报告
        report = analyzer.generate_evolution_report()
        print("✅ 演化报告生成成功:")
        print(report)
        
        # 测试可视化
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            analyzer.visualize_evolution(save_path="test_evolution_analysis.png")
            print("✅ 演化可视化测试成功")
        except Exception as e:
            print(f"⚠️  演化可视化测试失败: {e}")
        
        # 测试数据导出
        analyzer.export_evolution_data("test_evolution_data.json")
        print("✅ 数据导出测试成功")
        
        print("🎉 网络演化分析器测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_network_evolution_analyzer()
