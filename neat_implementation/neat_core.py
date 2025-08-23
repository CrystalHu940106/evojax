"""
NEAT Core Implementation
Implements the core NEAT (NeuroEvolution of Augmenting Topologies) algorithm.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random
import time


class NodeType(Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class ActivationFunction(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LINEAR = "linear"


@dataclass
class NodeGene:
    """Represents a node (neuron) in the network."""
    node_id: int
    node_type: NodeType
    activation: ActivationFunction = ActivationFunction.SIGMOID
    x_position: float = 0.0  # For visualization
    y_position: float = 0.0


@dataclass
class ConnectionGene:
    """Represents a connection between two nodes."""
    innovation_number: int
    input_node: int
    output_node: int
    weight: float
    enabled: bool = True
    
    def copy(self):
        return ConnectionGene(
            self.innovation_number,
            self.input_node,
            self.output_node,
            self.weight,
            self.enabled
        )


class InnovationTracker:
    """Tracks innovation numbers for consistent genetic markers."""
    
    def __init__(self):
        self.innovation_number = 0
        self.innovations: Dict[Tuple[int, int], int] = {}
        
    def get_innovation_number(self, input_node: int, output_node: int) -> int:
        """Get innovation number for a connection, creating new if needed."""
        key = (input_node, output_node)
        if key not in self.innovations:
            self.innovations[key] = self.innovation_number
            self.innovation_number += 1
        return self.innovations[key]
    
    def reset(self):
        """Reset for new generation."""
        self.innovation_number = 0
        self.innovations.clear()


class NEATGenome:
    """Represents a NEAT genome containing nodes and connections."""
    
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.species_id = None
        
        # Create input and output nodes
        self._create_initial_nodes()
        
    def _create_initial_nodes(self):
        """Create initial input and output nodes."""
        node_id = 0
        
        # Input nodes
        for i in range(self.input_size):
            self.nodes[node_id] = NodeGene(
                node_id=node_id,
                node_type=NodeType.INPUT,
                x_position=0.0,
                y_position=i / max(1, self.input_size - 1)
            )
            node_id += 1
            
        # Output nodes  
        for i in range(self.output_size):
            self.nodes[node_id] = NodeGene(
                node_id=node_id,
                node_type=NodeType.OUTPUT,
                x_position=1.0,
                y_position=i / max(1, self.output_size - 1)
            )
            node_id += 1
    
    def add_node(self, node_gene: NodeGene):
        """Add a node to the genome."""
        self.nodes[node_gene.node_id] = node_gene
        
    def add_connection(self, connection_gene: ConnectionGene):
        """Add a connection to the genome."""
        self.connections[connection_gene.innovation_number] = connection_gene
        
    def remove_connection(self, innovation_number: int):
        """Remove a connection from the genome."""
        if innovation_number in self.connections:
            del self.connections[innovation_number]
            
    def get_enabled_connections(self) -> List[ConnectionGene]:
        """Get all enabled connections."""
        return [conn for conn in self.connections.values() if conn.enabled]
    
    def has_connection(self, input_node: int, output_node: int) -> bool:
        """Check if connection exists between two nodes."""
        for conn in self.connections.values():
            if conn.input_node == input_node and conn.output_node == output_node:
                return True
        return False
    
    def copy(self) -> 'NEATGenome':
        """Create a deep copy of the genome."""
        new_genome = NEATGenome(self.input_size, self.output_size)
        new_genome.nodes = {k: NodeGene(v.node_id, v.node_type, v.activation, v.x_position, v.y_position)
                           for k, v in self.nodes.items()}
        new_genome.connections = {k: v.copy() for k, v in self.connections.items()}
        new_genome.fitness = self.fitness
        new_genome.adjusted_fitness = self.adjusted_fitness
        return new_genome
    
    def get_network_complexity(self) -> int:
        """Calculate network complexity (nodes + connections)."""
        return len(self.nodes) + len([c for c in self.connections.values() if c.enabled])


class NEATMutator:
    """Handles mutations for NEAT genomes."""
    
    def __init__(self, innovation_tracker: InnovationTracker):
        self.innovation_tracker = innovation_tracker
        
        # Mutation probabilities
        self.weight_mutation_rate = 0.8
        self.weight_perturbation_rate = 0.9  # vs complete replacement
        self.weight_perturbation_strength = 0.1
        self.add_connection_rate = 0.05
        self.add_node_rate = 0.03
        self.disable_connection_rate = 0.01
        
    def mutate_weights(self, genome: NEATGenome):
        """Mutate connection weights."""
        for connection in genome.connections.values():
            if random.random() < self.weight_mutation_rate:
                if random.random() < self.weight_perturbation_rate:
                    # Perturb existing weight
                    connection.weight += random.gauss(0, self.weight_perturbation_strength)
                else:
                    # Replace with new random weight
                    connection.weight = random.gauss(0, 1.0)
    
    def add_connection_mutation(self, genome: NEATGenome) -> bool:
        """Add a new connection between existing nodes."""
        if random.random() > self.add_connection_rate:
            return False
            
        # Get all possible connections
        input_nodes = [nid for nid, node in genome.nodes.items() 
                      if node.node_type in [NodeType.INPUT, NodeType.HIDDEN]]
        output_nodes = [nid for nid, node in genome.nodes.items() 
                       if node.node_type in [NodeType.HIDDEN, NodeType.OUTPUT]]
        
        # Try to find a valid connection
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts:
            input_node = random.choice(input_nodes)
            output_node = random.choice(output_nodes)
            
            # Check if connection is valid (no cycles, not already exists)
            if (input_node != output_node and 
                not genome.has_connection(input_node, output_node) and
                not self._creates_cycle_optimized(genome, input_node, output_node)):
                
                innovation_num = self.innovation_tracker.get_innovation_number(input_node, output_node)
                new_connection = ConnectionGene(
                    innovation_number=innovation_num,
                    input_node=input_node,
                    output_node=output_node,
                    weight=random.gauss(0, 1.0),
                    enabled=True
                )
                genome.add_connection(new_connection)
                return True
                
            attempts += 1
        return False
    
    def add_node_mutation(self, genome: NEATGenome) -> bool:
        """Add a new node by splitting an existing connection."""
        if random.random() > self.add_node_rate:
            return False
            
        enabled_connections = genome.get_enabled_connections()
        if not enabled_connections:
            return False
            
        # Choose random connection to split
        connection = random.choice(enabled_connections)
        
        # Disable the original connection
        connection.enabled = False
        
        # Create new node
        new_node_id = max(genome.nodes.keys()) + 1
        
        # 随机选择激活函数，增加多样性
        activation_functions = [
            ActivationFunction.SIGMOID,
            ActivationFunction.TANH,
            ActivationFunction.RELU,
            ActivationFunction.LINEAR
        ]
        selected_activation = random.choice(activation_functions)
        
        new_node = NodeGene(
            node_id=new_node_id,
            node_type=NodeType.HIDDEN,
            activation=selected_activation,  # 随机激活函数
            x_position=(genome.nodes[connection.input_node].x_position + 
                       genome.nodes[connection.output_node].x_position) / 2,
            y_position=(genome.nodes[connection.input_node].y_position + 
                       genome.nodes[connection.output_node].y_position) / 2
        )
        genome.add_node(new_node)
        
        # Create two new connections
        # Connection 1: input -> new node (weight = 1.0)
        conn1_innovation = self.innovation_tracker.get_innovation_number(
            connection.input_node, new_node_id)
        conn1 = ConnectionGene(
            innovation_number=conn1_innovation,
            input_node=connection.input_node,
            output_node=new_node_id,
            weight=1.0,
            enabled=True
        )
        
        # Connection 2: new node -> output (weight = original weight)
        conn2_innovation = self.innovation_tracker.get_innovation_number(
            new_node_id, connection.output_node)
        conn2 = ConnectionGene(
            innovation_number=conn2_innovation,
            input_node=new_node_id,
            output_node=connection.output_node,
            weight=connection.weight,
            enabled=True
        )
        
        genome.add_connection(conn1)
        genome.add_connection(conn2)
        return True
    
    def disable_connection_mutation(self, genome: NEATGenome):
        """Randomly disable a connection."""
        enabled_connections = genome.get_enabled_connections()
        if enabled_connections and random.random() < self.disable_connection_rate:
            connection = random.choice(enabled_connections)
            connection.enabled = False
    
    def mutate(self, genome: NEATGenome):
        """Apply all mutations to a genome."""
        self.mutate_weights(genome)
        self.add_connection_mutation(genome)
        self.add_node_mutation(genome)
        self.disable_connection_mutation(genome)
    
    def _creates_cycle(self, genome: NEATGenome, input_node: int, output_node: int) -> bool:
        """Check if adding a connection would create a cycle using topological sort."""
        # Create a temporary copy of the genome with the new connection
        temp_genome = genome.copy()
        
        # Add the potential connection temporarily
        temp_connection = ConnectionGene(
            innovation_number=-1,  # Temporary innovation number
            input_node=input_node,
            output_node=output_node,
            weight=0.0,
            enabled=True
        )
        temp_genome.add_connection(temp_connection)
        
        # Check for cycles using topological sort
        return self._has_cycle(temp_genome)
    
    def _has_cycle(self, genome: NEATGenome) -> bool:
        """Check if a genome has cycles using DFS-based cycle detection."""
        # Build adjacency list
        adjacency = {node_id: [] for node_id in genome.nodes.keys()}
        for conn in genome.connections.values():
            if conn.enabled:
                if conn.input_node in adjacency:
                    adjacency[conn.input_node].append(conn.output_node)
        
        # Track visited nodes and recursion stack
        visited = set()
        rec_stack = set()
        
        def dfs_cycle_detect(node_id: int) -> bool:
            """DFS to detect cycles."""
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in adjacency.get(node_id, []):
                if neighbor not in visited:
                    if dfs_cycle_detect(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Back edge found - cycle detected
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        # Check all nodes for cycles
        for node_id in genome.nodes.keys():
            if node_id not in visited:
                if dfs_cycle_detect(node_id):
                    return True
        
        return False
    
    def _creates_cycle_optimized(self, genome: NEATGenome, input_node: int, output_node: int) -> bool:
        """Optimized cycle detection using reachability analysis."""
        # Quick check: if output can reach input, adding connection creates cycle
        return self._can_reach(genome, output_node, input_node)
    
    def _can_reach(self, genome: NEATGenome, start_node: int, target_node: int) -> bool:
        """Check if start_node can reach target_node using BFS."""
        if start_node == target_node:
            return True
        
        # Build adjacency list for enabled connections only
        adjacency = {node_id: [] for node_id in genome.nodes.keys()}
        for conn in genome.connections.values():
            if conn.enabled:
                if conn.input_node in adjacency:
                    adjacency[conn.input_node].append(conn.output_node)
        
        # BFS to check reachability
        visited = set()
        queue = [start_node]
        visited.add(start_node)
        
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, []):
                if neighbor == target_node:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False


def activation_function(x: jnp.ndarray, func_type: ActivationFunction) -> jnp.ndarray:
    """Apply activation function."""
    if func_type == ActivationFunction.SIGMOID:
        return jax.nn.sigmoid(x)
    elif func_type == ActivationFunction.TANH:
        return jnp.tanh(x)
    elif func_type == ActivationFunction.RELU:
        return jax.nn.relu(x)
    elif func_type == ActivationFunction.LINEAR:
        return x
    else:
        return jax.nn.sigmoid(x)  # default


def crossover(parent1: NEATGenome, parent2: NEATGenome) -> NEATGenome:
    """Create offspring through crossover of two parent genomes."""
    # Debug: check input parameters
    if parent1 is None or parent2 is None:
        print(f"⚠️  Crossover: One of the parents is None: parent1={parent1}, parent2={parent2}")
        return None
    
    if parent1.fitness < parent2.fitness:
        parent1, parent2 = parent2, parent1  # Ensure parent1 is fitter
    
    child = NEATGenome(parent1.input_size, parent1.output_size)
    
    # Copy nodes from both parents (union)
    all_node_ids = set(parent1.nodes.keys()) | set(parent2.nodes.keys())
    for node_id in all_node_ids:
        if node_id in parent1.nodes:
            child.nodes[node_id] = NodeGene(
                node_id=parent1.nodes[node_id].node_id,
                node_type=parent1.nodes[node_id].node_type,
                activation=parent1.nodes[node_id].activation,
                x_position=parent1.nodes[node_id].x_position,
                y_position=parent1.nodes[node_id].y_position
            )
        else:
            child.nodes[node_id] = NodeGene(
                node_id=parent2.nodes[node_id].node_id,
                node_type=parent2.nodes[node_id].node_type,
                activation=parent2.nodes[node_id].activation,
                x_position=parent2.nodes[node_id].x_position,
                y_position=parent2.nodes[node_id].y_position
            )
    
    # Handle connections
    all_innovations = set(parent1.connections.keys()) | set(parent2.connections.keys())
    
    for innovation in all_innovations:
        if innovation in parent1.connections and innovation in parent2.connections:
            # Matching gene - randomly choose from either parent
            parent_conn = random.choice([parent1.connections[innovation], 
                                       parent2.connections[innovation]])
            child.connections[innovation] = parent_conn.copy()
        elif innovation in parent1.connections:
            # Disjoint/excess gene from fitter parent
            child.connections[innovation] = parent1.connections[innovation].copy()
        # Ignore disjoint/excess genes from less fit parent
    
    return child


def calculate_compatibility_distance(genome1: NEATGenome, genome2: NEATGenome,
                                   c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float:
    """Calculate compatibility distance between two genomes."""
    innovations1 = set(genome1.connections.keys())
    innovations2 = set(genome2.connections.keys())
    
    # Calculate excess and disjoint genes
    max_innovation1 = max(innovations1) if innovations1 else 0
    max_innovation2 = max(innovations2) if innovations2 else 0
    max_innovation = max(max_innovation1, max_innovation2)
    
    excess = 0
    disjoint = 0
    matching = 0
    weight_diff_sum = 0.0
    
    all_innovations = innovations1 | innovations2
    
    for innovation in all_innovations:
        if innovation in innovations1 and innovation in innovations2:
            # Matching gene
            matching += 1
            weight1 = genome1.connections[innovation].weight
            weight2 = genome2.connections[innovation].weight
            weight_diff_sum += abs(weight1 - weight2)
        else:
            # Non-matching gene
            if innovation > min(max_innovation1, max_innovation2):
                excess += 1
            else:
                disjoint += 1
    
    # Calculate average weight difference
    avg_weight_diff = weight_diff_sum / max(1, matching)
    
    # Normalize by genome size
    N = max(len(innovations1), len(innovations2), 1)
    
    # Calculate compatibility distance
    distance = (c1 * excess / N) + (c2 * disjoint / N) + (c3 * avg_weight_diff)
    
    return distance


class NEATConfig:
    """NEAT算法的配置参数"""
    
    def __init__(self):
        # 种群参数
        self.population_size: int = 150
        self.elite_size: int = 15
        
        # 突变参数 - 平衡权重和结构突变
        self.weight_mutation_rate: float = 0.6      # 降低权重突变率
        self.weight_mutation_power: float = 0.5     # 权重突变强度
        self.weight_mutation_std: float = 0.1       # 权重突变标准差
        
        # 结构突变参数 - 提高结构突变率
        self.add_connection_rate: float = 0.3       # 提高连接突变率
        self.add_node_rate: float = 0.2             # 提高节点突变率
        self.remove_connection_rate: float = 0.1    # 移除连接率
        self.remove_node_rate: float = 0.05         # 移除节点率
        
        # 连接参数
        self.connection_weight_range: Tuple[float, float] = (-3.0, 3.0)
        self.connection_bias_range: Tuple[float, float] = (-3.0, 3.0)
        
        # 激活函数参数
        self.activation_mutation_rate: float = 0.1
        self.available_activations: List[ActivationFunction] = [
            ActivationFunction.SIGMOID,
            ActivationFunction.TANH,
            ActivationFunction.RELU,
            ActivationFunction.LINEAR
        ]
        
        # 物种参数
        self.species_threshold: float = 3.0
        self.species_elite_size: int = 2
        self.species_stagnation_limit: int = 15
        
        # 适应度参数
        self.fitness_threshold: float = 100.0
        self.max_generations: int = 1000
        
        # 创新参数
        self.excess_coefficient: float = 1.0
        self.disjoint_coefficient: float = 1.0
        self.weight_coefficient: float = 0.4
        
        # 任务特定参数
        self.task_type: str = "slimevolley"
        self.curriculum_learning: bool = True
        self.adaptive_mutation: bool = True
    
    def get_slimevolley_config(self) -> 'NEATConfig':
        """获取针对SlimeVolley任务优化的配置"""
        config = NEATConfig()
        
        # SlimeVolley特定优化
        config.population_size = 200  # 更大的种群
        config.elite_size = 20
        
        # 平衡的突变策略
        config.weight_mutation_rate = 0.5      # 适中的权重突变
        config.add_connection_rate = 0.4       # 高连接突变
        config.add_node_rate = 0.25            # 高节点突变
        
        # 物种参数优化
        config.species_threshold = 2.5         # 更严格的物种划分
        config.species_stagnation_limit = 20   # 更长的停滞容忍
        
        # 任务特定设置
        config.task_type = "slimevolley"
        config.curriculum_learning = True
        config.adaptive_mutation = True
        
        return config
    
    def get_adaptive_config(self, generation: int, best_fitness: float) -> 'NEATConfig':
        """根据训练进度自适应调整配置"""
        config = NEATConfig()
        
        # 基于代数的自适应调整
        if generation < 50:
            # 早期阶段：鼓励探索
            config.weight_mutation_rate = 0.7
            config.add_connection_rate = 0.5
            config.add_node_rate = 0.3
            config.species_threshold = 2.0
        elif generation < 200:
            # 中期阶段：平衡探索和利用
            config.weight_mutation_rate = 0.6
            config.add_connection_rate = 0.4
            config.add_node_rate = 0.25
            config.species_threshold = 2.5
        else:
            # 后期阶段：偏向利用
            config.weight_mutation_rate = 0.4
            config.add_connection_rate = 0.3
            config.add_node_rate = 0.2
            config.species_threshold = 3.0
        
        # 基于适应度的自适应调整
        if best_fitness > 50:
            # 高适应度：减少结构突变，优化权重
            config.add_connection_rate *= 0.7
            config.add_node_rate *= 0.7
            config.weight_mutation_rate *= 1.2
        elif best_fitness < -10:
            # 低适应度：增加结构突变，探索新拓扑
            config.add_connection_rate *= 1.3
            config.add_node_rate *= 1.3
            config.weight_mutation_rate *= 0.8
        
        return config


class CurriculumLearning:
    """课程学习系统，逐步增加任务难度"""
    
    def __init__(self, task_type: str = "slimevolley"):
        self.task_type = task_type
        self.current_level = 0
        self.max_levels = 5
        self.level_progression = self._create_level_progression()
        
    def _create_level_progression(self) -> List[Dict]:
        """创建SlimeVolley的课程级别"""
        if self.task_type == "slimevolley":
            return [
                # 级别0：基础控制
                {
                    'name': '基础控制',
                    'description': '学习基本的移动和跳跃',
                    'max_steps': 1000,
                    'ball_speed_multiplier': 0.5,
                    'opponent_difficulty': 0.3,
                    'reward_multiplier': 1.0,
                    'required_fitness': -50
                },
                # 级别1：球跟踪
                {
                    'name': '球跟踪',
                    'description': '学习跟踪球的位置和运动',
                    'max_steps': 1500,
                    'ball_speed_multiplier': 0.7,
                    'opponent_difficulty': 0.5,
                    'reward_multiplier': 1.2,
                    'required_fitness': -30
                },
                # 级别2：基础击球
                {
                    'name': '基础击球',
                    'description': '学习基本的击球技能',
                    'max_steps': 2000,
                    'ball_speed_multiplier': 0.8,
                    'opponent_difficulty': 0.6,
                    'reward_multiplier': 1.5,
                    'required_fitness': -10
                },
                # 级别3：策略游戏
                {
                    'name': '策略游戏',
                    'description': '学习进攻和防守策略',
                    'max_steps': 2500,
                    'ball_speed_multiplier': 0.9,
                    'opponent_difficulty': 0.8,
                    'reward_multiplier': 1.8,
                    'required_fitness': 10
                },
                # 级别4：高级对战
                {
                    'name': '高级对战',
                    'description': '挑战高难度对手',
                    'max_steps': 3000,
                    'ball_speed_multiplier': 1.0,
                    'opponent_difficulty': 1.0,
                    'reward_multiplier': 2.0,
                    'required_fitness': 30
                }
            ]
        else:
            # 默认课程
            return [
                {
                    'name': '基础',
                    'description': '基础任务',
                    'max_steps': 1000,
                    'difficulty': 0.5,
                    'reward_multiplier': 1.0,
                    'required_fitness': 0
                }
            ]
    
    def get_current_level_config(self) -> Dict:
        """获取当前级别的配置"""
        if self.current_level < len(self.level_progression):
            return self.level_progression[self.current_level]
        return self.level_progression[-1]  # 返回最高级别
    
    def should_progress(self, best_fitness: float, generation: int) -> bool:
        """判断是否应该进入下一级别"""
        current_config = self.get_current_level_config()
        required_fitness = current_config.get('required_fitness', 0)
        
        # 检查适应度要求
        if best_fitness >= required_fitness:
            return True
        
        # 检查代数要求（避免过早升级）
        min_generations = 20 + self.current_level * 10
        if generation >= min_generations:
            return True
        
        return False
    
    def progress_to_next_level(self) -> bool:
        """进入下一级别"""
        if self.current_level < self.max_levels - 1:
            self.current_level += 1
            return True
        return False
    
    def get_level_summary(self) -> str:
        """获取当前级别的摘要"""
        config = self.get_current_level_config()
        return f"级别 {self.current_level}: {config['name']} - {config['description']}"
    
    def reset_to_level(self, level: int):
        """重置到指定级别"""
        self.current_level = max(0, min(level, self.max_levels - 1))
    
    def get_adaptive_opponent(self, base_opponent, current_fitness: float) -> Dict:
        """获取自适应难度的对手"""
        current_config = self.get_current_level_config()
        base_difficulty = current_config.get('opponent_difficulty', 0.5)
        
        # 基于当前适应度调整难度
        if current_fitness > 20:
            # 高适应度：增加难度
            adaptive_difficulty = min(base_difficulty * 1.3, 1.0)
        elif current_fitness < -20:
            # 低适应度：降低难度
            adaptive_difficulty = max(base_difficulty * 0.7, 0.1)
        else:
            adaptive_difficulty = base_difficulty
        
        return {
            'difficulty': adaptive_difficulty,
            'ball_speed': current_config.get('ball_speed_multiplier', 1.0),
            'max_steps': current_config.get('max_steps', 3000),
            'reward_multiplier': current_config.get('reward_multiplier', 1.0)
        }


class BehavioralDiversity:
    """行为多样性系统，鼓励智能体发展不同的策略"""
    
    def __init__(self, diversity_threshold: float = 0.3):
        self.diversity_threshold = diversity_threshold
        self.behavior_archive = []  # 存储不同的行为模式
        self.diversity_metrics = {
            'position_diversity': 0.0,
            'action_diversity': 0.0,
            'strategy_diversity': 0.0,
            'timing_diversity': 0.0
        }
    
    def calculate_behavior_signature(self, genome: NEATGenome, episode_data: Dict) -> Dict:
        """
        计算智能体的行为特征签名
        用于衡量行为多样性
        """
        behavior_signature = {
            'position_pattern': self._extract_position_pattern(episode_data),
            'action_pattern': self._extract_action_pattern(episode_data),
            'strategy_pattern': self._extract_strategy_pattern(episode_data),
            'timing_pattern': self._extract_timing_pattern(episode_data),
            'network_complexity': genome.get_network_complexity()
        }
        
        return behavior_signature
    
    def _extract_position_pattern(self, episode_data: Dict) -> List[float]:
        """提取位置模式特征"""
        if 'agent_positions' not in episode_data:
            return [0.0] * 10
        
        positions = episode_data['agent_positions']
        if len(positions) < 10:
            return [0.0] * 10
        
        # 计算位置统计特征
        x_positions = [pos[0] for pos in positions]
        y_positions = [pos[1] for pos in positions]
        
        pattern = [
            np.mean(x_positions),      # 平均X位置
            np.std(x_positions),       # X位置标准差
            np.mean(y_positions),      # 平均Y位置
            np.std(y_positions),       # Y位置标准差
            np.max(x_positions),       # 最大X位置
            np.min(x_positions),       # 最小X位置
            np.max(y_positions),       # 最大Y位置
            np.min(y_positions),       # 最小Y位置
            len(set([round(x, 1) for x in x_positions])),  # X位置多样性
            len(set([round(y, 1) for y in y_positions]))   # Y位置多样性
        ]
        
        return pattern
    
    def _extract_action_pattern(self, episode_data: Dict) -> List[float]:
        """提取动作模式特征"""
        if 'actions_taken' not in episode_data:
            return [0.0] * 8
        
        actions_taken = episode_data['actions_taken']
        effective_actions = episode_data.get('effective_actions', 0)
        
        # 计算动作统计特征
        pattern = [
            actions_taken,                    # 总动作数
            effective_actions,                # 有效动作数
            effective_actions / max(actions_taken, 1),  # 动作效率
            episode_data.get('forward_actions', 0),     # 前进动作数
            episode_data.get('backward_actions', 0),    # 后退动作数
            episode_data.get('jump_actions', 0),        # 跳跃动作数
            episode_data.get('idle_actions', 0),        # 空闲动作数
            episode_data.get('action_sequences', [])    # 动作序列模式
        ]
        
        return pattern
    
    def _extract_strategy_pattern(self, episode_data: Dict) -> List[float]:
        """提取策略模式特征"""
        pattern = [
            episode_data.get('offensive_plays', 0),     # 进攻次数
            episode_data.get('defensive_plays', 0),     # 防守次数
            episode_data.get('counter_attacks', 0),     # 反击次数
            episode_data.get('ball_control_time', 0),   # 控球时间
            episode_data.get('field_control', 0),       # 场地控制度
            episode_data.get('pressure_applied', 0),    # 施加压力
            episode_data.get('risk_taking', 0),         # 冒险行为
            episode_data.get('adaptation_speed', 0)     # 适应速度
        ]
        
        return pattern
    
    def _extract_timing_pattern(self, episode_data: Dict) -> List[float]:
        """提取时机模式特征"""
        pattern = [
            episode_data.get('reaction_time', 0),       # 反应时间
            episode_data.get('anticipation_time', 0),   # 预测时间
            episode_data.get('jump_timing_accuracy', 0), # 跳跃时机准确性
            episode_data.get('hit_timing_accuracy', 0),  # 击球时机准确性
            episode_data.get('movement_timing', 0),      # 移动时机
            episode_data.get('decision_delay', 0),       # 决策延迟
            episode_data.get('execution_speed', 0),      # 执行速度
            episode_data.get('timing_consistency', 0)    # 时机一致性
        ]
        
        return pattern
    
    def calculate_diversity_score(self, behavior_signature: Dict, 
                                 population_signatures: List[Dict]) -> float:
        """
        计算行为多样性分数
        基于与种群中其他智能体的行为差异
        """
        if not population_signatures:
            return 1.0  # 第一个智能体获得最高多样性分数
        
        total_diversity = 0.0
        comparisons = 0
        
        for other_signature in population_signatures:
            diversity = self._calculate_pairwise_diversity(
                behavior_signature, other_signature
            )
            total_diversity += diversity
            comparisons += 1
        
        avg_diversity = total_diversity / comparisons if comparisons > 0 else 0.0
        
        # 归一化到[0, 1]范围
        normalized_diversity = min(avg_diversity / self.diversity_threshold, 1.0)
        
        return normalized_diversity
    
    def _calculate_pairwise_diversity(self, sig1: Dict, sig2: Dict) -> float:
        """计算两个行为签名之间的多样性"""
        diversity = 0.0
        
        # 位置模式多样性
        pos_diff = np.mean(np.abs(np.array(sig1['position_pattern']) - 
                                 np.array(sig2['position_pattern'])))
        diversity += pos_diff * 0.3
        
        # 动作模式多样性
        act_diff = np.mean(np.abs(np.array(sig1['action_pattern']) - 
                                 np.array(sig2['action_pattern'])))
        diversity += act_diff * 0.25
        
        # 策略模式多样性
        strat_diff = np.mean(np.abs(np.array(sig1['strategy_pattern']) - 
                                   np.array(sig2['strategy_pattern'])))
        diversity += strat_diff * 0.25
        
        # 时机模式多样性
        timing_diff = np.mean(np.abs(np.array(sig1['timing_pattern']) - 
                                    np.array(sig2['timing_pattern'])))
        diversity += timing_diff * 0.2
        
        return diversity
    
    def update_diversity_metrics(self, population_signatures: List[Dict]):
        """更新多样性指标"""
        if len(population_signatures) < 2:
            return
        
        # 计算种群多样性指标
        diversity_scores = []
        for i, sig1 in enumerate(population_signatures):
            other_signatures = [sig2 for j, sig2 in enumerate(population_signatures) if i != j]
            diversity_score = self.calculate_diversity_score(sig1, other_signatures)
            diversity_scores.append(diversity_score)
        
        # 更新全局多样性指标
        self.diversity_metrics['overall_diversity'] = np.mean(diversity_scores)
        self.diversity_metrics['diversity_std'] = np.std(diversity_scores)
        
        # 更新行为档案
        self.behavior_archive.extend(population_signatures)
        
        # 保持档案大小合理
        if len(self.behavior_archive) > 1000:
            self.behavior_archive = self.behavior_archive[-500:]
    
    def get_diversity_bonus(self, behavior_signature: Dict, 
                           population_signatures: List[Dict]) -> float:
        """
        获取多样性奖励
        鼓励智能体发展独特的行为策略
        """
        diversity_score = self.calculate_diversity_score(behavior_signature, population_signatures)
        
        # 多样性奖励：独特的行为获得更高奖励
        diversity_bonus = diversity_score * 2.0
        
        # 额外奖励：非常独特的行为
        if diversity_score > 0.8:
            diversity_bonus += 1.0
        
        return diversity_bonus
    
    def suggest_behavior_exploration(self, genome: NEATGenome, 
                                   current_diversity: float) -> List[str]:
        """
        建议行为探索方向
        帮助智能体发展新的策略
        """
        suggestions = []
        
        if current_diversity < 0.3:
            suggestions.append("尝试不同的移动模式")
            suggestions.append("探索新的击球角度")
            suggestions.append("改变防守策略")
        
        if current_diversity < 0.5:
            suggestions.append("尝试预测性移动")
            suggestions.append("探索冒险性进攻")
            suggestions.append("改变跳跃时机")
        
        if current_diversity < 0.7:
            suggestions.append("尝试复杂的组合动作")
            suggestions.append("探索心理战术")
            suggestions.append("改变节奏控制")
        
        return suggestions


class IntermediateObjectives:
    """中间目标系统，定义SlimeVolley的具体目标"""
    
    def __init__(self):
        self.objectives = {
            'ball_control': {
                'maximize_ball_contacts': 0.0,
                'minimize_opponent_contacts': 0.0,
                'ball_control_time': 0.0,
                'successful_passes': 0.0
            },
            'position_control': {
                'center_court_control': 0.0,
                'force_opponent_difficult_positions': 0.0,
                'maintain_advantageous_position': 0.0,
                'field_coverage': 0.0
            },
            'tactical_play': {
                'offensive_pressure': 0.0,
                'defensive_solidarity': 0.0,
                'counter_attack_opportunities': 0.0,
                'momentum_control': 0.0
            },
            'technical_skills': {
                'jump_timing_accuracy': 0.0,
                'hit_angle_optimization': 0.0,
                'movement_efficiency': 0.0,
                'reaction_speed': 0.0
            }
        }
    
    def calculate_objective_scores(self, episode_data: Dict) -> Dict:
        """
        计算中间目标分数
        基于episode数据评估各个目标的完成情况
        """
        scores = {}
        
        # 球控制目标
        scores['ball_control'] = self._calculate_ball_control_scores(episode_data)
        
        # 位置控制目标
        scores['position_control'] = self._calculate_position_control_scores(episode_data)
        
        # 战术游戏目标
        scores['tactical_play'] = self._calculate_tactical_play_scores(episode_data)
        
        # 技术技能目标
        scores['technical_skills'] = self._calculate_technical_skills_scores(episode_data)
        
        return scores
    
    def _calculate_ball_control_scores(self, episode_data: Dict) -> Dict:
        """计算球控制相关分数"""
        scores = {}
        
        # 最大化球接触
        total_ball_contacts = episode_data.get('ball_contacts', 0)
        max_possible_contacts = episode_data.get('max_possible_contacts', 100)
        scores['maximize_ball_contacts'] = min(total_ball_contacts / max_possible_contacts, 1.0)
        
        # 最小化对手球接触
        opponent_contacts = episode_data.get('opponent_ball_contacts', 0)
        scores['minimize_opponent_contacts'] = max(0, 1.0 - opponent_contacts / max_possible_contacts)
        
        # 控球时间
        ball_control_time = episode_data.get('ball_control_time', 0)
        total_time = episode_data.get('total_time', 1)
        scores['ball_control_time'] = ball_control_time / total_time if total_time > 0 else 0
        
        # 成功传球
        successful_passes = episode_data.get('successful_passes', 0)
        total_passes = episode_data.get('total_passes', 1)
        scores['successful_passes'] = successful_passes / total_passes if total_passes > 0 else 0
        
        return scores
    
    def _calculate_position_control_scores(self, episode_data: Dict) -> Dict:
        """计算位置控制相关分数"""
        scores = {}
        
        # 中场控制
        center_court_time = episode_data.get('center_court_time', 0)
        total_time = episode_data.get('total_time', 1)
        scores['center_court_control'] = center_court_time / total_time if total_time > 0 else 0
        
        # 迫使对手到困难位置
        opponent_difficult_positions = episode_data.get('opponent_difficult_positions', 0)
        total_opponent_positions = episode_data.get('total_opponent_positions', 1)
        scores['force_opponent_difficult_positions'] = (
            opponent_difficult_positions / total_opponent_positions 
            if total_opponent_positions > 0 else 0
        )
        
        # 保持有利位置
        advantageous_position_time = episode_data.get('advantageous_position_time', 0)
        scores['maintain_advantageous_position'] = (
            advantageous_position_time / total_time if total_time > 0 else 0
        )
        
        # 场地覆盖
        field_coverage = episode_data.get('field_coverage', 0)
        scores['field_coverage'] = min(field_coverage / 100, 1.0)  # 归一化到[0,1]
        
        return scores
    
    def _calculate_tactical_play_scores(self, episode_data: Dict) -> Dict:
        """计算战术游戏相关分数"""
        scores = {}
        
        # 进攻压力
        offensive_pressure = episode_data.get('offensive_pressure', 0)
        scores['offensive_pressure'] = min(offensive_pressure / 10, 1.0)
        
        # 防守稳固性
        defensive_solidarity = episode_data.get('defensive_solidarity', 0)
        scores['defensive_solidarity'] = min(defensive_solidarity / 10, 1.0)
        
        # 反击机会
        counter_attack_opportunities = episode_data.get('counter_attack_opportunities', 0)
        scores['counter_attack_opportunities'] = min(counter_attack_opportunities / 5, 1.0)
        
        # 势头控制
        momentum_control = episode_data.get('momentum_control', 0)
        scores['momentum_control'] = min(momentum_control / 10, 1.0)
        
        return scores
    
    def _calculate_technical_skills_scores(self, episode_data: Dict) -> Dict:
        """计算技术技能相关分数"""
        scores = {}
        
        # 跳跃时机准确性
        jump_timing_accuracy = episode_data.get('jump_timing_accuracy', 0)
        scores['jump_timing_accuracy'] = min(jump_timing_accuracy / 100, 1.0)
        
        # 击球角度优化
        hit_angle_optimization = episode_data.get('hit_angle_optimization', 0)
        scores['hit_angle_optimization'] = min(hit_angle_optimization / 100, 1.0)
        
        # 移动效率
        movement_efficiency = episode_data.get('movement_efficiency', 0)
        scores['movement_efficiency'] = min(movement_efficiency / 100, 1.0)
        
        # 反应速度
        reaction_speed = episode_data.get('reaction_speed', 0)
        scores['reaction_speed'] = min(reaction_speed / 100, 1.0)
        
        return scores
    
    def get_objective_rewards(self, objective_scores: Dict) -> float:
        """
        根据中间目标分数计算奖励
        鼓励智能体完成多个目标
        """
        total_reward = 0.0
        
        # 球控制奖励 (权重: 0.3)
        ball_control_reward = sum(objective_scores['ball_control'].values()) / 4
        total_reward += ball_control_reward * 0.3
        
        # 位置控制奖励 (权重: 0.25)
        position_control_reward = sum(objective_scores['position_control'].values()) / 4
        total_reward += position_control_reward * 0.25
        
        # 战术游戏奖励 (权重: 0.25)
        tactical_play_reward = sum(objective_scores['tactical_play'].values()) / 4
        total_reward += tactical_play_reward * 0.25
        
        # 技术技能奖励 (权重: 0.2)
        technical_skills_reward = sum(objective_scores['technical_skills'].values()) / 4
        total_reward += technical_skills_reward * 0.2
        
        # 额外奖励：完成所有目标
        all_objectives_completed = all(
            score > 0.7 for category in objective_scores.values() 
            for score in category.values()
        )
        if all_objectives_completed:
            total_reward += 2.0
        
        return total_reward
    
    def get_objective_feedback(self, objective_scores: Dict) -> List[str]:
        """
        获取目标完成情况的反馈
        帮助智能体了解需要改进的方面
        """
        feedback = []
        
        # 分析球控制目标
        ball_control = objective_scores['ball_control']
        if ball_control['maximize_ball_contacts'] < 0.5:
            feedback.append("需要增加球接触次数")
        if ball_control['minimize_opponent_contacts'] < 0.5:
            feedback.append("需要减少对手的球接触")
        
        # 分析位置控制目标
        position_control = objective_scores['position_control']
        if position_control['center_court_control'] < 0.5:
            feedback.append("需要更好地控制中场")
        if position_control['force_opponent_difficult_positions'] < 0.5:
            feedback.append("需要迫使对手到困难位置")
        
        # 分析战术游戏目标
        tactical_play = objective_scores['tactical_play']
        if tactical_play['offensive_pressure'] < 0.5:
            feedback.append("需要增加进攻压力")
        if tactical_play['defensive_solidarity'] < 0.5:
            feedback.append("需要加强防守")
        
        # 分析技术技能目标
        technical_skills = objective_scores['technical_skills']
        if technical_skills['jump_timing_accuracy'] < 0.5:
            feedback.append("需要提高跳跃时机准确性")
        if technical_skills['hit_angle_optimization'] < 0.5:
            feedback.append("需要优化击球角度")
        
        return feedback


class PhysicsUnderstanding:
    """物理理解系统，帮助智能体理解SlimeVolley的物理机制"""
    
    def __init__(self):
        # 物理常量
        self.GRAVITY = -9.8 * 2 * 1.5  # SlimeVolley的重力
        self.BALL_RADIUS = 0.5
        self.PLAYER_RADIUS = 1.5
        self.TIMESTEP = 1/30.0
        
        # 物理理解指标
        self.physics_metrics = {
            'trajectory_prediction': 0.0,
            'momentum_understanding': 0.0,
            'collision_physics': 0.0,
            'timing_accuracy': 0.0
        }
    
    def predict_ball_trajectory(self, ball_state: Dict, time_steps: int = 10) -> List[Dict]:
        """
        预测球的轨迹
        基于当前状态和物理定律
        """
        trajectory = []
        current_state = ball_state.copy()
        
        for step in range(time_steps):
            # 更新位置
            current_state['x'] += current_state['vx'] * self.TIMESTEP
            current_state['y'] += current_state['vy'] * self.TIMESTEP
            
            # 更新速度（重力影响）
            current_state['vy'] += self.GRAVITY * self.TIMESTEP
            
            # 检查边界碰撞
            current_state = self._handle_boundary_collisions(current_state)
            
            # 记录状态
            trajectory.append(current_state.copy())
            
            # 检查是否落地
            if current_state['y'] <= self.BALL_RADIUS + 1.5:  # 地面高度
                break
        
        return trajectory
    
    def _handle_boundary_collisions(self, ball_state: Dict) -> Dict:
        """处理边界碰撞"""
        new_state = ball_state.copy()
        
        # 左右边界碰撞
        if new_state['x'] <= -12 + self.BALL_RADIUS:  # 左边界
            new_state['x'] = -12 + self.BALL_RADIUS
            new_state['vx'] = -new_state['vx'] * 0.8  # 能量损失
        
        if new_state['x'] >= 12 - self.BALL_RADIUS:  # 右边界
            new_state['x'] = 12 - self.BALL_RADIUS
            new_state['vx'] = -new_state['vx'] * 0.8  # 能量损失
        
        # 上下边界碰撞
        if new_state['y'] <= 1.5 + self.BALL_RADIUS:  # 地面
            new_state['y'] = 1.5 + self.BALL_RADIUS
            new_state['vy'] = -new_state['vy'] * 0.6  # 地面能量损失
        
        if new_state['y'] >= 24 - self.BALL_RADIUS:  # 天花板
            new_state['y'] = 24 - self.BALL_RADIUS
            new_state['vy'] = -new_state['vy'] * 0.8  # 天花板能量损失
        
        # 中间墙碰撞
        if (abs(new_state['x']) <= 1.0 + self.BALL_RADIUS and 
            new_state['y'] <= 3.5):  # 中间墙
            if new_state['x'] > 0:
                new_state['x'] = 1.0 + self.BALL_RADIUS
            else:
                new_state['x'] = -1.0 - self.BALL_RADIUS
            new_state['vx'] = -new_state['vx'] * 0.7  # 墙的能量损失
        
        return new_state
    
    def calculate_optimal_hit_angle(self, ball_state: Dict, target_position: Dict) -> Dict:
        """
        计算最佳击球角度
        考虑目标位置和物理约束
        """
        # 计算从球到目标的向量
        dx = target_position['x'] - ball_state['x']
        dy = target_position['y'] - ball_state['y']
        
        # 计算距离
        distance = np.sqrt(dx**2 + dy**2)
        
        # 计算最佳角度（考虑重力）
        if distance > 0:
            # 使用抛物线方程计算最佳角度
            optimal_angle = np.arctan2(dy, dx)
            
            # 考虑重力影响的调整
            gravity_adjustment = self.GRAVITY * distance / (2 * 10**2)  # 简化计算
            optimal_angle += gravity_adjustment
            
            # 限制角度范围
            optimal_angle = np.clip(optimal_angle, -np.pi/2, np.pi/2)
        else:
            optimal_angle = 0
        
        # 计算所需速度
        required_speed = 15.0  # 基础速度
        
        # 返回最佳击球参数
        return {
            'angle': optimal_angle,
            'speed': required_speed,
            'vx': required_speed * np.cos(optimal_angle),
            'vy': required_speed * np.sin(optimal_angle)
        }
    
    def calculate_jump_timing(self, ball_state: Dict, player_state: Dict) -> Dict:
        """
        计算最佳跳跃时机
        考虑球的位置、速度和玩家位置
        """
        # 预测球何时到达玩家位置
        ball_to_player = {
            'x': player_state['x'] - ball_state['x'],
            'y': player_state['y'] - ball_state['y']
        }
        
        # 计算到达时间
        if abs(ball_state['vx']) > 0.1:
            time_to_reach_x = ball_to_player['x'] / ball_state['vx']
        else:
            time_to_reach_x = float('inf')
        
        # 考虑重力影响的Y轴到达时间
        if ball_state['vy'] != 0:
            # 使用二次方程求解
            a = self.GRAVITY / 2
            b = ball_state['vy']
            c = ball_to_player['y']
            
            if b**2 - 4*a*c >= 0:
                time_to_reach_y = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                if time_to_reach_y < 0:
                    time_to_reach_y = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
            else:
                time_to_reach_y = float('inf')
        else:
            time_to_reach_y = float('inf')
        
        # 选择较早的到达时间
        time_to_reach = min(time_to_reach_x, time_to_reach_y)
        
        # 计算跳跃时机
        if time_to_reach > 0 and time_to_reach < 5.0:  # 合理的时间范围
            jump_timing = max(0, time_to_reach - 0.1)  # 提前0.1秒跳跃
            jump_recommended = True
        else:
            jump_timing = 0
            jump_recommended = False
        
        return {
            'jump_recommended': jump_recommended,
            'jump_timing': jump_timing,
            'time_to_reach': time_to_reach,
            'ball_to_player': ball_to_player
        }
    
    def analyze_collision_physics(self, ball_state: Dict, player_state: Dict) -> Dict:
        """
        分析碰撞物理
        计算碰撞后的状态变化
        """
        # 计算球和玩家的距离
        dx = ball_state['x'] - player_state['x']
        dy = ball_state['y'] - player_state['y']
        distance = np.sqrt(dx**2 + dy**2)
        
        # 检查是否发生碰撞
        collision_occurred = distance <= (self.BALL_RADIUS + self.PLAYER_RADIUS)
        
        if collision_occurred:
            # 计算碰撞角度
            collision_angle = np.arctan2(dy, dx)
            
            # 计算碰撞后的速度（简化模型）
            # 假设完全弹性碰撞，但有一些能量损失
            energy_loss = 0.8
            
            # 计算新的速度
            new_vx = ball_state['vx'] * energy_loss * np.cos(collision_angle)
            new_vy = ball_state['vy'] * energy_loss * np.sin(collision_angle)
            
            # 添加一些随机性
            new_vx += np.random.normal(0, 0.5)
            new_vy += np.random.normal(0, 0.5)
            
            collision_result = {
                'collision_occurred': True,
                'collision_angle': collision_angle,
                'new_vx': new_vx,
                'new_vy': new_vy,
                'energy_loss': energy_loss,
                'distance': distance
            }
        else:
            collision_result = {
                'collision_occurred': False,
                'distance': distance
            }
        
        return collision_result
    
    def calculate_physics_understanding_score(self, episode_data: Dict) -> float:
        """
        计算物理理解分数
        基于智能体在游戏中的物理表现
        """
        score = 0.0
        
        # 轨迹预测准确性
        trajectory_accuracy = episode_data.get('trajectory_prediction_accuracy', 0)
        score += trajectory_accuracy * 0.3
        
        # 动量理解
        momentum_understanding = episode_data.get('momentum_understanding', 0)
        score += momentum_understanding * 0.25
        
        # 碰撞物理
        collision_physics = episode_data.get('collision_physics', 0)
        score += collision_physics * 0.25
        
        # 时机准确性
        timing_accuracy = episode_data.get('timing_accuracy', 0)
        score += timing_accuracy * 0.2
        
        return min(score, 1.0)
    
    def get_physics_feedback(self, physics_score: float) -> List[str]:
        """
        获取物理理解的反馈
        帮助智能体改进物理理解
        """
        feedback = []
        
        if physics_score < 0.3:
            feedback.append("需要学习基本的物理定律")
            feedback.append("需要理解重力的影响")
            feedback.append("需要学习碰撞的基本原理")
        
        elif physics_score < 0.6:
            feedback.append("需要提高轨迹预测能力")
            feedback.append("需要优化跳跃时机")
            feedback.append("需要改进击球角度")
        
        elif physics_score < 0.8:
            feedback.append("需要掌握高级物理技巧")
            feedback.append("需要优化能量传递")
            feedback.append("需要提高预测精度")
        
        else:
            feedback.append("物理理解很好，继续保持")
            feedback.append("可以尝试更复杂的物理技巧")
        
        return feedback


class MixedOpponentTraining:
    """混合对手训练系统，支持多种AI难度和对手类型"""
    
    def __init__(self):
        self.opponent_types = {
            'baseline_ai': {
                'difficulty': 1.0,
                'description': '内置AI',
                'weight': 0.3
            },
            'random_agent': {
                'difficulty': 0.2,
                'description': '随机动作智能体',
                'weight': 0.1
            },
            'rule_based_agent': {
                'difficulty': 0.5,
                'description': '基于规则的智能体',
                'weight': 0.2
            },
            'previous_best': {
                'difficulty': 0.8,
                'description': '之前的最佳智能体',
                'weight': 0.2
            },
            'self_play': {
                'difficulty': 0.9,
                'description': '自我对战',
                'weight': 0.2
            }
        }
        
        self.opponent_history = []
        self.current_opponent = 'baseline_ai'
        self.opponent_rotation = []
        self.rotation_index = 0
    
    def select_opponent(self, current_fitness: float, generation: int, 
                       population_diversity: float) -> str:
        """
        选择训练对手
        基于当前适应度、代数和种群多样性
        """
        # 基于适应度的对手选择
        if current_fitness < -20:
            # 低适应度：选择简单对手
            candidates = ['random_agent', 'rule_based_agent']
            weights = [0.7, 0.3]
        elif current_fitness < 0:
            # 中等适应度：混合对手
            candidates = ['rule_based_agent', 'baseline_ai', 'previous_best']
            weights = [0.4, 0.4, 0.2]
        else:
            # 高适应度：选择困难对手
            candidates = ['baseline_ai', 'previous_best', 'self_play']
            weights = [0.3, 0.4, 0.3]
        
        # 基于代数的调整
        if generation < 50:
            # 早期：避免自我对战
            candidates = [c for c in candidates if c != 'self_play']
            weights = weights[:len(candidates)]
            weights = [w/sum(weights) for w in weights]
        
        # 基于种群多样性的调整
        if population_diversity < 0.3:
            # 低多样性：增加随机对手
            if 'random_agent' in candidates:
                idx = candidates.index('random_agent')
                weights[idx] *= 1.5
                weights = [w/sum(weights) for w in weights]
        
        # 选择对手
        selected_opponent = np.random.choice(candidates, p=weights)
        
        # 更新历史
        self.opponent_history.append({
            'generation': generation,
            'opponent': selected_opponent,
            'fitness': current_fitness,
            'diversity': population_diversity
        })
        
        self.current_opponent = selected_opponent
        return selected_opponent
    
    def get_opponent_config(self, opponent_type: str) -> Dict:
        """获取对手配置"""
        if opponent_type not in self.opponent_types:
            return self.opponent_types['baseline_ai']
        
        base_config = self.opponent_types[opponent_type].copy()
        
        # 动态调整难度
        if opponent_type == 'baseline_ai':
            # 内置AI难度可以动态调整
            base_config['dynamic_difficulty'] = True
            base_config['difficulty_range'] = [0.5, 1.5]
        elif opponent_type == 'rule_based_agent':
            # 基于规则的智能体可以调整策略
            base_config['strategy_variation'] = True
            base_config['strategy_count'] = 5
        
        return base_config
    
    def create_rule_based_agent(self, strategy_type: str = 'balanced') -> Dict:
        """
        创建基于规则的智能体
        实现不同的策略类型
        """
        strategies = {
            'defensive': {
                'description': '防守型策略',
                'position_weight': 0.8,
                'ball_tracking_weight': 0.9,
                'aggression_weight': 0.2,
                'risk_tolerance': 0.1
            },
            'aggressive': {
                'description': '进攻型策略',
                'position_weight': 0.4,
                'ball_tracking_weight': 0.7,
                'aggression_weight': 0.9,
                'risk_tolerance': 0.8
            },
            'balanced': {
                'description': '平衡型策略',
                'position_weight': 0.6,
                'ball_tracking_weight': 0.8,
                'aggression_weight': 0.6,
                'risk_tolerance': 0.5
            },
            'counter_attack': {
                'description': '反击型策略',
                'position_weight': 0.7,
                'ball_tracking_weight': 0.9,
                'aggression_weight': 0.5,
                'risk_tolerance': 0.6
            },
            'position_control': {
                'description': '位置控制策略',
                'position_weight': 0.9,
                'ball_tracking_weight': 0.6,
                'aggression_weight': 0.3,
                'risk_tolerance': 0.4
            }
        }
        
        return strategies.get(strategy_type, strategies['balanced'])
    
    def create_random_agent(self, skill_level: float = 0.2) -> Dict:
        """
        创建随机动作智能体
        基于技能水平调整随机性
        """
        return {
            'description': f'随机智能体 (技能水平: {skill_level:.1f})',
            'randomness': 1.0 - skill_level,
            'skill_level': skill_level,
            'action_probabilities': {
                'forward': 0.3,
                'backward': 0.3,
                'jump': 0.4
            }
        }
    
    def update_opponent_weights(self, performance_history: List[Dict]):
        """
        基于性能历史更新对手权重
        动态调整训练策略
        """
        if len(performance_history) < 10:
            return
        
        # 分析最近10代的性能
        recent_performance = performance_history[-10:]
        
        # 计算每种对手的平均性能
        opponent_performance = {}
        for record in recent_performance:
            opponent = record.get('opponent', 'unknown')
            fitness = record.get('fitness', 0)
            
            if opponent not in opponent_performance:
                opponent_performance[opponent] = []
            opponent_performance[opponent].append(fitness)
        
        # 计算平均性能
        for opponent, performances in opponent_performance.items():
            if len(performances) >= 3:  # 至少3次记录
                avg_performance = np.mean(performances)
                
                # 调整权重：性能越差，权重越高（提供更多挑战）
                if opponent in self.opponent_types:
                    current_weight = self.opponent_types[opponent]['weight']
                    
                    # 基于性能调整权重
                    if avg_performance < -10:
                        # 性能差：增加权重
                        new_weight = min(current_weight * 1.2, 0.5)
                    elif avg_performance > 20:
                        # 性能好：减少权重
                        new_weight = max(current_weight * 0.8, 0.1)
                    else:
                        # 性能中等：保持权重
                        new_weight = current_weight
                    
                    self.opponent_types[opponent]['weight'] = new_weight
        
        # 重新归一化权重
        total_weight = sum(self.opponent_types[opp]['weight'] for opp in self.opponent_types)
        for opponent in self.opponent_types:
            self.opponent_types[opponent]['weight'] /= total_weight
    
    def get_training_progression_plan(self, current_generation: int, 
                                    target_generations: int) -> List[Dict]:
        """
        获取训练进度计划
        规划不同阶段的对手策略
        """
        progression_plan = []
        
        # 阶段1：基础技能学习 (0-20%)
        phase1_end = int(target_generations * 0.2)
        if current_generation <= phase1_end:
            progression_plan.append({
                'phase': '基础技能学习',
                'generations': f'0-{phase1_end}',
                'opponents': ['random_agent', 'rule_based_agent'],
                'focus': '学习基本移动和球跟踪',
                'difficulty': '低'
            })
        
        # 阶段2：技能巩固 (20-50%)
        phase2_start = int(target_generations * 0.2)
        phase2_end = int(target_generations * 0.5)
        if current_generation > phase2_start and current_generation <= phase2_end:
            progression_plan.append({
                'phase': '技能巩固',
                'generations': f'{phase2_start}-{phase2_end}',
                'opponents': ['rule_based_agent', 'baseline_ai'],
                'focus': '巩固基本技能，学习简单策略',
                'difficulty': '中低'
            })
        
        # 阶段3：策略学习 (50-80%)
        phase3_start = int(target_generations * 0.5)
        phase3_end = int(target_generations * 0.8)
        if current_generation > phase3_start and current_generation <= phase3_end:
            progression_plan.append({
                'phase': '策略学习',
                'generations': f'{phase3_start}-{phase3_end}',
                'opponents': ['baseline_ai', 'previous_best'],
                'focus': '学习高级策略，提高反应速度',
                'difficulty': '中高'
            })
        
        # 阶段4：高级对战 (80-100%)
        phase4_start = int(target_generations * 0.8)
        if current_generation > phase4_start:
            progression_plan.append({
                'phase': '高级对战',
                'generations': f'{phase4_start}-{target_generations}',
                'opponents': ['previous_best', 'self_play'],
                'focus': '高级策略，自我对战优化',
                'difficulty': '高'
            })
        
        return progression_plan
    
    def get_opponent_analysis(self) -> Dict:
        """获取对手分析报告"""
        if not self.opponent_history:
            return {}
        
        analysis = {}
        
        # 按对手类型分组
        for opponent_type in self.opponent_types.keys():
            opponent_records = [r for r in self.opponent_history if r['opponent'] == opponent_type]
            
            if opponent_records:
                fitnesses = [r['fitness'] for r in opponent_records]
                diversities = [r['diversity'] for r in opponent_records]
                
                analysis[opponent_type] = {
                    'usage_count': len(opponent_records),
                    'avg_fitness': np.mean(fitnesses),
                    'fitness_std': np.std(fitnesses),
                    'avg_diversity': np.mean(diversities),
                    'last_used': max(r['generation'] for r in opponent_records),
                    'performance_trend': 'improving' if len(fitnesses) >= 2 and fitnesses[-1] > fitnesses[0] else 'stable'
                }
        
        return analysis


class TargetedSubSkillsTraining:
    """目标子技能训练系统，支持单独训练特定技能"""
    
    def __init__(self):
        self.sub_skills = {
            'ball_tracking': {
                'description': '球跟踪技能',
                'difficulty': 0.3,
                'required_fitness': -15.0,
                'training_focus': '学习跟踪球的运动轨迹',
                'evaluation_metrics': ['ball_distance', 'tracking_accuracy', 'reaction_time']
            },
            'positioning': {
                'description': '位置控制技能',
                'difficulty': 0.4,
                'required_fitness': -12.0,
                'training_focus': '学习控制智能体位置',
                'evaluation_metrics': ['position_accuracy', 'movement_efficiency', 'field_coverage']
            },
            'jumping_timing': {
                'description': '跳跃时机技能',
                'difficulty': 0.5,
                'required_fitness': -10.0,
                'training_focus': '学习跳跃的最佳时机',
                'evaluation_metrics': ['jump_accuracy', 'timing_precision', 'success_rate']
            },
            'ball_hitting': {
                'description': '击球技能',
                'difficulty': 0.6,
                'required_fitness': -8.0,
                'training_focus': '学习如何击球',
                'evaluation_metrics': ['hit_accuracy', 'power_control', 'direction_control']
            },
            'strategy_planning': {
                'description': '策略规划技能',
                'difficulty': 0.7,
                'required_fitness': -5.0,
                'training_focus': '学习制定游戏策略',
                'evaluation_metrics': ['strategy_effectiveness', 'adaptation_speed', 'decision_quality']
            },
            'defensive_play': {
                'description': '防守技能',
                'difficulty': 0.6,
                'required_fitness': -8.0,
                'training_focus': '学习防守技巧',
                'evaluation_metrics': ['defensive_coverage', 'pressure_resistance', 'counter_attack']
            },
            'offensive_play': {
                'description': '进攻技能',
                'difficulty': 0.7,
                'required_fitness': -5.0,
                'training_focus': '学习进攻技巧',
                'evaluation_metrics': ['offensive_pressure', 'scoring_efficiency', 'opportunity_creation']
            }
        }
        
        self.current_skill = 'ball_tracking'
        self.skill_progression = []
        self.skill_performance = {}
    
    def select_training_skill(self, current_fitness: float, 
                            skill_performance: Dict) -> str:
        """
        选择要训练的子技能
        基于当前适应度和技能表现
        """
        # 按难度排序技能
        sorted_skills = sorted(self.sub_skills.items(), 
                              key=lambda x: x[1]['difficulty'])
        
        # 找到当前应该训练的技能
        for skill_name, skill_info in sorted_skills:
            if current_fitness >= skill_info['required_fitness']:
                # 检查该技能是否已经掌握
                if skill_name in skill_performance:
                    skill_score = skill_performance[skill_name]
                    if skill_score < 0.7:  # 技能未完全掌握
                        return skill_name
                else:
                    # 新技能，开始训练
                    return skill_name
        
        # 如果所有基础技能都掌握了，选择最高级技能
        return sorted_skills[-1][0]
    
    def create_skill_specific_task(self, skill_name: str) -> Dict:
        """
        创建技能特定的训练任务
        为每个子技能设计专门的训练环境
        """
        if skill_name not in self.sub_skills:
            return self._create_default_task()
        
        skill_info = self.sub_skills[skill_name]
        
        if skill_name == 'ball_tracking':
            return self._create_ball_tracking_task()
        elif skill_name == 'positioning':
            return self._create_positioning_task()
        elif skill_name == 'jumping_timing':
            return self._create_jumping_timing_task()
        elif skill_name == 'ball_hitting':
            return self._create_ball_hitting_task()
        elif skill_name == 'strategy_planning':
            return self._create_strategy_planning_task()
        elif skill_name == 'defensive_play':
            return self._create_defensive_play_task()
        elif skill_name == 'offensive_play':
            return self._create_offensive_play_task()
        else:
            return self._create_default_task()
    
    def _create_ball_tracking_task(self) -> Dict:
        """创建球跟踪训练任务"""
        return {
            'task_type': 'ball_tracking',
            'description': '跟踪球的运动轨迹',
            'environment_modifications': {
                'ball_speed_multiplier': 0.5,  # 降低球速
                'opponent_difficulty': 0.1,    # 降低对手难度
                'max_steps': 300,              # 增加时间
                'reward_focus': 'ball_proximity'
            },
            'success_criteria': {
                'ball_distance_threshold': 2.0,
                'tracking_time_threshold': 0.8,
                'reaction_time_threshold': 0.3
            },
            'reward_structure': {
                'ball_proximity_bonus': 2.0,
                'tracking_accuracy_bonus': 1.5,
                'reaction_speed_bonus': 1.0
            }
        }
    
    def _create_positioning_task(self) -> Dict:
        """创建位置控制训练任务"""
        return {
            'task_type': 'positioning',
            'description': '控制智能体位置',
            'environment_modifications': {
                'ball_speed_multiplier': 0.3,
                'opponent_difficulty': 0.05,
                'max_steps': 400,
                'reward_focus': 'position_accuracy'
            },
            'success_criteria': {
                'position_accuracy_threshold': 0.8,
                'movement_efficiency_threshold': 0.7,
                'field_coverage_threshold': 0.6
            },
            'reward_structure': {
                'position_accuracy_bonus': 2.0,
                'movement_efficiency_bonus': 1.5,
                'field_coverage_bonus': 1.0
            }
        }
    
    def _create_jumping_timing_task(self) -> Dict:
        """创建跳跃时机训练任务"""
        return {
            'task_type': 'jumping_timing',
            'description': '掌握跳跃时机',
            'environment_modifications': {
                'ball_speed_multiplier': 0.4,
                'opponent_difficulty': 0.1,
                'max_steps': 350,
                'reward_focus': 'jump_timing'
            },
            'success_criteria': {
                'jump_accuracy_threshold': 0.8,
                'timing_precision_threshold': 0.7,
                'success_rate_threshold': 0.6
            },
            'reward_structure': {
                'jump_accuracy_bonus': 2.0,
                'timing_precision_bonus': 1.5,
                'success_rate_bonus': 1.0
            }
        }
    
    def _create_ball_hitting_task(self) -> Dict:
        """创建击球训练任务"""
        return {
            'task_type': 'ball_hitting',
            'description': '学习击球技巧',
            'environment_modifications': {
                'ball_speed_multiplier': 0.6,
                'opponent_difficulty': 0.2,
                'max_steps': 300,
                'reward_focus': 'hit_accuracy'
            },
            'success_criteria': {
                'hit_accuracy_threshold': 0.7,
                'power_control_threshold': 0.6,
                'direction_control_threshold': 0.6
            },
            'reward_structure': {
                'hit_accuracy_bonus': 2.0,
                'power_control_bonus': 1.5,
                'direction_control_bonus': 1.0
            }
        }
    
    def _create_strategy_planning_task(self) -> Dict:
        """创建策略规划训练任务"""
        return {
            'task_type': 'strategy_planning',
            'description': '制定游戏策略',
            'environment_modifications': {
                'ball_speed_multiplier': 0.8,
                'opponent_difficulty': 0.4,
                'max_steps': 250,
                'reward_focus': 'strategy_effectiveness'
            },
            'success_criteria': {
                'strategy_effectiveness_threshold': 0.6,
                'adaptation_speed_threshold': 0.7,
                'decision_quality_threshold': 0.6
            },
            'reward_structure': {
                'strategy_effectiveness_bonus': 2.0,
                'adaptation_speed_bonus': 1.5,
                'decision_quality_bonus': 1.0
            }
        }
    
    def _create_defensive_play_task(self) -> Dict:
        """创建防守训练任务"""
        return {
            'task_type': 'defensive_play',
            'description': '学习防守技巧',
            'environment_modifications': {
                'ball_speed_multiplier': 0.7,
                'opponent_difficulty': 0.3,
                'max_steps': 280,
                'reward_focus': 'defensive_coverage'
            },
            'success_criteria': {
                'defensive_coverage_threshold': 0.7,
                'pressure_resistance_threshold': 0.6,
                'counter_attack_threshold': 0.5
            },
            'reward_structure': {
                'defensive_coverage_bonus': 2.0,
                'pressure_resistance_bonus': 1.5,
                'counter_attack_bonus': 1.0
            }
        }
    
    def _create_offensive_play_task(self) -> Dict:
        """创建进攻训练任务"""
        return {
            'task_type': 'offensive_play',
            'description': '学习进攻技巧',
            'environment_modifications': {
                'ball_speed_multiplier': 0.8,
                'opponent_difficulty': 0.3,
                'max_steps': 280,
                'reward_focus': 'offensive_pressure'
            },
            'success_criteria': {
                'offensive_pressure_threshold': 0.6,
                'scoring_efficiency_threshold': 0.5,
                'opportunity_creation_threshold': 0.6
            },
            'reward_structure': {
                'offensive_pressure_bonus': 2.0,
                'scoring_efficiency_bonus': 1.5,
                'opportunity_creation_bonus': 1.0
            }
        }
    
    def _create_default_task(self) -> Dict:
        """创建默认训练任务"""
        return {
            'task_type': 'general',
            'description': '通用训练任务',
            'environment_modifications': {
                'ball_speed_multiplier': 1.0,
                'opponent_difficulty': 0.5,
                'max_steps': 200,
                'reward_focus': 'overall_performance'
            },
            'success_criteria': {
                'overall_performance_threshold': 0.5
            },
            'reward_structure': {
                'overall_performance_bonus': 1.0
            }
        }
    
    def evaluate_skill_performance(self, skill_name: str, episode_data: Dict) -> float:
        """
        评估特定技能的表现
        返回0-1之间的分数
        """
        if skill_name not in self.sub_skills:
            return 0.0
        
        skill_info = self.sub_skills[skill_name]
        evaluation_metrics = skill_info['evaluation_metrics']
        
        total_score = 0.0
        metric_count = 0
        
        for metric in evaluation_metrics:
            if metric in episode_data:
                metric_score = episode_data[metric]
                if isinstance(metric_score, (int, float)):
                    # 归一化到[0, 1]范围
                    normalized_score = min(max(metric_score, 0.0), 1.0)
                    total_score += normalized_score
                    metric_count += 1
        
        if metric_count == 0:
            return 0.0
        
        return total_score / metric_count
    
    def update_skill_progression(self, skill_name: str, performance: float, 
                               generation: int):
        """更新技能进度"""
        self.skill_progression.append({
            'generation': generation,
            'skill': skill_name,
            'performance': performance,
            'timestamp': time.time()
        })
        
        # 更新技能表现
        if skill_name not in self.skill_performance:
            self.skill_performance[skill_name] = []
        
        self.skill_performance[skill_name].append(performance)
        
        # 保持历史记录在合理范围内
        if len(self.skill_performance[skill_name]) > 50:
            self.skill_performance[skill_name] = self.skill_performance[skill_name][-25:]
    
    def get_skill_training_report(self) -> Dict:
        """获取技能训练报告"""
        report = {
            'current_skill': self.current_skill,
            'skill_progression': self.skill_progression[-10:],  # 最近10次
            'skill_performance_summary': {},
            'recommendations': []
        }
        
        # 分析每个技能的表现
        for skill_name, performances in self.skill_performance.items():
            if performances:
                avg_performance = np.mean(performances)
                recent_performance = np.mean(performances[-5:]) if len(performances) >= 5 else avg_performance
                
                report['skill_performance_summary'][skill_name] = {
                    'average_performance': avg_performance,
                    'recent_performance': recent_performance,
                    'performance_trend': 'improving' if recent_performance > avg_performance else 'stable',
                    'training_sessions': len(performances)
                }
        
        # 生成训练建议
        for skill_name, skill_info in self.sub_skills.items():
            if skill_name in self.skill_performance:
                current_performance = np.mean(self.skill_performance[skill_name][-3:]) if len(self.skill_performance[skill_name]) >= 3 else 0.0
                
                if current_performance < 0.5:
                    report['recommendations'].append(f"需要加强{skill_info['description']}训练")
                elif current_performance < 0.8:
                    report['recommendations'].append(f"继续巩固{skill_info['description']}")
                else:
                    report['recommendations'].append(f"{skill_info['description']}已掌握，可以学习新技能")
        
        return report
    
    def should_advance_to_next_skill(self, current_skill: str, 
                                   current_performance: float) -> bool:
        """判断是否应该进入下一个技能"""
        if current_skill not in self.sub_skills:
            return False
        
        skill_info = self.sub_skills[current_skill]
        
        # 检查是否达到技能要求
        if current_performance >= 0.8:  # 80%掌握度
            return True
        
        # 检查是否训练时间过长
        if current_skill in self.skill_performance:
            training_sessions = len(self.skill_performance[current_skill])
            if training_sessions > 20:  # 超过20次训练
                return True
        
        return False
    
    def get_next_skill(self, current_skill: str) -> str:
        """获取下一个要训练的技能"""
        sorted_skills = sorted(self.sub_skills.items(), 
                              key=lambda x: x[1]['difficulty'])
        
        current_index = -1
        for i, (skill_name, _) in enumerate(sorted_skills):
            if skill_name == current_skill:
                current_index = i
                break
        
        if current_index >= 0 and current_index < len(sorted_skills) - 1:
            return sorted_skills[current_index + 1][0]
        
        return current_skill  # 如果没有下一个，保持当前技能


class Species:
    """Represents a species in NEAT algorithm."""
    
    def __init__(self, species_id: int, representative: NEATGenome):
        self.species_id = species_id
        self.representative = representative
        self.members: List[NEATGenome] = [representative]
        self.best_fitness = representative.fitness
        self.generations_without_improvement = 0
        self.average_fitness = representative.fitness
        
    def add_member(self, genome: NEATGenome):
        """Add a genome to this species."""
        self.members.append(genome)
        self._update_fitness_stats()
    
    def remove_member(self, genome: NEATGenome):
        """Remove a genome from this species."""
        if genome in self.members:
            self.members.remove(genome)
            if self.members:  # If species not empty
                self._update_fitness_stats()
    
    def _update_fitness_stats(self):
        """Update fitness statistics for the species."""
        if not self.members:
            return
        
        fitnesses = [member.fitness for member in self.members]
        self.average_fitness = np.mean(fitnesses)
        self.best_fitness = max(fitnesses)
    
    def should_stagnate(self, stagnation_limit: int) -> bool:
        """Check if species should be marked as stagnant."""
        return self.generations_without_improvement >= stagnation_limit
    
    def select_representative(self):
        """Select a new representative for the species."""
        if self.members:
            # Select the genome with highest fitness as representative
            self.representative = max(self.members, key=lambda g: g.fitness)
    
    def get_size(self) -> int:
        """Get the number of members in this species."""
        return len(self.members)
    
    def is_empty(self) -> bool:
        """Check if species has no members."""
        return len(self.members) == 0


class NEATPopulation:
    """Manages a population of NEAT genomes with speciation."""
    
    def __init__(self, input_size: int, output_size: int, population_size: int = 150):
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        self.generation = 0
        
        # Population management
        self.population: List[NEATGenome] = []
        self.species: List[Species] = []
        self.innovation_tracker = InnovationTracker()
        self.mutator = NEATMutator(self.innovation_tracker)
        
        # NEAT parameters
        self.species_threshold = 3.0
        self.species_elite_size = 2
        self.species_stagnation_limit = 15
        self.elite_size = 15
        
        # Initialize population
        self._create_initial_population()
    
    def _create_initial_population(self):
        """Create initial population with minimal genomes."""
        try:
            from .neat_network import create_minimal_genome
        except ImportError:
            from neat_network import create_minimal_genome
        
        print(f"🧬 Creating initial NEAT population ({self.population_size} genomes)")
        
        for i in range(self.population_size):
            genome = create_minimal_genome(
                self.input_size, 
                self.output_size, 
                self.innovation_tracker
            )
            self.population.append(genome)
        
        # Initial speciation
        self._speciate_population()
        print(f"✅ Initial population created with {len(self.species)} species")
    
    def _speciate_population(self):
        """Divide population into species based on compatibility."""
        self.species.clear()
        
        for genome in self.population:
            # Find compatible species
            assigned = False
            for species in self.species:
                if self._is_compatible(genome, species.representative):
                    species.add_member(genome)
                    genome.species_id = species.species_id
                    assigned = True
                    break
            
            # Create new species if no compatible one found
            if not assigned:
                new_species_id = len(self.species)
                new_species = Species(new_species_id, genome)
                genome.species_id = new_species_id
                self.species.append(new_species)
    
    def _is_compatible(self, genome1: NEATGenome, genome2: NEATGenome) -> bool:
        """Check if two genomes are compatible for speciation."""
        # Debug: check if genomes are None
        if genome1 is None or genome2 is None:
            print(f"⚠️  One of the genomes is None: genome1={genome1}, genome2={genome2}")
            return False
        
        distance = calculate_compatibility_distance(genome1, genome2)
        return distance < self.species_threshold
    
    def evaluate_fitness(self, fitness_function):
        """Evaluate fitness for all genomes in the population."""
        print(f"🏆 Evaluating fitness for generation {self.generation}")
        
        for i, genome in enumerate(self.population):
            try:
                fitness = fitness_function(genome)
                genome.fitness = fitness
                if i % 10 == 0:
                    print(f"   Genome {i+1}/{len(self.population)}: {fitness:.3f}")
            except Exception as e:
                print(f"⚠️  Error evaluating genome {i}: {e}")
                genome.fitness = float('-inf')
        
        # Update species fitness statistics
        for species in self.species:
            species._update_fitness_stats()
    
    def select_parents(self) -> Tuple[NEATGenome, NEATGenome]:
        """Select two parents for reproduction."""
        # Debug: check population state
        if not self.population:
            print(f"⚠️  select_parents: Population is empty!")
            return None, None
        
        # Tournament selection
        tournament_size = 3
        parent1 = self._tournament_select(tournament_size)
        parent2 = self._tournament_select(tournament_size)
        
        # Debug: check selected parents
        if parent1 is None or parent2 is None:
            print(f"⚠️  select_parents: One of the parents is None: parent1={parent1}, parent2={parent2}")
            print(f"    Population size: {len(self.population)}")
            print(f"    Population fitnesses: {[g.fitness for g in self.population]}")
        
        # Ensure different parents
        while parent1 == parent2 and len(self.population) > 1:
            parent2 = self._tournament_select(tournament_size)
        
        return parent1, parent2
    
    def _tournament_select(self, tournament_size: int) -> NEATGenome:
        """Select a genome using tournament selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.fitness)
    
    def reproduce(self) -> List[NEATGenome]:
        """Create new population through reproduction."""
        new_population = []
        
        # Elitism: keep best genomes from each species
        for species in self.species:
            if species.get_size() > 0:
                # Sort by fitness and keep elites
                species.members.sort(key=lambda g: g.fitness, reverse=True)
                elite_count = min(self.species_elite_size, species.get_size())
                new_population.extend(species.members[:elite_count])
        
        # Fill remaining population with offspring
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            
            # Debug: check parents
            if parent1 is None or parent2 is None:
                print(f"⚠️  select_parents returned None: parent1={parent1}, parent2={parent2}")
                break
            
            # Crossover
            if random.random() < 0.75:  # 75% chance of crossover
                offspring = crossover(parent1, parent2)
                # Debug: check if crossover returned None
                if offspring is None:
                    print(f"⚠️  Crossover returned None, using parent1 instead")
                    offspring = parent1
            else:
                offspring = parent1
            
            # Debug: check offspring before mutation
            if offspring is None:
                print(f"⚠️  Offspring is None before mutation")
                break
            
            # Mutation
            self.mutator.mutate(offspring)
            
            # Debug: check offspring after mutation
            if offspring is None:
                print(f"⚠️  Offspring is None after mutation")
                break
            
            new_population.append(offspring)
        
        # Trim to exact population size
        new_population = new_population[:self.population_size]
        
        # Update population and generation
        self.population = new_population
        self.generation += 1
        
        # Re-speciate
        self._speciate_population()
        
        return new_population
    
    def get_best_genome(self) -> Optional[NEATGenome]:
        """Get the genome with highest fitness."""
        if not self.population:
            return None
        return max(self.population, key=lambda g: g.fitness)
    
    def get_population_stats(self) -> Dict:
        """Get statistics about the current population."""
        if not self.population:
            return {}
        
        fitnesses = [g.fitness for g in self.population]
        
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'species_count': len(self.species),
            'best_fitness': max(fitnesses),
            'average_fitness': np.mean(fitnesses),
            'fitness_std': np.std(fitnesses),
            'min_fitness': min(fitnesses)
        }
    
    def print_stats(self):
        """Print current population statistics."""
        stats = self.get_population_stats()
        if not stats:
            return
        
        print(f"\n📊 Generation {stats['generation']} Statistics:")
        print(f"   Population Size: {stats['population_size']}")
        print(f"   Species Count: {stats['species_count']}")
        print(f"   Best Fitness: {stats['best_fitness']:.3f}")
        print(f"   Average Fitness: {stats['average_fitness']:.3f}")
        print(f"   Fitness Std: {stats['fitness_std']:.3f}")
        print(f"   Min Fitness: {stats['min_fitness']:.3f}")


def test_cycle_detection():
    """Test the new cycle detection algorithms."""
    print("🧪 Testing Cycle Detection Algorithms...")
    
    # Create a simple genome for testing
    genome = NEATGenome(2, 1)
    mutator = NEATMutator(InnovationTracker())
    
    # Test 1: No cycles should be detected in initial genome
    print("  Test 1: Initial genome (should have no cycles)")
    has_cycle = mutator._has_cycle(genome)
    print(f"    Has cycle: {has_cycle} (Expected: False)")
    
    # Test 2: Add a valid connection (should not create cycle)
    print("  Test 2: Adding valid connection")
    can_add = not mutator._creates_cycle_optimized(genome, 0, 2)  # input 0 to hidden 2
    print(f"    Can add connection 0->2: {can_add} (Expected: True)")
    
    # Test 3: Add a connection that would create cycle
    print("  Test 3: Adding connection that creates cycle")
    # First add a hidden node
    hidden_node = NodeGene(
        node_id=2,
        node_type=NodeType.HIDDEN,
        activation=ActivationFunction.TANH,  # 使用TANH增加多样性
        x_position=0.5,
        y_position=0.5
    )
    genome.add_node(hidden_node)
    
    # Add connection from input to hidden
    conn1 = ConnectionGene(
        innovation_number=1,
        input_node=0,
        output_node=2,
        weight=1.0,
        enabled=True
    )
    genome.add_connection(conn1)
    
    # Add connection from hidden to output
    conn2 = ConnectionGene(
        innovation_number=2,
        input_node=2,
        output_node=1,
        weight=1.0,
        enabled=True
    )
    genome.add_connection(conn2)
    
    # Now try to add a connection that would create a cycle
    would_create_cycle = mutator._creates_cycle_optimized(genome, 2, 0)  # hidden 2 to input 0
    print(f"    Would create cycle with 2->0: {would_create_cycle} (Expected: True)")
    
    # Test 4: Test the full cycle detection
    print("  Test 4: Full cycle detection")
    full_cycle_detected = mutator._creates_cycle(genome, 2, 0)
    print(f"    Full cycle detection for 2->0: {full_cycle_detected} (Expected: True)")
    
    # Test 5: Test reachability
    print("  Test 5: Reachability analysis")
    can_reach = mutator._can_reach(genome, 2, 0)  # Can hidden 2 reach input 0?
    print(f"    Hidden 2 can reach input 0: {can_reach} (Expected: False)")
    can_reach_output = mutator._can_reach(genome, 0, 1)  # Can input 0 reach output 1?
    print(f"    Input 0 can reach output 1: {can_reach_output} (Expected: True)")
    
    print("✅ Cycle detection tests completed!")


if __name__ == "__main__":
    test_cycle_detection()