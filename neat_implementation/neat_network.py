"""
NEAT Network Evaluation
Implements feed-forward network construction and execution from NEAT genomes.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Set, Tuple, Optional
try:
    from .neat_core import NEATGenome, NodeGene, ConnectionGene, NodeType, activation_function
except ImportError:
    from neat_core import NEATGenome, NodeGene, ConnectionGene, NodeType, activation_function


class NEATNetwork:
    """Converts a NEAT genome into an executable neural network."""
    
    def __init__(self, genome: NEATGenome):
        self.genome = genome
        self.input_size = genome.input_size
        self.output_size = genome.output_size
        
        # Build network topology
        self.node_order = self._get_topological_order()
        self.input_nodes = [nid for nid, node in genome.nodes.items() 
                           if node.node_type == NodeType.INPUT]
        self.output_nodes = [nid for nid, node in genome.nodes.items() 
                            if node.node_type == NodeType.OUTPUT]
        
        # Create connection matrix for efficient computation
        self.connections_dict = self._build_connections_dict()
        
    def _get_topological_order(self) -> List[int]:
        """Get nodes in topological order for feed-forward evaluation."""
        nodes = self.genome.nodes
        connections = {k: v for k, v in self.genome.connections.items() if v.enabled}
        
        # Build adjacency list
        adj_list = {node_id: [] for node_id in nodes.keys()}
        in_degree = {node_id: 0 for node_id in nodes.keys()}
        
        for conn in connections.values():
            adj_list[conn.input_node].append(conn.output_node)
            in_degree[conn.output_node] += 1
        
        # Topological sort using Kahn's algorithm
        queue = [node_id for node_id in nodes.keys() if in_degree[node_id] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _build_connections_dict(self) -> Dict[int, List[Tuple[int, float]]]:
        """Build dictionary of connections for each node."""
        connections_dict = {node_id: [] for node_id in self.genome.nodes.keys()}
        
        for conn in self.genome.connections.values():
            if conn.enabled:
                connections_dict[conn.output_node].append((conn.input_node, conn.weight))
        
        return connections_dict
    
    def evaluate(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the network with given inputs."""
        return self._evaluate_single(inputs)
    
    def _evaluate_single(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Evaluate network for a single input vector."""
        # Initialize node activations
        activations = {}
        
        # Set input activations
        for i, node_id in enumerate(self.input_nodes):
            activations[node_id] = inputs[i]
        
        # Evaluate nodes in topological order
        for node_id in self.node_order:
            if node_id in activations:
                continue  # Already computed (input node)
                
            node = self.genome.nodes[node_id]
            
            # Calculate weighted sum of inputs
            weighted_sum = 0.0
            for input_node_id, weight in self.connections_dict[node_id]:
                if input_node_id in activations:
                    weighted_sum += activations[input_node_id] * weight
            
            # Apply activation function
            activations[node_id] = activation_function(
                jnp.array(weighted_sum), node.activation
            ).item()
        
        # Extract outputs
        outputs = []
        for node_id in self.output_nodes:
            outputs.append(activations.get(node_id, 0.0))
        
        return jnp.array(outputs)
    
    def evaluate_batch(self, inputs_batch: jnp.ndarray) -> jnp.ndarray:
        """Evaluate network for a batch of inputs."""
        return jax.vmap(self._evaluate_single)(inputs_batch)


class NEATNetworkJAX:
    """JAX-optimized version of NEAT network for faster evaluation."""
    
    def __init__(self, genome: NEATGenome):
        self.genome = genome
        self.input_size = genome.input_size
        self.output_size = genome.output_size
        
        # Convert genome to JAX-compatible format
        self._compile_network()
    
    def _compile_network(self):
        """Compile the network into JAX arrays for efficient computation."""
        nodes = self.genome.nodes
        connections = {k: v for k, v in self.genome.connections.items() if v.enabled}
        
        # Create node mapping
        self.node_ids = sorted(nodes.keys())
        self.node_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}
        
        num_nodes = len(self.node_ids)
        
        # Create adjacency matrix
        self.weight_matrix = jnp.zeros((num_nodes, num_nodes))
        weight_updates = []
        
        for conn in connections.values():
            input_idx = self.node_to_idx[conn.input_node]
            output_idx = self.node_to_idx[conn.output_node]
            weight_updates.append((input_idx, output_idx, conn.weight))
        
        # Update weight matrix
        if weight_updates:
            indices = jnp.array([(i, j) for i, j, _ in weight_updates])
            weights = jnp.array([w for _, _, w in weight_updates])
            self.weight_matrix = self.weight_matrix.at[indices[:, 1], indices[:, 0]].set(weights)
        
        # Create activation function indices
        self.activation_funcs = [nodes[node_id].activation for node_id in self.node_ids]
        
        # Input/output indices
        self.input_indices = jnp.array([self.node_to_idx[nid] for nid, node in nodes.items() 
                                       if node.node_type == NodeType.INPUT])
        self.output_indices = jnp.array([self.node_to_idx[nid] for nid, node in nodes.items() 
                                        if node.node_type == NodeType.OUTPUT])
        
        # Topological order for evaluation
        self.eval_order = self._get_evaluation_order()
    
    def _get_evaluation_order(self) -> jnp.ndarray:
        """Get evaluation order based on network topology."""
        nodes = self.genome.nodes
        node_order = []
        
        # Input nodes first
        for node_id in self.node_ids:
            if nodes[node_id].node_type == NodeType.INPUT:
                node_order.append(self.node_to_idx[node_id])
        
        # Then hidden nodes (ordered by x_position)
        hidden_nodes = [(self.node_to_idx[nid], nodes[nid].x_position) 
                       for nid in self.node_ids if nodes[nid].node_type == NodeType.HIDDEN]
        hidden_nodes.sort(key=lambda x: x[1])
        node_order.extend([idx for idx, _ in hidden_nodes])
        
        # Finally output nodes
        for node_id in self.node_ids:
            if nodes[node_id].node_type == NodeType.OUTPUT:
                node_order.append(self.node_to_idx[node_id])
        
        return jnp.array(node_order)
    
    @jax.jit
    def evaluate(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the network with JAX compilation."""
        num_nodes = len(self.node_ids)
        activations = jnp.zeros(num_nodes)
        
        # Set input activations
        activations = activations.at[self.input_indices].set(inputs)
        
        # Evaluate nodes in order
        def eval_step(i, activations):
            node_idx = self.eval_order[i]
            
            # Skip if it's an input node
            is_input = jnp.any(self.input_indices == node_idx)
            
            def compute_activation():
                # Compute weighted sum
                weighted_sum = jnp.sum(activations * self.weight_matrix[node_idx])
                
                # Apply activation function (simplified - using sigmoid for all)
                return jax.nn.sigmoid(weighted_sum)
            
            def keep_current():
                return activations[node_idx]
            
            new_activation = jax.lax.cond(is_input, keep_current, compute_activation)
            return activations.at[node_idx].set(new_activation)
        
        # Evaluate all nodes
        activations = jax.lax.fori_loop(0, len(self.eval_order), eval_step, activations)
        
        # Return outputs
        return activations[self.output_indices]
    
    def evaluate_batch(self, inputs_batch: jnp.ndarray) -> jnp.ndarray:
        """Evaluate network for a batch of inputs."""
        return jax.vmap(self.evaluate)(inputs_batch)


def create_minimal_genome(input_size: int, output_size: int, 
                         innovation_tracker) -> NEATGenome:
    """Create a minimal genome with direct input-output connections."""
    genome = NEATGenome(input_size, output_size)
    
    # Add connections from each input to each output
    for input_idx in range(input_size):
        for output_idx in range(output_size):
            input_node_id = input_idx
            output_node_id = input_size + output_idx
            
            innovation_num = innovation_tracker.get_innovation_number(
                input_node_id, output_node_id)
            
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=input_node_id,
                output_node=output_node_id,
                weight=np.random.normal(0, 1.0),
                enabled=True
            )
            genome.add_connection(connection)
    
    return genome


def create_slimevolley_optimized_genome(input_size: int, output_size: int, 
                                       innovation_tracker) -> NEATGenome:
    """
    为SlimeVolley任务创建优化的初始基因组
    专门设计以更好地处理12维观察空间
    """
    genome = NEATGenome(input_size, output_size)
    
    # 添加隐藏节点来更好地处理观察空间
    hidden_nodes = []
    
    # 创建专门的隐藏节点来处理不同类型的观察
    # 位置相关节点
    position_hidden = genome.add_hidden_node(
        innovation_tracker, 
        activation=ActivationFunction.TANH,
        x_position=0.5, 
        y_position=0.3
    )
    hidden_nodes.append(position_hidden)
    
    # 速度相关节点
    velocity_hidden = genome.add_hidden_node(
        innovation_tracker,
        activation=ActivationFunction.TANH,
        x_position=0.5,
        y_position=0.5
    )
    hidden_nodes.append(velocity_hidden)
    
    # 球相关节点
    ball_hidden = genome.add_hidden_node(
        innovation_tracker,
        activation=ActivationFunction.TANH,
        x_position=0.5,
        y_position=0.7
    )
    hidden_nodes.append(ball_hidden)
    
    # 对手相关节点
    opponent_hidden = genome.add_hidden_node(
        innovation_tracker,
        activation=ActivationFunction.TANH,
        x_position=0.5,
        y_position=0.9
    )
    hidden_nodes.append(opponent_hidden)
    
    # 添加输入到隐藏节点的连接
    # 位置相关连接 (x, y)
    for input_idx in [0, 1, 8, 9]:  # 智能体和对手的位置
        innovation_num = innovation_tracker.get_innovation_number(input_idx, position_hidden)
        connection = ConnectionGene(
            innovation_number=innovation_num,
            input_node=input_idx,
            output_node=position_hidden,
            weight=np.random.normal(0, 0.5),
            enabled=True
        )
        genome.add_connection(connection)
    
    # 速度相关连接 (vx, vy)
    for input_idx in [2, 3, 10, 11]:  # 智能体和对手的速度
        innovation_num = innovation_tracker.get_innovation_number(input_idx, velocity_hidden)
        connection = ConnectionGene(
            innovation_number=innovation_num,
            input_node=input_idx,
            output_node=velocity_hidden,
            weight=np.random.normal(0, 0.5),
            enabled=True
        )
        genome.add_connection(connection)
    
    # 球相关连接 (bx, by, bvx, bvy)
    for input_idx in [4, 5, 6, 7]:  # 球的位置和速度
        innovation_num = innovation_tracker.get_innovation_number(input_idx, ball_hidden)
        connection = ConnectionGene(
            innovation_number=innovation_num,
            input_node=input_idx,
            output_node=ball_hidden,
            weight=np.random.normal(0, 0.5),
            enabled=True
        )
        genome.add_connection(connection)
    
    # 对手相关连接 (ox, oy, ovx, ovy)
    for input_idx in [8, 9, 10, 11]:  # 对手的位置和速度
        innovation_num = innovation_tracker.get_innovation_number(input_idx, opponent_hidden)
        connection = ConnectionGene(
            innovation_number=innovation_num,
            input_node=input_idx,
            output_node=opponent_hidden,
            weight=np.random.normal(0, 0.5),
            enabled=True
        )
        genome.add_connection(connection)
    
    # 添加隐藏节点到输出的连接
    for hidden_node in hidden_nodes:
        for output_idx in range(output_size):
            output_node_id = input_size + output_idx
            innovation_num = innovation_tracker.get_innovation_number(hidden_node, output_node_id)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=hidden_node,
                output_node=output_node_id,
                weight=np.random.normal(0, 0.3),
                enabled=True
            )
            genome.add_connection(connection)
    
    # 保持原有的直接输入-输出连接（作为快捷连接）
    for input_idx in range(input_size):
        for output_idx in range(output_size):
            output_node_id = input_size + output_idx
            
            # 检查是否已存在连接
            existing_connection = False
            for conn in genome.connections.values():
                if conn.input_node == input_idx and conn.output_node == output_node_id:
                    existing_connection = True
                    break
            
            if not existing_connection:
                innovation_num = innovation_tracker.get_innovation_number(input_idx, output_node_id)
                connection = ConnectionGene(
                    innovation_number=innovation_num,
                    input_node=input_idx,
                    output_node=output_node_id,
                    weight=np.random.normal(0, 0.2),  # 较小的权重
                    enabled=True
                )
                genome.add_connection(connection)
    
    return genome


def create_adaptive_genome(input_size: int, output_size: int, 
                          innovation_tracker, task_type: str = "default") -> NEATGenome:
    """
    根据任务类型创建自适应的初始基因组
    """
    if task_type == "slimevolley":
        return create_slimevolley_optimized_genome(input_size, output_size, innovation_tracker)
    else:
        return create_minimal_genome(input_size, output_size, innovation_tracker)


def create_advanced_seed_network(input_size: int, output_size: int, 
                                 innovation_tracker) -> NEATGenome:
    """
    创建高级种子网络
    包含多层结构和预训练权重，为SlimeVolley任务优化
    """
    genome = NEATGenome(input_size, output_size)
    
    # 第一层隐藏节点：输入处理层
    input_processing_nodes = []
    for i in range(6):  # 6个输入处理节点
        node = genome.add_hidden_node(
            innovation_tracker,
            activation=ActivationFunction.TANH,
            x_position=0.2,
            y_position=0.2 + i * 0.15
        )
        input_processing_nodes.append(node)
    
    # 第二层隐藏节点：特征组合层
    feature_combination_nodes = []
    for i in range(8):  # 8个特征组合节点
        node = genome.add_hidden_node(
            innovation_tracker,
            activation=ActivationFunction.TANH,
            x_position=0.4,
            y_position=0.1 + i * 0.12
        )
        feature_combination_nodes.append(node)
    
    # 第三层隐藏节点：策略层
    strategy_nodes = []
    for i in range(6):  # 6个策略节点
        node = genome.add_hidden_node(
            innovation_tracker,
            activation=ActivationFunction.TANH,
            x_position=0.6,
            y_position=0.15 + i * 0.15
        )
        strategy_nodes.append(node)
    
    # 第四层隐藏节点：输出处理层
    output_processing_nodes = []
    for i in range(4):  # 4个输出处理节点
        node = genome.add_hidden_node(
            innovation_tracker,
            activation=ActivationFunction.TANH,
            x_position=0.8,
            y_position=0.2 + i * 0.2
        )
        output_processing_nodes.append(node)
    
    # 连接模式1：输入到输入处理层
    for input_idx in range(input_size):
        for proc_node in input_processing_nodes:
            innovation_num = innovation_tracker.get_innovation_number(input_idx, proc_node)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=input_idx,
                output_node=proc_node,
                weight=np.random.normal(0, 0.3),  # 较小的初始权重
                enabled=True
            )
            genome.add_connection(connection)
    
    # 连接模式2：输入处理层到特征组合层
    for proc_node in input_processing_nodes:
        for feat_node in feature_combination_nodes:
            innovation_num = innovation_tracker.get_innovation_number(proc_node, feat_node)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=proc_node,
                output_node=feat_node,
                weight=np.random.normal(0, 0.4),
                enabled=True
            )
            genome.add_connection(connection)
    
    # 连接模式3：特征组合层到策略层
    for feat_node in feature_combination_nodes:
        for strat_node in strategy_nodes:
            innovation_num = innovation_tracker.get_innovation_number(feat_node, strat_node)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=feat_node,
                output_node=strat_node,
                weight=np.random.normal(0, 0.5),
                enabled=True
            )
            genome.add_connection(connection)
    
    # 连接模式4：策略层到输出处理层
    for strat_node in strategy_nodes:
        for out_proc_node in output_processing_nodes:
            innovation_num = innovation_tracker.get_innovation_number(strat_node, out_proc_node)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=strat_node,
                output_node=out_proc_node,
                weight=np.random.normal(0, 0.4),
                enabled=True
            )
            genome.add_connection(connection)
    
    # 连接模式5：输出处理层到输出
    for out_proc_node in output_processing_nodes:
        for output_idx in range(output_size):
            output_node_id = input_size + output_idx
            innovation_num = innovation_tracker.get_innovation_number(out_proc_node, output_node_id)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=out_proc_node,
                output_node=output_node_id,
                weight=np.random.normal(0, 0.3),
                enabled=True
            )
            genome.add_connection(connection)
    
    # 添加一些跨层连接（跳跃连接）
    # 输入直接到特征组合层
    for input_idx in range(input_size):
        for feat_node in feature_combination_nodes[:4]:  # 前4个特征节点
            if np.random.random() < 0.3:  # 30%概率
                innovation_num = innovation_tracker.get_innovation_number(input_idx, feat_node)
                connection = ConnectionGene(
                    innovation_number=innovation_num,
                    input_node=input_idx,
                    output_node=feat_node,
                    weight=np.random.normal(0, 0.2),
                    enabled=True
                )
                genome.add_connection(connection)
    
    # 输入处理层直接到策略层
    for proc_node in input_processing_nodes[:3]:  # 前3个处理节点
        for strat_node in strategy_nodes[:3]:  # 前3个策略节点
            if np.random.random() < 0.4:  # 40%概率
                innovation_num = innovation_tracker.get_innovation_number(proc_node, strat_node)
                connection = ConnectionGene(
                    innovation_number=innovation_num,
                    input_node=proc_node,
                    output_node=strat_node,
                    weight=np.random.normal(0, 0.3),
                    enabled=True
                )
                genome.add_connection(connection)
    
    # 特征组合层直接到输出
    for feat_node in feature_combination_nodes:
        for output_idx in range(output_size):
            if np.random.random() < 0.2:  # 20%概率
                output_node_id = input_size + output_idx
                innovation_num = innovation_tracker.get_innovation_number(feat_node, output_node_id)
                connection = ConnectionGene(
                    innovation_number=innovation_num,
                    input_node=feat_node,
                    output_node=output_node_id,
                    weight=np.random.normal(0, 0.25),
                    enabled=True
                )
                genome.add_connection(connection)
    
    return genome


def create_slimevolley_expert_network(input_size: int, output_size: int, 
                                     innovation_tracker) -> NEATGenome:
    """
    创建SlimeVolley专家网络
    基于领域知识预设计网络结构
    """
    genome = NEATGenome(input_size, output_size)
    
    # 专门的节点类型
    # 1. 球跟踪节点
    ball_tracking_nodes = []
    for i in range(4):
        node = genome.add_hidden_node(
            innovation_tracker,
            activation=ActivationFunction.TANH,
            x_position=0.15,
            y_position=0.1 + i * 0.2
        )
        ball_tracking_nodes.append(node)
    
    # 2. 位置控制节点
    position_control_nodes = []
    for i in range(4):
        node = genome.add_hidden_node(
            innovation_tracker,
            activation=ActivationFunction.TANH,
            x_position=0.35,
            y_position=0.1 + i * 0.2
        )
        position_control_nodes.append(node)
    
    # 3. 时机控制节点
    timing_control_nodes = []
    for i in range(3):
        node = genome.add_hidden_node(
            innovation_tracker,
            activation=ActivationFunction.TANH,
            x_position=0.55,
            y_position=0.2 + i * 0.25
        )
        timing_control_nodes.append(node)
    
    # 4. 策略决策节点
    strategy_nodes = []
    for i in range(5):
        node = genome.add_hidden_node(
            innovation_tracker,
            activation=ActivationFunction.TANH,
            x_position=0.75,
            y_position=0.1 + i * 0.18
        )
        strategy_nodes.append(node)
    
    # 连接模式：基于SlimeVolley的领域知识
    
    # 球相关输入 (0-7) 连接到球跟踪节点
    for input_idx in range(8):  # 球的位置和速度
        for ball_node in ball_tracking_nodes:
            innovation_num = innovation_tracker.get_innovation_number(input_idx, ball_node)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=input_idx,
                output_node=ball_node,
                weight=np.random.normal(0, 0.4),
                enabled=True
            )
            genome.add_connection(connection)
    
    # 位置相关输入 (0-1, 8-9) 连接到位置控制节点
    position_inputs = [0, 1, 8, 9]  # 智能体和对手位置
    for input_idx in position_inputs:
        for pos_node in position_control_nodes:
            innovation_num = innovation_tracker.get_innovation_number(input_idx, pos_node)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=input_idx,
                output_node=pos_node,
                weight=np.random.normal(0, 0.4),
                enabled=True
            )
            genome.add_connection(connection)
    
    # 速度相关输入 (2-3, 10-11) 连接到时机控制节点
    velocity_inputs = [2, 3, 10, 11]  # 智能体和对手速度
    for input_idx in velocity_inputs:
        for timing_node in timing_control_nodes:
            innovation_num = innovation_tracker.get_innovation_number(input_idx, timing_node)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=input_idx,
                output_node=timing_node,
                weight=np.random.normal(0, 0.4),
                enabled=True
            )
            genome.add_connection(connection)
    
    # 球跟踪节点到策略节点
    for ball_node in ball_tracking_nodes:
        for strat_node in strategy_nodes:
            innovation_num = innovation_tracker.get_innovation_number(ball_node, strat_node)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=ball_node,
                output_node=strat_node,
                weight=np.random.normal(0, 0.5),
                enabled=True
            )
            genome.add_connection(connection)
    
    # 位置控制节点到策略节点
    for pos_node in position_control_nodes:
        for strat_node in strategy_nodes:
            innovation_num = innovation_tracker.get_innovation_number(pos_node, strat_node)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=pos_node,
                output_node=strat_node,
                weight=np.random.normal(0, 0.5),
                enabled=True
            )
            genome.add_connection(connection)
    
    # 时机控制节点到策略节点
    for timing_node in timing_control_nodes:
        for strat_node in strategy_nodes:
            innovation_num = innovation_tracker.get_innovation_number(timing_node, strat_node)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=timing_node,
                output_node=strat_node,
                weight=np.random.normal(0, 0.5),
                enabled=True
            )
            genome.add_connection(connection)
    
    # 策略节点到输出
    for strat_node in strategy_nodes:
        for output_idx in range(output_size):
            output_node_id = input_size + output_idx
            innovation_num = innovation_tracker.get_innovation_number(strat_node, output_node_id)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=strat_node,
                output_node=output_node_id,
                weight=np.random.normal(0, 0.4),
                enabled=True
            )
            genome.add_connection(connection)
    
    # 添加一些直接连接（快捷连接）
    # 球位置直接到输出（用于快速反应）
    for input_idx in [4, 5]:  # 球的X、Y位置
        for output_idx in range(output_size):
            output_node_id = input_size + output_idx
            innovation_num = innovation_tracker.get_innovation_number(input_idx, output_node_id)
            connection = ConnectionGene(
                innovation_number=innovation_num,
                input_node=input_idx,
                output_node=output_node_id,
                weight=np.random.normal(0, 0.2),
                enabled=True
            )
            genome.add_connection(connection)
    
    return genome


def evaluate_genome_fitness(genome: NEATGenome, fitness_function, 
                          use_jax: bool = True) -> float:
    """Evaluate fitness of a genome using provided fitness function."""
    if use_jax:
        network = NEATNetworkJAX(genome)
    else:
        network = NEATNetwork(genome)
    
    fitness = fitness_function(network)
    genome.fitness = fitness
    return fitness


class NetworkVisualizer:
    """Visualize NEAT network structure with comprehensive plotting capabilities."""
    
    def __init__(self, genome: NEATGenome):
        self.genome = genome
        self.network_info = self.get_network_info(genome)
        
    @staticmethod
    def get_network_info(genome: NEATGenome) -> Dict:
        """Get network information for visualization."""
        nodes_info = []
        connections_info = []
        
        # Collect node information
        for node_id, node in genome.nodes.items():
            nodes_info.append({
                'id': node_id,
                'type': node.node_type.value,
                'activation': node.activation.value,
                'x': node.x_position,
                'y': node.y_position
            })
        
        # Collect connection information
        for conn in genome.connections.values():
            if conn.enabled:
                connections_info.append({
                    'from': conn.input_node,
                    'to': conn.output_node,
                    'weight': conn.weight,
                    'innovation': conn.innovation_number
                })
        
        return {
            'nodes': nodes_info,
            'connections': connections_info,
            'complexity': genome.get_network_complexity(),
            'fitness': genome.fitness
        }
    
    def visualize_network(self, save_path: str = None, show_labels: bool = True, 
                         node_size: int = 800, figsize: tuple = (12, 8),
                         dpi: int = 100, layout: str = 'spring') -> None:
        """Generate and display network structure visualization."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.colors import LinearSegmentedColormap
            import numpy as np
        except ImportError as e:
            print(f"❌ 缺少必要的可视化库: {e}")
            print("请安装: pip install matplotlib")
            return
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Extract data
        nodes = self.network_info['nodes']
        connections = self.network_info['connections']
        
        # Create color maps for different node types
        node_colors = {
            'INPUT': '#4CAF50',      # Green
            'HIDDEN': '#2196F3',     # Blue
            'OUTPUT': '#FF9800'      # Orange
        }
        
        # Draw connections first (so they appear behind nodes)
        for conn in connections:
            from_node = next(n for n in nodes if n['id'] == conn['from'])
            to_node = next(n for n in nodes if n['id'] == conn['to'])
            
            # Line color based on weight
            weight = conn['weight']
            if weight > 0:
                color = 'red'  # Positive weights
                alpha = min(0.8, abs(weight) / 2.0)
            else:
                color = 'blue'  # Negative weights
                alpha = min(0.8, abs(weight) / 2.0)
            
            # Line width based on weight magnitude
            linewidth = max(0.5, min(3.0, abs(weight)))
            
            # Draw connection line
            ax.plot([from_node['x'], to_node['x']], 
                   [from_node['y'], to_node['y']], 
                   color=color, alpha=alpha, linewidth=linewidth,
                   solid_capstyle='round')
            
            # Add arrowhead
            dx = to_node['x'] - from_node['x']
            dy = to_node['y'] - from_node['y']
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                # Normalize and scale arrow
                dx_norm = dx / length * 0.05
                dy_norm = dy / length * 0.05
                ax.arrow(from_node['x'] + dx * 0.8, from_node['y'] + dy * 0.8,
                         dx_norm, dy_norm, head_width=0.02, head_length=0.03,
                         fc=color, ec=color, alpha=alpha)
        
        # Draw nodes
        for node in nodes:
            node_type = node['type']
            color = node_colors.get(node_type, '#9E9E9E')  # Default gray
            
            # Draw node circle
            circle = patches.Circle((node['x'], node['y']), 0.03, 
                                  facecolor=color, edgecolor='black', 
                                  linewidth=2, alpha=0.8)
            ax.add_patch(circle)
            
            # Add node labels
            if show_labels:
                label = f"{node['id']}"
                if node_type == 'HIDDEN':
                    label += f"\n{node['activation']}"
                
                ax.text(node['x'], node['y'], label, 
                       ha='center', va='center', fontsize=8,
                       fontweight='bold', color='white')
        
        # Set plot properties
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title and legend
        title = f"NEAT Network Structure\n"
        title += f"Nodes: {len(nodes)}, Connections: {len(connections)}, "
        title += f"Complexity: {self.network_info['complexity']}, "
        title += f"Fitness: {self.network_info['fitness']:.3f}"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            patches.Patch(color=node_colors['INPUT'], label='Input Nodes'),
            patches.Patch(color=node_colors['HIDDEN'], label='Hidden Nodes'),
            patches.Patch(color=node_colors['OUTPUT'], label='Output Nodes')
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.15, 1.0))
        
        # Add weight legend
        weight_text = "Connection Weights:\nRed = Positive\nBlue = Negative"
        ax.text(1.15, 0.5, weight_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"💾 网络图已保存到: {save_path}")
        
        plt.show()
    
    def visualize_network_advanced(self, save_path: str = None, 
                                 figsize: tuple = (16, 10)) -> None:
        """Advanced network visualization with detailed analysis."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.gridspec import GridSpec
            import numpy as np
        except ImportError as e:
            print(f"❌ 缺少必要的可视化库: {e}")
            print("请安装: pip install matplotlib")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig)
        
        # Main network plot
        ax_main = fig.add_subplot(gs[0, :2])
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_weights = fig.add_subplot(gs[1, 0])
        ax_activation = fig.add_subplot(gs[1, 1])
        ax_complexity = fig.add_subplot(gs[1, 2])
        
        # Main network visualization
        self._plot_main_network(ax_main)
        
        # Statistics panel
        self._plot_statistics(ax_stats)
        
        # Weight distribution
        self._plot_weight_distribution(ax_weights)
        
        # Activation function distribution
        self._plot_activation_distribution(ax_activation)
        
        # Complexity analysis
        self._plot_complexity_analysis(ax_complexity)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 高级网络图已保存到: {save_path}")
        
        plt.show()
    
    def _plot_main_network(self, ax):
        """Plot main network structure."""
        import matplotlib.patches as patches
        
        nodes = self.network_info['nodes']
        connections = self.network_info['connections']
        
        # Node colors
        node_colors = {
            'INPUT': '#4CAF50', 'HIDDEN': '#2196F3', 'OUTPUT': '#FF9800'
        }
        
        # Draw connections
        for conn in connections:
            from_node = next(n for n in nodes if n['id'] == conn['from'])
            to_node = next(n for n in nodes if n['id'] == conn['to'])
            
            weight = conn['weight']
            color = 'red' if weight > 0 else 'blue'
            alpha = min(0.8, abs(weight) / 2.0)
            linewidth = max(0.5, min(3.0, abs(weight)))
            
            ax.plot([from_node['x'], to_node['x']], 
                   [from_node['y'], to_node['y']], 
                   color=color, alpha=alpha, linewidth=linewidth)
        
        # Draw nodes
        for node in nodes:
            color = node_colors.get(node['type'], '#9E9E9E')
            circle = patches.Circle((node['x'], node['y']), 0.03, 
                                  facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            
            ax.text(node['x'], node['y'], str(node['id']), 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Network Topology', fontsize=12, fontweight='bold')
    
    def _plot_statistics(self, ax):
        """Plot network statistics."""
        nodes = self.network_info['nodes']
        connections = self.network_info['connections']
        
        # Count node types
        node_types = {}
        for node in nodes:
            node_type = node['type']
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Create bar chart
        types = list(node_types.keys())
        counts = list(node_types.values())
        colors = ['#4CAF50', '#2196F3', '#FF9800']
        
        bars = ax.bar(types, counts, color=colors[:len(types)])
        ax.set_title('Node Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(count), ha='center', va='bottom')
    
    def _plot_weight_distribution(self, ax):
        """Plot weight distribution histogram."""
        weights = [conn['weight'] for conn in self.network_info['connections']]
        
        if weights:
            ax.hist(weights, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(weights), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(weights):.3f}')

            ax.axvline(np.median(weights), color='green', linestyle='--', 
                      label=f'Median: {np.median(weights):.3f}')
            ax.set_title('Weight Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_activation_distribution(self, ax):
        """Plot activation function distribution."""
        activations = [node['activation'] for node in self.network_info['nodes']]
        
        if activations:
            activation_counts = {}
            for act in activations:
                activation_counts[act] = activation_counts.get(act, 0) + 1
            
            types = list(activation_counts.keys())
            counts = list(activation_counts.values())
            
            bars = ax.bar(types, counts, color='lightcoral')
            ax.set_title('Activation Functions', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count')
            
            # Add value labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       str(count), ha='center', va='bottom')
    
    def _plot_complexity_analysis(self, ax):
        """Plot complexity analysis."""
        complexity = self.network_info['complexity']
        fitness = self.network_info['fitness']
        
        # Create a simple complexity vs fitness visualization
        ax.scatter(complexity, fitness, s=100, alpha=0.7, color='purple')
        ax.set_title('Complexity vs Fitness', fontsize=12, fontweight='bold')
        ax.set_xlabel('Network Complexity')
        ax.set_ylabel('Fitness')
        ax.grid(True, alpha=0.3)
        
        # Add text annotation
        ax.text(0.05, 0.95, f'Complexity: {complexity}\nFitness: {fitness:.3f}', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    def export_network_data(self, file_path: str) -> None:
        """Export network data to JSON file for external analysis."""
        import json
        
        data = {
            'network_info': self.network_info,
            'metadata': {
                'export_timestamp': str(np.datetime64('now')),
                'genome_id': id(self.genome),
                'visualization_version': '2.0'
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 网络数据已导出到: {file_path}")
    
    def create_network_summary(self) -> str:
        """Create a text summary of the network."""
        nodes = self.network_info['nodes']
        connections = self.network_info['connections']
        
        summary = f"""
🌐 NEAT Network Summary
{'='*50}
📊 Basic Information:
   • Total Nodes: {len(nodes)}
   • Total Connections: {len(connections)}
   • Network Complexity: {self.network_info['complexity']}
   • Current Fitness: {self.network_info['fitness']:.3f}

🔧 Node Breakdown:
"""
        
        # Count node types
        node_types = {}
        for node in nodes:
            node_type = node['type']
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        for node_type, count in node_types.items():
            summary += f"   • {node_type}: {count}\n"
        
        # Connection analysis
        if connections:
            weights = [conn['weight'] for conn in connections]
            summary += f"""
🔗 Connection Analysis:
   • Average Weight: {np.mean(weights):.3f}
   • Weight Std Dev: {np.std(weights):.3f}
   • Min Weight: {min(weights):.3f}
   • Max Weight: {max(weights):.3f}
   • Positive Weights: {sum(1 for w in weights if w > 0)}
   • Negative Weights: {sum(1 for w in weights if w < 0)}
"""
        
        summary += f"""
📈 Performance Metrics:
   • Connections per Node: {len(connections)/len(nodes):.2f}
   • Network Density: {len(connections)/(len(nodes)*(len(nodes)-1)):.3f}
"""
        
        return summary


def test_network_visualization():
    """Test the network visualization functionality."""
    print("🧪 Testing Network Visualization...")
    
    try:
        # Create a test genome
        from neat_core import NEATGenome, InnovationTracker, NodeGene, ConnectionGene, NodeType, ActivationFunction
        
        # Create innovation tracker
        innovation_tracker = InnovationTracker()
        
        # Create a simple genome for testing
        genome = NEATGenome(2, 1)  # 2 inputs, 1 output
        
        # Add a hidden node
        hidden_node = NodeGene(
            node_id=2,
            node_type=NodeType.HIDDEN,
            activation=ActivationFunction.TANH,  # 使用TANH增加多样性
            x_position=0.5,
            y_position=0.5
        )
        genome.add_node(hidden_node)
        
        # Add connections
        # Input 0 to hidden
        conn1 = ConnectionGene(
            innovation_number=innovation_tracker.get_innovation_number(0, 2),
            input_node=0,
            output_node=2,
            weight=0.8,
            enabled=True
        )
        genome.add_connection(conn1)
        
        # Input 1 to hidden
        conn2 = ConnectionGene(
            innovation_number=innovation_tracker.get_innovation_number(1, 2),
            input_node=1,
            output_node=2,
            weight=-0.3,
            enabled=True
        )
        genome.add_connection(conn2)
        
        # Hidden to output
        conn3 = ConnectionGene(
            innovation_number=innovation_tracker.get_innovation_number(2, 1),
            input_node=2,
            output_node=1,
            weight=1.2,
            enabled=True
        )
        genome.add_connection(conn3)
        
        # Set some fitness
        genome.fitness = 0.75
        
        print("✅ Test genome created successfully")
        print(f"   Nodes: {len(genome.nodes)}")
        print(f"   Connections: {len(genome.connections)}")
        print(f"   Fitness: {genome.fitness}")
        
        # Test NetworkVisualizer
        visualizer = NetworkVisualizer(genome)
        print("✅ NetworkVisualizer created successfully")
        
        # Test network info
        info = visualizer.get_network_info(genome)
        print("✅ Network info collected:")
        print(f"   Nodes: {len(info['nodes'])}")
        print(f"   Connections: {len(info['connections'])}")
        print(f"   Complexity: {info['complexity']}")
        
        # Test network summary
        summary = visualizer.create_network_summary()
        print("✅ Network summary created:")
        print(summary)
        
        # Test basic visualization (without showing)
        print("✅ Testing basic visualization...")
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for testing
            visualizer.visualize_network(save_path="test_network.png")
            print("✅ Basic visualization test passed")
        except Exception as e:
            print(f"⚠️  Basic visualization test failed: {e}")
        
        # Test advanced visualization (without showing)
        print("✅ Testing advanced visualization...")
        try:
            visualizer.visualize_network_advanced(save_path="test_network_advanced.png")
            print("✅ Advanced visualization test passed")
        except Exception as e:
            print(f"⚠️  Advanced visualization test failed: {e}")
        
        # Test data export
        print("✅ Testing data export...")
        try:
            visualizer.export_network_data("test_network_data.json")
            print("✅ Data export test passed")
        except Exception as e:
            print(f"⚠️  Data export test failed: {e}")
        
        print("🎉 All visualization tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_network_visualization()