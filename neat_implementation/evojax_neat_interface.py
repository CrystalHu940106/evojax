"""
EvoJAX兼容的NEAT接口
解决JAX类型兼容性问题，确保自定义状态对象能被EvoJAX识别
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, List, Dict, Any
import os
import sys

# 导入EvoJAX基础类
from evojax.task.base import TaskState, VectorizedTask
from evojax.task.slimevolley import SlimeVolley, State as SlimeVolleyState
from evojax.policy.base import PolicyNetwork
from evojax.util import create_logger

# 导入NEAT实现
try:
    from .neat_core import NEATGenome, InnovationTracker
    from .neat_network import NEATNetworkJAX, NEATNetwork
except ImportError:
    from neat_core import NEATGenome, InnovationTracker
    from neat_network import NEATNetworkJAX, NEATNetwork


@jax.jit
def neat_network_forward(params, obs):
    """JIT编译的NEAT网络前向传播"""
    return NEATNetworkJAX.forward(params, obs)


class NEATEvoJAXPolicy(PolicyNetwork):
    """
    EvoJAX兼容的NEAT策略网络
    将NEAT基因组转换为EvoJAX可识别的策略
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__(input_dim, hidden_dims, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neat_network = None
        self.genome = None
        
    def set_genome(self, genome: NEATGenome):
        """设置NEAT基因组"""
        self.genome = genome
        # 将NEAT基因组转换为网络参数
        self.neat_network = NEATNetworkJAX.from_genome(genome)
        
    def get_actions(self, t_states, params=None, **kwargs):
        """获取动作，兼容EvoJAX接口"""
        if self.neat_network is None:
            raise ValueError("NEAT基因组未设置")
        
        # 提取观察数据
        obs = t_states.obs  # Shape: (batch_size, obs_dim)
        
        # 使用NEAT网络进行前向传播
        actions = self.neat_network.forward(obs)
        
        # 确保输出形状正确 (batch_size, action_dim)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        
        return actions


class NEATEvoJAXTask(VectorizedTask):
    """
    EvoJAX兼容的NEAT任务包装器
    解决自定义状态对象的JAX类型兼容性问题
    """
    
    def __init__(self, max_steps: int = 3000, test: bool = False):
        super().__init__()
        self.max_steps = max_steps
        self.test = test
        
        # 创建基础的SlimeVolley环境
        self.base_env = SlimeVolley(max_steps=max_steps, test=test)
        
        # 环境属性
        self.obs_shape = self.base_env.obs_shape
        self.act_shape = self.base_env.act_shape
        
        # 日志记录器
        self.logger = create_logger(name='NEATEvoJAXTask')
        
    def reset(self, key: jnp.ndarray) -> Tuple[SlimeVolleyState, jnp.ndarray]:
        """重置环境，返回兼容的EvoJAX状态"""
        # 使用基础环境的reset方法
        state, obs = self.base_env.reset(key)
        return state, obs
    
    def step(self, state: SlimeVolleyState, actions: jnp.ndarray) -> Tuple[SlimeVolleyState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """执行一步，返回兼容的EvoJAX状态"""
        # 使用基础环境的step方法
        next_state, obs, reward, done = self.base_env.step(state, actions)
        return next_state, obs, reward, done
    
    def get_obs(self, state: SlimeVolleyState) -> jnp.ndarray:
        """获取观察，兼容EvoJAX接口"""
        return self.base_env.get_obs(state)


class NEATEvoJAXInterface:
    """
    NEAT和EvoJAX之间的接口
    提供统一的训练接口，解决JAX类型兼容性问题
    """
    
    def __init__(self, max_steps: int = 3000, test: bool = False, use_jax: bool = True):
        self.max_steps = max_steps
        self.test = test
        self.use_jax = use_jax
        
        # 创建EvoJAX兼容的任务
        self.evojax_task = NEATEvoJAXTask(max_steps=max_steps, test=test)
        
        # 环境属性
        self.input_size = self.evojax_task.obs_shape[0]  # 12 observations
        self.output_size = self.evojax_task.act_shape[0]  # 3 actions
        
        # 策略网络
        self.policy = None
        
        # 日志记录器
        self.logger = create_logger(name='NEATEvoJAXInterface')
        
    def create_policy_from_genome(self, genome: NEATGenome) -> NEATEvoJAXPolicy:
        """从NEAT基因组创建EvoJAX兼容的策略"""
        policy = NEATEvoJAXPolicy(
            input_dim=self.input_size,
            hidden_dims=[],  # NEAT网络结构由基因组决定
            output_dim=self.output_size
        )
        policy.set_genome(genome)
        return policy
    
    def evaluate_genome(self, genome: NEATGenome, num_episodes: int = 1) -> float:
        """评估单个基因组"""
        if num_episodes <= 0:
            return 0.0
        
        total_fitness = 0.0
        
        for episode in range(num_episodes):
            # 创建策略
            policy = self.create_policy_from_genome(genome)
            
            # 重置环境
            key = jax.random.PRNGKey(episode)
            state, obs = self.evojax_task.reset(key)
            
            episode_reward = 0.0
            done = False
            step_count = 0
            
            while not done and step_count < self.max_steps:
                # 获取动作
                actions = policy.get_actions(
                    type('TaskStates', (), {'obs': obs.reshape(1, -1)})()
                )
                
                # 执行动作
                state, obs, reward, done = self.evojax_task.step(state, actions[0])
                
                episode_reward += float(reward)
                step_count += 1
                
                # 检查是否完成
                if done:
                    break
            
            total_fitness += episode_reward
        
        # 返回平均适应度
        return total_fitness / num_episodes
    
    def evaluate_population_batch(self, genomes: List[NEATGenome], 
                                num_episodes: int = 1, 
                                batch_size: int = 32) -> List[float]:
        """批量评估种群"""
        if not genomes:
            return []
        
        # 如果使用JAX且GPU可用，尝试批处理
        if self.use_jax and len(jax.devices('gpu')) > 0:
            try:
                return self._evaluate_batch_jax(genomes, num_episodes, batch_size)
            except Exception as e:
                self.logger.warning(f"JAX批处理失败，回退到串行评估: {e}")
                return self._evaluate_serial(genomes, num_episodes)
        else:
            return self._evaluate_serial(genomes, num_episodes)
    
    def _evaluate_batch_jax(self, genomes: List[NEATGenome], 
                           num_episodes: int, batch_size: int) -> List[float]:
        """使用JAX进行批处理评估"""
        fitnesses = []
        
        # 分批处理
        for i in range(0, len(genomes), batch_size):
            batch_genomes = genomes[i:i + batch_size]
            batch_fitnesses = []
            
            for genome in batch_genomes:
                fitness = self.evaluate_genome(genome, num_episodes)
                batch_fitnesses.append(fitness)
            
            fitnesses.extend(batch_fitnesses)
            
            # 显示进度
            if (i + batch_size) % (batch_size * 4) == 0:
                progress = min((i + batch_size) / len(genomes) * 100, 100)
                self.logger.info(f"批处理评估进度: {progress:.1f}%")
        
        return fitnesses
    
    def _evaluate_serial(self, genomes: List[NEATGenome], 
                        num_episodes: int) -> List[float]:
        """串行评估"""
        fitnesses = []
        
        for i, genome in enumerate(genomes):
            if i % 10 == 0:
                self.logger.info(f"串行评估进度: {i+1}/{len(genomes)}")
            
            fitness = self.evaluate_genome(genome, num_episodes)
            fitnesses.append(fitness)
        
        return fitnesses
    
    def get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'max_steps': self.max_steps,
            'test_mode': self.test,
            'use_jax': self.use_jax,
            'gpu_available': len(jax.devices('gpu')) > 0
        }


# 兼容性测试函数
def test_evojax_compatibility():
    """测试EvoJAX兼容性"""
    print("🧪 测试EvoJAX兼容性...")
    
    try:
        # 创建接口
        interface = NEATEvoJAXInterface(max_steps=300, test=True, use_jax=True)
        
        # 获取环境信息
        env_info = interface.get_environment_info()
        print(f"✅ 环境信息: {env_info}")
        
        # 测试基因组创建
        from neat_implementation.neat_network import create_minimal_genome
        innovation_tracker = InnovationTracker()
        test_genome = create_minimal_genome(
            interface.input_size, 
            interface.output_size, 
            innovation_tracker
        )
        print(f"✅ 测试基因组创建成功")
        
        # 测试策略创建
        policy = interface.create_policy_from_genome(test_genome)
        print(f"✅ 策略创建成功")
        
        # 测试评估
        fitness = interface.evaluate_genome(test_genome, num_episodes=1)
        print(f"✅ 基因组评估成功，适应度: {fitness:.3f}")
        
        print("🎉 EvoJAX兼容性测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ EvoJAX兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行兼容性测试
    test_evojax_compatibility()

