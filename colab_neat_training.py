#!/usr/bin/env python3
"""
Google Colab NEAT训练脚本 - 使用T4 GPU加速
专门为Colab环境优化，包含自动环境检测和GPU配置
"""

# 自动检测和配置Colab环境
import os
import sys
import time
import json
import pickle
from datetime import datetime

# Colab环境检测和配置
def setup_colab_environment():
    """设置Colab环境"""
    print("🚀 检测到Google Colab环境，正在配置...")
    
    # 检查是否在Colab中
    try:
        import google.colab
        IN_COLAB = True
        print("✅ 确认在Google Colab环境中")
    except ImportError:
        IN_COLAB = False
        print("⚠️  不在Colab环境中，将使用本地配置")
    
    if IN_COLAB:
        # 安装必要的包
        print("📦 安装必要的包...")
        os.system("pip install -q jax jaxlib flax optax")
        
        # 配置JAX使用GPU
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
        
        print("✅ Colab环境配置完成")
    
    return IN_COLAB

# 设置环境
IN_COLAB = setup_colab_environment()

# 导入必要的库
import numpy as np
import jax
import jax.numpy as jnp

# 检查GPU可用性
def check_gpu():
    """检查GPU可用性"""
    try:
        gpu_devices = jax.devices('gpu')
        if gpu_devices:
            print(f"🎮 检测到GPU: {len(gpu_devices)}个设备")
            for i, device in enumerate(gpu_devices):
                print(f"  - GPU {i}: {device}")
            return True
        else:
            print("⚠️  未检测到GPU，将使用CPU")
            return False
    except Exception as e:
        print(f"⚠️  GPU检测失败: {e}")
        return False

# 检查GPU
HAS_GPU = check_gpu()

# 添加当前目录到路径（Colab中需要）
if IN_COLAB:
    # 在Colab中，我们需要从GitHub克隆代码
    if not os.path.exists('evojax'):
        print("📥 从GitHub克隆evojax代码...")
        os.system("git clone https://github.com/CrystalHu940106/evojax.git")
        os.chdir('evojax')
    else:
        os.chdir('evojax')

# 导入NEAT实现
try:
    from neat_implementation.neat_core import NEATPopulation, NEATConfig
    from neat_implementation.slime_volleyball_neat import SlimeVolleyNEAT
    from neat_implementation.neat_network import NEATNetwork
    print("✅ NEAT实现导入成功")
except ImportError as e:
    print(f"❌ NEAT实现导入失败: {e}")
    print("请确保在正确的目录中运行此脚本")
    sys.exit(1)


class ColabNEATTrainer:
    """Colab优化的NEAT训练器"""
    
    def __init__(self):
        self.config = {
            'generations': 100,
            'population_size': 100,
            'episodes_per_individual': 1,
            'max_steps': 300,
            'use_gpu': HAS_GPU
        }
        
        print("🚀 Colab NEAT训练器启动")
        print("=" * 50)
        print(f"📊 训练配置:")
        print(f"  - 代数: {self.config['generations']}")
        print(f"  - 种群大小: {self.config['population_size']}")
        print(f"  - 每个体评估次数: {self.config['episodes_per_individual']}")
        print(f"  - 每局最大步数: {self.config['max_steps']}")
        print(f"  - 使用GPU: {self.config['use_gpu']}")
        print("=" * 50)
        
        # 初始化NEAT配置
        self.neat_config = NEATConfig()
        self.neat_config.population_size = self.config['population_size']
        
        # 设置优化的突变参数
        self.neat_config.weight_mutation_rate = 0.8
        self.neat_config.weight_mutation_power = 0.1
        self.neat_config.add_connection_rate = 0.1
        self.neat_config.add_node_rate = 0.05
        self.neat_config.species_threshold = 3.0
        
        # 初始化SlimeVolley环境
        self.slimevolley_neat = SlimeVolleyNEAT(
            use_jax=self.config['use_gpu'],  # 在Colab中使用JAX
            batch_config=None
        )
        
        # 初始化NEAT种群
        self.population = NEATPopulation(
            input_size=12,
            output_size=3,
            population_size=self.config['population_size']
        )
        
        # 设置NEAT参数
        self.population.species_threshold = self.neat_config.species_threshold
        self.population.species_elite_size = self.neat_config.species_elite_size
        self.population.species_stagnation_limit = self.neat_config.species_stagnation_limit
        self.population.elite_size = self.neat_config.elite_size
        
        # 训练统计
        self.training_stats = {
            'start_time': time.time(),
            'generations': [],
            'best_fitness_history': [],
            'avg_fitness_history': [],
            'species_count_history': [],
            'total_time': 0
        }
        
        print("✅ 训练器初始化完成")
        
    def evaluate_population(self, population):
        """评估种群"""
        print(f"🔍 评估种群...")
        
        # 获取实际的基因组列表
        genomes = population.population
        
        # 使用JAX批处理评估（如果在GPU上）
        if self.config['use_gpu']:
            try:
                print("🚀 使用JAX批处理评估...")
                fitnesses = self.slimevolley_neat.evaluate_population_batch(
                    genomes,
                    num_episodes=self.config['episodes_per_individual']
                )
            except Exception as e:
                print(f"⚠️  JAX批处理失败，回退到串行评估: {e}")
                fitnesses = self._evaluate_serial(genomes)
        else:
            fitnesses = self._evaluate_serial(genomes)
        
        # 设置适应度
        for genome, fitness in zip(genomes, fitnesses):
            genome.fitness = fitness
            
        return fitnesses
    
    def _evaluate_serial(self, genomes):
        """串行评估"""
        fitnesses = []
        for i, genome in enumerate(genomes):
            if i % 10 == 0:
                print(f"  评估个体 {i+1}/{len(genomes)}")
            fitness = self.slimevolley_neat.evaluate_genome(genome)
            fitnesses.append(fitness)
        return fitnesses
    
    def train_generation(self, generation_num):
        """训练一代"""
        print(f"\n🔄 第 {generation_num} 代训练开始")
        gen_start_time = time.time()
        
        # 评估当前种群
        fitnesses = self.evaluate_population(self.population)
        
        # 收集统计信息
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        species_count = len(self.population.species)
        
        # 显示当前代结果
        print(f"📊 第 {generation_num} 代结果:")
        print(f"  - 最佳适应度: {best_fitness:.3f}")
        print(f"  - 平均适应度: {avg_fitness:.3f}")
        print(f"  - 物种数量: {species_count}")
        
        # 记录统计
        self.training_stats['generations'].append({
            'generation': generation_num,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'species_count': species_count,
            'time': time.time() - gen_start_time
        })
        
        self.training_stats['best_fitness_history'].append(best_fitness)
        self.training_stats['avg_fitness_history'].append(avg_fitness)
        self.training_stats['species_count_history'].append(species_count)
        
        # 检查是否达到目标
        if best_fitness >= 5.0:  # 能稳定击败AI
            print(f"🎯 目标达成！最佳适应度: {best_fitness:.3f}")
            return True
        
        # 进化到下一代
        if generation_num < self.config['generations']:
            print(f"🔄 进化到第 {generation_num + 1} 代...")
            new_population = self.population.reproduce()
            # 确保新种群是NEATPopulation对象
            if isinstance(new_population, list):
                # 如果返回的是列表，需要重新包装
                self.population.population = new_population
            else:
                self.population = new_population
        
        gen_time = time.time() - gen_start_time
        print(f"⏱️  第 {generation_num} 代用时: {gen_time:.1f}秒")
        
        return False
    
    def save_checkpoint(self, generation_num):
        """保存检查点"""
        checkpoint_data = {
            'generation': generation_num,
            'population': self.population,
            'training_stats': self.training_stats,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = f"colab_neat_checkpoint_gen{generation_num}_{int(time.time())}.pkl"
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"💾 检查点已保存: {checkpoint_file}")
        except Exception as e:
            print(f"⚠️  保存检查点失败: {e}")
    
    def save_final_results(self):
        """保存最终结果"""
        results_file = f"colab_neat_results_{int(time.time())}.json"
        
        # 准备保存的数据
        save_data = {
            'config': self.config,
            'training_stats': {
                'total_generations': len(self.training_stats['generations']),
                'best_fitness_history': self.training_stats['best_fitness_history'],
                'avg_fitness_history': self.training_stats['avg_fitness_history'],
                'species_count_history': self.training_stats['species_count_history'],
                'total_time': time.time() - self.training_stats['start_time']
            },
            'final_population_size': len(self.population.population),
            'best_genome': None,  # 需要序列化
            'timestamp': datetime.now().isoformat(),
            'colab_info': {
                'in_colab': IN_COLAB,
                'has_gpu': HAS_GPU,
                'gpu_devices': [str(d) for d in jax.devices('gpu')] if HAS_GPU else []
            }
        }
        
        # 保存为JSON
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 训练结果已保存: {results_file}")
        return results_file
    
    def run_training(self):
        """运行完整训练"""
        print("\n🎬 开始100代NEAT训练...")
        print("目标：训练出能击败内置AI的SlimeVolley agent")
        print("=" * 50)
        
        try:
            for generation in range(1, self.config['generations'] + 1):
                # 训练当前代
                target_reached = self.train_generation(generation)
                
                # 每10代保存检查点
                if generation % 10 == 0:
                    self.save_checkpoint(generation)
                
                # 检查是否达到目标
                if target_reached:
                    print(f"🎉 训练提前完成！在第 {generation} 代达到目标")
                    break
                
                # 显示进度
                progress = (generation / self.config['generations']) * 100
                elapsed_time = time.time() - self.training_stats['start_time']
                estimated_total = elapsed_time / generation * self.config['generations']
                remaining_time = estimated_total - elapsed_time
                
                print(f"📈 进度: {progress:.1f}% ({generation}/{self.config['generations']})")
                print(f"⏱️  已用时: {elapsed_time/60:.1f}分钟")
                print(f"⏱️  预计剩余: {remaining_time/60:.1f}分钟")
                print("-" * 30)
            
            # 训练完成
            total_time = time.time() - self.training_stats['start_time']
            self.training_stats['total_time'] = total_time
            
            print(f"\n🎉 100代训练完成！")
            print(f"⏱️  总用时: {total_time/60:.1f}分钟 ({total_time/3600:.1f}小时)")
            print(f"🏆 最佳适应度: {max(self.training_stats['best_fitness_history']):.3f}")
            
            # 保存最终结果
            results_file = self.save_final_results()
            
            print(f"\n📊 训练统计:")
            print(f"  - 总代数: {len(self.training_stats['generations'])}")
            print(f"  - 最终种群大小: {len(self.population.population)}")
            print(f"  - 最终物种数量: {self.training_stats['species_count_history'][-1]}")
            print(f"  - 结果文件: {results_file}")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  训练被用户中断")
            print(f"⏱️  已用时: {time.time() - self.training_stats['start_time']:.1f}秒")
            self.save_checkpoint(len(self.training_stats['generations']))
            
        except Exception as e:
            print(f"\n❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            self.save_checkpoint(len(self.training_stats['generations']))


def main():
    """主函数"""
    print("🎮 SlimeVolley NEAT Colab训练")
    print("=" * 50)
    
    # 显示环境信息
    print(f"🌍 环境信息:")
    print(f"  - Colab环境: {IN_COLAB}")
    print(f"  - GPU可用: {HAS_GPU}")
    if HAS_GPU:
        print(f"  - GPU设备: {jax.devices('gpu')}")
    print(f"  - JAX版本: {jax.__version__}")
    print("=" * 50)
    
    # 创建训练器
    trainer = ColabNEATTrainer()
    
    # 开始训练
    trainer.run_training()
    
    print("\n🎯 训练完成！")


if __name__ == '__main__':
    main()
