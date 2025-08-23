"""
批处理评估系统使用示例
展示如何使用JAX批处理评估来大幅提升NEAT训练性能
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from neat_implementation.slime_volleyball_neat import SlimeVolleyNEAT
from neat_implementation.batch_config import (
    BatchEvaluationConfig, 
    create_optimized_batch_config,
    DEFAULT_BATCH_CONFIG
)
from neat_implementation.neat_core import NEATGenome, InnovationTracker
from neat_implementation.neat_network import create_minimal_genome


def create_test_population(size: int = 100) -> list:
    """创建测试种群"""
    print(f"📝 创建 {size} 个测试基因组...")
    
    innovation_tracker = InnovationTracker()
    population = []
    
    for i in range(size):
        genome = create_minimal_genome(
            input_size=12,  # SlimeVolley观察空间
            output_size=3,  # 动作空间
            innovation_tracker=innovation_tracker
        )
        
        # 随机初始化权重
        for conn in genome.connections.values():
            conn.weight = np.random.uniform(-2.0, 2.0)
        
        population.append(genome)
    
    print(f"✅ 成功创建 {len(population)} 个测试基因组")
    return population


def benchmark_evaluation_methods(population: list, num_episodes: int = 1):
    """对比不同评估方法的性能"""
    print("\n" + "="*60)
    print("🔬 评估方法性能对比")
    print("="*60)
    
    # 创建环境
    env = SlimeVolleyNEAT(max_steps=500, test=True, use_jax=True)
    
    # 1. 串行评估
    print("\n📊 1. 串行评估")
    print("-" * 30)
    
    start_time = time.time()
    serial_fitnesses = []
    
    for i, genome in enumerate(population):
        fitness = env.evaluate_genome(genome, num_episodes)
        serial_fitnesses.append(fitness)
        
        if (i + 1) % 20 == 0:
            progress = (i + 1) / len(population) * 100
            print(f"   进度: {progress:.1f}% ({i + 1}/{len(population)})")
    
    serial_time = time.time() - start_time
    print(f"⏱️  串行评估完成: {serial_time:.2f}s")
    print(f"📈 平均每基因组: {serial_time/len(population):.3f}s")
    
    # 2. 基础批处理评估
    print("\n🚀 2. 基础批处理评估 (批大小=32)")
    print("-" * 30)
    
    start_time = time.time()
    batch_fitnesses_32 = env.evaluate_population_batch(
        population, num_episodes, batch_size=32
    )
    batch_time_32 = time.time() - start_time
    
    print(f"⏱️  批处理评估完成: {batch_time_32:.2f}s")
    print(f"📈 平均每基因组: {batch_time_32/len(population):.3f}s")
    
    # 3. 智能批处理评估
    print("\n🧠 3. 智能批处理评估 (自动优化)")
    print("-" * 30)
    
    start_time = time.time()
    smart_batch_fitnesses = env.evaluate_population_batch(
        population, num_episodes, batch_size=None  # 自动优化
    )
    smart_batch_time = time.time() - start_time
    
    print(f"⏱️  智能批处理评估完成: {smart_batch_time:.2f}s")
    print(f"📈 平均每基因组: {smart_batch_time/len(population):.3f}s")
    
    # 4. 大批处理评估
    print("\n🔥 4. 大批处理评估 (批大小=64)")
    print("-" * 30)
    
    start_time = time.time()
    batch_fitnesses_64 = env.evaluate_population_batch(
        population, num_episodes, batch_size=64
    )
    batch_time_64 = time.time() - start_time
    
    print(f"⏱️  大批处理评估完成: {batch_time_64:.2f}s")
    print(f"📈 平均每基因组: {batch_time_64/len(population):.3f}s")
    
    # 性能分析
    print("\n📊 性能分析结果")
    print("=" * 40)
    
    methods = [
        ("串行评估", serial_time),
        ("批处理(32)", batch_time_32),
        ("智能批处理", smart_batch_time),
        ("批处理(64)", batch_time_64)
    ]
    
    # 计算性能提升
    for name, time_taken in methods:
        speedup = serial_time / time_taken
        efficiency = len(population) / time_taken
        print(f"{name:15s}: {time_taken:6.2f}s | {speedup:5.2f}x | {efficiency:6.2f} 基因组/秒")
    
    # 结果一致性检查
    print("\n✅ 结果一致性检查")
    print("-" * 30)
    
    all_fitnesses = [
        serial_fitnesses,
        batch_fitnesses_32,
        smart_batch_fitnesses,
        batch_fitnesses_64
    ]
    
    method_names = ["串行", "批处理(32)", "智能批处理", "批处理(64)"]
    
    for i, (name, fitnesses) in enumerate(zip(method_names, all_fitnesses)):
        if i == 0:
            continue
        
        # 与串行结果比较
        diff = np.mean(np.abs(np.array(serial_fitnesses) - np.array(fitnesses)))
        print(f"{name:15s}: 平均差异 {diff:.6f}")
        
        if diff < 1e-6:
            print(f"   ✅ 结果完全一致")
        elif diff < 1e-3:
            print(f"   ⚠️  结果基本一致")
        else:
            print(f"   ❌ 结果差异较大")
    
    return {
        'serial': serial_fitnesses,
        'batch_32': batch_fitnesses_32,
        'smart_batch': smart_batch_fitnesses,
        'batch_64': batch_fitnesses_64,
        'timings': {
            'serial': serial_time,
            'batch_32': batch_time_32,
            'smart_batch': smart_batch_time,
            'batch_64': batch_time_64
        }
    }


def test_custom_batch_config():
    """测试自定义批处理配置"""
    print("\n" + "="*60)
    print("⚙️ 自定义批处理配置测试")
    print("="*60)
    
    # 创建自定义配置
    custom_config = BatchEvaluationConfig(
        default_batch_size=16,
        max_batch_size=96,
        min_batch_size=8,
        enable_warmup=True,
        warmup_batches=2,
        enable_performance_monitoring=True,
        enable_memory_monitoring=True
    )
    
    print("📋 自定义配置:")
    print(f"   默认批大小: {custom_config.default_batch_size}")
    print(f"   最大批大小: {custom_config.max_batch_size}")
    print(f"   最小批大小: {custom_config.min_batch_size}")
    print(f"   启用预热: {custom_config.enable_warmup}")
    print(f"   预热批数: {custom_config.warmup_batches}")
    
    # 使用自定义配置创建环境
    env = SlimeVolleyNEAT(
        max_steps=300, 
        test=True, 
        use_jax=True,
        batch_config=custom_config
    )
    
    # 创建测试种群
    population = create_test_population(64)
    
    # 测试自定义配置的性能
    print(f"\n🧪 使用自定义配置测试 {len(population)} 个基因组...")
    
    start_time = time.time()
    fitnesses = env.evaluate_population_batch(
        population, num_episodes=1, batch_size=None
    )
    test_time = time.time() - start_time
    
    print(f"⏱️  自定义配置测试完成: {test_time:.2f}s")
    print(f"📈 平均每基因组: {test_time/len(population):.3f}s")
    
    return fitnesses, test_time


def test_memory_optimization():
    """测试内存优化功能"""
    print("\n" + "="*60)
    print("💾 内存优化测试")
    print("="*60)
    
    # 创建环境
    env = SlimeVolleyNEAT(max_steps=200, test=True, use_jax=True)
    
    # 测试不同种群大小的内存使用
    population_sizes = [32, 64, 128, 256]
    
    for size in population_sizes:
        print(f"\n📊 测试种群大小: {size}")
        
        # 创建种群
        population = create_test_population(size)
        
        # 估算内存使用
        estimated_memory = env.memory_monitor.estimate_batch_memory_usage(
            batch_size=size, genome_complexity=50.0
        )
        
        print(f"   估算内存使用: {estimated_memory:.2f} GB")
        
        # 实际测试
        try:
            start_time = time.time()
            fitnesses = env.evaluate_population_batch(
                population, num_episodes=1, batch_size=size
            )
            test_time = time.time() - start_time
            
            print(f"   实际测试时间: {test_time:.2f}s")
            print(f"   平均每基因组: {test_time/size:.3f}s")
            
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")


def main():
    """主函数"""
    print("🚀 NEAT批处理评估系统演示")
    print("=" * 60)
    
    # 检查JAX可用性
    if not jax.devices():
        print("❌ JAX不可用，无法运行批处理评估")
        return
    
    print(f"✅ JAX可用，检测到设备: {len(jax.devices())}")
    if jax.devices('gpu'):
        print(f"🎮 GPU设备: {len(jax.devices('gpu'))}")
    if jax.devices('cpu'):
        print(f"💻 CPU设备: {len(jax.devices('cpu'))}")
    
    # 创建测试种群
    population = create_test_population(100)
    
    # 性能对比测试
    results = benchmark_evaluation_methods(population, num_episodes=1)
    
    # 自定义配置测试
    custom_fitnesses, custom_time = test_custom_batch_config()
    
    # 内存优化测试
    test_memory_optimization()
    
    # 总结
    print("\n" + "="*60)
    print("🎯 测试总结")
    print("="*60)
    
    best_method = min(results['timings'].items(), key=lambda x: x[1])
    print(f"🏆 最佳性能: {best_method[0]} ({best_method[1]:.2f}s)")
    
    worst_method = max(results['timings'].items(), key=lambda x: x[1])
    print(f"🐌 最慢方法: {worst_method[0]} ({worst_method[1]:.2f}s)")
    
    max_speedup = results['timings']['serial'] / best_method[1]
    print(f"🚀 最大性能提升: {max_speedup:.2f}x")
    
    print(f"\n💡 建议:")
    print(f"   • 对于小种群 (<50): 使用串行评估")
    print(f"   • 对于中等种群 (50-200): 使用批大小32")
    print(f"   • 对于大种群 (>200): 使用智能批处理")
    print(f"   • 启用JAX预热以获得最佳性能")
    
    print(f"\n✅ 批处理评估系统测试完成!")


if __name__ == "__main__":
    main()
