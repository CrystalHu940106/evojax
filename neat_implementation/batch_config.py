"""
JAX批处理配置和优化设置
管理NEAT批处理评估的各种参数和性能优化选项
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp


@dataclass
class BatchEvaluationConfig:
    """批处理评估配置"""
    
    # 基础批处理参数
    default_batch_size: int = 32
    max_batch_size: int = 128
    min_batch_size: int = 8
    
    # 性能优化参数
    enable_jit_compilation: bool = True
    enable_xla_optimization: bool = True
    enable_memory_optimization: bool = True
    
    # 自适应批处理参数
    auto_adjust_batch_size: bool = True
    target_memory_usage_gb: float = 4.0  # 目标内存使用量
    max_memory_usage_gb: float = 8.0     # 最大内存使用量
    
    # 并行度参数
    num_parallel_episodes: int = 4
    enable_multi_gpu: bool = False
    gpu_memory_fraction: float = 0.8
    
    # 预热和缓存参数
    enable_warmup: bool = True
    warmup_batches: int = 3
    enable_result_caching: bool = True
    cache_size: int = 1000
    
    # 监控和调试参数
    enable_performance_monitoring: bool = True
    enable_memory_monitoring: bool = True
    enable_progress_bar: bool = True
    log_level: str = "INFO"
    
    # 错误处理参数
    max_retries: int = 3
    retry_delay: float = 0.1
    fallback_to_serial: bool = True
    
    def __post_init__(self):
        """验证配置参数"""
        if self.default_batch_size < self.min_batch_size:
            self.default_batch_size = self.min_batch_size
        if self.default_batch_size > self.max_batch_size:
            self.default_batch_size = self.max_batch_size


@dataclass
class JAXOptimizationConfig:
    """JAX优化配置"""
    
    # 编译优化
    enable_jit: bool = True
    enable_pmap: bool = False  # 对于单GPU，通常不需要pmap
    enable_vmap: bool = True
    
    # 内存优化
    enable_custom_jvp: bool = False
    enable_custom_vjp: bool = False
    
    # 数值稳定性
    enable_nan_check: bool = True
    enable_inf_check: bool = True
    
    # 并行优化
    num_parallel_calls: int = 4
    enable_async_dispatch: bool = True
    
    # 缓存设置
    max_cache_size: int = 1000
    enable_compilation_cache: bool = True


class BatchSizeOptimizer:
    """智能批处理大小优化器"""
    
    def __init__(self, config: BatchEvaluationConfig):
        self.config = config
        self.performance_history = []
        self.memory_history = []
        
    def optimize_batch_size(self, 
                          population_size: int,
                          available_memory_gb: float,
                          previous_performance: Optional[float] = None) -> int:
        """
        根据种群大小、可用内存和性能历史优化批处理大小
        """
        # 基础批大小计算
        base_batch_size = self._calculate_base_batch_size(population_size)
        
        # 内存约束调整
        memory_adjusted_size = self._adjust_for_memory(
            base_batch_size, available_memory_gb
        )
        
        # 性能历史调整
        if previous_performance and self.performance_history:
            performance_adjusted_size = self._adjust_for_performance(
                memory_adjusted_size, previous_performance
            )
        else:
            performance_adjusted_size = memory_adjusted_size
        
        # 确保在有效范围内
        final_batch_size = max(
            self.config.min_batch_size,
            min(performance_adjusted_size, self.config.max_batch_size)
        )
        
        # 记录决策
        self._record_decision(population_size, final_batch_size, available_memory_gb)
        
        return final_batch_size
    
    def _calculate_base_batch_size(self, population_size: int) -> int:
        """计算基础批处理大小"""
        if population_size < 50:
            return min(16, population_size)
        elif population_size < 100:
            return 32
        elif population_size < 200:
            return 48
        elif population_size < 500:
            return 64
        else:
            return 96
    
    def _adjust_for_memory(self, batch_size: int, available_memory_gb: float) -> int:
        """根据可用内存调整批处理大小"""
        # 估算每个基因组的内存使用量（粗略估计）
        estimated_memory_per_genome_gb = 0.01  # 10MB per genome
        
        max_batch_size_by_memory = int(
            available_memory_gb * self.config.gpu_memory_fraction / 
            estimated_memory_per_genome_gb
        )
        
        return min(batch_size, max_batch_size_by_memory)
    
    def _adjust_for_performance(self, batch_size: int, previous_performance: float) -> int:
        """根据性能历史调整批处理大小"""
        if not self.performance_history:
            return batch_size
        
        # 分析性能趋势
        recent_performance = self.performance_history[-5:]  # 最近5次
        if len(recent_performance) < 3:
            return batch_size
        
        # 如果性能在下降，减少批大小
        if recent_performance[-1] < recent_performance[-2]:
            return max(batch_size // 2, self.config.min_batch_size)
        
        # 如果性能在提升，增加批大小
        elif recent_performance[-1] > recent_performance[-2]:
            return min(batch_size * 2, self.config.max_batch_size)
        
        return batch_size
    
    def _record_decision(self, population_size: int, batch_size: int, 
                        available_memory_gb: float):
        """记录决策信息"""
        self.performance_history.append({
            'population_size': population_size,
            'batch_size': batch_size,
            'available_memory': available_memory_gb,
            'timestamp': jax.time.time()
        })


class MemoryMonitor:
    """内存使用监控器"""
    
    def __init__(self, config: BatchEvaluationConfig):
        self.config = config
        self.memory_usage_history = []
        
    def get_available_memory_gb(self) -> float:
        """获取可用内存（GB）"""
        try:
            # 尝试获取GPU内存信息
            if jax.devices('gpu'):
                device = jax.devices('gpu')[0]
                # 这里需要根据具体的JAX版本调整
                # 对于较新版本，可以使用 jax.device_get_memory_info
                return 8.0  # 默认假设8GB GPU
            else:
                # CPU环境
                import psutil
                return psutil.virtual_memory().available / (1024**3)
        except:
            # 如果无法获取，返回默认值
            return 4.0
    
    def estimate_batch_memory_usage(self, batch_size: int, 
                                  genome_complexity: float) -> float:
        """估算批处理的内存使用量"""
        # 基础内存使用量
        base_memory_mb = 50  # 基础环境内存
        
        # 每个基因组的内存使用量（基于复杂度）
        memory_per_genome_mb = base_memory_mb + genome_complexity * 0.1
        
        # 总内存使用量
        total_memory_mb = base_memory_mb + batch_size * memory_per_genome_mb
        
        return total_memory_mb / 1024  # 转换为GB
    
    def should_reduce_batch_size(self, current_batch_size: int, 
                               current_memory_usage_gb: float) -> bool:
        """判断是否应该减少批处理大小"""
        return current_memory_usage_gb > self.config.max_memory_usage_gb


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self, config: BatchEvaluationConfig):
        self.config = config
        self.profiling_data = {}
        
    def start_profiling(self, operation_name: str):
        """开始性能分析"""
        import time
        self.profiling_data[operation_name] = {
            'start_time': time.time(),
            'memory_start': self._get_current_memory_usage()
        }
    
    def end_profiling(self, operation_name: str) -> Dict:
        """结束性能分析并返回结果"""
        import time
        if operation_name not in self.profiling_data:
            return {}
        
        end_time = time.time()
        memory_end = self._get_current_memory_usage()
        
        profiling_result = {
            'duration': end_time - self.profiling_data[operation_name]['start_time'],
            'memory_delta': memory_end - self.profiling_data[operation_name]['memory_start'],
            'memory_peak': memory_end
        }
        
        # 清理数据
        del self.profiling_data[operation_name]
        
        return profiling_result
    
    def _get_current_memory_usage(self) -> float:
        """获取当前内存使用量"""
        try:
            import psutil
            return psutil.virtual_memory().used / (1024**3)
        except:
            return 0.0
    
    def generate_performance_report(self) -> str:
        """生成性能报告"""
        if not self.profiling_data:
            return "无性能数据"
        
        report = "📊 性能分析报告\n"
        report += "=" * 40 + "\n"
        
        for operation, data in self.profiling_data.items():
            report += f"操作: {operation}\n"
            report += f"  开始时间: {data['start_time']}\n"
            report += f"  内存起始: {data['memory_start']:.2f} GB\n"
            report += "\n"
        
        return report


# 默认配置实例
DEFAULT_BATCH_CONFIG = BatchEvaluationConfig()
DEFAULT_JAX_CONFIG = JAXOptimizationConfig()

# 配置工厂函数
def create_optimized_batch_config(
    population_size: int,
    available_memory_gb: float,
    use_gpu: bool = True
) -> BatchEvaluationConfig:
    """创建针对特定环境的优化批处理配置"""
    config = BatchEvaluationConfig()
    
    # 根据环境调整配置
    if use_gpu:
        config.enable_jit_compilation = True
        config.enable_xla_optimization = True
        config.max_batch_size = min(128, int(available_memory_gb * 10))
    else:
        config.enable_jit_compilation = False
        config.max_batch_size = 64
    
    # 根据种群大小调整
    if population_size < 100:
        config.default_batch_size = 16
    elif population_size < 500:
        config.default_batch_size = 32
    else:
        config.default_batch_size = 64
    
    return config
