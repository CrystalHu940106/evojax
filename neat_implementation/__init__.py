"""
NEAT Implementation Package
A complete implementation of NEAT (NeuroEvolution of Augmenting Topologies)
"""

from neat_implementation.neat_core import (
    NodeType, ActivationFunction, NodeGene, ConnectionGene, 
    InnovationTracker, NEATGenome, NEATMutator,
    crossover, calculate_compatibility_distance
)

from neat_implementation.neat_network import (
    NEATNetwork, NEATNetworkJAX, create_minimal_genome,
    evaluate_genome_fitness, NetworkVisualizer
)

from neat_implementation.neat_core import (
    Species, NEATPopulation
)

from neat_implementation.network_evolution_analyzer import (
    NetworkEvolutionAnalyzer, ComplexityMetrics, StructuralInnovation, EvolutionSnapshot
)

from neat_implementation.activation_function_manager import (
    ActivationFunctionManager
)

__all__ = [
    # Core classes
    'NodeType', 'ActivationFunction', 'NodeGene', 'ConnectionGene',
    'InnovationTracker', 'NEATGenome', 'NEATMutator',
    
    # Network classes
    'NEATNetwork', 'NEATNetworkJAX', 'NetworkVisualizer',
    
    # Population classes
    'Species', 'NEATPopulation',
    
    # Evolution analysis classes
    'NetworkEvolutionAnalyzer', 'ComplexityMetrics', 'StructuralInnovation', 'EvolutionSnapshot',
    
    # Activation function management
    'ActivationFunctionManager',
    
    # Utility functions
    'crossover', 'calculate_compatibility_distance',
    'create_minimal_genome', 'evaluate_genome_fitness'
]