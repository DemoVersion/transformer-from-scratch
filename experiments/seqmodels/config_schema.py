"""Pydantic configuration schema for sequence model experiments."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    embedding_size: int = Field(description="Dimension of token embeddings")
    context: int = Field(description="Maximum sequence length")


class TransformerModelConfig(ModelConfig):
    """Transformer-specific model configuration."""

    num_heads: int = Field(description="Number of attention heads")
    depth: int = Field(description="Number of transformer blocks")
    attention_type: Literal["default", "gpt2", "wide", "narrow", "relative"] = Field(
        default="default", description="Type of attention mechanism"
    )


class LSTMModelConfig(ModelConfig):
    """LSTM-specific model configuration."""

    hidden_size: int = Field(description="Hidden dimension of LSTM layers")
    num_layers: int = Field(description="Number of LSTM layers")
    dropout_rate: float = Field(
        default=0.25, description="Dropout rate for regularization"
    )


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    batch_size: int = Field(description="Training batch size")
    learning_rate: float = Field(description="Learning rate for optimizer")
    lr_warmup: int = Field(description="Number of warmup steps for learning rate")
    gradient_clipping: float = Field(
        default=1.0, description="Gradient clipping threshold"
    )
    num_batches: int = Field(description="Total number of training batches")


class EvaluationConfig(BaseModel):
    """Evaluation and testing configuration."""

    test_every: int = Field(description="Evaluate every N batches")
    test_subset: int = Field(description="Number of tokens to use for evaluation")
    test_batchsize: int = Field(description="Batch size for evaluation")
    sample_length: int = Field(description="Length of generated samples during eval")


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    dataset_name: str = Field(
        default="allenai/c4", description="HuggingFace dataset name"
    )
    dataset_config: str = Field(
        default="realnewslike", description="Dataset configuration/subset"
    )
    dataset_split: str = Field(default="train", description="Dataset split to use")
    tokenizer_path: str = Field(description="Path to save/load tokenizer")
    tokenizer_vocab_size: int = Field(description="BPE vocabulary size")
    tokenizer_training_docs: int = Field(
        description="Number of documents to train tokenizer on"
    )
    target_tokens: int = Field(description="Target number of tokens in dataset")
    train_ratio: float = Field(default=0.9, description="Training data ratio")
    val_ratio: float = Field(default=0.05, description="Validation data ratio")
    test_ratio: float = Field(default=0.05, description="Test data ratio")


class ExperimentConfig(BaseModel):
    """General experiment configuration."""

    random_seed: int = Field(default=1, description="Random seed (negative for random)")
    final: bool = Field(default=False, description="Use test set instead of validation")
    tensorboard_dir: str = Field(
        default="./runs", description="TensorBoard log directory"
    )


class TransformerExperimentConfig(BaseModel):
    """Complete configuration for transformer experiments."""

    model: TransformerModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    dataset: DatasetConfig
    experiment: ExperimentConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TransformerExperimentConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


class LSTMExperimentConfig(BaseModel):
    """Complete configuration for LSTM experiments."""

    model: LSTMModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    dataset: DatasetConfig
    experiment: ExperimentConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LSTMExperimentConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
