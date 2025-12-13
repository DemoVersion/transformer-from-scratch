"""Configuration constants for playground experiments."""

# Model hyperparameters
EMBEDDING_SIZE = 128
NUM_HEADS = 8
DEPTH = 12
CONTEXT = 256
ATTENTION_TYPE = "default"  # Options: "default", "gpt2", "wide", "narrow", "relative"

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
LR_WARMUP = 5000
GRADIENT_CLIPPING = 1.0
NUM_BATCHES = 1_000_000

# Evaluation/testing
TEST_EVERY = 1500
TEST_SUBSET = 100000
TEST_BATCHSIZE = 64
SAMPLE_LENGTH = 600

# Dataset configuration
DATASET_NAME = "allenai/c4"
DATASET_CONFIG = "realnewslike"
DATASET_SPLIT = "train"
TOKENIZER_PATH = "playground/bpe-tokenizer"
TOKENIZER_VOCAB_SIZE = 8000
TOKENIZER_TRAINING_DOCS = 10000
TARGET_TOKENS = 10_000_000
DATASET_TRAIN_RATIO = 0.9
DATASET_VAL_RATIO = 0.05
DATASET_TEST_RATIO = 0.05

# Paths
TENSORBOARD_DIR = "./runs"

# Other settings
RANDOM_SEED = 1  # Negative for random
FINAL = False  # Use test set instead of validation
