"""
Production-ready PyTorch implementation of a hybrid CRNN + MLP model
for Alzheimer's Disease classification from speech prosody.

This model fuses deep time-frequency representations from log-Mel spectrograms
with interpretable acoustic features (eGeMAPS) to perform subject-level
cognitive state classification (AD / MCI / Control).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from typing import Tuple, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Type


def get_device(device_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate device for training (GPU if available, else CPU).
    
    Args:
        device_id: Optional GPU device ID (0, 1, etc.). If None, uses default GPU.
    
    Returns:
        torch.device: CUDA device if available, else CPU device
    """
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f"cuda:{device_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def get_model_save_dir() -> str:
    """
    Get the default directory for saving models.
    
    Returns:
        Path to models directory (created if it doesn't exist)
    """
    # Get project root directory (one level up from scripts folder)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def save_model(
    model: "HybridCRNNModel",
    filepath: str,
    additional_info: Optional[Dict] = None,
) -> None:
    """
    Standalone function to save a model.
    
    Args:
        model: The HybridCRNNModel instance to save
        filepath: Path to save the model
        additional_info: Optional dictionary with additional info to save
    
    Example:
        save_model(model, "models/best_model.pth", {"epoch": 50, "val_acc": 0.85})
    """
    model.save(filepath, additional_info)


def load_model(
    filepath: str,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Tuple["HybridCRNNModel", Dict]:
    """
    Standalone function to load a saved model.
    
    Args:
        filepath: Path to the saved model file
        device: Device to load the model on
        strict: Whether to strictly enforce state_dict keys
    
    Returns:
        Tuple of (loaded_model, additional_info_dict)
    
    Example:
        model, info = load_model("models/best_model.pth", device=torch.device("cuda"))
    """
    return HybridCRNNModel.load(filepath, device, strict)


class CNNEncoder(nn.Module):
    """
    CNN Frontend for processing log-Mel spectrograms.
    
    Learns local time-frequency patterns and reduces spatial dimensionality
    through 2-3 convolutional blocks.
    
    Input: [batch_size, 1, mel_bins, time_frames]
    Output: [batch_size, out_channels, reduced_mel_bins, reduced_time_frames]
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_blocks: int = 3,
        mel_bins: int = 128,
        kernel_size: Tuple[int, int] = (3, 3),
        pool_size: Tuple[int, int] = (2, 2),
    ):
        """
        Args:
            in_channels: Input channels (1 for grayscale spectrogram)
            base_channels: Base number of channels for first conv block
            num_blocks: Number of convolutional blocks (2-3 recommended)
            mel_bins: Number of Mel frequency bins (for shape tracking)
            kernel_size: Convolution kernel size (height, width)
            pool_size: Max pooling size (height, width)
        """
        super(CNNEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.mel_bins = mel_bins
        
        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        
        channels = in_channels
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)  # Double channels each block
            
            # Conv2d → BatchNorm2d → ReLU → MaxPool2d
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size[0] // 2, kernel_size[1] // 2),  # Same padding
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=pool_size),
            )
            
            self.conv_blocks.append(block)
            channels = out_channels
        
        self.out_channels = channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN encoder.
        
        Args:
            x: Input spectrogram [batch_size, 1, mel_bins, time_frames]
        
        Returns:
            Encoded features [batch_size, out_channels, reduced_mel_bins, reduced_time_frames]
        """
        # x: [batch_size, 1, mel_bins, time_frames]
        for block in self.conv_blocks:
            x = block(x)
            # After each block: spatial dimensions reduced by pool_size
        
        # Output: [batch_size, out_channels, reduced_mel_bins, reduced_time_frames]
        return x


class CRNNEncoder(nn.Module):
    """
    Recurrent Temporal Modeling (CRNN) for capturing prosodic patterns.
    
    After CNN processing, collapses frequency dimension and treats time axis
    as sequence. Uses bidirectional LSTM/GRU to capture:
    - Prosodic drift
    - Temporal instability
    - Long-range pause patterns
    
    Input: [batch_size, channels, mel_bins, time_frames]
    Output: [batch_size, hidden_size * 2] (bidirectional)
    """
    
    def __init__(
        self,
        in_channels: int,
        rnn_input_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_gru: bool = False,
    ):
        """
        Args:
            in_channels: Number of channels from CNN encoder (for reference)
            rnn_input_size: Input size for RNN after frequency collapse projection
            hidden_size: Hidden size for RNN cells
            num_layers: Number of RNN layers
            dropout: Dropout rate (applied between RNN layers)
            use_gru: If True, use GRU; if False, use LSTM
        """
        super(CRNNEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.rnn_input_size = rnn_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gru = use_gru
        
        # Projection layer: dynamically sized input → fixed rnn_input_size
        # This will be initialized on first forward pass when we know the exact input size
        # Using register_buffer to ensure it's properly handled, but we'll create it as a module
        self._projection_initialized = False
        self.projection = None
        
        # Bidirectional RNN for temporal modeling
        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_class(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        
        # Output size: hidden_size * 2 (bidirectional)
        self.output_size = hidden_size * 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CRNN encoder.
        
        Args:
            x: CNN features [batch_size, channels, mel_bins, time_frames]
        
        Returns:
            Temporal embedding [batch_size, hidden_size * 2]
        """
        batch_size = x.size(0)
        channels = x.size(1)
        mel_bins = x.size(2)
        time_frames = x.size(3)
        
        # Collapse frequency dimension: [batch_size, channels, mel_bins, time_frames]
        # → [batch_size, channels * mel_bins, time_frames]
        # → [batch_size, time_frames, channels * mel_bins]
        x = x.view(batch_size, channels * mel_bins, time_frames)
        x = x.transpose(1, 2)  # [batch_size, time_frames, channels * mel_bins]
        
        # Initialize projection layer on first forward pass if needed
        input_dim = channels * mel_bins
        if not self._projection_initialized:
            # Create and register projection layer
            self.projection = nn.Linear(input_dim, self.rnn_input_size).to(x.device)
            self.add_module('projection', self.projection)
            self._projection_initialized = True
        
        # Project to RNN input size
        x = self.projection(x)  # [batch_size, time_frames, rnn_input_size]
        
        # Process through bidirectional RNN
        # Output: [batch_size, time_frames, hidden_size * 2]
        rnn_out, _ = self.rnn(x)
        
        # Global temporal pooling: take mean over time dimension
        # This aggregates temporal information into a fixed-size representation
        # Alternative: use last timestep or attention mechanism
        temporal_embedding = rnn_out.mean(dim=1)  # [batch_size, hidden_size * 2]
        
        return temporal_embedding


class eGeMAPSEncoder(nn.Module):
    """
    MLP encoder for eGeMAPS acoustic features.
    
    Learns compact, interpretable embeddings from fixed-length acoustic features.
    Uses multiple layers with ReLU activation and dropout for regularization.
    
    Input: [batch_size, feature_dim] (≈88 features)
    Output: [batch_size, embedding_dim]
    """
    
    def __init__(
        self,
        input_dim: int = 88,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
        embedding_dim: int = 64,
    ):
        """
        Args:
            input_dim: Dimension of eGeMAPS features (typically 88)
            hidden_dims: Tuple of hidden layer dimensions (at least 2 layers)
            dropout: Dropout rate for regularization
            embedding_dim: Final embedding dimension
        """
        super(eGeMAPSEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        
        # Build MLP layers: Linear → ReLU → Dropout (repeat 2× minimum)
        layers = []
        
        # First layer: input → first hidden
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Final projection to embedding dimension
        layers.extend([
            nn.Linear(prev_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ])
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through eGeMAPS encoder.
        
        Args:
            x: eGeMAPS features [batch_size, input_dim]
        
        Returns:
            Embedding [batch_size, embedding_dim]
        """
        # x: [batch_size, input_dim]
        embedding = self.mlp(x)
        # Output: [batch_size, embedding_dim]
        return embedding


class FusionClassifier(nn.Module):
    """
    Feature fusion and classification head.
    
    Concatenates CRNN and eGeMAPS embeddings, applies fusion layer with dropout,
    and produces class logits.
    
    Input: 
        - CRNN embedding: [batch_size, crnn_dim]
        - eGeMAPS embedding: [batch_size, egemaps_dim]
    Output: [batch_size, num_classes] (logits, no softmax)
    """
    
    def __init__(
        self,
        crnn_dim: int,
        egemaps_dim: int,
        fusion_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        """
        Args:
            crnn_dim: Dimension of CRNN embedding
            egemaps_dim: Dimension of eGeMAPS embedding
            fusion_dim: Dimension of fusion layer
            num_classes: Number of output classes (3: AD / MCI / Control)
            dropout: Dropout rate for fusion layer
        """
        super(FusionClassifier, self).__init__()
        
        self.crnn_dim = crnn_dim
        self.egemaps_dim = egemaps_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        
        # Fusion layer: concatenated features → fusion_dim
        self.fusion = nn.Sequential(
            nn.Linear(crnn_dim + egemaps_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Classification head: fusion_dim → num_classes (logits only)
        self.classifier = nn.Linear(fusion_dim, num_classes)
    
    def forward(
        self,
        crnn_embedding: torch.Tensor,
        egemaps_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through fusion and classification.
        
        Args:
            crnn_embedding: CRNN temporal embedding [batch_size, crnn_dim]
            egemaps_embedding: eGeMAPS acoustic embedding [batch_size, egemaps_dim]
        
        Returns:
            Class logits [batch_size, num_classes] (no softmax applied)
        """
        # Concatenate embeddings
        # crnn_embedding: [batch_size, crnn_dim]
        # egemaps_embedding: [batch_size, egemaps_dim]
        fused = torch.cat([crnn_embedding, egemaps_embedding], dim=1)
        # fused: [batch_size, crnn_dim + egemaps_dim]
        
        # Apply fusion layer
        fused = self.fusion(fused)
        # fused: [batch_size, fusion_dim]
        
        # Generate logits (no softmax - applied in loss function)
        logits = self.classifier(fused)
        # logits: [batch_size, num_classes]
        
        return logits


class HybridCRNNModel(nn.Module):
    """
    Complete hybrid CRNN + MLP model for Alzheimer's Disease classification.
    
    Architecture:
    1. CNN encoder processes log-Mel spectrograms
    2. CRNN encoder captures temporal prosodic patterns
    3. eGeMAPS encoder processes acoustic features
    4. Fusion classifier combines embeddings and predicts classes
    
    Inputs:
        - spectrogram: [batch_size, 1, mel_bins, time_frames]
        - egemaps_features: [batch_size, egemaps_dim]
    
    Output:
        - logits: [batch_size, num_classes] (AD / MCI / Control)
    """
    
    def __init__(
        self,
        # Spectrogram parameters
        mel_bins: int = 128,
        # CNN parameters
        cnn_base_channels: int = 32,
        cnn_num_blocks: int = 3,
        cnn_kernel_size: Tuple[int, int] = (3, 3),
        cnn_pool_size: Tuple[int, int] = (2, 2),
        # CRNN parameters
        crnn_hidden_size: int = 128,
        crnn_num_layers: int = 2,
        crnn_dropout: float = 0.3,
        use_gru: bool = False,
        # eGeMAPS parameters
        egemaps_input_dim: int = 88,
        egemaps_hidden_dims: Tuple[int, ...] = (128, 64),
        egemaps_embedding_dim: int = 64,
        egemaps_dropout: float = 0.3,
        # Fusion parameters
        fusion_dim: int = 128,
        fusion_dropout: float = 0.3,
        # Classification parameters
        num_classes: int = 3,
    ):
        """
        Initialize hybrid CRNN + MLP model.
        
        Args:
            mel_bins: Number of Mel frequency bins in spectrogram
            cnn_base_channels: Base channels for CNN encoder
            cnn_num_blocks: Number of CNN blocks (2-3 recommended)
            cnn_kernel_size: Convolution kernel size
            cnn_pool_size: Max pooling size
            crnn_hidden_size: Hidden size for RNN cells
            crnn_num_layers: Number of RNN layers
            crnn_dropout: Dropout rate for RNN
            use_gru: Use GRU instead of LSTM
            egemaps_input_dim: Dimension of eGeMAPS features
            egemaps_hidden_dims: Hidden layer dimensions for eGeMAPS MLP
            egemaps_embedding_dim: Final eGeMAPS embedding dimension
            egemaps_dropout: Dropout rate for eGeMAPS MLP
            fusion_dim: Dimension of fusion layer
            fusion_dropout: Dropout rate for fusion layer
            num_classes: Number of output classes (3: AD / MCI / Control)
        """
        super(HybridCRNNModel, self).__init__()
        
        # CNN encoder for spectrogram processing
        self.cnn_encoder = CNNEncoder(
            in_channels=1,
            base_channels=cnn_base_channels,
            num_blocks=cnn_num_blocks,
            mel_bins=mel_bins,
            kernel_size=cnn_kernel_size,
            pool_size=cnn_pool_size,
        )
        
        # CRNN encoder for temporal modeling
        # Uses projection layer to handle variable input sizes after frequency collapse
        self.crnn_encoder = CRNNEncoder(
            in_channels=cnn_base_channels * (2 ** (cnn_num_blocks - 1)),  # Final CNN channels
            rnn_input_size=crnn_hidden_size,  # Project to same size as hidden for efficiency
            hidden_size=crnn_hidden_size,
            num_layers=crnn_num_layers,
            dropout=crnn_dropout,
            use_gru=use_gru,
        )
        
        # eGeMAPS encoder for acoustic features
        self.egemaps_encoder = eGeMAPSEncoder(
            input_dim=egemaps_input_dim,
            hidden_dims=egemaps_hidden_dims,
            dropout=egemaps_dropout,
            embedding_dim=egemaps_embedding_dim,
        )
        
        # Fusion classifier
        self.fusion_classifier = FusionClassifier(
            crnn_dim=crnn_hidden_size * 2,  # Bidirectional
            egemaps_dim=egemaps_embedding_dim,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            dropout=fusion_dropout,
        )
        
        # Store configuration for saving/loading
        self.mel_bins = mel_bins
        self.num_classes = num_classes
        self.config = {
            "mel_bins": mel_bins,
            "cnn_base_channels": cnn_base_channels,
            "cnn_num_blocks": cnn_num_blocks,
            "cnn_kernel_size": cnn_kernel_size,
            "cnn_pool_size": cnn_pool_size,
            "crnn_hidden_size": crnn_hidden_size,
            "crnn_num_layers": crnn_num_layers,
            "crnn_dropout": crnn_dropout,
            "use_gru": use_gru,
            "egemaps_input_dim": egemaps_input_dim,
            "egemaps_hidden_dims": egemaps_hidden_dims,
            "egemaps_embedding_dim": egemaps_embedding_dim,
            "egemaps_dropout": egemaps_dropout,
            "fusion_dim": fusion_dim,
            "fusion_dropout": fusion_dropout,
            "num_classes": num_classes,
        }
    
    def forward(
        self,
        spectrogram: torch.Tensor,
        egemaps_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through complete model.
        
        Args:
            spectrogram: Log-Mel spectrogram [batch_size, 1, mel_bins, time_frames]
            egemaps_features: eGeMAPS acoustic features [batch_size, egemaps_input_dim]
        
        Returns:
            Class logits [batch_size, num_classes] (no softmax applied)
        """
        # Input validation
        assert spectrogram.dim() == 4, f"Spectrogram must be 4D, got {spectrogram.dim()}D"
        assert spectrogram.size(1) == 1, f"Spectrogram must have 1 channel, got {spectrogram.size(1)}"
        assert egemaps_features.dim() == 2, f"eGeMAPS features must be 2D, got {egemaps_features.dim()}D"
        
        batch_size = spectrogram.size(0)
        
        # 1. CNN encoder: process spectrogram
        # Input: [batch_size, 1, mel_bins, time_frames]
        cnn_features = self.cnn_encoder(spectrogram)
        # Output: [batch_size, cnn_channels, reduced_mel_bins, reduced_time_frames]
        
        # 2. CRNN encoder: temporal modeling
        # Input: [batch_size, cnn_channels, reduced_mel_bins, reduced_time_frames]
        crnn_embedding = self.crnn_encoder(cnn_features)
        # Output: [batch_size, crnn_hidden_size * 2]
        
        # 3. eGeMAPS encoder: acoustic features
        # Input: [batch_size, egemaps_input_dim]
        egemaps_embedding = self.egemaps_encoder(egemaps_features)
        # Output: [batch_size, egemaps_embedding_dim]
        
        # 4. Fusion and classification
        # Inputs: [batch_size, crnn_dim], [batch_size, egemaps_dim]
        logits = self.fusion_classifier(crnn_embedding, egemaps_embedding)
        # Output: [batch_size, num_classes] (logits, no softmax)
        
        return logits
    
    def get_model_info(self) -> dict:
        """
        Get model configuration and parameter count.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Get device information
        device = next(self.parameters()).device
        device_type = "CUDA" if device.type == "cuda" else "CPU"
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "mel_bins": self.mel_bins,
            "num_classes": self.num_classes,
            "cnn_blocks": self.cnn_encoder.num_blocks,
            "crnn_hidden_size": self.crnn_encoder.hidden_size,
            "crnn_layers": self.crnn_encoder.num_layers,
            "device": device_type,
            "device_id": device.index if device.type == "cuda" else None,
        }
    
    def save(self, filepath: str, additional_info: Optional[Dict] = None) -> None:
        """
        Save the model state dict and configuration to disk.
        
        Args:
            filepath: Path to save the model (should end with .pth or .pt)
            additional_info: Optional dictionary with additional info to save
                            (e.g., training metrics, epoch, optimizer state)
        
        Example:
            model.save("models/best_model.pth", {"epoch": 50, "val_acc": 0.85})
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        # Prepare save dictionary
        save_dict = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }
        
        # Add additional info if provided
        if additional_info is not None:
            save_dict["additional_info"] = additional_info
        
        # Save model
        torch.save(save_dict, filepath)
        print(f"✅ Model saved to: {filepath}")
        
        # Also save config as JSON for easy inspection
        config_path = filepath.replace(".pth", "_config.json").replace(".pt", "_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"✅ Config saved to: {config_path}")
    
    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None, strict: bool = True) -> Tuple["HybridCRNNModel", Dict]:
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model file
            device: Device to load the model on (if None, uses saved device or CPU)
            strict: Whether to strictly enforce that the keys in state_dict match
        
        Returns:
            Tuple of (loaded_model, additional_info_dict)
        
        Example:
            model, info = HybridCRNNModel.load("models/best_model.pth", device=torch.device("cuda"))
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=device if device else "cpu")
        
        # Extract components
        model_state_dict = checkpoint["model_state_dict"]
        config = checkpoint["config"]
        additional_info = checkpoint.get("additional_info", {})
        
        # Create model with saved configuration
        model = cls(**config)
        
        # Load state dict
        model.load_state_dict(model_state_dict, strict=strict)
        
        # Move to device if specified
        if device is not None:
            model = model.to(device)
        
        print(f"✅ Model loaded from: {filepath}")
        if additional_info:
            print(f"   Additional info: {list(additional_info.keys())}")
        
        return model, additional_info


# =========================
# EXAMPLE USAGE
# =========================
if __name__ == "__main__":
    # Example: Create model and test forward pass with GPU support
    
    # =========================
    # DEVICE SETUP (GPU/CPU)
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Device Configuration")
    print("=" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Using Device: {device}")
    print("=" * 60)
    
    # Model configuration
    model = HybridCRNNModel(
        mel_bins=128,
        cnn_base_channels=32,
        cnn_num_blocks=3,
        egemaps_input_dim=88,
        num_classes=3,
        crnn_hidden_size=128,
    )
    
    # Move model to GPU (or CPU if GPU not available)
    model = model.to(device)
    print(f"\nModel moved to: {device}")
    
    # Print model information
    info = model.get_model_info()
    print("\n" + "=" * 60)
    print("Hybrid CRNN + MLP Model")
    print("=" * 60)
    for key, value in info.items():
        if isinstance(value, int):
            print(f"{key}: {value:,}")
        elif value is None:
            print(f"{key}: N/A")
        else:
            print(f"{key}: {value}")
    print("=" * 60)
    
    # Test forward pass
    batch_size = 4
    mel_bins = 128
    time_frames = 100  # Example: 10 seconds at 10ms hop length
    egemaps_dim = 88
    
    # Create dummy inputs and move to same device as model
    spectrogram = torch.randn(batch_size, 1, mel_bins, time_frames).to(device)
    egemaps_features = torch.randn(batch_size, egemaps_dim).to(device)
    
    print(f"\nInputs moved to: {device}")
    print(f"  Spectrogram device: {spectrogram.device}")
    print(f"  eGeMAPS features device: {egemaps_features.device}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(spectrogram, egemaps_features)
    
    print(f"\nOutput device: {logits.device}")
    print(f"\nInput shapes:")
    print(f"  Spectrogram: {spectrogram.shape}")
    print(f"  eGeMAPS features: {egemaps_features.shape}")
    print(f"\nOutput shape:")
    print(f"  Logits: {logits.shape}")
    print(f"\nLogits (first sample):")
    print(f"  {logits[0]}")
    print(f"\nProbabilities (after softmax):")
    probs = F.softmax(logits[0], dim=0)
    print(f"  AD: {probs[0]:.4f}, MCI: {probs[1]:.4f}, Control: {probs[2]:.4f}")
    print("=" * 60)
    
    # Verify GPU usage
    if torch.cuda.is_available():
        print(f"\n✅ GPU Training Ready!")
        print(f"   Model is on: {next(model.parameters()).device}")
        print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"   Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print(f"\n⚠️  GPU not available - using CPU")
    print("=" * 60)
    
    # =========================
    # DEMONSTRATE MODEL SAVING
    # =========================
    print("\n" + "=" * 60)
    print("Model Saving & Loading Demo")
    print("=" * 60)
    
    # Get default model save directory
    models_dir = get_model_save_dir()
    model_path = os.path.join(models_dir, "test_model.pth")
    
    # Save model with additional info (e.g., training metrics)
    additional_info = {
        "epoch": 50,
        "val_accuracy": 0.85,
        "train_loss": 0.32,
        "val_loss": 0.45,
        "best_epoch": 45,
    }
    
    print(f"\nSaving model to: {model_path}")
    model.save(model_path, additional_info=additional_info)
    
    # Load the saved model
    print(f"\nLoading model from: {model_path}")
    loaded_model, loaded_info = HybridCRNNModel.load(model_path, device=device)
    
    # Verify loaded model works
    print("\nTesting loaded model...")
    loaded_model.eval()
    with torch.no_grad():
        loaded_logits = loaded_model(spectrogram, egemaps_features)
    
    # Check if outputs match (should be very close)
    logits_match = torch.allclose(logits, loaded_logits, atol=1e-6)
    print(f"✅ Loaded model outputs match original: {logits_match}")
    
    # Display loaded additional info
    if loaded_info:
        print(f"\nLoaded additional info:")
        for key, value in loaded_info.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Model saving and loading verified!")
    print("=" * 60)

