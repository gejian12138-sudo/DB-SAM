# Configuration for DB-SAM training and inference.
# All paths are relative to the project root.

config_dict = {
    # Data directories
    'train_dir': './dataset/data/train',
    # Random seed for reproducibility
    'random_seed': 42,
    # Path to the original SAM ViT-B checkpoint (download separately, see README)
    'checkpoint_path': './checkpoints/sam_vit_b_01ec64.pth',
    # Input image size
    'img_size': 1024,
    # Pixel mean for image normalization (ImageNet stats)
    'pixel_mean': [123.675, 116.28, 103.53],
    # Pixel std for image normalization (ImageNet stats)
    'pixel_std': [58.395, 57.12, 57.375],
    # Working directory for logs and model checkpoints
    'work_dir': 'workdir',
}
