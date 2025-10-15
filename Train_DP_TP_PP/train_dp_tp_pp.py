# """
# Pipeline parallel training implementation - Fixed version
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.distributed as dist
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import time
# import numpy as np
# import random

# # Import from utilities package  
# from ..utilities.utils import *
# from ..utilities.Dataloader import CustomDataset, mnist_transform
# from ..utilities.model import Model

# # Import pipeline parallelism components
# from QuintNet.PipelineParallelism import PipelineParallelWrapper, ProcessGroupManager, PipelineTrainer


# # Debug utility functions
# def debug_model_partitioning(model_wrapper, rank):
#     """Debug model partitioning across stages"""
#     print(f"\n[PARTITION DEBUG Rank {rank}] Model Partitioning Analysis:")
    
#     # Count parameters in local stage
#     local_params = sum(p.numel() for p in model_wrapper.local_module.parameters())
    
#     # Gather parameter counts from all ranks
#     param_counts = [None] * model_wrapper.world_size
#     dist.all_gather_object(param_counts, local_params, group=model_wrapper.pp_group)
    
#     if rank == 0:
#         total_params = sum(param_counts)
#         print(f"  Total parameters: {total_params:,}")
#         for stage_id, count in enumerate(param_counts):
#             percentage = (count / total_params) * 100
#             print(f"  Stage {stage_id}: {count:,} parameters ({percentage:.1f}%)")
        
#         # Check balance
#         min_params = min(param_counts)
#         max_params = max(param_counts)
#         imbalance = (max_params - min_params) / min_params * 100
#         print(f"  Load imbalance: {imbalance:.1f}%")
        
# def train_epoch(pipeline_trainer, train_loader, device, rank, pp_size):
#     """Train model for one epoch with pipeline parallelism."""
#     pipeline_trainer.model.train()
#     running_loss = 0.0
#     running_accuracy = 0.0
#     num_batches = 0
    
#     # Progress bar only on first rank
#     if rank == 0:
#         pbar = tqdm(train_loader, desc="Training")
#     else:
#         pbar = train_loader
    
#     for batch_idx, batch in enumerate(pbar):
#         images = batch['image']
#         labels = batch['label']
        
#         # All ranks process the batch
#         loss, accuracy = pipeline_trainer.train_step(images, labels)
        
#         # Only last rank gets loss/accuracy values
#         if rank == pp_size - 1 and loss is not None:
#             running_loss += loss
#             running_accuracy += accuracy
#             num_batches += 1
            
#             # Update progress bar
#             if rank == 0:
#                 pbar.set_postfix({
#                     'Loss': f'{loss:.4f}',
#                     'Acc': f'{accuracy:.2f}%'
#                 })
    
#     # Calculate averages (only meaningful on last rank)
#     if rank == pp_size - 1 and num_batches > 0:
        
#         return avg_loss, avg_accuracy
#     else:
#         return None, None
    
# def train_epoch_afab_optimised(pipeline_trainer, train_loader, device, rank, pgm, pp_size, epoch):
#     """Train model for one epoch with pipeline parallelism."""
#     pipeline_trainer.model.train()
#     loss, acc = pipeline_trainer.train_all_forward_and_backward_optimised(train_loader, pgm, epoch)
#     if rank == pp_size - 1:
#         return loss, acc
#     else:
#         return None, None
    
# def train_epoch_onef_oneb(pipeline_trainer, train_loader, device, rank, pgm, pp_size, epoch):
#     """Train model for one epoch with pipeline parallelism."""
#     pipeline_trainer.model.train()
#     loss, acc = pipeline_trainer.train_1f1b_three_phase(train_loader, epoch)
#     if rank == pp_size - 1:
#         return loss, acc
#     else:
#         return None, None


# def validate(pipeline_trainer, val_loader, rank):
#     """Validate model with pipeline parallelism."""
#     avg_loss, avg_accuracy = pipeline_trainer.evaluate(val_loader)
#     return avg_loss, avg_accuracy

# def print_debug_norms(model, rank, point_in_time: str):
#     """
#     Prints the L2 norm of the model's parameters and their gradients.

#     Args:
#         model: The local model module for the current rank.
#         rank: The rank of the current process.
#         point_in_time: A string describing when this is being called (e.g., "Before step").
#     """
#     total_param_norm = 0.0
#     total_grad_norm = 0.0
    
#     # Ensure we only access parameters of the local stage
#     params = list(model.parameters())
#     if not params:
#         print(f"[DEBUG Rank {rank}] ({point_in_time}): No parameters on this rank.")
#         return

#     for p in params:
#         if p.requires_grad:
#             param_norm = p.detach().data.norm(2)
#             total_param_norm += param_norm.item() ** 2
            
#             if p.grad is not None:
#                 grad_norm = p.grad.detach().data.norm(2)
#                 total_grad_norm += grad_norm.item() ** 2

#     total_param_norm = total_param_norm ** 0.5
#     total_grad_norm = total_grad_norm ** 0.5

#     print(f"[DEBUG Rank {rank}] ({point_in_time}): "
#           f"Param Norm = {total_param_norm:.4f}, "
#           f"Grad Norm = {total_grad_norm:.4f}")

# def verify_pipeline_split(pp_model, rank, pp_group):
#     """Verify that the pipeline split is correct."""
#     print(f"\n{'='*60}")
#     print(f"[Rank {rank}] Pipeline Split Verification")
#     print(f"{'='*60}")
    
#     # Print local module structure
#     print(f"Local module type: {type(pp_model.local_module)}")
#     if isinstance(pp_model.local_module, nn.Sequential):
#         print(f"Sequential with {len(pp_model.local_module)} modules:")
#         for i, module in enumerate(pp_model.local_module):
#             print(f"  {i}: {type(module).__name__}")
#     else:
#         print(f"Single module: {type(pp_model.local_module).__name__}")
    
#     # Count parameters
#     param_count = sum(p.numel() for p in pp_model.local_module.parameters())
#     print(f"Parameters: {param_count:,}")
    
#     # Test with dummy data to verify shapes
#     if rank == 0:
#         # Test first stage with image input
#         dummy_input = torch.randn(2, 1, 28, 28).to(f'cuda:{rank}')
#         try:
#             with torch.no_grad():
#                 output = pp_model(dummy_input)
#             print(f"Input shape: {dummy_input.shape}")
#             print(f"Output shape: {output.shape}")
#         except Exception as e:
#             print(f"❌ ERROR during forward: {e}")
#     else:
#         # Test subsequent stages - need to know expected input shape
#         # For ViT after embedding: [B, num_patches, hidden_dim]
#         num_patches = (28 // 4) ** 2  # 49 patches for 28x28 image with patch_size=4
#         dummy_input = torch.randn(2, num_patches, 64).to(f'cuda:{rank}')
#         try:
#             with torch.no_grad():
#                 output = pp_model(dummy_input)
#             print(f"Input shape: {dummy_input.shape}")
#             print(f"Output shape: {output.shape}")
            
#             # Last stage should output [B, num_classes]
#             if rank == 1:  # Last stage
#                 expected_shape = (2, 10)  # 10 classes for MNIST
#                 if output.shape != expected_shape:
#                     print(f"⚠️  WARNING: Expected output shape {expected_shape}, got {output.shape}")
#         except Exception as e:
#             print(f"❌ ERROR during forward: {e}")
    
#     print(f"{'='*60}\n")
    
    
# def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, rank, pp_size, pp_group, pgm, debug_level=2):
#     """Complete training loop with pipeline parallelism."""
#     criterion = nn.CrossEntropyLoss()
    
#     # STEP 1: Synchronize initial weights
#     if rank == 0:
#         print("\n" + "="*60)
#         print("INITIALIZING PIPELINE PARALLEL TRAINING")
#         print("="*60)
#         print("Step 1: Synchronizing model weights across all ranks...")
    
#     for param in model.parameters():
#         dist.broadcast(param.data, src=0, group=pp_group)
    
#     dist.barrier(group=pp_group)
#     if rank == 0:
#         print("✓ Model weights synchronized\n")
    
#     # STEP 2: Create PipelineParallelWrapper
#     if rank == 0:
#         print("Step 2: Creating pipeline parallel wrapper...")
#     pp_model = PipelineParallelWrapper(model, pgm).to(device)
    
#     # STEP 3: Verify the split
#     verify_pipeline_split(pp_model, rank, pp_group)
#     dist.barrier(group=pp_group)
    
#     # STEP 4: Verify data consistency
#     if rank == 0:
#         print("Step 3: Verifying data consistency...")
    
#     # Get first batch and check if all ranks have the same data
#     first_batch = next(iter(train_loader))
    
#     # Check image hash
#     img_sum = first_batch['image'].sum().item()
#     label_sum = first_batch['label'].sum().item()
    
#     img_sums = [None] * pp_size
#     label_sums = [None] * pp_size
#     dist.all_gather_object(img_sums, img_sum, group=pp_group)
#     dist.all_gather_object(label_sums, label_sum, group=pp_group)
    
#     if rank == 0:
#         if len(set(img_sums)) == 1 and len(set(label_sums)) == 1:
#             print(f"✓ Data is consistent across all ranks")
#             print(f"  Image sum: {img_sums[0]:.4f}")
#             print(f"  Label sum: {label_sums[0]:.4f}")
#         else:
#             print(f"❌ WARNING: Data differs across ranks!")
#             print(f"  Image sums: {img_sums}")
#             print(f"  Label sums: {label_sums}")
#             print(f"  This will cause training to fail!")
#         print("="*60 + "\n")
    
#     dist.barrier(group=pp_group)
    
#     # Create optimizer for local parameters
#     optimizer = optim.Adam(pp_model.parameters(), lr=learning_rate)
    
#     # Create PipelineTrainer  
#     pipeline_trainer = PipelineTrainer(pp_model, pp_group, criterion, device, optimizer=optimizer, max_grad_norm=1.0)
    
#     # Rest of training loop stays the same...
#     patience = 5
#     min_improvement = 0.01
#     best_val_acc = 0.0
#     best_model_state = None
#     epochs_without_improvement = 0
    
#     train_losses = []
#     val_losses = []
#     train_accs = []
#     val_accs = []
    
#     for epoch in range(num_epochs):
#         if rank == 0:
#             print(f"\nEpoch {epoch+1}/{num_epochs}")
#             print("-" * 50)
        
#         # Train
#         train_loss, train_acc = train_epoch_onef_oneb(pipeline_trainer, train_loader, device, rank, pgm, pp_size, epoch)
        
#         # Validate
#         val_loss, val_acc = validate(pipeline_trainer, val_loader, rank)
        
#         # Only last rank has metrics
#         if rank == pp_size - 1:
#             train_losses.append(train_loss)
#             train_accs.append(train_acc)
#             val_losses.append(val_loss)
#             val_accs.append(val_acc)
            
#             improved = val_acc > best_val_acc + min_improvement
            
#             if improved:
#                 best_val_acc = val_acc
#                 epochs_without_improvement = 0
#                 best_model_state = {
#                     'model_state_dict': pp_model.state_dict(),
#                     'epoch': epoch,
#                     'val_acc': val_acc
#                 }
#                 improved_tensor = torch.tensor([1.0], device=device)
#             else:
#                 epochs_without_improvement += 1
#                 improved_tensor = torch.tensor([0.0], device=device)
            
#             dist.broadcast(improved_tensor, src=pp_size-1, group=pp_group)
            
#             status = " 🎉 NEW BEST!" if improved else ""
#             print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
#             print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%{status}")
#             print(f"Best Val Acc: {best_val_acc:.2f}%, Patience: {patience - epochs_without_improvement}")
            
#             if epochs_without_improvement >= patience:
#                 print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
#                 break
#         else:
#             improved_tensor = torch.tensor([0.0], device=device)
#             dist.broadcast(improved_tensor, src=pp_size-1, group=pp_group)
        
#         dist.barrier(group=pp_group)
    
#     if best_model_state is not None and rank == pp_size - 1:
#         pp_model.load_state_dict(best_model_state['model_state_dict'])
#         print(f"\n🏆 Training finished. Best model loaded (Val Acc: {best_val_acc:.2f}%)")
    
#     if rank == pp_size - 1:
#         return {
#             'train_losses': train_losses,
#             'val_losses': val_losses,
#             'train_accs': train_accs,
#             'val_accs': val_accs,
#             'best_val_acc': best_val_acc
#         }
#     else:
#         return None
    
    
# def synchronize_model_weights(model, rank, pp_group):
#     """Broadcast model weights from rank 0 to all ranks to ensure consistency."""
#     print(f"[Rank {rank}] Synchronizing model weights...")
    
#     for param in model.parameters():
#         # Broadcast each parameter from rank 0
#         dist.broadcast(param.data, src=0, group=pp_group)
    
#     print(f"[Rank {rank}] Model weights synchronized.")
    
# def main():
#     # Set seeds for reproducibility
#     def set_seed(seed=42):
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         np.random.seed(seed)
#         random.seed(seed)
#         torch.backends.cudnn.deterministic = True
    
#     # Initialize distributed
#     dist.init_process_group(backend="nccl")
    
#     # Get the global rank to set the device
#     global_rank = dist.get_rank()
    
#     # 2. Set the CUDA device for this specific process. THIS IS CRITICAL.
#     torch.cuda.set_device(global_rank)
    
#     # --- For your current setup ---
#     pp_size = 4
#     tp_size = 1 # You are not using tensor parallelism yet
    
#     # 3. NOW that devices are set, create the subgroups.
#     print(f"Rank {global_rank}: Initializing ProcessGroupManager...")
#     pgm = ProcessGroupManager(pp_size=pp_size, tp_size=tp_size)
#     pp_group = pgm.get_pp_group()
    
#     # Use the rank within the pipeline group for your logic
#     rank = pgm.get_pp_rank()
#     world_size = dist.get_world_size(pp_group) # This is your pipeline size
    
#     device = torch.device(f"cuda:{rank}")
    
#     print(f"Global Rank {global_rank} is Pipeline Rank {rank} on device {device}")
    
    
#     # Configuration
#     config = {
#         'dataset_path': '/mnt/dataset/mnist/',  # Changed from '/workspace/dataset/'
#         'batch_size': 8,
#         'num_epochs': 4,
#         'learning_rate': 0.0001,
#         'num_workers': 1
#     }
    
#     device = torch.device(f"cuda:{rank}")
    
#     if rank == 0:
#         print(f"Using {world_size} GPUs for pipeline parallelism")
#         print(f"Device assignment: Rank {rank} -> {device}")
    
#     # Set seeds
#     set_seed(42)
    
#     # Load datasets
#     train_dataset = CustomDataset(
#         config['dataset_path'], 
#         split='train', 
#         transform=mnist_transform
#     )
#     val_dataset = CustomDataset(
#         config['dataset_path'], 
#         split='test', 
#         transform=mnist_transform
#     )
    
#     # Create dataloaders with same data for all ranks
#     def worker_init_fn(worker_id):
#         set_seed(42)
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         num_workers=config['num_workers'],
#         worker_init_fn=worker_init_fn,
#         persistent_workers=True,
#         generator=torch.Generator().manual_seed(42),
#         drop_last=True  # Important for pipeline parallelism
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config['batch_size'],
#         shuffle=False,
#         num_workers=config['num_workers'],
#         persistent_workers=True,
#         drop_last=False
#     )
    
#     # Verify data consistency
#     if rank == 0:
#         print("Verifying data consistency across ranks...")
    
#     first_batch = next(iter(train_loader))
#     batch_sum = first_batch['image'].sum().item()
    
#     gathered_sums = [None] * world_size
#     dist.all_gather_object(gathered_sums, batch_sum)
    
#     if rank == 0:
#         if len(set(gathered_sums)) == 1:
#             print("✓ All ranks have identical data")
#         else:
#             print("✗ WARNING: Data differs across ranks!")
#             print(f"Batch sums: {gathered_sums}")
    
#     # Create model (full model on each rank, will be split by wrapper)
#     model = Model(
#         img_size=28,
#         patch_size=4,
#         hidden_dim=64,
#         in_channels=1,
#         n_heads=4,
#         depth=8
#         ).to(device)
    
    
#     if rank == 0:
#         # Count total parameters before splitting
#         total_params = sum(p.numel() for p in model.parameters())
#         print(f"Total model parameters: {total_params:,}")
    
#     if rank == 0:
#         print("\n🔄 Synchronizing model weights across all pipeline stages...")
#     synchronize_model_weights(model, rank, pp_group)
#     dist.barrier(group=pp_group)  # Ensure all ranks finish synchronization
#     if rank == 0:
#         print("✓ Model weights synchronized.\n")
        
#     # Train model
#     start_time = time.time()
#     results = train_model(
#         model,
#         train_loader,
#         val_loader,
#         config['num_epochs'],
#         config['learning_rate'],
#         device,
#         rank,
#         pp_size,
#         pp_group,
#         pgm
#     )
    
#     training_time = time.time() - start_time
    
#     if rank == pp_size - 1 and results is not None:
#         print(f"\nTraining completed in {training_time//60:.0f}m {training_time%60:.0f}s")
#         print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    
#     # Cleanup
#     dist.barrier()
#     dist.destroy_process_group()


# if __name__ == "__main__":
#     main()
    
    
    


# # def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, rank, pp_size, pp_group, pgm, debug_level=2):
# #     """Complete training loop with pipeline parallelism."""
# #     criterion = nn.CrossEntropyLoss()
    
# #     # Create PipelineParallelWrapper first
# #     pp_model = PipelineParallelWrapper(model, pgm).to(device)
    
# #     if rank == 0:
# #         print("\n--- Checking requires_grad status for all parameters ---")
# #     # This check needs to be run on each rank for its local parameters
# #     for name, p in pp_model.local_module.named_parameters():
# #         if "attention.K.bias" in name:
# #             print(f"Rank {rank}: {name:<40} | requires_grad: {p.requires_grad}")
# #     # Create optimizer for local parameters only
# #     optimizer = optim.Adam(pp_model.parameters(), lr=learning_rate)
    
# #     # Create PipelineTrainer
# #     pipeline_trainer = PipelineTrainer(pp_model, pp_group, criterion, device, optimizer=optimizer)
    
# #     # Training parameters
# #     patience = 5
# #     min_improvement = 0.01
# #     best_val_acc = 0.0
# #     best_model_state = None
# #     epochs_without_improvement = 0
    
# #     train_losses = []
# #     val_losses = []
# #     train_accs = []
# #     val_accs = []
    
# #     for epoch in range(num_epochs):
# #         print_debug_norms(pp_model, rank, f"Start of Epoch {epoch+1}")
# #         if rank == 0:
# #             print(f"\nEpoch {epoch+1}/{num_epochs}")
# #             print("-" * 50)
        
# #         # Train
# #         train_loss, train_acc = train_epoch_onef_oneb(pipeline_trainer, train_loader, device, rank, pgm, pp_size, epoch)
        
# #         # Validate
# #         val_loss, val_acc = validate(pipeline_trainer, val_loader, rank)
        
# #         # Only last rank has metrics
# #         if rank == pp_size - 1:
# #             train_losses.append(train_loss)
# #             train_accs.append(train_acc)
# #             val_losses.append(val_loss)
# #             val_accs.append(val_acc)
            
# #             # Check for improvement
# #             improved = val_acc > best_val_acc + min_improvement
            
# #             if improved:
# #                 best_val_acc = val_acc
# #                 epochs_without_improvement = 0
                
# #                 # Save best model state (local stage only)
# #                 best_model_state = {
# #                     'model_state_dict': pp_model.state_dict(),
# #                     # 'optimizer_state_dict': optimizer.state_dict(),
# #                     'epoch': epoch,
# #                     'val_acc': val_acc
# #                 }
                
# #                 # Notify other ranks about improvement
# #                 improved_tensor = torch.tensor([1.0], device=device)
# #             else:
# #                 epochs_without_improvement += 1
# #                 improved_tensor = torch.tensor([0.0], device=device)
            
# #             # Broadcast improvement status
# #             dist.broadcast(improved_tensor, src=pp_size-1, group=pp_group)
            
# #             # Print metrics
# #             status = " 🎉 NEW BEST!" if improved else ""
# #             print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
# #             print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%{status}")
# #             print(f"Best Val Acc: {best_val_acc:.2f}%, Patience: {patience - epochs_without_improvement}")
            
# #             # Early stopping check
# #             if epochs_without_improvement >= patience:
# #                 print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
# #                 break
# #         else:
# #             # Other ranks wait for improvement signal
# #             improved_tensor = torch.tensor([0.0], device=device)
# #             dist.broadcast(improved_tensor, src=pp_size-1, group=pp_group)
        
# #         # Synchronize all ranks
# #         dist.barrier(group=pp_group)
    
# #     # Load best model if available
# #     if best_model_state is not None and rank == pp_size - 1:
# #         pp_model.load_state_dict(best_model_state['model_state_dict'])
# #         print(f"\n🏆 Training finished. Best model loaded (Val Acc: {best_val_acc:.2f}%)")
    
# #     # Return results
# #     if rank == pp_size - 1:
# #         return {
# #             'train_losses': train_losses,
# #             'val_losses': val_losses,
# #             'train_accs': train_accs,
# #             'val_accs': val_accs,
# #             'best_val_acc': best_val_acc
# #         }
# #     else:
# #         return None

"""
Pipeline Parallel Training for MNIST Classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
import random
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from utilities
from utilities.utils import *
from utilities.Dataloader import CustomDataset, mnist_transform
from utilities.model import Model

# Import pipeline parallelism components
from QuintNet.PipelineParallelism import (
    ProcessGroupManager,
    PipelineParallelWrapper,
    PipelineTrainer
)


class PipelineDataLoader:
    """
    Wrapper for DataLoader to support gradient accumulation.
    """
    def __init__(self, dataloader, grad_acc_steps):
        self.dataloader = dataloader
        self.grad_acc_steps = grad_acc_steps
        self.iterator = iter(dataloader)
    
    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self
    
    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        
        # Convert to expected format
        return {
            "images": batch['image'],
            "labels": batch['label']
        }
    
    def __len__(self):
        return len(self.dataloader)


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synchronize_model_weights(model, rank, pp_group):
    """Broadcast model weights from rank 0 to all ranks."""
    if rank == 0:
        print("Synchronizing model weights across all ranks...")
    
    for param in model.parameters():
        dist.broadcast(param.data, src=0, group=pp_group)
    
    dist.barrier(group=pp_group)
    if rank == 0:
        print("✓ Model weights synchronized\n")


def train_epoch(pipeline_trainer, pipeline_loader, tensor_shapes, device, dtype, rank, pp_size, epoch, schedule):
    """Train one epoch with accuracy tracking."""
    pipeline_trainer.model.train()
    
    num_batches = len(pipeline_loader) // pipeline_loader.grad_acc_steps
    total_loss = 0.0
    total_acc = 0.0
    
    # Progress bar only on last rank
    if rank == pp_size - 1:
        pbar = tqdm(range(num_batches), desc=f"Training Epoch {epoch+1}")
    
    for step in range(num_batches):
        # Choose training schedule
        if schedule == "afab":
            loss, acc = pipeline_trainer.train_step_afab(
                pipeline_loader, tensor_shapes, device, dtype
            )
        else:  # 1f1b
            loss, acc = pipeline_trainer.train_step_1f1b(
                pipeline_loader, tensor_shapes, device, dtype
            )
        
        # Accumulate metrics (only on last rank)
        if rank == pp_size - 1:
            if loss is not None:
                total_loss += loss
                total_acc += acc
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Acc': f'{acc:.2f}%'
            })
            pbar.update(1)
    
    if rank == pp_size - 1:
        pbar.close()
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        return avg_loss, avg_acc
    
    return None, None


def validate(pipeline_trainer, val_loader, tensor_shapes, device, dtype, rank, pp_size):
    """Validate the model with accuracy tracking."""
    pipeline_trainer.model.eval()
    
    val_loss, val_acc = pipeline_trainer.evaluate(
        val_loader, 
        tensor_shapes, 
        device, 
        dtype
    )
    
    return val_loss, val_acc


def train_model(config, pgm):
    """Main training function with accuracy tracking."""
    rank = pgm.get_pp_rank()
    pp_size = pgm.get_pp_world_size()
    pp_group = pgm.get_pp_group()
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print("\n" + "="*60)
        print("PIPELINE PARALLEL TRAINING")
        print("="*60)
        print(f"Pipeline stages: {pp_size}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Gradient accumulation steps: {config['grad_acc_steps']}")
        print(f"Schedule: {config['schedule'].upper()}")
        print("="*60 + "\n")
    
    # Set seeds
    set_seed(42)
    
    # Load datasets
    train_dataset = CustomDataset(
        config['dataset_path'],
        split='train',
        transform=mnist_transform
    )
    val_dataset = CustomDataset(
        config['dataset_path'],
        split='test',
        transform=mnist_transform
    )
    
    # Create dataloaders
    def worker_init_fn(worker_id):
        set_seed(42 + worker_id)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn,
        persistent_workers=True if config['num_workers'] > 0 else False,
        generator=torch.Generator().manual_seed(42),
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True
    )
    
    # Create model
    model = Model(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        hidden_dim=config['hidden_dim'],
        in_channels=config['in_channels'],
        n_heads=config['n_heads'],
        depth=config['depth']
    ).to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total model parameters: {total_params:,}\n")
    
    # Synchronize weights
    synchronize_model_weights(model, rank, pp_group)
    
    # Create pipeline wrapper
    pp_model = PipelineParallelWrapper(model, pgm).to(device)
    
    # Create optimizer and criterion
    optimizer = optim.Adam(pp_model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Create pipeline trainer
    pipeline_trainer = PipelineTrainer(
        pp_model,
        pp_group,
        criterion,
        device,
        optimizer=optimizer,
        max_grad_norm=config['max_grad_norm']
    )
    
    # Create pipeline dataloader
    pipeline_train_loader = PipelineDataLoader(train_loader, config['grad_acc_steps'])
    
    # Calculate tensor shapes for communication
    num_patches = (config['img_size'] // config['patch_size']) ** 2
    tensor_shapes = (config['batch_size'], num_patches + 1, config['hidden_dim'])
    dtype = torch.float32
    
    # Training loop
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(config['num_epochs']):
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            pipeline_trainer,
            pipeline_train_loader,
            tensor_shapes,
            device,
            dtype,
            rank,
            pp_size,
            epoch,
            config['schedule']
        )
        
        # Validate
        val_loss, val_acc = validate(
            pipeline_trainer,
            val_loader,
            tensor_shapes,
            device,
            dtype,
            rank,
            pp_size
        )
        
        # Print metrics and track best model
        if rank == pp_size - 1:
            # Store metrics
            if train_loss is not None:
                train_losses.append(train_loss)
                train_accs.append(train_acc)
            
            if val_loss is not None:
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            
            # Print metrics
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Check for improvement
            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                print(f"  🎉 New Best Validation Accuracy!")
            else:
                epochs_without_improvement += 1
            
            print(f"  Best Val Acc: {best_val_acc:.2f}%")
            print(f"  Patience: {config['patience'] - epochs_without_improvement}/{config['patience']}")
            print(f"{'='*50}")
            
            # Early stopping
            if epochs_without_improvement >= config['patience']:
                print(f"\n⚠ Early stopping triggered")
                break
        
        dist.barrier(group=pp_group)
    
    # Print final results
    if rank == pp_size - 1:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Final Train Accuracy: {train_accs[-1]:.2f}%")
        print(f"Final Val Accuracy: {val_accs[-1]:.2f}%")
        print(f"{'='*60}\n")
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc
        }
    
    return None



def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    
    global_rank = dist.get_rank()
    torch.cuda.set_device(global_rank)
    
    # Create process group manager
    pp_size = int(os.environ.get("PP_SIZE", 4))
    pgm = ProcessGroupManager(pp_size=pp_size, tp_size=1)
    
    # Configuration
    config = {
        'dataset_path': os.environ.get('DATASET_PATH', '/mnt/dataset/mnist/'),
        'batch_size': 8,
        'num_workers': 2,
        'img_size': 28,
        'patch_size': 4,
        'hidden_dim': 64,
        'in_channels': 1,
        'n_heads': 4,
        'depth': 8,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'grad_acc_steps': 4,
        'max_grad_norm': 1.0,
        'patience': 5,
        'schedule': os.environ.get('SCHEDULE', '1f1b'),
    }
    
    # Train
    start_time = time.time()
    train_model(config, pgm)
    
    if pgm.get_pp_rank() == 0:
        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed//60:.0f}m {elapsed%60:.0f}s")
    
    # Cleanup
    dist.barrier(group=pgm.get_pp_group())
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
