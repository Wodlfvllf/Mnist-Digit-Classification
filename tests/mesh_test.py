import torch
import torch.distributed as dist
from typing import Dict, List
import sys
from QuintNet.src import init_mesh, MeshGenerator

def test_mesh_shape_and_ranks(mesh: MeshGenerator):
    """
    Test 1: Verify mesh tensor has correct shape and contains all ranks.
    
    Validates:
    - Mesh shape matches mesh_dim
    - All ranks from 0 to world_size-1 are present
    - No duplicate ranks
    """
    print(f"\n{'='*60}")
    print(f"TEST 1: Mesh Shape and Rank Uniqueness")
    print(f"{'='*60}")
    
    world_size = dist.get_world_size()
    my_rank = dist.get_rank()
    
    # Check mesh shape
    expected_shape = mesh.mesh_dim
    actual_shape = tuple(mesh.mesh.shape)
    
    assert actual_shape == expected_shape, \
        f"Rank {my_rank}: Mesh shape mismatch! Expected {expected_shape}, got {actual_shape}"
    
    # Check all ranks are present
    mesh_ranks = set(mesh.mesh.flatten().tolist())
    expected_ranks = set(range(world_size))
    
    assert mesh_ranks == expected_ranks, \
        f"Rank {my_rank}: Missing or extra ranks! Expected {expected_ranks}, got {mesh_ranks}"
    
    # Check no duplicates
    assert len(mesh_ranks) == world_size, \
        f"Rank {my_rank}: Duplicate ranks detected!"
    
    if my_rank == 0:
        print(f"✓ Mesh shape: {actual_shape}")
        print(f"✓ All {world_size} ranks present: {sorted(mesh_ranks)}")
        print(f"✓ No duplicate ranks")


def test_process_group_sizes(mesh: MeshGenerator):
    """
    Test 2: Verify each process group has the correct size.
    
    Validates:
    - DP group size = dp_size
    - PP group size = pp_size
    - TP group size = tp_size
    """
    print(f"\n{'='*60}")
    print(f"TEST 2: Process Group Sizes")
    print(f"{'='*60}")
    
    my_rank = dist.get_rank()
    
    expected_sizes = {
        'dp': mesh.dp_size,
        'pp': mesh.pp_size,
        'tp': mesh.tp_size
    }
    
    for dim_name, expected_size in expected_sizes.items():
        group = mesh.get_group(dim_name)
        actual_size = dist.get_world_size(group=group)
        
        assert actual_size == expected_size, \
            f"Rank {my_rank}: {dim_name.upper()} group size mismatch! " \
            f"Expected {expected_size}, got {actual_size}"
    
    if my_rank == 0:
        print(f"✓ DP group size: {expected_sizes['dp']}")
        print(f"✓ PP group size: {expected_sizes['pp']}")
        print(f"✓ TP group size: {expected_sizes['tp']}")


def test_coordinates_consistency(mesh: MeshGenerator):
    """
    Test 3: Verify coordinate calculation methods are consistent.
    
    Validates:
    - get_coordinates() matches get_coordinates_tensor_search()
    - Coordinates are within valid bounds
    """
    print(f"\n{'='*60}")
    print(f"TEST 3: Coordinate Calculation Consistency")
    print(f"{'='*60}")
    
    world_size = dist.get_world_size()
    my_rank = dist.get_rank()
    
    for rank in range(world_size):
        coords_math = mesh.get_coordinates(rank)
        coords_tensor = mesh.get_coordinates_tensor_search(rank)
        
        assert coords_math == coords_tensor, \
            f"Rank {my_rank}: Coordinate mismatch for rank {rank}! " \
            f"Math: {coords_math}, Tensor: {coords_tensor}"
        
        # Verify coordinates are within bounds
        dp_coord, tp_coord, pp_coord = coords_math
        assert 0 <= dp_coord < mesh.dp_size, \
            f"Rank {my_rank}: DP coordinate {dp_coord} out of bounds [0, {mesh.dp_size})"
        assert 0 <= tp_coord < mesh.tp_size, \
            f"Rank {my_rank}: TP coordinate {tp_coord} out of bounds [0, {mesh.tp_size})"
        assert 0 <= pp_coord < mesh.pp_size, \
            f"Rank {my_rank}: PP coordinate {pp_coord} out of bounds [0, {mesh.pp_size})"
    
    if my_rank == 0:
        print(f"✓ Coordinate methods consistent for all {world_size} ranks")
        print(f"✓ All coordinates within valid bounds")


def test_group_membership(mesh: MeshGenerator):
    """
    Test 4: Verify correct group membership for each rank.
    
    Validates:
    - Each rank belongs to exactly one group per dimension
    - Group members share correct coordinates
    """
    print(f"\n{'='*60}")
    print(f"TEST 4: Group Membership Validation")
    print(f"{'='*60}")
    
    world_size = dist.get_world_size()
    my_rank = dist.get_rank()
    
    # Get my coordinates
    my_coords = mesh.get_coordinates_tensor_search(my_rank)
    dp_coord, pp_coord, tp_coord = my_coords
    
    # Test DP group: all ranks should share same pp and tp
    dp_group = mesh.get_group('dp')
    dp_ranks = []
    for rank in range(world_size):
        coords = mesh.get_coordinates_tensor_search(rank)
        if coords[1] == pp_coord and coords[2] == tp_coord:  # same pp, tp
            dp_ranks.append(rank)
    
    # Verify I'm in this DP group
    assert my_rank in dp_ranks, \
        f"Rank {my_rank}: Not in expected DP group {dp_ranks}!"
    assert len(dp_ranks) == mesh.dp_size, \
        f"Rank {my_rank}: DP group size mismatch! Expected {mesh.dp_size}, got {len(dp_ranks)}"
    
    # Test PP group: all ranks should share same dp and tp
    pp_group = mesh.get_group('pp')
    pp_ranks = []
    for rank in range(world_size):
        coords = mesh.get_coordinates_tensor_search(rank)
        if coords[0] == dp_coord and coords[2] == tp_coord:  # same dp, tp
            pp_ranks.append(rank)
    
    assert my_rank in pp_ranks, \
        f"Rank {my_rank}: Not in expected PP group {pp_ranks}!"
    assert len(pp_ranks) == mesh.pp_size, \
        f"Rank {my_rank}: PP group size mismatch! Expected {mesh.pp_size}, got {len(pp_ranks)}"
    
    # Test TP group: all ranks should share same dp and pp
    tp_group = mesh.get_group('tp')
    tp_ranks = []
    for rank in range(world_size):
        coords = mesh.get_coordinates_tensor_search(rank)
        if coords[0] == dp_coord and coords[1] == pp_coord:  # same dp, pp
            tp_ranks.append(rank)
    
    assert my_rank in tp_ranks, \
        f"Rank {my_rank}: Not in expected TP group {tp_ranks}!"
    assert len(tp_ranks) == mesh.tp_size, \
        f"Rank {my_rank}: TP group size mismatch! Expected {mesh.tp_size}, got {len(tp_ranks)}"
    
    if my_rank == 0:
        print(f"✓ All ranks have correct group memberships")
        print(f"✓ Group members share correct coordinates")


def test_group_communication(mesh: MeshGenerator):
    """
    Test 5: Verify actual communication works within each group.
    
    Validates:
    - all_reduce works correctly in each dimension
    - Communication is isolated to correct group members
    """
    print(f"\n{'='*60}")
    print(f"TEST 5: Group Communication Test")
    print(f"{'='*60}")
    
    my_rank = dist.get_rank()
    
    for dim_name in ['dp', 'pp', 'tp']:
        group = mesh.get_group(dim_name)
        group_size = dist.get_world_size(group=group)
        
        # Create a tensor with this rank's value
        tensor = torch.tensor([my_rank], dtype=torch.float32).cuda()
        
        # Perform all_reduce (sum)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        
        # Calculate expected sum based on group membership
        group_ranks = []
        my_coords = mesh.get_coordinates_tensor_search(my_rank)
        
        for rank in range(dist.get_world_size()):
            coords = mesh.get_coordinates_tensor_search(rank)
            
            # Check if rank is in same group
            if dim_name == 'dp':
                if coords[1] == my_coords[1] and coords[2] == my_coords[2]:
                    group_ranks.append(rank)
            elif dim_name == 'pp':
                if coords[0] == my_coords[0] and coords[2] == my_coords[2]:
                    group_ranks.append(rank)
            else:  # tp
                if coords[0] == my_coords[0] and coords[1] == my_coords[1]:
                    group_ranks.append(rank)
        
        expected_sum = sum(group_ranks)
        actual_sum = tensor.item()
        
        assert abs(actual_sum - expected_sum) < 1e-5, \
            f"Rank {my_rank}: {dim_name.upper()} group communication failed! " \
            f"Expected sum {expected_sum}, got {actual_sum}"
    
    if my_rank == 0:
        print(f"✓ DP group communication works")
        print(f"✓ PP group communication works")
        print(f"✓ TP group communication works")


def test_rank_to_coordinate_bijection(mesh: MeshGenerator):
    """
    Test 6: Verify one-to-one mapping between ranks and coordinates.
    
    Validates:
    - Each rank maps to unique coordinates
    - Each coordinate maps back to correct rank
    """
    print(f"\n{'='*60}")
    print(f"TEST 6: Rank-Coordinate Bijection")
    print(f"{'='*60}")
    
    world_size = dist.get_world_size()
    my_rank = dist.get_rank()
    
    seen_coords = set()
    
    for rank in range(world_size):
        coords = tuple(mesh.get_coordinates_tensor_search(rank))
        
        # Check uniqueness
        assert coords not in seen_coords, \
            f"Rank {my_rank}: Duplicate coordinates {coords} for rank {rank}!"
        seen_coords.add(coords)
        
        # Verify reverse mapping
        dp_coord, pp_coord, tp_coord = coords
        rank_from_mesh = mesh.mesh[dp_coord, pp_coord, tp_coord].item()
        
        assert rank_from_mesh == rank, \
            f"Rank {my_rank}: Coordinate {coords} maps to rank {rank_from_mesh}, " \
            f"expected {rank}!"
    
    if my_rank == 0:
        print(f"✓ All {world_size} ranks have unique coordinates")
        print(f"✓ Reverse mapping (coordinates → rank) is correct")


def test_cross_dimension_independence(mesh: MeshGenerator):
    """
    Test 7: Verify process groups are independent across dimensions.
    
    Validates:
    - DP, PP, TP groups don't overlap (except for the rank itself)
    - Each dimension partitions the world correctly
    """
    print(f"\n{'='*60}")
    print(f"TEST 7: Cross-Dimension Independence")
    print(f"{'='*60}")
    
    world_size = dist.get_world_size()
    my_rank = dist.get_rank()
    
    # Build all groups for each dimension
    all_groups = {'dp': [], 'pp': [], 'tp': []}
    
    for rank in range(world_size):
        coords = mesh.get_coordinates_tensor_search(rank)
        dp_coord, pp_coord, tp_coord = coords
        
        # DP groups
        dp_group_id = (pp_coord, tp_coord)
        if dp_group_id not in [g[0] for g in all_groups['dp']]:
            all_groups['dp'].append((dp_group_id, []))
        for g in all_groups['dp']:
            if g[0] == dp_group_id:
                g[1].append(rank)
        
        # PP groups
        pp_group_id = (dp_coord, tp_coord)
        if pp_group_id not in [g[0] for g in all_groups['pp']]:
            all_groups['pp'].append((pp_group_id, []))
        for g in all_groups['pp']:
            if g[0] == pp_group_id:
                g[1].append(rank)
        
        # TP groups
        tp_group_id = (dp_coord, pp_coord)
        if tp_group_id not in [g[0] for g in all_groups['tp']]:
            all_groups['tp'].append((tp_group_id, []))
        for g in all_groups['tp']:
            if g[0] == tp_group_id:
                g[1].append(rank)
    
    # Verify each dimension partitions the world
    for dim_name, groups in all_groups.items():
        all_ranks_in_dim = set()
        for _, group_ranks in groups:
            # Check no overlaps within dimension
            assert len(set(group_ranks) & all_ranks_in_dim) == 0, \
                f"Rank {my_rank}: Overlapping groups in {dim_name} dimension!"
            all_ranks_in_dim.update(group_ranks)
        
        # Check all ranks are covered
        assert all_ranks_in_dim == set(range(world_size)), \
            f"Rank {my_rank}: {dim_name} dimension doesn't cover all ranks!"
    
    if my_rank == 0:
        print(f"✓ DP dimension partitions world correctly ({len(all_groups['dp'])} groups)")
        print(f"✓ PP dimension partitions world correctly ({len(all_groups['pp'])} groups)")
        print(f"✓ TP dimension partitions world correctly ({len(all_groups['tp'])} groups)")


def run_all_tests(mesh: MeshGenerator):
    """
    Run all mesh initialization tests.
    
    Usage:
        mesh = init_mesh(mesh_dim=(2, 2, 2), mesh_name=('dp', 'pp', 'tp'))
        run_all_tests(mesh)
    """
    my_rank = dist.get_rank()
    
    try:
        test_mesh_shape_and_ranks(mesh)
        dist.barrier()
        
        test_process_group_sizes(mesh)
        dist.barrier()
        
        test_coordinates_consistency(mesh)
        dist.barrier()
        
        test_group_membership(mesh)
        dist.barrier()
        
        test_group_communication(mesh)
        dist.barrier()
        
        test_rank_to_coordinate_bijection(mesh)
        dist.barrier()
        
        test_cross_dimension_independence(mesh)
        dist.barrier()
        
        if my_rank == 0:
            print(f"\n{'='*60}")
            print(f"🎉 ALL TESTS PASSED!")
            print(f"{'='*60}")
            print(f"Mesh Configuration:")
            print(f"  - DP size: {mesh.dp_size}")
            print(f"  - PP size: {mesh.pp_size}")
            print(f"  - TP size: {mesh.tp_size}")
            print(f"  - Total GPUs: {dist.get_world_size()}")
            print(f"{'='*60}\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED on rank {my_rank}: {e}")
        dist.destroy_process_group()
        sys.exit(1)


# # Example usage in your training script
# if __name__ == "__main__":
#     # Initialize mesh
#     mesh = init_mesh(
#         device_type='cuda',
#         mesh_dim=(2, 2, 2),  # 8 GPUs: 2 DP, 2 PP, 2 TP
#         mesh_name=('dp', 'pp', 'tp')
#     )
    
#     # Run all tests
#     run_all_tests(mesh)
    
#     # Continue with training...
