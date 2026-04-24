use foundation::{BlockPos, ChunkCoord};
use meshing::mesh_chunk;
use voxels::{AIR_BLOCK, STONE_BLOCK};
use world::VoxelWorld;

fn main() {
    println!("Adventure Quest - Rust Voxel Prototype");

    let mut world = VoxelWorld::new(12345);

    println!("Loading chunks...");
    world.load_chunks_around_origin(1);

    println!("Loaded chunks: {}", world.chunks.len());

    let target_chunk = ChunkCoord::new(0, 0, 0);

    let mesh = mesh_chunk(&world, target_chunk).expect("Target chunk should exist");

    println!("Chunk {:?} mesh:", target_chunk);
    println!("  vertices: {}", mesh.vertices.len());
    println!("  indices: {}", mesh.indices.len());
    println!("  triangles: {}", mesh.triangle_count());
    println!("  visible faces: {}", mesh.visible_face_count);

    println!("Editing block...");

    let edit_pos = BlockPos::new(0, 20, 0);
    world.set_block(edit_pos, AIR_BLOCK);

    let place_pos = BlockPos::new(1, 20, 0);
    world.set_block(place_pos, STONE_BLOCK);

    let edited_mesh = mesh_chunk(&world, target_chunk).expect("Target chunk should exist");

    println!("After edit:");
    println!("  vertices: {}", edited_mesh.vertices.len());
    println!("  indices: {}", edited_mesh.indices.len());
    println!("  triangles: {}", edited_mesh.triangle_count());
    println!("  visible faces: {}", edited_mesh.visible_face_count);

    if let Some(chunk) = world.get_chunk(target_chunk) {
        println!("Chunk revision: {}", chunk.revision);
        println!("Chunk dirty mask: {:08b}", chunk.subchunk_dirty_mask);
    }
}