use foundation::{BlockPos, ChunkCoord};
use std::ops::Range;
use voxels::{
    subchunk_bit, subchunk_index, BlockId, Chunk, AIR_BLOCK, CHUNK_SIZE, SUBCHUNK_COUNT,
    SUBCHUNK_SIZE,
};
use world::VoxelWorld;

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub block_id: BlockId,
}

#[derive(Debug, Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub visible_face_count: u32,
    pub subchunk_visible_mask: u8,
}

impl MeshData {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn has_visible_geometry(&self) -> bool {
        self.subchunk_visible_mask != 0
    }
}

pub struct ChunkMeshInput {
    pub coord: ChunkCoord,
    pub revision: u32,
    pub center: Chunk,
    pub neighbors: [Option<Chunk>; 6],
}

impl ChunkMeshInput {
    pub fn from_world(world: &VoxelWorld, coord: ChunkCoord) -> Option<Self> {
        let center = world.get_chunk(coord)?.clone();
        let revision = center.revision;
        let neighbors = neighbor_chunk_coords(coord)
            .map(|neighbor_coord| world.get_chunk(neighbor_coord).cloned());

        Some(Self {
            coord,
            revision,
            center,
            neighbors,
        })
    }
}

pub fn mesh_chunk(world: &VoxelWorld, coord: ChunkCoord) -> Option<MeshData> {
    let chunk = world.get_chunk(coord)?;

    if chunk.is_empty() {
        return Some(MeshData::default());
    }

    let mut mesh = MeshData::default();
    mesh_block_range(
        chunk,
        coord,
        0..CHUNK_SIZE,
        0..CHUNK_SIZE,
        0..CHUNK_SIZE,
        &mut mesh,
        |chunk, coord, x, y, z, dir| neighbor_block(world, chunk, coord, x, y, z, dir),
    );

    Some(mesh)
}

pub fn mesh_chunk_input(input: &ChunkMeshInput) -> MeshData {
    if input.center.is_empty() {
        return MeshData::default();
    }

    let mut mesh = MeshData::default();
    mesh_block_range(
        &input.center,
        input.coord,
        0..CHUNK_SIZE,
        0..CHUNK_SIZE,
        0..CHUNK_SIZE,
        &mut mesh,
        |chunk, coord, x, y, z, dir| snapshot_neighbor_block(input, chunk, coord, x, y, z, dir),
    );

    mesh
}

pub fn mesh_subchunk(world: &VoxelWorld, coord: ChunkCoord, subchunk: usize) -> Option<MeshData> {
    if subchunk >= SUBCHUNK_COUNT {
        return None;
    }

    let chunk = world.get_chunk(coord)?;

    if chunk.is_subchunk_empty(subchunk) {
        return Some(MeshData::default());
    }

    let (x_range, y_range, z_range) = subchunk_ranges(subchunk);
    let mut mesh = MeshData::default();

    mesh_block_range(
        chunk,
        coord,
        x_range,
        y_range,
        z_range,
        &mut mesh,
        |chunk, coord, x, y, z, dir| neighbor_block(world, chunk, coord, x, y, z, dir),
    );

    Some(mesh)
}

pub fn mesh_dirty_subchunks(
    world: &VoxelWorld,
    coord: ChunkCoord,
) -> Option<Vec<(usize, MeshData)>> {
    let chunk = world.get_chunk(coord)?;
    let dirty_mask = chunk.subchunk_dirty_mask;
    let mut meshes = Vec::new();

    for subchunk in 0..SUBCHUNK_COUNT {
        if dirty_mask & subchunk_bit(subchunk) != 0 {
            let mesh = mesh_subchunk(world, coord, subchunk)
                .expect("subchunk came from the valid subchunk range");
            meshes.push((subchunk, mesh));
        }
    }

    Some(meshes)
}

const ALL_DIRECTIONS: [Direction; 6] = [
    Direction::PosX,
    Direction::NegX,
    Direction::PosY,
    Direction::NegY,
    Direction::PosZ,
    Direction::NegZ,
];

fn mesh_block_range(
    chunk: &Chunk,
    coord: ChunkCoord,
    x_range: Range<usize>,
    y_range: Range<usize>,
    z_range: Range<usize>,
    mesh: &mut MeshData,
    neighbor_block: impl Fn(&Chunk, ChunkCoord, usize, usize, usize, Direction) -> BlockId,
) {
    for y in y_range {
        for z in z_range.clone() {
            for x in x_range.clone() {
                let block = chunk.get_block(x, y, z);

                if block == AIR_BLOCK {
                    continue;
                }

                let world_x = coord.x * CHUNK_SIZE as i32 + x as i32;
                let world_y = coord.y * CHUNK_SIZE as i32 + y as i32;
                let world_z = coord.z * CHUNK_SIZE as i32 + z as i32;
                let world_pos = BlockPos::new(world_x, world_y, world_z);

                for dir in ALL_DIRECTIONS {
                    let neighbor = neighbor_block(chunk, coord, x, y, z, dir);

                    if neighbor == AIR_BLOCK {
                        let subchunk = subchunk_index(x, y, z);
                        mesh.subchunk_visible_mask |= subchunk_bit(subchunk);
                        add_face(mesh, world_pos, dir, block);
                    }
                }
            }
        }
    }
}

fn neighbor_chunk_coords(coord: ChunkCoord) -> [ChunkCoord; 6] {
    [
        coord.offset(1, 0, 0),
        coord.offset(-1, 0, 0),
        coord.offset(0, 1, 0),
        coord.offset(0, -1, 0),
        coord.offset(0, 0, 1),
        coord.offset(0, 0, -1),
    ]
}

fn subchunk_ranges(subchunk: usize) -> (Range<usize>, Range<usize>, Range<usize>) {
    (
        subchunk_axis_range(subchunk & 1 != 0),
        subchunk_axis_range(subchunk & 2 != 0),
        subchunk_axis_range(subchunk & 4 != 0),
    )
}

fn subchunk_axis_range(upper_half: bool) -> Range<usize> {
    if upper_half {
        SUBCHUNK_SIZE..CHUNK_SIZE
    } else {
        0..SUBCHUNK_SIZE
    }
}

fn neighbor_block(
    world: &VoxelWorld,
    chunk: &Chunk,
    coord: ChunkCoord,
    x: usize,
    y: usize,
    z: usize,
    dir: Direction,
) -> BlockId {
    match dir {
        Direction::PosX if x + 1 < CHUNK_SIZE => chunk.get_block(x + 1, y, z),
        Direction::NegX if x > 0 => chunk.get_block(x - 1, y, z),
        Direction::PosY if y + 1 < CHUNK_SIZE => chunk.get_block(x, y + 1, z),
        Direction::NegY if y > 0 => chunk.get_block(x, y - 1, z),
        Direction::PosZ if z + 1 < CHUNK_SIZE => chunk.get_block(x, y, z + 1),
        Direction::NegZ if z > 0 => chunk.get_block(x, y, z - 1),
        Direction::PosX => neighbor_chunk_block(world, coord.offset(1, 0, 0), 0, y, z),
        Direction::NegX => {
            neighbor_chunk_block(world, coord.offset(-1, 0, 0), CHUNK_SIZE - 1, y, z)
        }
        Direction::PosY => neighbor_chunk_block(world, coord.offset(0, 1, 0), x, 0, z),
        Direction::NegY => {
            neighbor_chunk_block(world, coord.offset(0, -1, 0), x, CHUNK_SIZE - 1, z)
        }
        Direction::PosZ => neighbor_chunk_block(world, coord.offset(0, 0, 1), x, y, 0),
        Direction::NegZ => {
            neighbor_chunk_block(world, coord.offset(0, 0, -1), x, y, CHUNK_SIZE - 1)
        }
    }
}

fn snapshot_neighbor_block(
    input: &ChunkMeshInput,
    chunk: &Chunk,
    _coord: ChunkCoord,
    x: usize,
    y: usize,
    z: usize,
    dir: Direction,
) -> BlockId {
    match dir {
        Direction::PosX if x + 1 < CHUNK_SIZE => chunk.get_block(x + 1, y, z),
        Direction::NegX if x > 0 => chunk.get_block(x - 1, y, z),
        Direction::PosY if y + 1 < CHUNK_SIZE => chunk.get_block(x, y + 1, z),
        Direction::NegY if y > 0 => chunk.get_block(x, y - 1, z),
        Direction::PosZ if z + 1 < CHUNK_SIZE => chunk.get_block(x, y, z + 1),
        Direction::NegZ if z > 0 => chunk.get_block(x, y, z - 1),
        Direction::PosX => snapshot_neighbor_chunk_block(input, 0, 0, y, z),
        Direction::NegX => snapshot_neighbor_chunk_block(input, 1, CHUNK_SIZE - 1, y, z),
        Direction::PosY => snapshot_neighbor_chunk_block(input, 2, x, 0, z),
        Direction::NegY => snapshot_neighbor_chunk_block(input, 3, x, CHUNK_SIZE - 1, z),
        Direction::PosZ => snapshot_neighbor_chunk_block(input, 4, x, y, 0),
        Direction::NegZ => snapshot_neighbor_chunk_block(input, 5, x, y, CHUNK_SIZE - 1),
    }
}

fn snapshot_neighbor_chunk_block(
    input: &ChunkMeshInput,
    neighbor_index: usize,
    x: usize,
    y: usize,
    z: usize,
) -> BlockId {
    input.neighbors[neighbor_index]
        .as_ref()
        .map(|chunk| chunk.get_block(x, y, z))
        .unwrap_or(AIR_BLOCK)
}

fn neighbor_chunk_block(
    world: &VoxelWorld,
    coord: ChunkCoord,
    x: usize,
    y: usize,
    z: usize,
) -> BlockId {
    world
        .get_chunk(coord)
        .map(|chunk| chunk.get_block(x, y, z))
        .unwrap_or(AIR_BLOCK)
}

fn add_face(mesh: &mut MeshData, pos: BlockPos, dir: Direction, block_id: BlockId) {
    let base_index = mesh.vertices.len() as u32;

    let x = pos.x as f32;
    let y = pos.y as f32;
    let z = pos.z as f32;

    let p000 = [x, y, z];
    let p100 = [x + 1.0, y, z];
    let p010 = [x, y + 1.0, z];
    let p110 = [x + 1.0, y + 1.0, z];
    let p001 = [x, y, z + 1.0];
    let p101 = [x + 1.0, y, z + 1.0];
    let p011 = [x, y + 1.0, z + 1.0];
    let p111 = [x + 1.0, y + 1.0, z + 1.0];

    let (normal, positions) = match dir {
        Direction::PosX => ([1.0, 0.0, 0.0], [p100, p101, p111, p110]),
        Direction::NegX => ([-1.0, 0.0, 0.0], [p001, p000, p010, p011]),
        Direction::PosY => ([0.0, 1.0, 0.0], [p010, p110, p111, p011]),
        Direction::NegY => ([0.0, -1.0, 0.0], [p001, p101, p100, p000]),
        Direction::PosZ => ([0.0, 0.0, 1.0], [p101, p001, p011, p111]),
        Direction::NegZ => ([0.0, 0.0, -1.0], [p000, p100, p110, p010]),
    };

    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    for i in 0..4 {
        mesh.vertices.push(Vertex {
            position: positions[i],
            normal,
            uv: uvs[i],
            block_id,
        });
    }

    mesh.indices.extend_from_slice(&[
        base_index,
        base_index + 2,
        base_index + 1,
        base_index,
        base_index + 3,
        base_index + 2,
    ]);

    mesh.visible_face_count += 1;
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxels::{subchunk_index, Chunk, STONE_BLOCK};

    fn world_with_empty_chunk(coord: ChunkCoord) -> VoxelWorld {
        let mut world = VoxelWorld::new(0);
        world.chunks.insert(coord, Chunk::new_empty(coord));
        world
    }

    #[test]
    fn missing_chunk_has_no_mesh() {
        let world = VoxelWorld::new(0);

        assert!(mesh_chunk(&world, ChunkCoord::new(0, 0, 0)).is_none());
        assert!(mesh_subchunk(&world, ChunkCoord::new(0, 0, 0), 0).is_none());
    }

    #[test]
    fn single_block_meshes_all_six_faces() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);

        world.set_block(BlockPos::new(0, 0, 0), STONE_BLOCK);

        let mesh = mesh_chunk(&world, coord).expect("chunk exists");

        assert_eq!(mesh.visible_face_count, 6);
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
        assert_eq!(mesh.triangle_count(), 12);
        assert_eq!(mesh.subchunk_visible_mask, subchunk_bit(0));
        assert!(mesh.has_visible_geometry());
    }

    #[test]
    fn adjacent_blocks_cull_the_shared_face() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);

        world.set_block(BlockPos::new(0, 0, 0), STONE_BLOCK);
        world.set_block(BlockPos::new(1, 0, 0), STONE_BLOCK);

        let mesh = mesh_chunk(&world, coord).expect("chunk exists");

        assert_eq!(mesh.visible_face_count, 10);
        assert_eq!(mesh.triangle_count(), 20);
        assert_eq!(mesh.subchunk_visible_mask, subchunk_bit(0));
    }

    #[test]
    fn subchunk_mesh_limits_output_to_requested_subchunk() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);

        world.set_block(BlockPos::new(17, 0, 0), STONE_BLOCK);

        let empty_mesh = mesh_subchunk(&world, coord, 0).expect("chunk exists");
        let filled_mesh = mesh_subchunk(&world, coord, 1).expect("chunk exists");

        assert_eq!(empty_mesh.visible_face_count, 0);
        assert_eq!(empty_mesh.subchunk_visible_mask, 0);
        assert!(!empty_mesh.has_visible_geometry());
        assert_eq!(filled_mesh.visible_face_count, 6);
        assert_eq!(filled_mesh.subchunk_visible_mask, subchunk_bit(1));
    }

    #[test]
    fn dirty_subchunk_meshes_follow_chunk_dirty_mask() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);

        world.set_block(BlockPos::new(17, 0, 0), STONE_BLOCK);

        let meshes = mesh_dirty_subchunks(&world, coord).expect("chunk exists");

        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].0, subchunk_index(17, 0, 0));
        assert_eq!(meshes[0].1.visible_face_count, 6);
        assert_eq!(meshes[0].1.subchunk_visible_mask, subchunk_bit(1));
    }

    #[test]
    fn subchunk_boundary_meshes_cull_cross_subchunk_faces() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);

        world.set_block(BlockPos::new(15, 0, 0), STONE_BLOCK);
        world.set_block(BlockPos::new(16, 0, 0), STONE_BLOCK);

        let left_mesh =
            mesh_subchunk(&world, coord, subchunk_index(15, 0, 0)).expect("chunk exists");
        let right_mesh =
            mesh_subchunk(&world, coord, subchunk_index(16, 0, 0)).expect("chunk exists");

        assert_eq!(left_mesh.visible_face_count, 5);
        assert_eq!(right_mesh.visible_face_count, 5);
        assert_eq!(left_mesh.subchunk_visible_mask, subchunk_bit(0));
        assert_eq!(right_mesh.subchunk_visible_mask, subchunk_bit(1));
    }

    #[test]
    fn chunk_boundary_meshes_cull_faces_against_loaded_neighbor_chunks() {
        let coord = ChunkCoord::new(0, 0, 0);
        let neighbor_coord = ChunkCoord::new(1, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world
            .chunks
            .insert(neighbor_coord, Chunk::new_empty(neighbor_coord));

        world.set_block(BlockPos::new(31, 0, 0), STONE_BLOCK);
        world.set_block(BlockPos::new(32, 0, 0), STONE_BLOCK);

        let origin_mesh = mesh_chunk(&world, coord).expect("origin chunk exists");
        let neighbor_mesh = mesh_chunk(&world, neighbor_coord).expect("neighbor chunk exists");

        assert_eq!(origin_mesh.visible_face_count, 5);
        assert_eq!(neighbor_mesh.visible_face_count, 5);
        assert_eq!(origin_mesh.subchunk_visible_mask, subchunk_bit(1));
        assert_eq!(neighbor_mesh.subchunk_visible_mask, subchunk_bit(0));
    }

    #[test]
    fn snapshot_mesh_matches_live_world_mesh() {
        let coord = ChunkCoord::new(0, 0, 0);
        let neighbor_coord = ChunkCoord::new(1, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world
            .chunks
            .insert(neighbor_coord, Chunk::new_empty(neighbor_coord));

        world.set_block(BlockPos::new(31, 0, 0), STONE_BLOCK);
        world.set_block(BlockPos::new(32, 0, 0), STONE_BLOCK);

        let input = ChunkMeshInput::from_world(&world, coord).expect("snapshot exists");
        let live_mesh = mesh_chunk(&world, coord).expect("live mesh exists");
        let snapshot_mesh = mesh_chunk_input(&input);

        assert_eq!(input.revision, world.get_chunk(coord).unwrap().revision);
        assert_eq!(
            snapshot_mesh.visible_face_count,
            live_mesh.visible_face_count
        );
        assert_eq!(snapshot_mesh.indices.len(), live_mesh.indices.len());
        assert_eq!(
            snapshot_mesh.subchunk_visible_mask,
            live_mesh.subchunk_visible_mask
        );
    }

    #[test]
    fn generated_triangles_are_wound_toward_their_face_normals() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);

        world.set_block(BlockPos::new(0, 0, 0), STONE_BLOCK);

        let mesh = mesh_chunk(&world, coord).expect("chunk exists");

        for triangle in mesh.indices.chunks_exact(3) {
            let a = mesh.vertices[triangle[0] as usize];
            let b = mesh.vertices[triangle[1] as usize];
            let c = mesh.vertices[triangle[2] as usize];
            let normal = triangle_normal(a.position, b.position, c.position);

            assert!(
                dot(normal, a.normal) > 0.0,
                "triangle normal {:?} should face vertex normal {:?}",
                normal,
                a.normal
            );
        }
    }

    fn triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
        cross(subtract(b, a), subtract(c, a))
    }

    fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
}
