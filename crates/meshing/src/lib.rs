use foundation::{BlockPos, ChunkCoord};
use voxels::{BlockId, AIR_BLOCK, CHUNK_SIZE};
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
}

impl MeshData {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

pub fn mesh_chunk(world: &VoxelWorld, coord: ChunkCoord) -> Option<MeshData> {
    let chunk = world.get_chunk(coord)?;

    if chunk.is_empty() {
        return Some(MeshData::default());
    }

    let mut mesh = MeshData::default();

    for y in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let block = chunk.get_block(x, y, z);

                if block == AIR_BLOCK {
                    continue;
                }

                let world_x = coord.x * CHUNK_SIZE as i32 + x as i32;
                let world_y = coord.y * CHUNK_SIZE as i32 + y as i32;
                let world_z = coord.z * CHUNK_SIZE as i32 + z as i32;

                let world_pos = BlockPos::new(world_x, world_y, world_z);

                for dir in ALL_DIRECTIONS {
                    let neighbor_pos = offset_pos(world_pos, dir);
                    let neighbor = world.get_block(neighbor_pos);

                    if neighbor == AIR_BLOCK {
                        add_face(&mut mesh, world_pos, dir, block);
                    }
                }
            }
        }
    }

    Some(mesh)
}

const ALL_DIRECTIONS: [Direction; 6] = [
    Direction::PosX,
    Direction::NegX,
    Direction::PosY,
    Direction::NegY,
    Direction::PosZ,
    Direction::NegZ,
];

fn offset_pos(pos: BlockPos, dir: Direction) -> BlockPos {
    match dir {
        Direction::PosX => BlockPos::new(pos.x + 1, pos.y, pos.z),
        Direction::NegX => BlockPos::new(pos.x - 1, pos.y, pos.z),
        Direction::PosY => BlockPos::new(pos.x, pos.y + 1, pos.z),
        Direction::NegY => BlockPos::new(pos.x, pos.y - 1, pos.z),
        Direction::PosZ => BlockPos::new(pos.x, pos.y, pos.z + 1),
        Direction::NegZ => BlockPos::new(pos.x, pos.y, pos.z - 1),
    }
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
        base_index + 1,
        base_index + 2,
        base_index,
        base_index + 2,
        base_index + 3,
    ]);

    mesh.visible_face_count += 1;
}