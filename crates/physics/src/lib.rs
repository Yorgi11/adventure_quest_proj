use foundation::{BlockPos, ChunkCoord};
use voxels::{world_to_chunk_coord, world_to_local_block, BlockId, AIR_BLOCK};
use world::VoxelWorld;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelRayHit {
    pub world_block: BlockPos,
    pub chunk_coord: ChunkCoord,
    pub local_block: (usize, usize, usize),
    pub block_id: BlockId,
    pub face_normal: [i32; 3],
    pub distance: f32,
}

impl VoxelRayHit {
    pub fn placement_block(&self) -> BlockPos {
        BlockPos::new(
            self.world_block.x + self.face_normal[0],
            self.world_block.y + self.face_normal[1],
            self.world_block.z + self.face_normal[2],
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct AxisState {
    step: i32,
    t_max: f32,
    t_delta: f32,
}

pub fn raycast_blocks(
    world: &VoxelWorld,
    origin: [f32; 3],
    direction: [f32; 3],
    max_distance: f32,
) -> Option<VoxelRayHit> {
    if max_distance < 0.0 {
        return None;
    }

    let direction = normalize(direction)?;
    let mut block = BlockPos::new(
        origin[0].floor() as i32,
        origin[1].floor() as i32,
        origin[2].floor() as i32,
    );

    let mut x_axis = axis_state(origin[0], direction[0]);
    let mut y_axis = axis_state(origin[1], direction[1]);
    let mut z_axis = axis_state(origin[2], direction[2]);

    let mut distance = 0.0;
    let mut face_normal = [0, 0, 0];

    while distance <= max_distance {
        let block_id = world.get_block(block);

        if block_id != AIR_BLOCK {
            let chunk_coord = world_to_chunk_coord(block.x, block.y, block.z);
            let local_block = world_to_local_block(block.x, block.y, block.z);

            return Some(VoxelRayHit {
                world_block: block,
                chunk_coord,
                local_block,
                block_id,
                face_normal,
                distance,
            });
        }

        let axis = next_axis(x_axis.t_max, y_axis.t_max, z_axis.t_max);
        let next_distance = match axis {
            0 => x_axis.t_max,
            1 => y_axis.t_max,
            _ => z_axis.t_max,
        };

        if next_distance > max_distance {
            break;
        }

        distance = next_distance;

        match axis {
            0 => {
                block.x += x_axis.step;
                face_normal = [-x_axis.step, 0, 0];
                x_axis.t_max += x_axis.t_delta;
            }
            1 => {
                block.y += y_axis.step;
                face_normal = [0, -y_axis.step, 0];
                y_axis.t_max += y_axis.t_delta;
            }
            _ => {
                block.z += z_axis.step;
                face_normal = [0, 0, -z_axis.step];
                z_axis.t_max += z_axis.t_delta;
            }
        }
    }

    None
}

fn normalize(direction: [f32; 3]) -> Option<[f32; 3]> {
    let length_squared =
        direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];

    if length_squared <= f32::EPSILON {
        return None;
    }

    let inv_length = 1.0 / length_squared.sqrt();

    Some([
        direction[0] * inv_length,
        direction[1] * inv_length,
        direction[2] * inv_length,
    ])
}

fn axis_state(origin: f32, direction: f32) -> AxisState {
    if direction > 0.0 {
        let next_boundary = origin.floor() + 1.0;

        AxisState {
            step: 1,
            t_max: (next_boundary - origin) / direction,
            t_delta: 1.0 / direction,
        }
    } else if direction < 0.0 {
        let next_boundary = origin.floor();

        AxisState {
            step: -1,
            t_max: (origin - next_boundary) / -direction,
            t_delta: -1.0 / direction,
        }
    } else {
        AxisState {
            step: 0,
            t_max: f32::INFINITY,
            t_delta: f32::INFINITY,
        }
    }
}

fn next_axis(x: f32, y: f32, z: f32) -> usize {
    if x <= y && x <= z {
        0
    } else if y <= z {
        1
    } else {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use foundation::ChunkCoord;
    use voxels::{Chunk, STONE_BLOCK};

    fn world_with_empty_chunk(coord: ChunkCoord) -> VoxelWorld {
        let mut world = VoxelWorld::new(0);
        world.chunks.insert(coord, Chunk::new_empty(coord));
        world
    }

    #[test]
    fn raycast_hits_first_solid_block() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world.set_block(BlockPos::new(3, 0, 0), STONE_BLOCK);

        let hit = raycast_blocks(&world, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 10.0)
            .expect("ray should hit block");

        assert_eq!(hit.world_block, BlockPos::new(3, 0, 0));
        assert_eq!(hit.local_block, (3, 0, 0));
        assert_eq!(hit.block_id, STONE_BLOCK);
        assert_eq!(hit.face_normal, [-1, 0, 0]);
        assert_eq!(hit.placement_block(), BlockPos::new(2, 0, 0));
        assert!((hit.distance - 2.5).abs() < 0.0001);
    }

    #[test]
    fn raycast_returns_none_when_hit_is_past_max_distance() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world.set_block(BlockPos::new(3, 0, 0), STONE_BLOCK);

        let hit = raycast_blocks(&world, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 2.4);

        assert!(hit.is_none());
    }

    #[test]
    fn raycast_hits_block_containing_origin() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world.set_block(BlockPos::new(0, 0, 0), STONE_BLOCK);

        let hit = raycast_blocks(&world, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 10.0)
            .expect("ray starts inside block");

        assert_eq!(hit.world_block, BlockPos::new(0, 0, 0));
        assert_eq!(hit.face_normal, [0, 0, 0]);
        assert_eq!(hit.distance, 0.0);
    }

    #[test]
    fn raycast_supports_negative_world_blocks() {
        let coord = ChunkCoord::new(-1, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world.set_block(BlockPos::new(-2, 0, 0), STONE_BLOCK);

        let hit = raycast_blocks(&world, [1.5, 0.5, 0.5], [-1.0, 0.0, 0.0], 10.0)
            .expect("ray should hit negative block");

        assert_eq!(hit.world_block, BlockPos::new(-2, 0, 0));
        assert_eq!(hit.chunk_coord, ChunkCoord::new(-1, 0, 0));
        assert_eq!(hit.local_block, (30, 0, 0));
        assert_eq!(hit.face_normal, [1, 0, 0]);
        assert!((hit.distance - 2.5).abs() < 0.0001);
    }

    #[test]
    fn raycast_ignores_zero_length_direction() {
        let world = VoxelWorld::new(0);

        assert!(raycast_blocks(&world, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 10.0).is_none());
    }
}
