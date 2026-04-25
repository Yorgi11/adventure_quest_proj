use foundation::{BlockPos, ChunkCoord};
use voxels::{world_to_chunk_coord, world_to_local_block, BlockId, AIR_BLOCK};
use world::VoxelWorld;

const COLLISION_EPSILON: f32 = 0.001;
const MAX_COLLISION_STEP: f32 = 0.45;

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AabbCollisionResult {
    pub center: [f32; 3],
    pub collided: [bool; 3],
    pub on_ground: bool,
}

pub fn move_aabb_through_voxels(
    world: &VoxelWorld,
    center: [f32; 3],
    half_extents: [f32; 3],
    delta: [f32; 3],
) -> AabbCollisionResult {
    let max_delta = delta.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    let steps = (max_delta / MAX_COLLISION_STEP).ceil().max(1.0) as usize;
    let mut center = center;
    let mut step_delta = [
        delta[0] / steps as f32,
        delta[1] / steps as f32,
        delta[2] / steps as f32,
    ];
    let mut collided = [false; 3];
    let mut on_ground = false;

    for _ in 0..steps {
        let result = move_aabb_collision_step(world, center, half_extents, step_delta);

        center = result.center;
        on_ground |= result.on_ground;

        for axis in 0..3 {
            if result.collided[axis] {
                collided[axis] = true;
                step_delta[axis] = 0.0;
            }
        }
    }

    AabbCollisionResult {
        center,
        collided,
        on_ground,
    }
}

fn move_aabb_collision_step(
    world: &VoxelWorld,
    center: [f32; 3],
    half_extents: [f32; 3],
    delta: [f32; 3],
) -> AabbCollisionResult {
    let mut center = center;
    let mut collided = [false; 3];
    let mut on_ground = false;

    for axis in 0..3 {
        if delta[axis] == 0.0 {
            continue;
        }

        center[axis] += delta[axis];

        if let Some(corrected_axis) =
            collision_axis_correction(world, center, half_extents, axis, delta[axis])
        {
            center[axis] = corrected_axis;
            collided[axis] = true;

            if axis == 1 && delta[axis] < 0.0 {
                on_ground = true;
            }
        }
    }

    AabbCollisionResult {
        center,
        collided,
        on_ground,
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

fn collision_axis_correction(
    world: &VoxelWorld,
    center: [f32; 3],
    half_extents: [f32; 3],
    axis: usize,
    delta: f32,
) -> Option<f32> {
    let bounds = aabb_bounds(center, half_extents);
    let min_block = block_floor(bounds.0);
    let max_block = block_floor([
        bounds.1[0] - COLLISION_EPSILON,
        bounds.1[1] - COLLISION_EPSILON,
        bounds.1[2] - COLLISION_EPSILON,
    ]);
    let mut corrected = center[axis];
    let mut hit = false;

    for y in min_block.y..=max_block.y {
        for z in min_block.z..=max_block.z {
            for x in min_block.x..=max_block.x {
                let block = BlockPos::new(x, y, z);

                if world.get_block(block) == AIR_BLOCK {
                    continue;
                }

                if !aabb_intersects_block(bounds, block) {
                    continue;
                }

                hit = true;

                if delta > 0.0 {
                    corrected = corrected.min(block_axis_min(block, axis) - half_extents[axis]);
                } else {
                    corrected = corrected.max(block_axis_max(block, axis) + half_extents[axis]);
                }
            }
        }
    }

    hit.then_some(
        corrected
            + if delta > 0.0 {
                -COLLISION_EPSILON
            } else {
                COLLISION_EPSILON
            },
    )
}

fn aabb_bounds(center: [f32; 3], half_extents: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    (
        [
            center[0] - half_extents[0],
            center[1] - half_extents[1],
            center[2] - half_extents[2],
        ],
        [
            center[0] + half_extents[0],
            center[1] + half_extents[1],
            center[2] + half_extents[2],
        ],
    )
}

fn block_floor(pos: [f32; 3]) -> BlockPos {
    BlockPos::new(
        pos[0].floor() as i32,
        pos[1].floor() as i32,
        pos[2].floor() as i32,
    )
}

fn aabb_intersects_block(bounds: ([f32; 3], [f32; 3]), block: BlockPos) -> bool {
    let block_min = [block.x as f32, block.y as f32, block.z as f32];
    let block_max = [block_min[0] + 1.0, block_min[1] + 1.0, block_min[2] + 1.0];

    bounds.0[0] < block_max[0]
        && bounds.1[0] > block_min[0]
        && bounds.0[1] < block_max[1]
        && bounds.1[1] > block_min[1]
        && bounds.0[2] < block_max[2]
        && bounds.1[2] > block_min[2]
}

fn block_axis_min(block: BlockPos, axis: usize) -> f32 {
    match axis {
        0 => block.x as f32,
        1 => block.y as f32,
        _ => block.z as f32,
    }
}

fn block_axis_max(block: BlockPos, axis: usize) -> f32 {
    block_axis_min(block, axis) + 1.0
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

    #[test]
    fn aabb_moves_freely_without_solid_blocks() {
        let world = world_with_empty_chunk(ChunkCoord::new(0, 0, 0));

        let result =
            move_aabb_through_voxels(&world, [0.5, 2.0, 0.5], [0.3, 0.9, 0.3], [1.0, -0.5, 0.25]);

        assert_vec3_close(result.center, [1.5, 1.5, 0.75]);
        assert_eq!(result.collided, [false, false, false]);
        assert!(!result.on_ground);
    }

    #[test]
    fn aabb_lands_on_solid_floor() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world.set_block(BlockPos::new(0, 0, 0), STONE_BLOCK);

        let result =
            move_aabb_through_voxels(&world, [0.5, 2.5, 0.5], [0.3, 0.9, 0.3], [0.0, -2.0, 0.0]);

        assert!((result.center[1] - 1.901).abs() < 0.0001);
        assert_eq!(result.collided, [false, true, false]);
        assert!(result.on_ground);
    }

    #[test]
    fn aabb_does_not_tunnel_through_floor_on_large_delta() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world.set_block(BlockPos::new(0, 0, 0), STONE_BLOCK);

        let result =
            move_aabb_through_voxels(&world, [0.5, 5.0, 0.5], [0.3, 0.9, 0.3], [0.0, -10.0, 0.0]);

        assert!((result.center[1] - 1.901).abs() < 0.0001);
        assert_eq!(result.collided, [false, true, false]);
        assert!(result.on_ground);
    }

    #[test]
    fn aabb_stops_against_solid_wall() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world.set_block(BlockPos::new(2, 1, 0), STONE_BLOCK);

        let result =
            move_aabb_through_voxels(&world, [0.5, 1.5, 0.5], [0.3, 0.9, 0.3], [3.0, 0.0, 0.0]);

        assert!((result.center[0] - 1.699).abs() < 0.0001);
        assert_eq!(result.collided, [true, false, false]);
        assert!(!result.on_ground);
    }

    fn assert_vec3_close(actual: [f32; 3], expected: [f32; 3]) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 0.0001,
                "axis {axis}: expected {}, got {}",
                expected[axis],
                actual[axis]
            );
        }
    }
}
