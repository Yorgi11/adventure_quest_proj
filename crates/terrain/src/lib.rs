use foundation::ChunkCoord;
use voxels::{
    BlockId, Chunk, AIR_BLOCK, CHUNK_SIZE, COAL_ORE_BLOCK, DIRT_BLOCK, GRASS_BLOCK, IRON_ORE_BLOCK,
    STONE_BLOCK,
};

const BASE_HEIGHT: i32 = 16;
const HEIGHT_VARIATION: f32 = 12.0;
const SURFACE_DIRT_DEPTH: i32 = 4;
const CAVE_SURFACE_BUFFER: i32 = 6;
const CAVE_CELL_SIZE: i32 = 12;
const CAVE_THRESHOLD: f32 = 0.72;
const COAL_MAX_HEIGHT: i32 = 48;
const IRON_MAX_HEIGHT: i32 = 16;
const COAL_RARITY: f32 = 0.985;
const IRON_RARITY: f32 = 0.992;

pub struct WorldGenerator {
    pub seed: u64,
}

impl WorldGenerator {
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub fn generate_chunk(&self, coord: ChunkCoord) -> Chunk {
        let mut chunk = Chunk::new_empty(coord);

        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let world_x = coord.x * CHUNK_SIZE as i32 + x as i32;
                    let world_y = coord.y * CHUNK_SIZE as i32 + y as i32;
                    let world_z = coord.z * CHUNK_SIZE as i32 + z as i32;

                    let block = self.generate_block(world_x, world_y, world_z);
                    chunk.set_block(x, y, z, block);
                }
            }
        }

        chunk.clear_dirty();
        chunk.revision = 0;
        chunk
    }

    fn generate_block(&self, world_x: i32, world_y: i32, world_z: i32) -> BlockId {
        let height = self.terrain_height(world_x, world_z);

        if world_y > height {
            return AIR_BLOCK;
        }

        if world_y == height {
            return GRASS_BLOCK;
        }

        if world_y >= height - SURFACE_DIRT_DEPTH {
            return DIRT_BLOCK;
        }

        if self.should_carve_cave(world_x, world_y, world_z, height) {
            return AIR_BLOCK;
        }

        self.generate_ore(world_x, world_y, world_z)
            .unwrap_or(STONE_BLOCK)
    }

    fn terrain_height(&self, world_x: i32, world_z: i32) -> i32 {
        let noise = self.value_noise_2d(world_x, world_z);
        let height_variation = (noise * HEIGHT_VARIATION) as i32;

        BASE_HEIGHT + height_variation
    }

    fn should_carve_cave(&self, world_x: i32, world_y: i32, world_z: i32, height: i32) -> bool {
        if world_y > height - CAVE_SURFACE_BUFFER {
            return false;
        }

        self.value_noise_3d(world_x, world_y, world_z, CAVE_CELL_SIZE) > CAVE_THRESHOLD
    }

    fn generate_ore(&self, world_x: i32, world_y: i32, world_z: i32) -> Option<BlockId> {
        if world_y <= IRON_MAX_HEIGHT
            && self.hash_3d_to_unit(world_x, world_y, world_z, 0x1A2B_3C4D_5E6F_7788) > IRON_RARITY
        {
            return Some(IRON_ORE_BLOCK);
        }

        if world_y <= COAL_MAX_HEIGHT
            && self.hash_3d_to_unit(world_x, world_y, world_z, 0x9988_7766_5544_3322) > COAL_RARITY
        {
            return Some(COAL_ORE_BLOCK);
        }

        None
    }

    fn value_noise_2d(&self, x: i32, z: i32) -> f32 {
        let cell_size = 16;

        let x0 = x.div_euclid(cell_size);
        let z0 = z.div_euclid(cell_size);

        let local_x = x.rem_euclid(cell_size) as f32 / cell_size as f32;
        let local_z = z.rem_euclid(cell_size) as f32 / cell_size as f32;

        let h00 = self.hash_to_unit(x0, z0);
        let h10 = self.hash_to_unit(x0 + 1, z0);
        let h01 = self.hash_to_unit(x0, z0 + 1);
        let h11 = self.hash_to_unit(x0 + 1, z0 + 1);

        let sx = smoothstep(local_x);
        let sz = smoothstep(local_z);

        let a = lerp(h00, h10, sx);
        let b = lerp(h01, h11, sx);

        lerp(a, b, sz)
    }

    fn value_noise_3d(&self, x: i32, y: i32, z: i32, cell_size: i32) -> f32 {
        let x0 = x.div_euclid(cell_size);
        let y0 = y.div_euclid(cell_size);
        let z0 = z.div_euclid(cell_size);

        let local_x = x.rem_euclid(cell_size) as f32 / cell_size as f32;
        let local_y = y.rem_euclid(cell_size) as f32 / cell_size as f32;
        let local_z = z.rem_euclid(cell_size) as f32 / cell_size as f32;

        let sx = smoothstep(local_x);
        let sy = smoothstep(local_y);
        let sz = smoothstep(local_z);

        let c000 = self.hash_3d_to_unit(x0, y0, z0, 0);
        let c100 = self.hash_3d_to_unit(x0 + 1, y0, z0, 0);
        let c010 = self.hash_3d_to_unit(x0, y0 + 1, z0, 0);
        let c110 = self.hash_3d_to_unit(x0 + 1, y0 + 1, z0, 0);
        let c001 = self.hash_3d_to_unit(x0, y0, z0 + 1, 0);
        let c101 = self.hash_3d_to_unit(x0 + 1, y0, z0 + 1, 0);
        let c011 = self.hash_3d_to_unit(x0, y0 + 1, z0 + 1, 0);
        let c111 = self.hash_3d_to_unit(x0 + 1, y0 + 1, z0 + 1, 0);

        let x00 = lerp(c000, c100, sx);
        let x10 = lerp(c010, c110, sx);
        let x01 = lerp(c001, c101, sx);
        let x11 = lerp(c011, c111, sx);

        let y0 = lerp(x00, x10, sy);
        let y1 = lerp(x01, x11, sy);

        lerp(y0, y1, sz)
    }

    fn hash_to_unit(&self, x: i32, z: i32) -> f32 {
        let mut value = self.seed;

        let x_bits = x as i64 as u64;
        let z_bits = z as i64 as u64;

        value ^= x_bits.wrapping_add(0x9E37_79B9_7F4A_7C15);
        value = value.rotate_left(13);

        value ^= z_bits.wrapping_add(0xBF58_476D_1CE4_E5B9);
        value = value.wrapping_mul(0x94D0_49BB_1331_11EB);

        (value & 0xFFFF) as f32 / 65535.0
    }

    fn hash_3d_to_unit(&self, x: i32, y: i32, z: i32, salt: u64) -> f32 {
        let mut value = self.seed ^ salt;

        value ^= (x as i64 as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        value = value.rotate_left(13);
        value ^= (y as i64 as u64).wrapping_add(0xBF58_476D_1CE4_E5B9);
        value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= (z as i64 as u64).wrapping_add(0xD6E8_FD9B_5F36_6DAB);
        value = value.rotate_left(17);

        (value & 0xFFFF) as f32 / 65535.0
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_generation_supports_negative_chunk_coordinates() {
        let generator = WorldGenerator::new(12345);

        let chunk = generator.generate_chunk(ChunkCoord::new(-1, -1, -1));

        assert_eq!(chunk.coord, ChunkCoord::new(-1, -1, -1));
    }

    #[test]
    fn terrain_generation_is_deterministic_for_same_seed() {
        let first = WorldGenerator::new(12345).generate_chunk(ChunkCoord::new(0, 0, 0));
        let second = WorldGenerator::new(12345).generate_chunk(ChunkCoord::new(0, 0, 0));

        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    assert_eq!(
                        first.get_block(x, y, z),
                        second.get_block(x, y, z),
                        "block mismatch at {x}, {y}, {z}"
                    );
                }
            }
        }
    }

    #[test]
    fn terrain_generation_places_basic_ores() {
        let chunk = WorldGenerator::new(12345).generate_chunk(ChunkCoord::new(0, 0, 0));
        let mut coal_blocks = 0;
        let mut iron_blocks = 0;

        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    match chunk.get_block(x, y, z) {
                        COAL_ORE_BLOCK => coal_blocks += 1,
                        IRON_ORE_BLOCK => iron_blocks += 1,
                        _ => {}
                    }
                }
            }
        }

        assert!(coal_blocks > 0);
        assert!(iron_blocks > 0);
    }

    #[test]
    fn terrain_generation_carves_underground_caves() {
        let chunk = WorldGenerator::new(12345).generate_chunk(ChunkCoord::new(0, 0, 0));
        let mut underground_air_blocks = 0;

        for y in 0..8 {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    if chunk.get_block(x, y, z) == AIR_BLOCK {
                        underground_air_blocks += 1;
                    }
                }
            }
        }

        assert!(underground_air_blocks > 0);
    }
}
