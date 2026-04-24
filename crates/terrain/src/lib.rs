use foundation::ChunkCoord;
use voxels::{
    BlockId, Chunk, AIR_BLOCK, CHUNK_SIZE, DIRT_BLOCK, GRASS_BLOCK, STONE_BLOCK,
};

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

        if world_y >= height - 4 {
            return DIRT_BLOCK;
        }

        STONE_BLOCK
    }

    fn terrain_height(&self, world_x: i32, world_z: i32) -> i32 {
        let noise = self.value_noise_2d(world_x, world_z);

        let base_height = 16;
        let height_variation = (noise * 12.0) as i32;

        base_height + height_variation
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
}