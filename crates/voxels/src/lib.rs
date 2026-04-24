use foundation::ChunkCoord;

pub type BlockId = u16;

pub const AIR_BLOCK: BlockId = 0;
pub const DIRT_BLOCK: BlockId = 1;
pub const GRASS_BLOCK: BlockId = 2;
pub const STONE_BLOCK: BlockId = 3;

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_AREA: usize = CHUNK_SIZE * CHUNK_SIZE;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

pub const SUBCHUNK_SIZE: usize = 16;
pub const SUBCHUNK_VOLUME: usize = SUBCHUNK_SIZE * SUBCHUNK_SIZE * SUBCHUNK_SIZE;
pub const SUBCHUNK_COUNT: usize = 8;

#[inline(always)]
pub const fn block_index(x: usize, y: usize, z: usize) -> usize {
    x + z * CHUNK_SIZE + y * CHUNK_AREA
}

#[inline(always)]
pub const fn subchunk_index(x: usize, y: usize, z: usize) -> usize {
    let sx = if x >= SUBCHUNK_SIZE { 1 } else { 0 };
    let sy = if y >= SUBCHUNK_SIZE { 1 } else { 0 };
    let sz = if z >= SUBCHUNK_SIZE { 1 } else { 0 };

    sx + sy * 2 + sz * 4
}

#[inline(always)]
pub const fn subchunk_bit(index: usize) -> u8 {
    1u8 << index
}

#[inline(always)]
pub fn world_to_chunk_coord(x: i32, y: i32, z: i32) -> ChunkCoord {
    ChunkCoord {
        x: x.div_euclid(CHUNK_SIZE as i32),
        y: y.div_euclid(CHUNK_SIZE as i32),
        z: z.div_euclid(CHUNK_SIZE as i32),
    }
}

#[inline(always)]
pub fn world_to_local_block(x: i32, y: i32, z: i32) -> (usize, usize, usize) {
    (
        x.rem_euclid(CHUNK_SIZE as i32) as usize,
        y.rem_euclid(CHUNK_SIZE as i32) as usize,
        z.rem_euclid(CHUNK_SIZE as i32) as usize,
    )
}

pub struct Chunk {
    pub coord: ChunkCoord,

    blocks: Box<[BlockId; CHUNK_VOLUME]>,

    pub solid_block_count: u32,
    pub subchunk_solid_counts: [u16; SUBCHUNK_COUNT],

    pub subchunk_occupancy_mask: u8,
    pub subchunk_dirty_mask: u8,
    pub subchunk_visible_mask: u8,
    pub subchunk_full_solid_mask: u8,

    pub revision: u32,
}

impl Chunk {
    pub fn new_empty(coord: ChunkCoord) -> Self {
        Self {
            coord,
            blocks: Box::new([AIR_BLOCK; CHUNK_VOLUME]),

            solid_block_count: 0,
            subchunk_solid_counts: [0; SUBCHUNK_COUNT],

            subchunk_occupancy_mask: 0,
            subchunk_dirty_mask: 0,
            subchunk_visible_mask: 0,
            subchunk_full_solid_mask: 0,

            revision: 0,
        }
    }

    #[inline(always)]
    pub fn get_block(&self, x: usize, y: usize, z: usize) -> BlockId {
        self.blocks[block_index(x, y, z)]
    }

    #[inline(always)]
    pub fn set_block_raw(&mut self, x: usize, y: usize, z: usize, block: BlockId) {
        let index = block_index(x, y, z);
        self.blocks[index] = block;
    }

    pub fn set_block(&mut self, x: usize, y: usize, z: usize, new_block: BlockId) {
        debug_assert!(x < CHUNK_SIZE);
        debug_assert!(y < CHUNK_SIZE);
        debug_assert!(z < CHUNK_SIZE);

        let index = block_index(x, y, z);
        let old_block = self.blocks[index];

        if old_block == new_block {
            return;
        }

        let old_is_air = old_block == AIR_BLOCK;
        let new_is_air = new_block == AIR_BLOCK;

        let sub = subchunk_index(x, y, z);
        let bit = subchunk_bit(sub);

        self.blocks[index] = new_block;

        match (old_is_air, new_is_air) {
            (true, false) => {
                self.solid_block_count += 1;
                self.subchunk_solid_counts[sub] += 1;
                self.subchunk_occupancy_mask |= bit;
            }
            (false, true) => {
                self.solid_block_count -= 1;
                self.subchunk_solid_counts[sub] -= 1;

                if self.subchunk_solid_counts[sub] == 0 {
                    self.subchunk_occupancy_mask &= !bit;
                }
            }
            _ => {}
        }

        if self.subchunk_solid_counts[sub] as usize == SUBCHUNK_VOLUME {
            self.subchunk_full_solid_mask |= bit;
        } else {
            self.subchunk_full_solid_mask &= !bit;
        }

        self.mark_subchunk_dirty(sub);
        self.revision = self.revision.wrapping_add(1);
    }

    #[inline(always)]
    pub fn mark_subchunk_dirty(&mut self, sub: usize) {
        self.subchunk_dirty_mask |= subchunk_bit(sub);
    }

    #[inline(always)]
    pub fn clear_dirty(&mut self) {
        self.subchunk_dirty_mask = 0;
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.solid_block_count == 0
    }

    #[inline(always)]
    pub fn is_subchunk_empty(&self, sub: usize) -> bool {
        self.subchunk_solid_counts[sub] == 0
    }

    #[inline(always)]
    pub fn is_subchunk_full(&self, sub: usize) -> bool {
        self.subchunk_solid_counts[sub] as usize == SUBCHUNK_VOLUME
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_world_coordinates_convert_correctly() {
        let coord = world_to_chunk_coord(-1, -1, -1);
        assert_eq!(coord, ChunkCoord::new(-1, -1, -1));

        let local = world_to_local_block(-1, -1, -1);
        assert_eq!(local, (31, 31, 31));
    }

    #[test]
    fn setting_block_updates_masks() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));

        chunk.set_block(0, 0, 0, STONE_BLOCK);

        assert_eq!(chunk.solid_block_count, 1);
        assert_eq!(chunk.subchunk_solid_counts[0], 1);
        assert_eq!(chunk.subchunk_occupancy_mask, 1);
        assert_eq!(chunk.subchunk_dirty_mask, 1);
        assert_eq!(chunk.revision, 1);
    }

    #[test]
    fn removing_block_updates_masks() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));

        chunk.set_block(0, 0, 0, STONE_BLOCK);
        chunk.set_block(0, 0, 0, AIR_BLOCK);

        assert_eq!(chunk.solid_block_count, 0);
        assert_eq!(chunk.subchunk_solid_counts[0], 0);
        assert_eq!(chunk.subchunk_occupancy_mask, 0);
        assert_eq!(chunk.revision, 2);
    }
}