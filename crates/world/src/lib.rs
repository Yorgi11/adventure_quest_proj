use std::collections::HashMap;

use foundation::{BlockPos, ChunkCoord};
use voxels::{
    subchunk_index, world_to_chunk_coord, world_to_local_block, BlockId, Chunk, AIR_BLOCK,
    CHUNK_SIZE, SUBCHUNK_SIZE,
};
use terrain::WorldGenerator;

pub struct VoxelWorld {
    pub chunks: HashMap<ChunkCoord, Chunk>,
    generator: WorldGenerator,
}

impl VoxelWorld {
    pub fn new(seed: u64) -> Self {
        Self {
            chunks: HashMap::new(),
            generator: WorldGenerator::new(seed),
        }
    }

    pub fn load_chunk(&mut self, coord: ChunkCoord) {
        if self.chunks.contains_key(&coord) {
            return;
        }

        let chunk = self.generator.generate_chunk(coord);
        self.chunks.insert(coord, chunk);
    }

    pub fn load_chunks_around_origin(&mut self, radius: i32) {
        for y in -radius..=radius {
            for z in -radius..=radius {
                for x in -radius..=radius {
                    self.load_chunk(ChunkCoord::new(x, y, z));
                }
            }
        }
    }

    pub fn get_chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
    }

    pub fn get_chunk_mut(&mut self, coord: ChunkCoord) -> Option<&mut Chunk> {
        self.chunks.get_mut(&coord)
    }

    pub fn get_block(&self, pos: BlockPos) -> BlockId {
        let chunk_coord = world_to_chunk_coord(pos.x, pos.y, pos.z);
        let (lx, ly, lz) = world_to_local_block(pos.x, pos.y, pos.z);

        self.chunks
            .get(&chunk_coord)
            .map(|chunk| chunk.get_block(lx, ly, lz))
            .unwrap_or(AIR_BLOCK)
    }

    pub fn set_block(&mut self, pos: BlockPos, block: BlockId) {
        let chunk_coord = world_to_chunk_coord(pos.x, pos.y, pos.z);
        let (lx, ly, lz) = world_to_local_block(pos.x, pos.y, pos.z);

        self.load_chunk(chunk_coord);

        if let Some(chunk) = self.chunks.get_mut(&chunk_coord) {
            chunk.set_block(lx, ly, lz, block);
        }

        self.mark_neighbor_dirty_if_needed(chunk_coord, lx, ly, lz);
    }

    fn mark_neighbor_dirty_if_needed(
        &mut self,
        chunk_coord: ChunkCoord,
        lx: usize,
        ly: usize,
        lz: usize,
    ) {
        self.mark_adjacent_subchunks_dirty(chunk_coord, lx, ly, lz);
        self.mark_adjacent_chunks_dirty(chunk_coord, lx, ly, lz);
    }

    fn mark_adjacent_subchunks_dirty(
        &mut self,
        chunk_coord: ChunkCoord,
        lx: usize,
        ly: usize,
        lz: usize,
    ) {
        let Some(chunk) = self.chunks.get_mut(&chunk_coord) else {
            return;
        };

        let current_sub = subchunk_index(lx, ly, lz);

        if lx == SUBCHUNK_SIZE - 1 && lx + 1 < CHUNK_SIZE {
            chunk.mark_subchunk_dirty(subchunk_index(lx + 1, ly, lz));
        }

        if lx == SUBCHUNK_SIZE && lx > 0 {
            chunk.mark_subchunk_dirty(subchunk_index(lx - 1, ly, lz));
        }

        if ly == SUBCHUNK_SIZE - 1 && ly + 1 < CHUNK_SIZE {
            chunk.mark_subchunk_dirty(subchunk_index(lx, ly + 1, lz));
        }

        if ly == SUBCHUNK_SIZE && ly > 0 {
            chunk.mark_subchunk_dirty(subchunk_index(lx, ly - 1, lz));
        }

        if lz == SUBCHUNK_SIZE - 1 && lz + 1 < CHUNK_SIZE {
            chunk.mark_subchunk_dirty(subchunk_index(lx, ly, lz + 1));
        }

        if lz == SUBCHUNK_SIZE && lz > 0 {
            chunk.mark_subchunk_dirty(subchunk_index(lx, ly, lz - 1));
        }

        chunk.mark_subchunk_dirty(current_sub);
    }

    fn mark_adjacent_chunks_dirty(
        &mut self,
        chunk_coord: ChunkCoord,
        lx: usize,
        ly: usize,
        lz: usize,
    ) {
        if lx == 0 {
            self.mark_chunk_border_dirty(chunk_coord.offset(-1, 0, 0), CHUNK_SIZE - 1, ly, lz);
        }

        if lx == CHUNK_SIZE - 1 {
            self.mark_chunk_border_dirty(chunk_coord.offset(1, 0, 0), 0, ly, lz);
        }

        if ly == 0 {
            self.mark_chunk_border_dirty(chunk_coord.offset(0, -1, 0), lx, CHUNK_SIZE - 1, lz);
        }

        if ly == CHUNK_SIZE - 1 {
            self.mark_chunk_border_dirty(chunk_coord.offset(0, 1, 0), lx, 0, lz);
        }

        if lz == 0 {
            self.mark_chunk_border_dirty(chunk_coord.offset(0, 0, -1), lx, ly, CHUNK_SIZE - 1);
        }

        if lz == CHUNK_SIZE - 1 {
            self.mark_chunk_border_dirty(chunk_coord.offset(0, 0, 1), lx, ly, 0);
        }
    }

    fn mark_chunk_border_dirty(
        &mut self,
        neighbor_coord: ChunkCoord,
        lx: usize,
        ly: usize,
        lz: usize,
    ) {
        if let Some(neighbor) = self.chunks.get_mut(&neighbor_coord) {
            let sub = subchunk_index(lx, ly, lz);
            neighbor.mark_subchunk_dirty(sub);
        }
    }
}