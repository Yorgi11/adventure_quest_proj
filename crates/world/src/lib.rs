use std::collections::HashMap;

use foundation::{BlockPos, ChunkCoord};
use terrain::WorldGenerator;
use voxels::{
    subchunk_index, world_to_chunk_coord, world_to_local_block, BlockId, Chunk, AIR_BLOCK,
    CHUNK_SIZE, SUBCHUNK_SIZE,
};

pub struct VoxelWorld {
    pub chunks: HashMap<ChunkCoord, Chunk>,
    generator: WorldGenerator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldStreamingSettings {
    pub horizontal_load_radius: i32,
    pub vertical_load_radius: i32,
    pub render_radius: i32,
    pub simulation_radius: i32,
    pub mob_ai_radius: i32,
    pub save_unload_radius: i32,
}

impl WorldStreamingSettings {
    pub const fn new(
        horizontal_load_radius: i32,
        vertical_load_radius: i32,
        render_radius: i32,
        simulation_radius: i32,
        mob_ai_radius: i32,
        save_unload_radius: i32,
    ) -> Self {
        Self {
            horizontal_load_radius,
            vertical_load_radius,
            render_radius,
            simulation_radius,
            mob_ai_radius,
            save_unload_radius,
        }
    }

    pub const fn prototype() -> Self {
        Self::new(1, 1, 1, 1, 1, 2)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkStreamingUpdate {
    pub center_chunk: ChunkCoord,
    pub requested_loads: usize,
    pub newly_loaded: usize,
    pub total_loaded: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockEdit {
    pub pos: BlockPos,
    pub block: BlockId,
}

impl BlockEdit {
    pub const fn new(pos: BlockPos, block: BlockId) -> Self {
        Self { pos, block }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockEditSummary {
    pub requested_edits: usize,
    pub changed_blocks: usize,
}

#[derive(Debug, Clone, Copy)]
struct ChangedBlock {
    chunk_coord: ChunkCoord,
    lx: usize,
    ly: usize,
    lz: usize,
}

impl VoxelWorld {
    pub fn new(seed: u64) -> Self {
        Self {
            chunks: HashMap::new(),
            generator: WorldGenerator::new(seed),
        }
    }

    pub fn generate_chunk(seed: u64, coord: ChunkCoord) -> Chunk {
        WorldGenerator::new(seed).generate_chunk(coord)
    }

    pub const fn seed(&self) -> u64 {
        self.generator.seed
    }

    pub fn load_chunk(&mut self, coord: ChunkCoord) -> bool {
        if self.chunks.contains_key(&coord) {
            return false;
        }

        let chunk = Self::generate_chunk(self.seed(), coord);
        self.insert_chunk(chunk)
    }

    pub fn insert_chunk(&mut self, chunk: Chunk) -> bool {
        if self.chunks.contains_key(&chunk.coord) {
            return false;
        }

        self.chunks.insert(chunk.coord, chunk);
        true
    }

    pub fn load_chunks_around_origin(&mut self, radius: i32) -> ChunkStreamingUpdate {
        self.load_chunks_around_chunk(
            ChunkCoord::new(0, 0, 0),
            WorldStreamingSettings::new(radius, radius, radius, radius, radius, radius + 1),
        )
    }

    pub fn load_chunks_around_block(
        &mut self,
        center: BlockPos,
        settings: WorldStreamingSettings,
    ) -> ChunkStreamingUpdate {
        let center_chunk = world_to_chunk_coord(center.x, center.y, center.z);
        self.load_chunks_around_chunk(center_chunk, settings)
    }

    pub fn load_chunks_around_chunk(
        &mut self,
        center: ChunkCoord,
        settings: WorldStreamingSettings,
    ) -> ChunkStreamingUpdate {
        let horizontal_radius = settings.horizontal_load_radius.max(0);
        let vertical_radius = settings.vertical_load_radius.max(0);

        let mut requested_loads = 0;
        let mut newly_loaded = 0;

        for dy in -vertical_radius..=vertical_radius {
            for dz in -horizontal_radius..=horizontal_radius {
                for dx in -horizontal_radius..=horizontal_radius {
                    requested_loads += 1;

                    if self.load_chunk(center.offset(dx, dy, dz)) {
                        newly_loaded += 1;
                    }
                }
            }
        }

        ChunkStreamingUpdate {
            center_chunk: center,
            requested_loads,
            newly_loaded,
            total_loaded: self.chunks.len(),
        }
    }

    pub fn get_chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
    }

    pub fn get_chunk_mut(&mut self, coord: ChunkCoord) -> Option<&mut Chunk> {
        self.chunks.get_mut(&coord)
    }

    pub fn set_chunk_visible_mask(&mut self, coord: ChunkCoord, visible_mask: u8) -> bool {
        let Some(chunk) = self.chunks.get_mut(&coord) else {
            return false;
        };

        chunk.subchunk_visible_mask = visible_mask;
        true
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
        if let Some(change) = self.apply_block_edit(BlockEdit::new(pos, block)) {
            self.mark_neighbor_dirty_if_needed(change.chunk_coord, change.lx, change.ly, change.lz);
        }
    }

    pub fn set_blocks<I>(&mut self, edits: I) -> BlockEditSummary
    where
        I: IntoIterator<Item = BlockEdit>,
    {
        let mut requested_edits = 0;
        let mut changed_blocks = Vec::new();

        for edit in edits {
            requested_edits += 1;

            if let Some(change) = self.apply_block_edit(edit) {
                changed_blocks.push(change);
            }
        }

        for change in &changed_blocks {
            self.mark_neighbor_dirty_if_needed(change.chunk_coord, change.lx, change.ly, change.lz);
        }

        BlockEditSummary {
            requested_edits,
            changed_blocks: changed_blocks.len(),
        }
    }

    fn apply_block_edit(&mut self, edit: BlockEdit) -> Option<ChangedBlock> {
        let pos = edit.pos;
        let chunk_coord = world_to_chunk_coord(pos.x, pos.y, pos.z);
        let (lx, ly, lz) = world_to_local_block(pos.x, pos.y, pos.z);

        self.load_chunk(chunk_coord);

        let changed = self
            .chunks
            .get_mut(&chunk_coord)
            .map(|chunk| chunk.set_block(lx, ly, lz, edit.block))
            .unwrap_or(false);

        if changed {
            Some(ChangedBlock {
                chunk_coord,
                lx,
                ly,
                lz,
            })
        } else {
            None
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use voxels::{subchunk_bit, Chunk, STONE_BLOCK};

    fn world_with_empty_chunk(coord: ChunkCoord) -> VoxelWorld {
        let mut world = VoxelWorld::new(0);
        world.chunks.insert(coord, Chunk::new_empty(coord));
        world
    }

    #[test]
    fn setting_block_loads_missing_chunk() {
        let mut world = VoxelWorld::new(0);
        let pos = BlockPos::new(0, 0, 0);

        world.set_block(pos, STONE_BLOCK);

        assert_eq!(world.get_block(pos), STONE_BLOCK);
        assert!(world.get_chunk(ChunkCoord::new(0, 0, 0)).is_some());
    }

    #[test]
    fn inserting_existing_chunk_reports_no_change() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);

        assert!(!world.insert_chunk(Chunk::new_empty(coord)));
        assert_eq!(world.chunks.len(), 1);
    }

    #[test]
    fn generated_chunk_uses_requested_seed_and_coord() {
        let coord = ChunkCoord::new(2, -1, 3);
        let chunk = VoxelWorld::generate_chunk(123, coord);

        assert_eq!(chunk.coord, coord);
        assert_eq!(VoxelWorld::new(123).seed(), 123);
    }

    #[test]
    fn streaming_loads_chunks_around_origin() {
        let mut world = VoxelWorld::new(0);

        let first_update = world.load_chunks_around_origin(1);
        let second_update = world.load_chunks_around_origin(1);

        assert_eq!(first_update.center_chunk, ChunkCoord::new(0, 0, 0));
        assert_eq!(first_update.requested_loads, 27);
        assert_eq!(first_update.newly_loaded, 27);
        assert_eq!(first_update.total_loaded, 27);
        assert_eq!(second_update.requested_loads, 27);
        assert_eq!(second_update.newly_loaded, 0);
        assert_eq!(second_update.total_loaded, 27);
    }

    #[test]
    fn streaming_loads_around_world_block_position() {
        let mut world = VoxelWorld::new(0);
        let settings = WorldStreamingSettings::new(1, 0, 1, 1, 1, 2);

        let update = world.load_chunks_around_block(BlockPos::new(-1, 64, 33), settings);

        assert_eq!(update.center_chunk, ChunkCoord::new(-1, 2, 1));
        assert_eq!(update.requested_loads, 9);
        assert_eq!(update.newly_loaded, 9);
        assert!(world.get_chunk(ChunkCoord::new(-2, 2, 0)).is_some());
        assert!(world.get_chunk(ChunkCoord::new(0, 2, 2)).is_some());
        assert!(world.get_chunk(ChunkCoord::new(-1, 3, 1)).is_none());
    }

    #[test]
    fn streaming_clamps_negative_load_radii_to_center_chunk() {
        let mut world = VoxelWorld::new(0);
        let settings = WorldStreamingSettings::new(-4, -2, 0, 0, 0, 0);

        let update = world.load_chunks_around_chunk(ChunkCoord::new(3, -2, 1), settings);

        assert_eq!(update.center_chunk, ChunkCoord::new(3, -2, 1));
        assert_eq!(update.requested_loads, 1);
        assert_eq!(update.newly_loaded, 1);
        assert!(world.get_chunk(ChunkCoord::new(3, -2, 1)).is_some());
    }

    #[test]
    fn editing_at_subchunk_boundary_marks_both_subchunks_dirty() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);

        world.set_block(BlockPos::new(15, 0, 0), STONE_BLOCK);

        let chunk = world.get_chunk(coord).expect("chunk exists");
        let expected_mask = subchunk_bit(0) | subchunk_bit(1);

        assert_eq!(chunk.subchunk_dirty_mask & expected_mask, expected_mask);
    }

    #[test]
    fn chunk_visible_mask_can_be_updated_after_meshing() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);

        assert!(world.set_chunk_visible_mask(coord, 0b0000_0101));
        assert!(!world.set_chunk_visible_mask(ChunkCoord::new(99, 0, 0), 0b1111_1111));

        let chunk = world.get_chunk(coord).expect("chunk exists");

        assert_eq!(chunk.subchunk_visible_mask, 0b0000_0101);
    }

    #[test]
    fn no_op_edit_does_not_mark_chunk_dirty() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);

        world.set_block(BlockPos::new(0, 0, 0), AIR_BLOCK);

        let chunk = world.get_chunk(coord).expect("chunk exists");

        assert_eq!(chunk.subchunk_dirty_mask, 0);
        assert_eq!(chunk.revision, 0);
    }

    #[test]
    fn batch_edit_applies_multiple_changes_and_reports_summary() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);

        let summary = world.set_blocks([
            BlockEdit::new(BlockPos::new(0, 0, 0), STONE_BLOCK),
            BlockEdit::new(BlockPos::new(1, 0, 0), STONE_BLOCK),
            BlockEdit::new(BlockPos::new(2, 0, 0), AIR_BLOCK),
        ]);

        let chunk = world.get_chunk(coord).expect("chunk exists");

        assert_eq!(summary.requested_edits, 3);
        assert_eq!(summary.changed_blocks, 2);
        assert_eq!(world.get_block(BlockPos::new(0, 0, 0)), STONE_BLOCK);
        assert_eq!(world.get_block(BlockPos::new(1, 0, 0)), STONE_BLOCK);
        assert_eq!(chunk.solid_block_count, 2);
        assert_eq!(chunk.revision, 2);
    }

    #[test]
    fn batch_edit_loads_all_target_chunks() {
        let mut world = VoxelWorld::new(0);

        let summary = world.set_blocks([
            BlockEdit::new(BlockPos::new(0, 1024, 0), STONE_BLOCK),
            BlockEdit::new(BlockPos::new(32, 1024, 0), STONE_BLOCK),
        ]);

        assert_eq!(summary.changed_blocks, 2);
        assert_eq!(
            world.get_chunk(ChunkCoord::new(0, 32, 0)).unwrap().revision,
            1
        );
        assert_eq!(
            world.get_chunk(ChunkCoord::new(1, 32, 0)).unwrap().revision,
            1
        );
    }

    #[test]
    fn editing_at_chunk_boundary_marks_loaded_neighbor_dirty() {
        let coord = ChunkCoord::new(0, 0, 0);
        let neighbor_coord = ChunkCoord::new(1, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world
            .chunks
            .insert(neighbor_coord, Chunk::new_empty(neighbor_coord));

        world.set_block(BlockPos::new(31, 0, 0), STONE_BLOCK);

        let neighbor = world
            .get_chunk(neighbor_coord)
            .expect("neighbor chunk exists");

        assert_eq!(
            neighbor.subchunk_dirty_mask & subchunk_bit(0),
            subchunk_bit(0)
        );
    }

    #[test]
    fn editing_at_chunk_boundary_does_not_load_neighbor_chunk() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);

        world.set_block(BlockPos::new(31, 0, 0), STONE_BLOCK);

        assert!(world.get_chunk(ChunkCoord::new(1, 0, 0)).is_none());
    }

    #[test]
    fn batch_edit_marks_neighbor_loaded_by_same_batch_dirty() {
        let mut world = VoxelWorld::new(0);

        world.set_blocks([
            BlockEdit::new(BlockPos::new(31, 1024, 0), STONE_BLOCK),
            BlockEdit::new(BlockPos::new(32, 1024, 0), STONE_BLOCK),
        ]);

        let origin = world.get_chunk(ChunkCoord::new(0, 32, 0)).unwrap();
        let neighbor = world.get_chunk(ChunkCoord::new(1, 32, 0)).unwrap();

        assert_eq!(
            origin.subchunk_dirty_mask & subchunk_bit(1),
            subchunk_bit(1)
        );
        assert_eq!(
            neighbor.subchunk_dirty_mask & subchunk_bit(0),
            subchunk_bit(0)
        );
    }
}
