use foundation::BlockPos;
use physics::{raycast_blocks, VoxelRayHit};
use voxels::{BlockId, AIR_BLOCK, DIRT_BLOCK, GRASS_BLOCK, STONE_BLOCK};
use world::{BlockEdit, BlockEditSummary, VoxelWorld};

pub const HOTBAR_SLOT_COUNT: usize = 9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hotbar {
    slots: [BlockId; HOTBAR_SLOT_COUNT],
    selected_slot: usize,
}

impl Hotbar {
    pub const fn new(slots: [BlockId; HOTBAR_SLOT_COUNT]) -> Self {
        Self {
            slots,
            selected_slot: 0,
        }
    }

    pub const fn starter() -> Self {
        Self::new([
            DIRT_BLOCK,
            GRASS_BLOCK,
            STONE_BLOCK,
            AIR_BLOCK,
            AIR_BLOCK,
            AIR_BLOCK,
            AIR_BLOCK,
            AIR_BLOCK,
            AIR_BLOCK,
        ])
    }

    pub fn select_slot(&mut self, slot: usize) -> bool {
        if slot >= HOTBAR_SLOT_COUNT {
            return false;
        }

        self.selected_slot = slot;
        true
    }

    pub const fn selected_slot(&self) -> usize {
        self.selected_slot
    }

    pub const fn selected_block(&self) -> BlockId {
        self.slots[self.selected_slot]
    }

    pub fn set_slot(&mut self, slot: usize, block: BlockId) -> bool {
        let Some(target) = self.slots.get_mut(slot) else {
            return false;
        };

        *target = block;
        true
    }

    pub const fn slots(&self) -> &[BlockId; HOTBAR_SLOT_COUNT] {
        &self.slots
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockInteraction {
    Miss,
    Break {
        hit: VoxelRayHit,
        summary: BlockEditSummary,
    },
    Place {
        hit: VoxelRayHit,
        placed_block: BlockPos,
        block: BlockId,
        summary: BlockEditSummary,
    },
    NoPlaceableBlockSelected,
    InvalidPlacementFace {
        hit: VoxelRayHit,
    },
}

pub fn break_target_block(
    world: &mut VoxelWorld,
    origin: [f32; 3],
    direction: [f32; 3],
    reach_distance: f32,
) -> BlockInteraction {
    let Some(hit) = raycast_blocks(world, origin, direction, reach_distance) else {
        return BlockInteraction::Miss;
    };

    let summary = world.set_blocks([BlockEdit::new(hit.world_block, AIR_BLOCK)]);

    BlockInteraction::Break { hit, summary }
}

pub fn place_selected_block(
    world: &mut VoxelWorld,
    hotbar: &Hotbar,
    origin: [f32; 3],
    direction: [f32; 3],
    reach_distance: f32,
) -> BlockInteraction {
    place_block_from_selection(
        world,
        hotbar.selected_block(),
        origin,
        direction,
        reach_distance,
    )
}

pub fn place_block_from_selection(
    world: &mut VoxelWorld,
    block: BlockId,
    origin: [f32; 3],
    direction: [f32; 3],
    reach_distance: f32,
) -> BlockInteraction {
    if block == AIR_BLOCK {
        return BlockInteraction::NoPlaceableBlockSelected;
    }

    let Some(hit) = raycast_blocks(world, origin, direction, reach_distance) else {
        return BlockInteraction::Miss;
    };

    if hit.face_normal == [0, 0, 0] {
        return BlockInteraction::InvalidPlacementFace { hit };
    }

    let placed_block = hit.placement_block();
    let summary = world.set_blocks([BlockEdit::new(placed_block, block)]);

    BlockInteraction::Place {
        hit,
        placed_block,
        block,
        summary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use foundation::ChunkCoord;
    use voxels::Chunk;

    fn world_with_empty_chunk(coord: ChunkCoord) -> VoxelWorld {
        let mut world = VoxelWorld::new(0);
        world.chunks.insert(coord, Chunk::new_empty(coord));
        world
    }

    #[test]
    fn hotbar_selects_and_sets_slots() {
        let mut hotbar = Hotbar::starter();

        assert_eq!(hotbar.selected_slot(), 0);
        assert_eq!(hotbar.selected_block(), DIRT_BLOCK);
        assert!(hotbar.select_slot(2));
        assert_eq!(hotbar.selected_block(), STONE_BLOCK);
        assert!(hotbar.set_slot(2, GRASS_BLOCK));
        assert_eq!(hotbar.selected_block(), GRASS_BLOCK);
        assert!(!hotbar.select_slot(HOTBAR_SLOT_COUNT));
        assert!(!hotbar.set_slot(HOTBAR_SLOT_COUNT, STONE_BLOCK));
    }

    #[test]
    fn break_target_block_sets_hit_block_to_air() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world.set_block(BlockPos::new(3, 0, 0), STONE_BLOCK);

        let result = break_target_block(&mut world, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 10.0);

        let BlockInteraction::Break { hit, summary } = result else {
            panic!("expected break interaction");
        };

        assert_eq!(hit.world_block, BlockPos::new(3, 0, 0));
        assert_eq!(summary.changed_blocks, 1);
        assert_eq!(world.get_block(BlockPos::new(3, 0, 0)), AIR_BLOCK);
    }

    #[test]
    fn break_target_block_misses_when_ray_hits_nothing() {
        let mut world = world_with_empty_chunk(ChunkCoord::new(0, 0, 0));

        let result = break_target_block(&mut world, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 10.0);

        assert!(matches!(result, BlockInteraction::Miss));
    }

    #[test]
    fn place_selected_block_uses_hotbar_block_against_hit_face() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world.set_block(BlockPos::new(3, 0, 0), STONE_BLOCK);

        let mut hotbar = Hotbar::starter();
        hotbar.select_slot(1);

        let result =
            place_selected_block(&mut world, &hotbar, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 10.0);

        let BlockInteraction::Place {
            placed_block,
            block,
            summary,
            ..
        } = result
        else {
            panic!("expected place interaction");
        };

        assert_eq!(placed_block, BlockPos::new(2, 0, 0));
        assert_eq!(block, GRASS_BLOCK);
        assert_eq!(summary.changed_blocks, 1);
        assert_eq!(world.get_block(placed_block), GRASS_BLOCK);
    }

    #[test]
    fn place_selected_block_rejects_air_slot() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world.set_block(BlockPos::new(3, 0, 0), STONE_BLOCK);

        let mut hotbar = Hotbar::starter();
        hotbar.select_slot(3);

        let result =
            place_selected_block(&mut world, &hotbar, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 10.0);

        assert!(matches!(result, BlockInteraction::NoPlaceableBlockSelected));
        assert_eq!(world.get_block(BlockPos::new(2, 0, 0)), AIR_BLOCK);
    }

    #[test]
    fn place_selected_block_rejects_origin_inside_hit_block() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunk(coord);
        world.set_block(BlockPos::new(0, 0, 0), STONE_BLOCK);

        let hotbar = Hotbar::starter();

        let result =
            place_selected_block(&mut world, &hotbar, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 10.0);

        assert!(matches!(
            result,
            BlockInteraction::InvalidPlacementFace { .. }
        ));
        assert_eq!(world.get_block(BlockPos::new(0, 0, 0)), STONE_BLOCK);
    }
}
