use crate::{
    BlockId, AIR_BLOCK, COAL_ORE_BLOCK, DIRT_BLOCK, GRASS_BLOCK, IRON_ORE_BLOCK, STONE_BLOCK,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockProperties {
    pub id: BlockId,
    pub label: &'static str,
    pub color: [f32; 4],
    pub texture: Option<BlockTextureRegion>,
    pub hotbar_texture: Option<BlockTextureRegion>,
    pub break_hp: f32,
    pub is_opaque: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockTextureRegion {
    pub bounds: TextureBounds,
    pub offset: [f32; 2],
}

impl BlockTextureRegion {
    pub const fn new(bounds: TextureBounds, offset: [f32; 2]) -> Self {
        Self { bounds, offset }
    }

    pub const fn from_index(index: usize) -> Self {
        let atlas_column = index % BLOCK_TEXTURE_ATLAS_COLUMNS;
        let atlas_row = index / BLOCK_TEXTURE_ATLAS_COLUMNS;

        Self::from_atlas_cell(atlas_column, atlas_row)
    }

    pub const fn from_atlas_cell(column: usize, row: usize) -> Self {
        let min_x = column as f32 * BLOCK_TEXTURE_WIDTH_PX / BLOCK_TEXTURE_ATLAS_WIDTH_PX;
        let min_y = row as f32 * BLOCK_TEXTURE_HEIGHT_PX / BLOCK_TEXTURE_ATLAS_HEIGHT_PX;

        let size_x = BLOCK_TEXTURE_WIDTH_PX / BLOCK_TEXTURE_ATLAS_WIDTH_PX;
        let size_y = BLOCK_TEXTURE_HEIGHT_PX / BLOCK_TEXTURE_ATLAS_HEIGHT_PX;

        Self {
            bounds: TextureBounds::new([min_x, min_y], [size_x, size_y]),
            offset: [0.0, 0.0],
        }
    }

    pub fn uv(self, local_uv: [f32; 2]) -> [f32; 2] {
        [
            self.bounds.min[0] + self.offset[0] + local_uv[0].clamp(0.0, 1.0) * self.bounds.size[0],
            self.bounds.min[1] + self.offset[1] + local_uv[1].clamp(0.0, 1.0) * self.bounds.size[1],
        ]
    }

    pub fn face_uv(self, face: CubeFace, local_uv: [f32; 2]) -> [f32; 2] {
        let (column, row) = face.cross_tile();

        let tile_width = self.bounds.size[0] / CROSS_UV_COLUMNS as f32;
        let tile_height = self.bounds.size[1] / CROSS_UV_ROWS as f32;

        [
            self.bounds.min[0]
                + self.offset[0]
                + column as f32 * tile_width
                + local_uv[0].clamp(0.0, 1.0) * tile_width,
            self.bounds.min[1]
                + self.offset[1]
                + row as f32 * tile_height
                + local_uv[1].clamp(0.0, 1.0) * tile_height,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureBounds {
    pub min: [f32; 2],
    pub size: [f32; 2],
}

impl TextureBounds {
    pub const fn new(min: [f32; 2], size: [f32; 2]) -> Self {
        Self { min, size }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeFace {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl CubeFace {
    const fn cross_tile(self) -> (usize, usize) {
        match self {
            Self::PosY => (1, 0),
            Self::NegX => (0, 1),
            Self::PosZ => (1, 1),
            Self::PosX => (2, 1),
            Self::NegY => (1, 2),
            Self::NegZ => (1, 3),
        }
    }
}

pub const BLOCK_TEXTURE_ATLAS_PATH: &str = "assets/textures/blocksTexture.png";

pub const BLOCK_TEXTURE_WIDTH_PX: f32 = 24.0;
pub const BLOCK_TEXTURE_HEIGHT_PX: f32 = 32.0;

pub const BLOCK_TEXTURE_ATLAS_COLUMNS: usize = 5;
pub const BLOCK_TEXTURE_ATLAS_ROWS: usize = 2;

pub const BLOCK_TEXTURE_ATLAS_WIDTH_PX: f32 =
    BLOCK_TEXTURE_WIDTH_PX * BLOCK_TEXTURE_ATLAS_COLUMNS as f32;

pub const BLOCK_TEXTURE_ATLAS_HEIGHT_PX: f32 =
    BLOCK_TEXTURE_HEIGHT_PX * BLOCK_TEXTURE_ATLAS_ROWS as f32;

pub const CROSS_UV_COLUMNS: usize = 3;
pub const CROSS_UV_ROWS: usize = 4;

const DIRT_TEXTURE_INDEX: usize = 0;
const GRASS_TEXTURE_INDEX: usize = 1;
const STONE_TEXTURE_INDEX: usize = 2;
const COAL_ORE_TEXTURE_INDEX: usize = 3;
const IRON_ORE_TEXTURE_INDEX: usize = 4;
const HOTBAR_TEXTURE_ROW: usize = 1;

const DIRT_TEXTURE: BlockTextureRegion = BlockTextureRegion::from_index(DIRT_TEXTURE_INDEX);
const GRASS_TEXTURE: BlockTextureRegion = BlockTextureRegion::from_index(GRASS_TEXTURE_INDEX);
const STONE_TEXTURE: BlockTextureRegion = BlockTextureRegion::from_index(STONE_TEXTURE_INDEX);
const COAL_ORE_TEXTURE: BlockTextureRegion = BlockTextureRegion::from_index(COAL_ORE_TEXTURE_INDEX);
const IRON_ORE_TEXTURE: BlockTextureRegion = BlockTextureRegion::from_index(IRON_ORE_TEXTURE_INDEX);
const DIRT_HOTBAR_TEXTURE: BlockTextureRegion =
    BlockTextureRegion::from_atlas_cell(DIRT_TEXTURE_INDEX, HOTBAR_TEXTURE_ROW);
const GRASS_HOTBAR_TEXTURE: BlockTextureRegion =
    BlockTextureRegion::from_atlas_cell(GRASS_TEXTURE_INDEX, HOTBAR_TEXTURE_ROW);
const STONE_HOTBAR_TEXTURE: BlockTextureRegion =
    BlockTextureRegion::from_atlas_cell(STONE_TEXTURE_INDEX, HOTBAR_TEXTURE_ROW);
const COAL_ORE_HOTBAR_TEXTURE: BlockTextureRegion =
    BlockTextureRegion::from_atlas_cell(COAL_ORE_TEXTURE_INDEX, HOTBAR_TEXTURE_ROW);
const IRON_ORE_HOTBAR_TEXTURE: BlockTextureRegion =
    BlockTextureRegion::from_atlas_cell(IRON_ORE_TEXTURE_INDEX, HOTBAR_TEXTURE_ROW);

pub const AIR_PROPERTIES: BlockProperties = BlockProperties {
    id: AIR_BLOCK,
    label: "EMPTY",
    color: [0.0, 0.0, 0.0, 0.0],
    texture: None,
    hotbar_texture: None,
    break_hp: 0.0,
    is_opaque: false,
};

pub const DIRT_PROPERTIES: BlockProperties = BlockProperties {
    id: DIRT_BLOCK,
    label: "DIRT",
    color: [0.43, 0.25, 0.13, 1.0],
    texture: Some(DIRT_TEXTURE),
    hotbar_texture: Some(DIRT_HOTBAR_TEXTURE),
    break_hp: 0.55,
    is_opaque: true,
};

pub const GRASS_PROPERTIES: BlockProperties = BlockProperties {
    id: GRASS_BLOCK,
    label: "GRASS",
    color: [0.20, 0.55, 0.22, 1.0],
    texture: Some(GRASS_TEXTURE),
    hotbar_texture: Some(GRASS_HOTBAR_TEXTURE),
    break_hp: 0.65,
    is_opaque: true,
};

pub const STONE_PROPERTIES: BlockProperties = BlockProperties {
    id: STONE_BLOCK,
    label: "STONE",
    color: [0.48, 0.49, 0.50, 1.0],
    texture: Some(STONE_TEXTURE),
    hotbar_texture: Some(STONE_HOTBAR_TEXTURE),
    break_hp: 2.4,
    is_opaque: true,
};

pub const COAL_ORE_PROPERTIES: BlockProperties = BlockProperties {
    id: COAL_ORE_BLOCK,
    label: "COAL ORE",
    color: [0.11, 0.12, 0.12, 1.0],
    texture: Some(COAL_ORE_TEXTURE),
    hotbar_texture: Some(COAL_ORE_HOTBAR_TEXTURE),
    break_hp: 2.8,
    is_opaque: true,
};

pub const IRON_ORE_PROPERTIES: BlockProperties = BlockProperties {
    id: IRON_ORE_BLOCK,
    label: "IRON ORE",
    color: [0.72, 0.48, 0.31, 1.0],
    texture: Some(IRON_ORE_TEXTURE),
    hotbar_texture: Some(IRON_ORE_HOTBAR_TEXTURE),
    break_hp: 3.2,
    is_opaque: true,
};

pub const UNKNOWN_BLOCK_PROPERTIES: BlockProperties = BlockProperties {
    id: u16::MAX,
    label: "UNKNOWN",
    color: [0.74, 0.22, 0.72, 1.0],
    texture: None,
    hotbar_texture: None,
    break_hp: 1.0,
    is_opaque: true,
};

pub const BLOCK_PROPERTIES: [BlockProperties; 6] = [
    AIR_PROPERTIES,
    DIRT_PROPERTIES,
    GRASS_PROPERTIES,
    STONE_PROPERTIES,
    COAL_ORE_PROPERTIES,
    IRON_ORE_PROPERTIES,
];

pub fn block_properties(block: BlockId) -> BlockProperties {
    BLOCK_PROPERTIES
        .get(block as usize)
        .copied()
        .filter(|properties| properties.id == block)
        .unwrap_or(UNKNOWN_BLOCK_PROPERTIES)
}

pub fn block_label(block: BlockId) -> &'static str {
    block_properties(block).label
}

pub fn block_color_rgba(block: BlockId) -> [f32; 4] {
    block_properties(block).color
}

pub fn block_color_rgb(block: BlockId) -> [f32; 3] {
    let color = block_color_rgba(block);
    [color[0], color[1], color[2]]
}

pub fn block_texture(block: BlockId) -> Option<BlockTextureRegion> {
    block_properties(block).texture
}

pub fn block_has_texture(block: BlockId) -> bool {
    block_texture(block).is_some()
}

pub fn block_hotbar_texture(block: BlockId) -> Option<BlockTextureRegion> {
    block_properties(block).hotbar_texture
}

pub fn block_has_hotbar_texture(block: BlockId) -> bool {
    block_hotbar_texture(block).is_some()
}

pub fn block_hotbar_uvs(block: BlockId) -> Option<[[f32; 2]; 4]> {
    block_hotbar_texture(block).map(|texture| texture_face_uvs(texture, CubeFace::PosY))
}

pub fn block_face_uv(block: BlockId, face: CubeFace, local_uv: [f32; 2]) -> Option<[f32; 2]> {
    block_texture(block).map(|texture| texture.face_uv(face, local_uv))
}

pub fn texture_region_uvs(region: BlockTextureRegion) -> [[f32; 2]; 4] {
    [
        region.uv([0.0, 0.0]),
        region.uv([1.0, 0.0]),
        region.uv([1.0, 1.0]),
        region.uv([0.0, 1.0]),
    ]
}

pub fn texture_face_uvs(region: BlockTextureRegion, face: CubeFace) -> [[f32; 2]; 4] {
    [
        region.face_uv(face, [0.0, 0.0]),
        region.face_uv(face, [1.0, 0.0]),
        region.face_uv(face, [1.0, 1.0]),
        region.face_uv(face, [0.0, 1.0]),
    ]
}

pub fn block_break_hp(block: BlockId) -> f32 {
    block_properties(block).break_hp
}

pub fn block_is_opaque(block: BlockId) -> bool {
    block_properties(block).is_opaque
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_block_properties_are_lookup_table_backed() {
        assert_eq!(block_label(STONE_BLOCK), "STONE");
        assert_eq!(block_color_rgba(DIRT_BLOCK), DIRT_PROPERTIES.color);
        assert_eq!(block_break_hp(IRON_ORE_BLOCK), IRON_ORE_PROPERTIES.break_hp);
        assert!(block_has_texture(STONE_BLOCK));
        assert!(block_has_hotbar_texture(STONE_BLOCK));
        assert!(block_is_opaque(GRASS_BLOCK));
    }

    #[test]
    fn air_is_not_opaque_or_breakable() {
        assert_eq!(block_break_hp(AIR_BLOCK), 0.0);
        assert!(!block_has_texture(AIR_BLOCK));
        assert!(!block_has_hotbar_texture(AIR_BLOCK));
        assert!(!block_is_opaque(AIR_BLOCK));
    }

    #[test]
    fn unknown_blocks_have_debug_properties() {
        let properties = block_properties(65000);

        assert_eq!(properties.label, "UNKNOWN");
        assert!(properties.is_opaque);
        assert!(properties.break_hp > 0.0);
    }

    #[test]
    fn cross_texture_uvs_use_expected_face_tiles() {
        let uv = block_face_uv(STONE_BLOCK, CubeFace::PosZ, [0.5, 0.5])
            .expect("stone should have texture coordinates");
        assert!((uv[0] - 0.5).abs() < 0.0001);
        assert!((uv[1] - 0.1875).abs() < 0.0001);
    }

    #[test]
    fn hotbar_texture_uvs_use_second_atlas_row() {
        let uvs = block_hotbar_uvs(STONE_BLOCK).expect("stone should have hotbar uvs");

        assert!((uvs[0][0] - 0.46666667).abs() < 0.0001);
        assert!((uvs[0][1] - 0.5).abs() < 0.0001);
        assert!((uvs[2][0] - 0.53333336).abs() < 0.0001);
        assert!((uvs[2][1] - 0.625).abs() < 0.0001);
    }

    #[test]
    fn hotbar_texture_uses_top_square_of_cube_cross() {
        let hotbar = block_hotbar_texture(STONE_BLOCK).expect("stone should have hotbar texture");
        let expected_top_face = texture_face_uvs(hotbar, CubeFace::PosY);

        assert_eq!(block_hotbar_uvs(STONE_BLOCK), Some(expected_top_face));
    }

    #[test]
    fn texture_offsets_move_face_uvs_inside_bounds() {
        let texture =
            BlockTextureRegion::new(TextureBounds::new([0.1, 0.2], [0.4, 0.3]), [0.01, 0.02]);
        let uv = texture.face_uv(CubeFace::NegY, [1.0, 1.0]);

        assert!((uv[0] - 0.37666667).abs() < 0.0001);
        assert!((uv[1] - 0.445).abs() < 0.0001);
    }

    #[test]
    fn block_texture_indices_map_to_expected_pixel_columns() {
        let dirt = DIRT_TEXTURE;
        let grass = GRASS_TEXTURE;
        let stone = STONE_TEXTURE;
        let coal = COAL_ORE_TEXTURE;
        let iron = IRON_ORE_TEXTURE;

        assert!((dirt.bounds.min[0] - 0.0).abs() < 0.0001);
        assert!((grass.bounds.min[0] - 0.2).abs() < 0.0001);
        assert!((stone.bounds.min[0] - 0.4).abs() < 0.0001);
        assert!((coal.bounds.min[0] - 0.6).abs() < 0.0001);
        assert!((iron.bounds.min[0] - 0.8).abs() < 0.0001);
        assert!((dirt.bounds.min[1] - 0.0).abs() < 0.0001);

        assert!((dirt.bounds.size[0] - 0.2).abs() < 0.0001);
        assert!((dirt.bounds.size[1] - 0.5).abs() < 0.0001);
    }
}
