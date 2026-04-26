pub const WORLD_SEED: u64 = 12345;

pub const SAVE_ROOT_DIR: &str = "saves";
pub const SETTINGS_FILE_NAME: &str = "settings.cfg";
pub const DEFAULT_WORLD_DIR: &str = "default_world";
pub const CHUNK_SAVE_FILE_NAME: &str = "chunks.aqchunks";

pub const DEFAULT_MOUSE_SENSITIVITY: f32 = 0.0025;
pub const MIN_MOUSE_SENSITIVITY: f32 = 0.0001;
pub const MAX_MOUSE_SENSITIVITY: f32 = 0.05;

pub const DEFAULT_RENDER_CHUNK_DISTANCE: i32 = 4;
pub const MIN_RENDER_CHUNK_DISTANCE: i32 = 5;
pub const MAX_RENDER_CHUNK_DISTANCE: i32 = 30;
pub const MAX_VERTICAL_RENDER_CHUNK_DISTANCE: i32 = 2;

pub const INITIAL_SYNC_CHUNK_RADIUS: i32 = 1;
pub const MAX_ACTIVE_WORLD_CHUNKS: usize = 640;
pub const MAX_RENDERED_CHUNK_MESHES: usize = 512;
pub const MAX_STREAMING_COORDS_PER_CENTER: usize = 640;
pub const MAX_NEW_RENDER_MESHES_QUEUED_PER_FRAME: usize = 3;
pub const MAX_DIRTY_MESH_QUEUE_BACKLOG: usize = 64;
pub const MAX_PENDING_NORMAL_MESH_JOBS: usize = 8;
pub const MAX_PENDING_GENERATION_JOBS: usize = 8;
pub const DIRTY_MESH_JOBS_PER_FRAME: usize = 1;
pub const CHUNK_GENERATION_JOBS_PER_FRAME: usize = 1;
