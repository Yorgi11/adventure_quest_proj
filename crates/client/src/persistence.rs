use std::{
    collections::HashMap,
    fs::{self, File},
    io::{self, BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
};

use foundation::ChunkCoord;
use voxels::{BlockId, Chunk, AIR_BLOCK, CHUNK_VOLUME};
#[cfg(test)]
use world::VoxelWorld;

use crate::config;

const CHUNK_SAVE_MAGIC: &[u8; 8] = b"AQCHNK1\0";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SavedSettings {
    pub mouse_sensitivity: f32,
    pub render_chunk_distance: i32,
}

pub fn default_save_dir() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(config::SAVE_ROOT_DIR)
}

pub fn load_settings(save_dir: &Path) -> io::Result<Option<SavedSettings>> {
    let path = settings_path(save_dir);

    if !path.exists() {
        return Ok(None);
    }

    let mut mouse_sensitivity = None;
    let mut render_chunk_distance = None;

    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };

        match key.trim() {
            "mouse_sensitivity" => mouse_sensitivity = value.trim().parse::<f32>().ok(),
            "render_chunk_distance" => render_chunk_distance = value.trim().parse::<i32>().ok(),
            _ => {}
        }
    }

    Ok(Some(SavedSettings {
        mouse_sensitivity: mouse_sensitivity.unwrap_or(config::DEFAULT_MOUSE_SENSITIVITY),
        render_chunk_distance: render_chunk_distance
            .unwrap_or(config::DEFAULT_RENDER_CHUNK_DISTANCE),
    }))
}

pub fn save_settings(save_dir: &Path, settings: SavedSettings) -> io::Result<()> {
    fs::create_dir_all(save_dir)?;

    let mut file = File::create(settings_path(save_dir))?;
    writeln!(file, "mouse_sensitivity={:.6}", settings.mouse_sensitivity)?;
    writeln!(
        file,
        "render_chunk_distance={}",
        settings.render_chunk_distance
    )?;

    Ok(())
}

pub fn load_saved_chunks(save_dir: &Path) -> io::Result<HashMap<ChunkCoord, Chunk>> {
    let path = chunk_save_path(save_dir);

    if !path.exists() {
        return Ok(HashMap::new());
    }

    let mut file = File::open(path)?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;

    if &magic != CHUNK_SAVE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid Adventure Quest chunk save header",
        ));
    }

    let _seed = read_u64(&mut file)?;
    let chunk_count = read_u32(&mut file)?;
    let mut chunks = HashMap::with_capacity(chunk_count as usize);

    for _ in 0..chunk_count {
        let coord = ChunkCoord::new(
            read_i32(&mut file)?,
            read_i32(&mut file)?,
            read_i32(&mut file)?,
        );
        let revision = read_u32(&mut file)?;
        let mut blocks = Box::new([AIR_BLOCK; CHUNK_VOLUME]);

        for block in blocks.iter_mut() {
            *block = read_u16(&mut file)?;
        }

        chunks.insert(coord, Chunk::from_blocks(coord, blocks, revision));
    }

    Ok(chunks)
}

#[cfg(test)]
pub fn save_modified_chunks(save_dir: &Path, world: &VoxelWorld) -> io::Result<usize> {
    fs::create_dir_all(world_save_dir(save_dir))?;

    let modified_chunks: Vec<&Chunk> = world
        .chunks
        .iter()
        .filter_map(|(coord, chunk)| {
            let generated = VoxelWorld::generate_chunk(world.seed(), *coord);
            (chunk.blocks() != generated.blocks()).then_some(chunk)
        })
        .collect();

    write_chunks(save_dir, world.seed(), modified_chunks.iter().copied())?;

    Ok(modified_chunks.len())
}

pub fn save_chunks(
    save_dir: &Path,
    seed: u64,
    chunks: &HashMap<ChunkCoord, Chunk>,
) -> io::Result<usize> {
    write_chunks(save_dir, seed, chunks.values())?;

    Ok(chunks.len())
}

fn write_chunks<'a>(
    save_dir: &Path,
    seed: u64,
    chunks: impl IntoIterator<Item = &'a Chunk>,
) -> io::Result<()> {
    fs::create_dir_all(world_save_dir(save_dir))?;

    let mut chunks: Vec<&Chunk> = chunks.into_iter().collect();
    chunks.sort_by_key(|chunk| (chunk.coord.y, chunk.coord.z, chunk.coord.x));
    let mut file = File::create(chunk_save_path(save_dir))?;
    file.write_all(CHUNK_SAVE_MAGIC)?;
    write_u64(&mut file, seed)?;
    write_u32(&mut file, chunks.len() as u32)?;

    for chunk in chunks {
        write_i32(&mut file, chunk.coord.x)?;
        write_i32(&mut file, chunk.coord.y)?;
        write_i32(&mut file, chunk.coord.z)?;
        write_u32(&mut file, chunk.revision)?;

        for block in chunk.blocks().iter().copied() {
            write_u16(&mut file, block)?;
        }
    }

    Ok(())
}

fn settings_path(save_dir: &Path) -> PathBuf {
    save_dir.join(config::SETTINGS_FILE_NAME)
}

fn world_save_dir(save_dir: &Path) -> PathBuf {
    save_dir.join(config::DEFAULT_WORLD_DIR)
}

fn chunk_save_path(save_dir: &Path) -> PathBuf {
    world_save_dir(save_dir).join(config::CHUNK_SAVE_FILE_NAME)
}

fn read_i32(reader: &mut impl Read) -> io::Result<i32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u16(reader: &mut impl Read) -> io::Result<BlockId> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_i32(writer: &mut impl Write, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u16(writer: &mut impl Write, value: BlockId) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use foundation::BlockPos;
    use voxels::STONE_BLOCK;

    use super::*;

    fn temp_save_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        std::env::temp_dir().join(format!("aq_{name}_{unique}"))
    }

    #[test]
    fn settings_round_trip_to_text_file() {
        let dir = temp_save_dir("settings");
        let settings = SavedSettings {
            mouse_sensitivity: 0.004,
            render_chunk_distance: 4,
        };

        save_settings(&dir, settings).unwrap();

        assert_eq!(load_settings(&dir).unwrap(), Some(settings));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn unmodified_world_saves_no_chunks() {
        let dir = temp_save_dir("unmodified_world");
        let mut world = VoxelWorld::new(99);
        world.load_chunk(ChunkCoord::new(0, 0, 0));

        let saved = save_modified_chunks(&dir, &world).unwrap();
        let chunks = load_saved_chunks(&dir).unwrap();

        assert_eq!(saved, 0);
        assert!(chunks.is_empty());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn modified_chunk_round_trips() {
        let dir = temp_save_dir("modified_world");
        let mut world = VoxelWorld::new(99);
        let pos = BlockPos::new(0, 48, 0);

        world.set_block(pos, STONE_BLOCK);

        let saved = save_modified_chunks(&dir, &world).unwrap();
        let chunks = load_saved_chunks(&dir).unwrap();
        let coord = ChunkCoord::new(0, 1, 0);

        assert_eq!(saved, 1);
        assert_eq!(chunks.get(&coord).unwrap().get_block(0, 16, 0), STONE_BLOCK);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn explicit_saved_chunk_map_round_trips() {
        let dir = temp_save_dir("saved_map");
        let coord = ChunkCoord::new(1, 0, -2);
        let mut chunks = HashMap::new();
        let mut chunk = Chunk::new_empty(coord);
        chunk.set_block(3, 4, 5, STONE_BLOCK);
        chunks.insert(coord, chunk);

        let saved = save_chunks(&dir, 99, &chunks).unwrap();
        let loaded = load_saved_chunks(&dir).unwrap();

        assert_eq!(saved, 1);
        assert_eq!(loaded.get(&coord).unwrap().get_block(3, 4, 5), STONE_BLOCK);

        let _ = fs::remove_dir_all(dir);
    }
}
