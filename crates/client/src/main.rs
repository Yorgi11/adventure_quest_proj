use std::{
    collections::VecDeque,
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};

use foundation::{BlockPos, ChunkCoord};
use gameplay::{
    break_target_block, place_selected_block, BlockInteraction, Hotbar, HOTBAR_SLOT_COUNT,
};
use meshing::{mesh_chunk, mesh_dirty_subchunks, MeshData};
use physics::raycast_blocks;
use renderer::{ChunkMeshUpload, ClearRenderer, RendererOptions, VoxelCamera};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalPosition, PhysicalSize},
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};
use world::{VoxelWorld, WorldStreamingSettings};

fn main() -> Result<(), Box<dyn Error>> {
    let options = ClientOptions::from_args(std::env::args().skip(1));

    run_voxel_prototype();

    if options.no_window {
        return Ok(());
    }

    run_window(options)
}

fn run_window(options: ClientOptions) -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = AdventureQuestApp::new(options);

    event_loop.run_app(&mut app)?;

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClientOptions {
    no_window: bool,
    show_fps: bool,
    renderer: RendererOptions,
}

impl ClientOptions {
    fn from_args<I>(args: I) -> Self
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut options = Self::default();

        for arg in args {
            match arg.as_ref() {
                "--no-window" => options.no_window = true,
                "--fps-window" => options.show_fps = true,
                "--no-vsync" => options.renderer.vsync = false,
                _ => {}
            }
        }

        options
    }
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            no_window: false,
            show_fps: false,
            renderer: RendererOptions::default(),
        }
    }
}

struct AdventureQuestApp {
    window: Option<Arc<Window>>,
    renderer: Option<ClearRenderer>,
    input: InputState,
    camera: VoxelCamera,
    last_frame: Option<Instant>,
    show_fps: bool,
    renderer_options: RendererOptions,
    fps_counter: FpsCounter,
    world: Option<VoxelWorld>,
    hotbar: Hotbar,
    mouse_look_active: bool,
    dirty_mesh_queue: DirtyMeshQueue,
}

impl AdventureQuestApp {
    fn new(options: ClientOptions) -> Self {
        Self {
            window: None,
            renderer: None,
            input: InputState::default(),
            camera: VoxelCamera::looking_at_chunk_origin(),
            last_frame: None,
            show_fps: options.show_fps,
            renderer_options: options.renderer,
            fps_counter: FpsCounter::default(),
            world: None,
            hotbar: Hotbar::starter(),
            mouse_look_active: false,
            dirty_mesh_queue: DirtyMeshQueue::default(),
        }
    }
}

impl ApplicationHandler for AdventureQuestApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attributes = Window::default_attributes()
            .with_title("Adventure Quest")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));

        let window = match event_loop.create_window(window_attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("Failed to create window: {error}");
                event_loop.exit();
                return;
            }
        };

        window.focus_window();
        self.capture_mouse(&window);

        let mut renderer = match pollster::block_on(ClearRenderer::new_with_options(
            window.clone(),
            self.renderer_options,
        )) {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("Failed to initialize renderer: {error}");
                event_loop.exit();
                return;
            }
        };

        renderer.set_camera(self.camera);
        renderer.set_crosshair_enabled(true);
        if self.show_fps {
            renderer.set_fps_overlay(Some(0));
        }

        let window_world = build_window_world();

        if window_world.meshes.is_empty() {
            eprintln!("No chunk meshes were available for the first rendered frame");
        } else {
            renderer.upload_chunk_meshes(window_world.meshes.iter().map(WindowChunkMesh::upload));
            println!(
                "Uploaded {} chunk meshes to renderer ({} indices)",
                renderer.mesh_count(),
                renderer.index_count()
            );
        }

        self.world = Some(window_world.world);
        self.renderer = Some(renderer);
        self.window = Some(window.clone());
        window.request_redraw();
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let DeviceEvent::MouseMotion { delta } = event else {
            return;
        };

        self.apply_mouse_look(delta);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.clone() else {
            return;
        };

        if window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Focused(focused) => {
                if focused {
                    self.capture_mouse(&window);
                } else {
                    self.mouse_look_active = false;
                    window.set_cursor_visible(true);
                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;

                    if key == KeyCode::Escape && pressed {
                        event_loop.exit();
                    } else if pressed && self.select_hotbar_key(key) {
                    } else {
                        self.input.set_key(key, pressed);
                    }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                self.capture_mouse(&window);

                match button {
                    MouseButton::Left => self.interact_with_target(BlockAction::Break),
                    MouseButton::Right => self.interact_with_target(BlockAction::Place),
                    _ => {}
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size);
                }
            }
            WindowEvent::RedrawRequested => {
                self.advance_camera();
                self.process_dirty_mesh_queue();

                let fps_update = if self.show_fps {
                    self.fps_counter.record_frame(Instant::now())
                } else {
                    None
                };

                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.set_camera(self.camera);
                    if let Some(fps) = fps_update {
                        renderer.set_fps_overlay(Some(fps));
                    }
                    let _status = renderer.render();
                }

                window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

impl AdventureQuestApp {
    fn advance_camera(&mut self) {
        let now = Instant::now();
        let dt = self
            .last_frame
            .replace(now)
            .map(|last_frame| now.duration_since(last_frame).as_secs_f32().min(0.05))
            .unwrap_or(0.0);

        if dt == 0.0 {
            return;
        }

        let movement_speed = if self.input.fast { 42.0 } else { 18.0 };

        let forward = self.input.forward_axis() * movement_speed * dt;
        let right = self.input.right_axis() * movement_speed * dt;
        let up = self.input.up_axis() * movement_speed * dt;

        self.camera.translate_local(forward, right, up);
    }

    fn apply_mouse_look(&mut self, delta: (f64, f64)) {
        if !self.mouse_look_active {
            return;
        }

        let yaw = delta.0 as f32 * MOUSE_SENSITIVITY;
        let pitch = -(delta.1 as f32) * MOUSE_SENSITIVITY;

        self.camera.rotate(yaw, pitch);
    }

    fn capture_mouse(&mut self, window: &Window) {
        window.focus_window();
        let _ = window.set_cursor_position(window_center_position(window.inner_size()));
        window.set_cursor_visible(false);

        match window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
        {
            Ok(()) => {
                let _ = window.set_cursor_position(window_center_position(window.inner_size()));
                self.mouse_look_active = true;
            }
            Err(error) => {
                self.mouse_look_active = false;
                window.set_cursor_visible(true);
                eprintln!("Mouse capture unavailable: {error}");
            }
        }
    }

    fn select_hotbar_key(&mut self, key: KeyCode) -> bool {
        let Some(slot) = hotbar_slot_for_key(key) else {
            return false;
        };

        if self.hotbar.select_slot(slot) {
            println!(
                "Selected hotbar slot {} (block id {})",
                slot + 1,
                self.hotbar.selected_block()
            );
        }

        true
    }

    fn interact_with_target(&mut self, action: BlockAction) {
        let Some(world) = self.world.as_mut() else {
            return;
        };

        let origin = self.camera.position;
        let direction = self.camera.forward_direction();
        let result = match action {
            BlockAction::Break => break_target_block(world, origin, direction, PLAYER_REACH),
            BlockAction::Place => {
                place_selected_block(world, &self.hotbar, origin, direction, PLAYER_REACH)
            }
        };
        let changed_blocks = changed_block_count(result);

        print_window_interaction(result);

        if changed_blocks == 0 {
            return;
        }

        self.dirty_mesh_queue.enqueue_dirty(world);
    }

    fn process_dirty_mesh_queue(&mut self) {
        let Some(world) = self.world.as_mut() else {
            return;
        };
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        self.dirty_mesh_queue.enqueue_dirty(world);

        for _ in 0..DIRTY_MESH_UPLOADS_PER_FRAME {
            let Some(coord) = self.dirty_mesh_queue.pop_dirty(world) else {
                break;
            };

            upload_window_chunk_mesh(world, renderer, coord);
        }
    }
}

fn window_center_position(size: PhysicalSize<u32>) -> PhysicalPosition<f64> {
    PhysicalPosition::new(size.width as f64 * 0.5, size.height as f64 * 0.5)
}

const PLAYER_REACH: f32 = 8.0;
const MOUSE_SENSITIVITY: f32 = 0.0025;
const DIRTY_MESH_UPLOADS_PER_FRAME: usize = 1;

#[derive(Debug, Clone, Copy)]
enum BlockAction {
    Break,
    Place,
}

#[derive(Debug, Default)]
struct FpsCounter {
    sample_start: Option<Instant>,
    frames_in_sample: u32,
}

impl FpsCounter {
    const SAMPLE_DURATION: Duration = Duration::from_millis(250);

    fn record_frame(&mut self, now: Instant) -> Option<u32> {
        let sample_start = self.sample_start.get_or_insert(now);
        self.frames_in_sample += 1;

        let elapsed = now.duration_since(*sample_start);
        if elapsed < Self::SAMPLE_DURATION {
            return None;
        }

        let fps = (self.frames_in_sample as f64 / elapsed.as_secs_f64()).round() as u32;
        self.sample_start = Some(now);
        self.frames_in_sample = 0;

        Some(fps)
    }
}

#[derive(Debug, Default)]
struct DirtyMeshQueue {
    pending: VecDeque<ChunkCoord>,
}

impl DirtyMeshQueue {
    fn enqueue_dirty(&mut self, world: &VoxelWorld) {
        for coord in dirty_window_chunk_coords(world) {
            self.enqueue(coord);
        }
    }

    fn enqueue(&mut self, coord: ChunkCoord) {
        if !self.pending.contains(&coord) {
            self.pending.push_back(coord);
        }
    }

    fn pop_dirty(&mut self, world: &VoxelWorld) -> Option<ChunkCoord> {
        while let Some(coord) = self.pending.pop_front() {
            let is_dirty = world
                .get_chunk(coord)
                .map(|chunk| chunk.subchunk_dirty_mask != 0)
                .unwrap_or(false);

            if is_dirty {
                return Some(coord);
            }
        }

        None
    }

    #[cfg(test)]
    fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    fast: bool,
}

impl InputState {
    fn set_key(&mut self, key: KeyCode, pressed: bool) {
        match key {
            KeyCode::KeyW => self.forward = pressed,
            KeyCode::KeyS => self.backward = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::ControlLeft | KeyCode::ControlRight => self.down = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.fast = pressed,
            _ => {}
        }
    }

    fn forward_axis(self) -> f32 {
        axis(self.forward, self.backward)
    }

    fn right_axis(self) -> f32 {
        axis(self.right, self.left)
    }

    fn up_axis(self) -> f32 {
        axis(self.up, self.down)
    }
}

fn axis(positive: bool, negative: bool) -> f32 {
    match (positive, negative) {
        (true, false) => 1.0,
        (false, true) => -1.0,
        _ => 0.0,
    }
}

fn hotbar_slot_for_key(key: KeyCode) -> Option<usize> {
    let slot = match key {
        KeyCode::Digit1 => 0,
        KeyCode::Digit2 => 1,
        KeyCode::Digit3 => 2,
        KeyCode::Digit4 => 3,
        KeyCode::Digit5 => 4,
        KeyCode::Digit6 => 5,
        KeyCode::Digit7 => 6,
        KeyCode::Digit8 => 7,
        KeyCode::Digit9 => 8,
        _ => return None,
    };

    (slot < HOTBAR_SLOT_COUNT).then_some(slot)
}

struct WindowWorldState {
    world: VoxelWorld,
    meshes: Vec<WindowChunkMesh>,
}

struct WindowChunkMesh {
    coord: ChunkCoord,
    revision: u32,
    visible_mask: u8,
    mesh: MeshData,
}

impl WindowChunkMesh {
    fn upload(&self) -> ChunkMeshUpload<'_> {
        ChunkMeshUpload {
            coord: self.coord,
            revision: self.revision,
            visible_mask: self.visible_mask,
            mesh: &self.mesh,
        }
    }
}

fn build_window_world() -> WindowWorldState {
    let mut world = VoxelWorld::new(12345);
    let player_block = BlockPos::new(0, 40, 0);

    world.load_chunks_around_block(player_block, WorldStreamingSettings::prototype());

    let meshes = build_window_meshes(&mut world);

    WindowWorldState { world, meshes }
}

fn build_window_meshes(world: &mut VoxelWorld) -> Vec<WindowChunkMesh> {
    let mut coords: Vec<ChunkCoord> = world.chunks.keys().copied().collect();
    coords.sort_by_key(|coord| (coord.y, coord.z, coord.x));

    let mut meshes = Vec::new();

    for coord in coords {
        let Some(mesh) = mesh_chunk(&world, coord) else {
            continue;
        };

        let revision = world
            .get_chunk(coord)
            .map(|chunk| chunk.revision)
            .unwrap_or_default();
        let visible_mask = mesh.subchunk_visible_mask;

        if let Some(chunk) = world.get_chunk_mut(coord) {
            chunk.subchunk_visible_mask = visible_mask;
            chunk.clear_dirty();
        }

        if !mesh.indices.is_empty() {
            meshes.push(WindowChunkMesh {
                coord,
                revision,
                visible_mask,
                mesh,
            });
        }
    }

    meshes
}

fn dirty_window_chunk_coords(world: &VoxelWorld) -> Vec<ChunkCoord> {
    let mut coords: Vec<ChunkCoord> = world
        .chunks
        .iter()
        .filter_map(|(coord, chunk)| (chunk.subchunk_dirty_mask != 0).then_some(*coord))
        .collect();

    coords.sort_by_key(|coord| (coord.y, coord.z, coord.x));
    coords
}

fn upload_window_chunk_mesh(
    world: &mut VoxelWorld,
    renderer: &mut ClearRenderer,
    coord: ChunkCoord,
) {
    let Some(mesh) = mesh_chunk(world, coord) else {
        renderer.remove_chunk_mesh(coord);
        return;
    };
    let revision = world
        .get_chunk(coord)
        .map(|chunk| chunk.revision)
        .unwrap_or_default();
    let visible_mask = mesh.subchunk_visible_mask;

    if let Some(chunk) = world.get_chunk_mut(coord) {
        chunk.subchunk_visible_mask = visible_mask;
        chunk.clear_dirty();
    }

    renderer.upload_chunk_mesh(ChunkMeshUpload {
        coord,
        revision,
        visible_mask,
        mesh: &mesh,
    });
}

fn print_window_interaction(interaction: BlockInteraction) {
    match interaction {
        BlockInteraction::Break { hit, summary } => {
            if summary.changed_blocks > 0 {
                println!("Broke block {:?} ({})", hit.world_block, hit.block_id);
            }
        }
        BlockInteraction::Place {
            placed_block,
            block,
            summary,
            ..
        } => {
            if summary.changed_blocks > 0 {
                println!("Placed block {block} at {:?}", placed_block);
            }
        }
        BlockInteraction::Miss => println!("No block in reach"),
        BlockInteraction::NoPlaceableBlockSelected => println!("Selected hotbar slot is empty"),
        BlockInteraction::InvalidPlacementFace { .. } => println!("Cannot place from inside block"),
    }
}

fn run_voxel_prototype() {
    println!("Adventure Quest - Rust Voxel Prototype");

    let mut world = VoxelWorld::new(12345);
    let player_block = BlockPos::new(0, 40, 0);
    let streaming_settings = WorldStreamingSettings::prototype();

    println!("Streaming chunks...");
    let streaming_update = world.load_chunks_around_block(player_block, streaming_settings);

    println!(
        "Loaded chunks: {} (new: {}, requested: {}, center: {:?})",
        streaming_update.total_loaded,
        streaming_update.newly_loaded,
        streaming_update.requested_loads,
        streaming_update.center_chunk
    );

    let ray_origin = [0.5, 40.0, 0.5];
    let ray_direction = [0.0, -1.0, 0.0];

    if let Some(hit) = raycast_blocks(&world, ray_origin, ray_direction, 80.0) {
        println!("Raycast hit:");
        println!("  block: {:?}", hit.world_block);
        println!("  face normal: {:?}", hit.face_normal);
        println!("  placement block: {:?}", hit.placement_block());
        println!("  distance: {:.2}", hit.distance);
    }

    let target_chunk = ChunkCoord::new(0, 0, 0);

    let mesh = mesh_chunk(&world, target_chunk).expect("Target chunk should exist");

    println!("Chunk {:?} mesh:", target_chunk);
    println!("  vertices: {}", mesh.vertices.len());
    println!("  indices: {}", mesh.indices.len());
    println!("  triangles: {}", mesh.triangle_count());
    println!("  visible faces: {}", mesh.visible_face_count);
    println!("  visible subchunks: {:08b}", mesh.subchunk_visible_mask);

    println!("Editing block...");

    let mut changed_blocks = 0;
    let mut hotbar = Hotbar::starter();
    hotbar.select_slot(1);

    let break_result = break_target_block(&mut world, ray_origin, ray_direction, 80.0);
    changed_blocks += changed_block_count(break_result);

    let place_result = place_selected_block(&mut world, &hotbar, ray_origin, ray_direction, 80.0);
    changed_blocks += changed_block_count(place_result);

    let dirty_meshes =
        mesh_dirty_subchunks(&world, target_chunk).expect("Target chunk should exist");

    let edited_mesh = mesh_chunk(&world, target_chunk).expect("Target chunk should exist");

    println!("After edit:");
    println!("  selected hotbar slot: {}", hotbar.selected_slot());
    println!("  selected block id: {}", hotbar.selected_block());
    println!("  changed blocks: {}", changed_blocks);
    println!("  dirty subchunk meshes: {}", dirty_meshes.len());
    for (subchunk, mesh) in &dirty_meshes {
        println!(
            "    subchunk {subchunk}: {} visible faces",
            mesh.visible_face_count
        );
    }
    println!("  vertices: {}", edited_mesh.vertices.len());
    println!("  indices: {}", edited_mesh.indices.len());
    println!("  triangles: {}", edited_mesh.triangle_count());
    println!("  visible faces: {}", edited_mesh.visible_face_count);
    println!(
        "  visible subchunks: {:08b}",
        edited_mesh.subchunk_visible_mask
    );

    world.set_chunk_visible_mask(target_chunk, edited_mesh.subchunk_visible_mask);

    if let Some(chunk) = world.get_chunk(target_chunk) {
        println!("Chunk revision: {}", chunk.revision);
        println!("Chunk dirty mask: {:08b}", chunk.subchunk_dirty_mask);
        println!("Chunk visible mask: {:08b}", chunk.subchunk_visible_mask);
    }
}

fn changed_block_count(interaction: BlockInteraction) -> usize {
    match interaction {
        BlockInteraction::Break { summary, .. } | BlockInteraction::Place { summary, .. } => {
            summary.changed_blocks
        }
        BlockInteraction::Miss
        | BlockInteraction::NoPlaceableBlockSelected
        | BlockInteraction::InvalidPlacementFace { .. } => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxels::{Chunk, STONE_BLOCK};

    fn world_with_dirty_chunk(coord: ChunkCoord) -> VoxelWorld {
        let mut world = VoxelWorld::new(0);
        let mut chunk = Chunk::new_empty(coord);
        chunk.set_block(0, 0, 0, STONE_BLOCK);
        world.chunks.insert(coord, chunk);
        world
    }

    #[test]
    fn dirty_mesh_queue_deduplicates_dirty_chunks() {
        let coord = ChunkCoord::new(0, 0, 0);
        let world = world_with_dirty_chunk(coord);
        let mut queue = DirtyMeshQueue::default();

        queue.enqueue_dirty(&world);
        queue.enqueue_dirty(&world);

        assert_eq!(queue.pending_count(), 1);
        assert_eq!(queue.pop_dirty(&world), Some(coord));
        assert_eq!(queue.pop_dirty(&world), None);
    }

    #[test]
    fn dirty_mesh_queue_skips_chunks_that_are_no_longer_dirty() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_dirty_chunk(coord);
        let mut queue = DirtyMeshQueue::default();

        queue.enqueue_dirty(&world);
        world.get_chunk_mut(coord).unwrap().clear_dirty();

        assert_eq!(queue.pop_dirty(&world), None);
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn window_center_position_uses_inner_size_midpoint() {
        let center = window_center_position(PhysicalSize::new(1280, 720));

        assert_eq!(center, PhysicalPosition::new(640.0, 360.0));
    }

    #[test]
    fn client_options_parse_window_flags() {
        let options = ClientOptions::from_args(["--fps-window", "--no-vsync"]);

        assert!(!options.no_window);
        assert!(options.show_fps);
        assert_eq!(options.renderer, RendererOptions::new(false));
    }

    #[test]
    fn client_options_parse_no_window() {
        let options = ClientOptions::from_args(["--no-window"]);

        assert!(options.no_window);
        assert!(!options.show_fps);
        assert_eq!(options.renderer, RendererOptions::default());
    }
}
