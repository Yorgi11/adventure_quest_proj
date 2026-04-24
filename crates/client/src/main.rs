use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};

use foundation::{BlockPos, ChunkCoord};
use gameplay::{break_target_block, place_selected_block, BlockInteraction, Hotbar};
use meshing::{mesh_chunk, mesh_dirty_subchunks, MeshData};
use physics::raycast_blocks;
use renderer::{ChunkMeshUpload, ClearRenderer, VoxelCamera};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
use world::{VoxelWorld, WorldStreamingSettings};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    run_voxel_prototype();

    if args.iter().any(|arg| arg == "--no-window") {
        return Ok(());
    }

    run_window(args.iter().any(|arg| arg == "--fps-window"))
}

fn run_window(show_fps: bool) -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = AdventureQuestApp::new(show_fps);

    event_loop.run_app(&mut app)?;

    Ok(())
}

struct AdventureQuestApp {
    window: Option<Arc<Window>>,
    renderer: Option<ClearRenderer>,
    input: InputState,
    camera: VoxelCamera,
    last_frame: Option<Instant>,
    show_fps: bool,
    fps_counter: FpsCounter,
}

impl AdventureQuestApp {
    fn new(show_fps: bool) -> Self {
        Self {
            window: None,
            renderer: None,
            input: InputState::default(),
            camera: VoxelCamera::looking_at_chunk_origin(),
            last_frame: None,
            show_fps,
            fps_counter: FpsCounter::default(),
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

        let mut renderer = match pollster::block_on(ClearRenderer::new(window.clone())) {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("Failed to initialize renderer: {error}");
                event_loop.exit();
                return;
            }
        };

        renderer.set_camera(self.camera);
        if self.show_fps {
            renderer.set_fps_overlay(Some(0));
        }

        let meshes = build_window_meshes();

        if meshes.is_empty() {
            eprintln!("No chunk meshes were available for the first rendered frame");
        } else {
            renderer.upload_chunk_meshes(meshes.iter().map(WindowChunkMesh::upload));
            println!(
                "Uploaded {} chunk meshes to renderer ({} indices)",
                renderer.mesh_count(),
                renderer.index_count()
            );
        }

        self.renderer = Some(renderer);
        self.window = Some(window.clone());
        window.request_redraw();
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
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;

                    if key == KeyCode::Escape && pressed {
                        event_loop.exit();
                    } else {
                        self.input.set_key(key, pressed);
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size);
                }
            }
            WindowEvent::RedrawRequested => {
                self.advance_camera();

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
        let rotation_speed = 1.75;

        let forward = self.input.forward_axis() * movement_speed * dt;
        let right = self.input.right_axis() * movement_speed * dt;
        let up = self.input.up_axis() * movement_speed * dt;
        let yaw = self.input.yaw_axis() * rotation_speed * dt;
        let pitch = self.input.pitch_axis() * rotation_speed * dt;

        self.camera.translate_local(forward, right, up);
        self.camera.rotate(yaw, pitch);
    }
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

#[derive(Debug, Default, Clone, Copy)]
struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    yaw_left: bool,
    yaw_right: bool,
    pitch_up: bool,
    pitch_down: bool,
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
            KeyCode::ArrowLeft => self.yaw_left = pressed,
            KeyCode::ArrowRight => self.yaw_right = pressed,
            KeyCode::ArrowUp => self.pitch_up = pressed,
            KeyCode::ArrowDown => self.pitch_down = pressed,
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

    fn yaw_axis(self) -> f32 {
        axis(self.yaw_right, self.yaw_left)
    }

    fn pitch_axis(self) -> f32 {
        axis(self.pitch_up, self.pitch_down)
    }
}

fn axis(positive: bool, negative: bool) -> f32 {
    match (positive, negative) {
        (true, false) => 1.0,
        (false, true) => -1.0,
        _ => 0.0,
    }
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

fn build_window_meshes() -> Vec<WindowChunkMesh> {
    let mut world = VoxelWorld::new(12345);
    let player_block = BlockPos::new(0, 40, 0);

    world.load_chunks_around_block(player_block, WorldStreamingSettings::prototype());

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

        world.set_chunk_visible_mask(coord, visible_mask);

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
