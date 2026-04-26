use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet, VecDeque},
    error::Error,
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

mod config;
mod persistence;

use foundation::{BlockPos, ChunkCoord};
use gameplay::{
    break_target_block, place_selected_block, BlockInteraction, Hotbar, HOTBAR_SLOT_COUNT,
};
use meshing::{mesh_chunk, mesh_chunk_input, mesh_dirty_subchunks, ChunkMeshInput, MeshData};
use physics::{move_aabb_through_voxels, raycast_blocks};
use renderer::{
    ChunkMeshUpload, ClearRenderer, RendererOptions, UiOverlay, UiOverlayItem, UiRect, UiText,
    VoxelCamera,
};
use voxels::{world_to_chunk_coord, Chunk, SUBCHUNK_COUNT};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalPosition, PhysicalSize},
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};
use world::{VoxelWorld, WorldStreamingSettings};

use persistence::SavedSettings;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppScreen {
    MainMenu,
    WorldList,
    Settings,
    PauseMenu,
    InGame,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuAction {
    Play,
    Settings,
    CreateWorld,
    Back,
    Resume,
    QuitToMenu,
    QuitGame,
    MouseSensitivity,
    ChunkViewDistance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SettingsField {
    MouseSensitivity,
    ChunkViewDistance,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct GameSettings {
    mouse_sensitivity: f32,
    chunk_view_distance: i32,
}

impl GameSettings {
    fn streaming_settings(self) -> WorldStreamingSettings {
        let radius = self.chunk_view_distance.clamp(
            config::MIN_RENDER_CHUNK_DISTANCE,
            config::MAX_RENDER_CHUNK_DISTANCE,
        );
        let vertical_radius = radius.min(config::MAX_VERTICAL_RENDER_CHUNK_DISTANCE);

        WorldStreamingSettings::new(radius, vertical_radius, radius, radius, radius, radius + 1)
    }

    fn from_saved(settings: SavedSettings) -> Self {
        Self {
            mouse_sensitivity: settings
                .mouse_sensitivity
                .clamp(config::MIN_MOUSE_SENSITIVITY, config::MAX_MOUSE_SENSITIVITY),
            chunk_view_distance: settings.render_chunk_distance.clamp(
                config::MIN_RENDER_CHUNK_DISTANCE,
                config::MAX_RENDER_CHUNK_DISTANCE,
            ),
        }
    }

    fn saved(self) -> SavedSettings {
        SavedSettings {
            mouse_sensitivity: self.mouse_sensitivity,
            render_chunk_distance: self.chunk_view_distance,
        }
    }
}

impl Default for GameSettings {
    fn default() -> Self {
        Self {
            mouse_sensitivity: config::DEFAULT_MOUSE_SENSITIVITY,
            chunk_view_distance: config::DEFAULT_RENDER_CHUNK_DISTANCE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SettingsInputs {
    mouse_sensitivity: String,
    chunk_view_distance: String,
}

impl SettingsInputs {
    fn from_settings(settings: GameSettings) -> Self {
        Self {
            mouse_sensitivity: format!("{:.4}", settings.mouse_sensitivity),
            chunk_view_distance: settings.chunk_view_distance.to_string(),
        }
    }
}

struct AdventureQuestApp {
    window: Option<Arc<Window>>,
    renderer: Option<ClearRenderer>,
    screen: AppScreen,
    input: InputState,
    camera: VoxelCamera,
    last_frame: Option<Instant>,
    show_fps: bool,
    renderer_options: RendererOptions,
    fps_counter: FpsCounter,
    world: Option<VoxelWorld>,
    player: PlayerController,
    hotbar: Hotbar,
    noclip_enabled: bool,
    mouse_look_active: bool,
    mouse_position: PhysicalPosition<f64>,
    settings: GameSettings,
    settings_inputs: SettingsInputs,
    active_settings_field: Option<SettingsField>,
    settings_back_screen: AppScreen,
    save_dir: PathBuf,
    saved_chunks: HashMap<ChunkCoord, Chunk>,
    modified_chunk_coords: HashSet<ChunkCoord>,
    world_save_dirty: bool,
    dirty_mesh_queue: DirtyMeshQueue,
    streaming: ChunkStreamingState,
    chunk_jobs: ChunkJobQueue,
}

impl AdventureQuestApp {
    fn new(options: ClientOptions) -> Self {
        let save_dir = persistence::default_save_dir();
        let settings = load_saved_game_settings(&save_dir);
        let saved_chunks = load_saved_world_chunks(&save_dir);

        Self {
            window: None,
            renderer: None,
            screen: AppScreen::MainMenu,
            input: InputState::default(),
            camera: VoxelCamera::looking_at_chunk_origin(),
            last_frame: None,
            show_fps: options.show_fps,
            renderer_options: options.renderer,
            fps_counter: FpsCounter::default(),
            world: None,
            player: PlayerController::from_camera(VoxelCamera::looking_at_chunk_origin()),
            hotbar: Hotbar::starter(),
            noclip_enabled: false,
            mouse_look_active: false,
            mouse_position: PhysicalPosition::new(0.0, 0.0),
            settings,
            settings_inputs: SettingsInputs::from_settings(settings),
            active_settings_field: None,
            settings_back_screen: AppScreen::MainMenu,
            save_dir,
            saved_chunks,
            modified_chunk_coords: HashSet::new(),
            world_save_dirty: false,
            dirty_mesh_queue: DirtyMeshQueue::default(),
            streaming: ChunkStreamingState::default(),
            chunk_jobs: ChunkJobQueue::default(),
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
        renderer.set_crosshair_enabled(false);
        if self.show_fps {
            renderer.set_fps_overlay(Some(0));
        }
        renderer.set_ui_overlay(
            build_menu_layout(
                self.screen,
                &self.settings_inputs,
                self.active_settings_field,
                renderer.size(),
            )
            .overlay,
        );

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
            WindowEvent::CloseRequested => {
                if self.screen == AppScreen::Settings {
                    self.apply_settings_from_inputs();
                } else {
                    self.save_settings();
                }
                self.save_current_world_if_dirty();
                event_loop.exit();
            }
            WindowEvent::Focused(focused) => {
                if focused {
                    if self.screen == AppScreen::InGame {
                        self.capture_mouse(&window);
                    }
                } else {
                    self.release_mouse(&window);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_position = position;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if self.screen != AppScreen::InGame {
                    self.handle_menu_keyboard(event_loop, &window, &event);
                    return;
                }

                if let PhysicalKey::Code(key) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;

                    if key == KeyCode::Escape && pressed {
                        self.open_pause_menu(&window);
                    } else if key == KeyCode::KeyV && pressed {
                        self.toggle_noclip();
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
                if self.screen == AppScreen::InGame {
                    self.capture_mouse(&window);

                    match button {
                        MouseButton::Left => self.interact_with_target(BlockAction::Break),
                        MouseButton::Right => self.interact_with_target(BlockAction::Place),
                        _ => {}
                    }
                } else if button == MouseButton::Left {
                    self.handle_menu_click(event_loop, &window);
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size);
                }
                self.refresh_menu_overlay();
            }
            WindowEvent::RedrawRequested => {
                if self.screen == AppScreen::InGame {
                    self.advance_player();
                    self.process_completed_chunk_jobs();
                    self.process_chunk_streaming();
                    self.schedule_dirty_mesh_jobs();
                } else {
                    self.last_frame = None;
                }

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
    fn advance_player(&mut self) {
        let now = Instant::now();
        let dt = self
            .last_frame
            .replace(now)
            .map(|last_frame| now.duration_since(last_frame).as_secs_f32().min(0.05))
            .unwrap_or(0.0);

        if dt == 0.0 {
            return;
        }

        if self.noclip_enabled {
            self.advance_noclip_camera(dt);
            self.player = PlayerController::from_camera(self.camera);
            return;
        }

        let Some(world) = self.world.as_ref() else {
            return;
        };

        self.player
            .step(world, self.camera.yaw_radians, self.input, dt);
        self.camera.position = self.player.camera_position();
    }

    fn advance_noclip_camera(&mut self, dt: f32) {
        let movement_speed = if self.input.fast { 42.0 } else { 18.0 };

        let forward = self.input.forward_axis() * movement_speed * dt;
        let right = self.input.right_axis() * movement_speed * dt;
        let up = self.input.up_axis() * movement_speed * dt;

        self.camera.translate_local(forward, right, up);
    }

    fn apply_mouse_look(&mut self, delta: (f64, f64)) {
        if self.screen != AppScreen::InGame || !self.mouse_look_active {
            return;
        }

        let yaw = delta.0 as f32 * self.settings.mouse_sensitivity;
        let pitch = -(delta.1 as f32) * self.settings.mouse_sensitivity;

        self.camera.rotate(yaw, pitch);
    }

    fn capture_mouse(&mut self, window: &Window) {
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

    fn release_mouse(&mut self, window: &Window) {
        self.mouse_look_active = false;
        window.set_cursor_visible(true);
        let _ = window.set_cursor_grab(CursorGrabMode::None);
    }

    fn handle_menu_click(&mut self, event_loop: &ActiveEventLoop, window: &Window) {
        let Some(size) = self.renderer.as_ref().map(ClearRenderer::size) else {
            return;
        };
        let layout = build_menu_layout(
            self.screen,
            &self.settings_inputs,
            self.active_settings_field,
            size,
        );

        let Some(action) = layout
            .hits
            .iter()
            .find(|hit| hit.rect.contains(self.mouse_position))
            .map(|hit| hit.action)
        else {
            if self.screen == AppScreen::Settings {
                self.active_settings_field = None;
                self.refresh_menu_overlay();
            }
            return;
        };

        match action {
            MenuAction::Play => {
                self.screen = AppScreen::WorldList;
                self.active_settings_field = None;
                self.release_mouse(window);
                self.refresh_menu_overlay();
            }
            MenuAction::Settings => {
                self.settings_back_screen = if self.screen == AppScreen::PauseMenu {
                    AppScreen::PauseMenu
                } else {
                    AppScreen::MainMenu
                };
                self.screen = AppScreen::Settings;
                self.settings_inputs = SettingsInputs::from_settings(self.settings);
                self.active_settings_field = Some(SettingsField::MouseSensitivity);
                self.release_mouse(window);
                self.refresh_menu_overlay();
            }
            MenuAction::CreateWorld => {
                self.apply_settings_from_inputs();
                self.start_existing_world(window);
            }
            MenuAction::Back => {
                if self.screen == AppScreen::Settings {
                    self.apply_settings_from_inputs();
                    self.screen = self.settings_back_screen;
                } else {
                    self.screen = AppScreen::MainMenu;
                }
                self.active_settings_field = None;
                self.release_mouse(window);
                self.refresh_menu_overlay();
            }
            MenuAction::Resume => {
                self.resume_game(window);
            }
            MenuAction::QuitToMenu => {
                self.quit_to_main_menu(window);
            }
            MenuAction::QuitGame => {
                self.save_settings();
                self.save_current_world_if_dirty();
                event_loop.exit();
            }
            MenuAction::MouseSensitivity => {
                self.active_settings_field = Some(SettingsField::MouseSensitivity);
                self.refresh_menu_overlay();
            }
            MenuAction::ChunkViewDistance => {
                self.active_settings_field = Some(SettingsField::ChunkViewDistance);
                self.refresh_menu_overlay();
            }
        }
    }

    fn handle_menu_keyboard(
        &mut self,
        event_loop: &ActiveEventLoop,
        window: &Window,
        event: &KeyEvent,
    ) {
        let pressed = event.state == ElementState::Pressed;

        if self.screen == AppScreen::Settings && self.handle_settings_keyboard(event) {
            self.refresh_menu_overlay();
            return;
        }

        if !pressed {
            return;
        }

        let PhysicalKey::Code(key) = event.physical_key else {
            return;
        };

        if key != KeyCode::Escape {
            return;
        }

        match self.screen {
            AppScreen::MainMenu => {}
            AppScreen::PauseMenu => self.resume_game(window),
            AppScreen::Settings => {
                self.apply_settings_from_inputs();
                self.screen = self.settings_back_screen;
                self.active_settings_field = None;
                self.refresh_menu_overlay();
            }
            AppScreen::WorldList => {
                self.screen = AppScreen::MainMenu;
                self.active_settings_field = None;
                self.refresh_menu_overlay();
            }
            AppScreen::InGame => event_loop.exit(),
        }
    }

    fn handle_settings_keyboard(&mut self, event: &KeyEvent) -> bool {
        if event.state != ElementState::Pressed {
            return false;
        }

        if let PhysicalKey::Code(key) = event.physical_key {
            match key {
                KeyCode::Tab => {
                    self.active_settings_field = Some(match self.active_settings_field {
                        Some(SettingsField::MouseSensitivity) => SettingsField::ChunkViewDistance,
                        _ => SettingsField::MouseSensitivity,
                    });
                    return true;
                }
                KeyCode::Backspace => {
                    if let Some(text) = self.active_settings_text_mut() {
                        text.pop();
                    }
                    return true;
                }
                KeyCode::Enter => {
                    self.apply_settings_from_inputs();
                    return true;
                }
                _ => {}
            }
        }

        let Some(text) = event.text.as_deref() else {
            return false;
        };

        self.push_settings_text(text)
    }

    fn push_settings_text(&mut self, text: &str) -> bool {
        let Some(field) = self.active_settings_field else {
            return false;
        };

        let mut accepted_any = false;

        for character in text.chars() {
            if !is_settings_character_allowed(field, character) {
                continue;
            }

            let Some(target) = self.active_settings_text_mut() else {
                continue;
            };

            if field == SettingsField::MouseSensitivity && character == '.' && target.contains('.')
            {
                continue;
            }

            if target.len() >= settings_input_limit(field) {
                continue;
            }

            target.push(character);
            accepted_any = true;
        }

        accepted_any
    }

    fn active_settings_text_mut(&mut self) -> Option<&mut String> {
        match self.active_settings_field {
            Some(SettingsField::MouseSensitivity) => {
                Some(&mut self.settings_inputs.mouse_sensitivity)
            }
            Some(SettingsField::ChunkViewDistance) => {
                Some(&mut self.settings_inputs.chunk_view_distance)
            }
            None => None,
        }
    }

    fn apply_settings_from_inputs(&mut self) {
        let mouse_sensitivity = self
            .settings_inputs
            .mouse_sensitivity
            .parse::<f32>()
            .ok()
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(self.settings.mouse_sensitivity)
            .clamp(config::MIN_MOUSE_SENSITIVITY, config::MAX_MOUSE_SENSITIVITY);
        let chunk_view_distance = self
            .settings_inputs
            .chunk_view_distance
            .parse::<i32>()
            .unwrap_or(self.settings.chunk_view_distance)
            .clamp(
                config::MIN_RENDER_CHUNK_DISTANCE,
                config::MAX_RENDER_CHUNK_DISTANCE,
            );

        self.settings = GameSettings {
            mouse_sensitivity,
            chunk_view_distance,
        };
        self.settings_inputs = SettingsInputs::from_settings(self.settings);
        self.streaming
            .set_settings(self.settings.streaming_settings());
        self.save_settings();
    }

    fn save_settings(&self) {
        if let Err(error) = persistence::save_settings(&self.save_dir, self.settings.saved()) {
            eprintln!("Failed to save settings: {error}");
        }
    }

    fn save_current_world_if_dirty(&mut self) {
        if !self.world_save_dirty {
            return;
        }

        let Some(world) = self.world.as_ref() else {
            self.world_save_dirty = false;
            self.modified_chunk_coords.clear();
            return;
        };

        for coord in self.modified_chunk_coords.iter().copied() {
            let Some(chunk) = world.get_chunk(coord) else {
                self.saved_chunks.remove(&coord);
                continue;
            };

            let generated = VoxelWorld::generate_chunk(world.seed(), coord);

            if chunk.blocks() == generated.blocks() {
                self.saved_chunks.remove(&coord);
            } else {
                self.saved_chunks.insert(coord, chunk.clone());
            }
        }

        match persistence::save_chunks(&self.save_dir, world.seed(), &self.saved_chunks) {
            Ok(saved_chunks) => {
                println!("Saved {saved_chunks} modified chunks");
                self.world_save_dirty = false;
                self.modified_chunk_coords.clear();
            }
            Err(error) => eprintln!("Failed to save world chunks: {error}"),
        }
    }

    fn start_existing_world(&mut self, window: &Window) {
        self.release_mouse(window);
        self.saved_chunks = load_saved_world_chunks(&self.save_dir);
        self.camera = VoxelCamera::looking_at_chunk_origin();
        self.player = PlayerController::from_camera(self.camera);
        self.input = InputState::default();
        self.last_frame = None;
        self.noclip_enabled = false;
        self.world_save_dirty = false;
        self.modified_chunk_coords.clear();
        self.dirty_mesh_queue = DirtyMeshQueue::default();
        self.streaming = ChunkStreamingState::new(self.settings.streaming_settings());
        self.chunk_jobs = ChunkJobQueue::default();

        let window_world =
            build_window_world(self.settings.streaming_settings(), &self.saved_chunks);

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_camera(self.camera);
            renderer.set_ui_overlay(UiOverlay::default());
            renderer.set_crosshair_enabled(true);
            renderer.upload_chunk_meshes(window_world.meshes.iter().map(WindowChunkMesh::upload));

            if window_world.meshes.is_empty() {
                eprintln!("No chunk meshes were available for the first rendered frame");
            } else {
                println!(
                    "Uploaded {} chunk meshes to renderer ({} indices)",
                    renderer.mesh_count(),
                    renderer.index_count()
                );
            }
        }

        self.world = Some(window_world.world);
        self.screen = AppScreen::InGame;
        self.capture_mouse(window);
    }

    fn open_pause_menu(&mut self, window: &Window) {
        self.input = InputState::default();
        self.last_frame = None;
        self.active_settings_field = None;
        self.screen = AppScreen::PauseMenu;
        self.release_mouse(window);
        self.refresh_menu_overlay();
    }

    fn resume_game(&mut self, window: &Window) {
        if self.world.is_none() {
            self.quit_to_main_menu(window);
            return;
        }

        self.input = InputState::default();
        self.last_frame = None;
        self.active_settings_field = None;
        self.screen = AppScreen::InGame;

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_ui_overlay(UiOverlay::default());
            renderer.set_crosshair_enabled(true);
        }

        self.capture_mouse(window);
    }

    fn quit_to_main_menu(&mut self, window: &Window) {
        self.save_current_world_if_dirty();
        self.release_mouse(window);
        self.world = None;
        self.input = InputState::default();
        self.last_frame = None;
        self.noclip_enabled = false;
        self.active_settings_field = None;
        self.settings_back_screen = AppScreen::MainMenu;
        self.dirty_mesh_queue = DirtyMeshQueue::default();
        self.streaming = ChunkStreamingState::new(self.settings.streaming_settings());
        self.chunk_jobs = ChunkJobQueue::default();
        self.world_save_dirty = false;
        self.modified_chunk_coords.clear();
        self.screen = AppScreen::MainMenu;

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.clear_chunk_meshes();
            renderer.set_crosshair_enabled(false);
        }

        self.refresh_menu_overlay();
    }

    fn refresh_menu_overlay(&mut self) {
        if self.screen == AppScreen::InGame {
            return;
        }

        let Some(size) = self.renderer.as_ref().map(ClearRenderer::size) else {
            return;
        };
        let overlay = build_menu_layout(
            self.screen,
            &self.settings_inputs,
            self.active_settings_field,
            size,
        )
        .overlay;

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_crosshair_enabled(false);
            renderer.set_ui_overlay(overlay);
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

    fn toggle_noclip(&mut self) {
        self.noclip_enabled = !self.noclip_enabled;
        self.player = PlayerController::from_camera(self.camera);

        if self.noclip_enabled {
            println!("Noclip enabled");
        } else {
            println!("Noclip disabled");
        }
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
        let priority_chunks = interaction_priority_chunks(result);
        let modified_chunks = interaction_modified_chunks(result);

        print_window_interaction(result);

        if changed_blocks == 0 {
            return;
        }

        self.world_save_dirty = true;
        self.modified_chunk_coords.extend(modified_chunks);
        self.dirty_mesh_queue
            .enqueue_priority(priority_chunks.iter().copied());

        for coord in priority_chunks {
            let Some(input) = ChunkMeshInput::from_world(world, coord) else {
                continue;
            };

            self.chunk_jobs.enqueue_mesh_priority(input);
        }
    }

    fn process_completed_chunk_jobs(&mut self) {
        let Some(world) = self.world.as_mut() else {
            return;
        };
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        for result in self.chunk_jobs.drain_results() {
            match result {
                ChunkJobResult::Generated { coord, chunk } => {
                    let chunk = self.saved_chunks.get(&coord).cloned().unwrap_or(chunk);

                    if world.insert_chunk(chunk) {
                        for dirty_coord in mark_streamed_chunk_dirty(world, coord) {
                            self.dirty_mesh_queue.enqueue(dirty_coord);
                        }
                    }
                }
                ChunkJobResult::Meshed {
                    coord,
                    revision,
                    mesh,
                    ..
                } => {
                    if !chunk_within_render_distance(
                        camera_chunk_coord(self.camera),
                        coord,
                        self.settings.streaming_settings(),
                    ) {
                        continue;
                    }

                    apply_mesh_job_result(
                        world,
                        renderer,
                        &mut self.dirty_mesh_queue,
                        coord,
                        revision,
                        mesh,
                    );
                }
            }
        }
    }

    fn schedule_dirty_mesh_jobs(&mut self) {
        let Some(world) = self.world.as_mut() else {
            return;
        };
        let center = camera_chunk_coord(self.camera);
        let settings = self.settings.streaming_settings();

        self.dirty_mesh_queue
            .enqueue_dirty_within(world, center, settings);

        for _ in 0..config::DIRTY_MESH_JOBS_PER_FRAME {
            if self.chunk_jobs.normal_mesh_pending_count() >= config::MAX_PENDING_NORMAL_MESH_JOBS {
                break;
            }

            let Some(coord) = self.dirty_mesh_queue.pop_dirty(world) else {
                break;
            };

            if !chunk_within_render_distance(center, coord, settings) {
                continue;
            }

            if self.chunk_jobs.is_mesh_pending(coord) {
                continue;
            }

            let Some(input) = ChunkMeshInput::from_world(world, coord) else {
                continue;
            };

            self.chunk_jobs.enqueue_mesh(input);
        }
    }

    fn process_chunk_streaming(&mut self) {
        let Some(seed) = self.world.as_ref().map(VoxelWorld::seed) else {
            return;
        };

        let center = camera_chunk_coord(self.camera);
        if self.streaming.update_center(center) {
            println!("Streaming around chunk {:?}", center);
        }

        self.prune_loaded_chunks();
        self.sync_render_radius();

        for _ in 0..config::CHUNK_GENERATION_JOBS_PER_FRAME {
            if self.chunk_jobs.generation_pending_count() >= config::MAX_PENDING_GENERATION_JOBS {
                break;
            }

            let Some(world) = self.world.as_ref() else {
                break;
            };
            let Some(coord) = self.streaming.pop_missing(world) else {
                break;
            };

            if self.chunk_jobs.is_generation_pending(coord) {
                continue;
            }

            self.chunk_jobs.enqueue_generation(seed, coord);
        }
    }

    fn prune_loaded_chunks(&mut self) {
        let Some(world) = self.world.as_mut() else {
            return;
        };

        if world.chunks.len() <= config::MAX_ACTIVE_WORLD_CHUNKS {
            return;
        }

        let center = camera_chunk_coord(self.camera);
        let overflow = world.chunks.len() - config::MAX_ACTIVE_WORLD_CHUNKS;
        let mut removable_coords: Vec<ChunkCoord> = world
            .chunks
            .keys()
            .copied()
            .filter(|coord| !self.modified_chunk_coords.contains(coord))
            .collect();

        removable_coords.sort_by_key(|coord| Reverse(chunk_distance_key(center, *coord)));

        let mut removed = 0;

        for coord in removable_coords.into_iter().take(overflow) {
            if world.chunks.remove(&coord).is_some() {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.remove_chunk_mesh(coord);
                }
                removed += 1;
            }
        }

        if removed > 0 {
            println!("Pruned {removed} far loaded chunks");
        }
    }

    fn sync_render_radius(&mut self) {
        let center = camera_chunk_coord(self.camera);
        let settings = self.settings.streaming_settings();
        let rendered_coords: Vec<ChunkCoord> = self
            .renderer
            .as_ref()
            .map(|renderer| renderer.chunk_mesh_info().map(|info| info.coord).collect())
            .unwrap_or_default();

        let mut retained_rendered_coords = Vec::new();
        let mut meshes_to_remove = Vec::new();

        for coord in rendered_coords {
            if chunk_within_render_distance(center, coord, settings) {
                retained_rendered_coords.push(coord);
            } else {
                meshes_to_remove.push(coord);
            }
        }

        retained_rendered_coords.sort_by_key(|coord| chunk_distance_key(center, *coord));

        if retained_rendered_coords.len() > config::MAX_RENDERED_CHUNK_MESHES {
            meshes_to_remove
                .extend(retained_rendered_coords.drain(config::MAX_RENDERED_CHUNK_MESHES..));
        }

        if let Some(renderer) = self.renderer.as_mut() {
            for coord in &meshes_to_remove {
                renderer.remove_chunk_mesh(*coord);
            }
        }

        let rendered_lookup: HashSet<ChunkCoord> =
            retained_rendered_coords.iter().copied().collect();
        let available_render_slots =
            config::MAX_RENDERED_CHUNK_MESHES.saturating_sub(rendered_lookup.len());

        if available_render_slots == 0 {
            return;
        }

        let Some(world) = self.world.as_mut() else {
            return;
        };

        let available_queue_slots = config::MAX_DIRTY_MESH_QUEUE_BACKLOG
            .saturating_sub(self.dirty_mesh_queue.pending_count());
        if available_queue_slots == 0 {
            return;
        }

        let mut chunks_to_mesh: Vec<ChunkCoord> = world
            .chunks
            .keys()
            .copied()
            .filter(|coord| chunk_within_render_distance(center, *coord, settings))
            .filter(|coord| !rendered_lookup.contains(coord))
            .collect();
        chunks_to_mesh.sort_by_key(|coord| chunk_distance_key(center, *coord));
        chunks_to_mesh.truncate(
            config::MAX_NEW_RENDER_MESHES_QUEUED_PER_FRAME
                .min(available_queue_slots)
                .min(available_render_slots),
        );

        let mut queued_coords = Vec::new();

        for coord in chunks_to_mesh {
            if mark_all_subchunks_dirty(world, coord) {
                queued_coords.push(coord);
            }
        }

        self.dirty_mesh_queue.enqueue_priority(queued_coords);
    }
}

fn window_center_position(size: PhysicalSize<u32>) -> PhysicalPosition<f64> {
    PhysicalPosition::new(size.width as f64 * 0.5, size.height as f64 * 0.5)
}

fn load_saved_game_settings(save_dir: &std::path::Path) -> GameSettings {
    match persistence::load_settings(save_dir) {
        Ok(Some(settings)) => GameSettings::from_saved(settings),
        Ok(None) => GameSettings::default(),
        Err(error) => {
            eprintln!("Failed to load saved settings: {error}");
            GameSettings::default()
        }
    }
}

fn load_saved_world_chunks(save_dir: &std::path::Path) -> HashMap<ChunkCoord, Chunk> {
    match persistence::load_saved_chunks(save_dir) {
        Ok(chunks) => chunks,
        Err(error) => {
            eprintln!("Failed to load saved world chunks: {error}");
            HashMap::new()
        }
    }
}

fn camera_chunk_coord(camera: VoxelCamera) -> ChunkCoord {
    let pos = camera_block_position(camera.position);
    world_to_chunk_coord(pos.x, pos.y, pos.z)
}

fn camera_block_position(position: [f32; 3]) -> BlockPos {
    BlockPos::new(
        position[0].floor() as i32,
        position[1].floor() as i32,
        position[2].floor() as i32,
    )
}

fn streaming_chunk_coords(
    center: ChunkCoord,
    settings: WorldStreamingSettings,
) -> VecDeque<ChunkCoord> {
    let horizontal_radius = settings.horizontal_load_radius.max(0);
    let vertical_radius = settings.vertical_load_radius.max(0);
    let mut coords = Vec::new();

    for dy in -vertical_radius..=vertical_radius {
        for dz in -horizontal_radius..=horizontal_radius {
            for dx in -horizontal_radius..=horizontal_radius {
                if dx * dx + dz * dz > horizontal_radius * horizontal_radius {
                    continue;
                }

                coords.push(center.offset(dx, dy, dz));
            }
        }
    }

    coords.sort_by_key(|coord| chunk_distance_key(center, *coord));
    coords.truncate(config::MAX_STREAMING_COORDS_PER_CENTER);
    coords.into()
}

fn chunk_distance_key(center: ChunkCoord, coord: ChunkCoord) -> i32 {
    let dx = coord.x - center.x;
    let dy = coord.y - center.y;
    let dz = coord.z - center.z;

    dx * dx + dy * dy + dz * dz
}

fn chunk_within_render_distance(
    center: ChunkCoord,
    coord: ChunkCoord,
    settings: WorldStreamingSettings,
) -> bool {
    let horizontal_radius = settings.render_radius.max(0);
    let vertical_radius = settings.vertical_load_radius.max(0);
    let dx = (coord.x - center.x).abs();
    let dy = (coord.y - center.y).abs();
    let dz = (coord.z - center.z).abs();

    dy <= vertical_radius && dx * dx + dz * dz <= horizontal_radius * horizontal_radius
}

fn interaction_priority_chunks(interaction: BlockInteraction) -> Vec<ChunkCoord> {
    let mut coords = Vec::new();

    match interaction {
        BlockInteraction::Break { hit, .. } => {
            push_unique_chunk_coord(&mut coords, hit.chunk_coord);
            for coord in neighbor_chunk_coords(hit.chunk_coord) {
                push_unique_chunk_coord(&mut coords, coord);
            }
        }
        BlockInteraction::Place {
            hit, placed_block, ..
        } => {
            let placed_coord = world_to_chunk_coord(placed_block.x, placed_block.y, placed_block.z);

            push_unique_chunk_coord(&mut coords, placed_coord);
            push_unique_chunk_coord(&mut coords, hit.chunk_coord);

            for coord in neighbor_chunk_coords(placed_coord) {
                push_unique_chunk_coord(&mut coords, coord);
            }
            for coord in neighbor_chunk_coords(hit.chunk_coord) {
                push_unique_chunk_coord(&mut coords, coord);
            }
        }
        BlockInteraction::Miss
        | BlockInteraction::NoPlaceableBlockSelected
        | BlockInteraction::InvalidPlacementFace { .. } => {}
    }

    coords
}

fn interaction_modified_chunks(interaction: BlockInteraction) -> Vec<ChunkCoord> {
    match interaction {
        BlockInteraction::Break { hit, .. } => vec![hit.chunk_coord],
        BlockInteraction::Place { placed_block, .. } => {
            vec![world_to_chunk_coord(
                placed_block.x,
                placed_block.y,
                placed_block.z,
            )]
        }
        BlockInteraction::Miss
        | BlockInteraction::NoPlaceableBlockSelected
        | BlockInteraction::InvalidPlacementFace { .. } => Vec::new(),
    }
}

fn push_unique_chunk_coord(coords: &mut Vec<ChunkCoord>, coord: ChunkCoord) {
    if !coords.contains(&coord) {
        coords.push(coord);
    }
}

fn mark_streamed_chunk_dirty(world: &mut VoxelWorld, coord: ChunkCoord) -> Vec<ChunkCoord> {
    let mut dirty_coords = Vec::new();

    for dirty_coord in [coord]
        .into_iter()
        .chain(neighbor_chunk_coords(coord).into_iter())
    {
        if mark_all_subchunks_dirty(world, dirty_coord) {
            dirty_coords.push(dirty_coord);
        }
    }

    dirty_coords
}

fn neighbor_chunk_coords(coord: ChunkCoord) -> [ChunkCoord; 6] {
    [
        coord.offset(1, 0, 0),
        coord.offset(-1, 0, 0),
        coord.offset(0, 1, 0),
        coord.offset(0, -1, 0),
        coord.offset(0, 0, 1),
        coord.offset(0, 0, -1),
    ]
}

fn mark_all_subchunks_dirty(world: &mut VoxelWorld, coord: ChunkCoord) -> bool {
    let Some(chunk) = world.get_chunk_mut(coord) else {
        return false;
    };

    for subchunk in 0..SUBCHUNK_COUNT {
        chunk.mark_subchunk_dirty(subchunk);
    }

    true
}

const PLAYER_REACH: f32 = 8.0;
const PLAYER_HALF_EXTENTS: [f32; 3] = [0.3, 0.9, 0.3];
const PLAYER_EYE_HEIGHT: f32 = 1.62;
const PLAYER_WALK_SPEED: f32 = 5.5;
const PLAYER_SPRINT_SPEED: f32 = 8.5;
const PLAYER_JUMP_SPEED: f32 = 8.0;
const PLAYER_GRAVITY: f32 = 24.0;
const PLAYER_MAX_FALL_SPEED: f32 = 48.0;

#[derive(Debug, Clone)]
struct MenuLayout {
    overlay: UiOverlay,
    hits: Vec<MenuHitRect>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MenuHitRect {
    action: MenuAction,
    rect: ScreenRect,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ScreenRect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl ScreenRect {
    const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    fn centered(window_width: f32, y: f32, width: f32, height: f32) -> Self {
        Self::new((window_width - width) * 0.5, y, width, height)
    }

    fn contains(self, position: PhysicalPosition<f64>) -> bool {
        let x = position.x as f32;
        let y = position.y as f32;

        x >= self.x && x <= self.x + self.width && y >= self.y && y <= self.y + self.height
    }
}

fn build_menu_layout(
    screen: AppScreen,
    settings_inputs: &SettingsInputs,
    active_field: Option<SettingsField>,
    size: PhysicalSize<u32>,
) -> MenuLayout {
    let width = size.width.max(1) as f32;
    let height = size.height.max(1) as f32;
    let mut overlay = UiOverlay::default();
    let mut hits = Vec::new();

    push_ui_rect(
        &mut overlay,
        ScreenRect::new(0.0, 0.0, width, height),
        [0.025, 0.032, 0.036, 1.0],
    );

    match screen {
        AppScreen::MainMenu => {
            push_centered_text(
                &mut overlay,
                "ADVENTURE QUEST",
                5.0,
                width,
                height * 0.16,
                [1.0, 1.0, 1.0, 1.0],
            );
            push_button(
                &mut overlay,
                &mut hits,
                MenuAction::Play,
                ScreenRect::centered(width, height * 0.38, 320.0, 58.0),
                "PLAY",
            );
            push_button(
                &mut overlay,
                &mut hits,
                MenuAction::Settings,
                ScreenRect::centered(width, height * 0.50, 320.0, 58.0),
                "SETTINGS",
            );
            push_button(
                &mut overlay,
                &mut hits,
                MenuAction::QuitGame,
                ScreenRect::centered(width, height * 0.62, 320.0, 58.0),
                "QUIT GAME",
            );
        }
        AppScreen::WorldList => {
            push_centered_text(
                &mut overlay,
                "SAVED WORLDS",
                4.0,
                width,
                height * 0.14,
                [1.0, 1.0, 1.0, 1.0],
            );
            push_centered_text(
                &mut overlay,
                "NO SAVED WORLDS YET",
                2.0,
                width,
                height * 0.28,
                [0.72, 0.78, 0.80, 1.0],
            );
            push_button(
                &mut overlay,
                &mut hits,
                MenuAction::CreateWorld,
                ScreenRect::centered(width, height * 0.42, 440.0, 58.0),
                "CREATE NEW WORLD",
            );
            push_button(
                &mut overlay,
                &mut hits,
                MenuAction::Back,
                ScreenRect::centered(width, height * 0.54, 240.0, 52.0),
                "BACK",
            );
        }
        AppScreen::PauseMenu => {
            push_centered_text(
                &mut overlay,
                "PAUSED",
                5.0,
                width,
                height * 0.14,
                [1.0, 1.0, 1.0, 1.0],
            );
            push_button(
                &mut overlay,
                &mut hits,
                MenuAction::Resume,
                ScreenRect::centered(width, height * 0.34, 360.0, 58.0),
                "RESUME",
            );
            push_button(
                &mut overlay,
                &mut hits,
                MenuAction::Settings,
                ScreenRect::centered(width, height * 0.46, 360.0, 58.0),
                "SETTINGS",
            );
            push_button(
                &mut overlay,
                &mut hits,
                MenuAction::QuitToMenu,
                ScreenRect::centered(width, height * 0.58, 360.0, 58.0),
                "QUIT TO MENU",
            );
        }
        AppScreen::Settings => {
            push_centered_text(
                &mut overlay,
                "SETTINGS",
                4.0,
                width,
                height * 0.12,
                [1.0, 1.0, 1.0, 1.0],
            );

            let label_x = (width * 0.5 - 390.0).max(28.0);
            let field_x = (width * 0.5 + 90.0).min(width - 300.0).max(label_x + 310.0);
            let first_y = height * 0.30;
            let second_y = height * 0.42;

            push_text(
                &mut overlay,
                label_x,
                first_y + 11.0,
                2.4,
                "MOUSE SENSITIVITY",
                [0.86, 0.91, 0.93, 1.0],
            );
            push_textbox(
                &mut overlay,
                &mut hits,
                MenuAction::MouseSensitivity,
                ScreenRect::new(field_x, first_y, 260.0, 52.0),
                &settings_inputs.mouse_sensitivity,
                active_field == Some(SettingsField::MouseSensitivity),
            );

            push_text(
                &mut overlay,
                label_x,
                second_y + 11.0,
                2.4,
                "RENDER CHUNK DISTANCE",
                [0.86, 0.91, 0.93, 1.0],
            );
            push_textbox(
                &mut overlay,
                &mut hits,
                MenuAction::ChunkViewDistance,
                ScreenRect::new(field_x, second_y, 260.0, 52.0),
                &settings_inputs.chunk_view_distance,
                active_field == Some(SettingsField::ChunkViewDistance),
            );

            push_button(
                &mut overlay,
                &mut hits,
                MenuAction::Back,
                ScreenRect::centered(width, height * 0.62, 240.0, 52.0),
                "BACK",
            );
        }
        AppScreen::InGame => {}
    }

    MenuLayout { overlay, hits }
}

fn push_button(
    overlay: &mut UiOverlay,
    hits: &mut Vec<MenuHitRect>,
    action: MenuAction,
    rect: ScreenRect,
    label: &str,
) {
    push_ui_rect(overlay, rect, [0.13, 0.17, 0.19, 1.0]);
    push_ui_border(overlay, rect, 2.0, [0.42, 0.52, 0.56, 1.0]);
    push_centered_text_in_rect(overlay, rect, label, 2.8, [1.0, 1.0, 1.0, 1.0]);
    hits.push(MenuHitRect { action, rect });
}

fn push_textbox(
    overlay: &mut UiOverlay,
    hits: &mut Vec<MenuHitRect>,
    action: MenuAction,
    rect: ScreenRect,
    value: &str,
    active: bool,
) {
    push_ui_rect(
        overlay,
        rect,
        if active {
            [0.070, 0.105, 0.118, 1.0]
        } else {
            [0.035, 0.044, 0.050, 1.0]
        },
    );
    push_ui_border(
        overlay,
        rect,
        if active { 3.0 } else { 2.0 },
        if active {
            [0.98, 0.78, 0.30, 1.0]
        } else {
            [0.34, 0.42, 0.46, 1.0]
        },
    );
    push_text(
        overlay,
        rect.x + 14.0,
        rect.y + 13.0,
        2.6,
        value,
        [1.0, 1.0, 1.0, 1.0],
    );

    if active {
        let caret_x =
            (rect.x + 14.0 + text_pixel_width(value, 2.6) + 5.0).min(rect.x + rect.width - 16.0);
        push_ui_rect(
            overlay,
            ScreenRect::new(caret_x, rect.y + 10.0, 2.0, rect.height - 20.0),
            [1.0, 0.92, 0.62, 1.0],
        );
    }

    hits.push(MenuHitRect { action, rect });
}

fn push_ui_rect(overlay: &mut UiOverlay, rect: ScreenRect, color: [f32; 4]) {
    overlay.items.push(UiOverlayItem::Rect(UiRect::new(
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        color,
    )));
}

fn push_ui_border(overlay: &mut UiOverlay, rect: ScreenRect, thickness: f32, color: [f32; 4]) {
    push_ui_rect(
        overlay,
        ScreenRect::new(rect.x, rect.y, rect.width, thickness),
        color,
    );
    push_ui_rect(
        overlay,
        ScreenRect::new(
            rect.x,
            rect.y + rect.height - thickness,
            rect.width,
            thickness,
        ),
        color,
    );
    push_ui_rect(
        overlay,
        ScreenRect::new(rect.x, rect.y, thickness, rect.height),
        color,
    );
    push_ui_rect(
        overlay,
        ScreenRect::new(
            rect.x + rect.width - thickness,
            rect.y,
            thickness,
            rect.height,
        ),
        color,
    );
}

fn push_centered_text(
    overlay: &mut UiOverlay,
    text: &str,
    scale: f32,
    width: f32,
    y: f32,
    color: [f32; 4],
) {
    let x = (width - text_pixel_width(text, scale)) * 0.5;
    push_text(overlay, x.max(8.0), y, scale, text, color);
}

fn push_centered_text_in_rect(
    overlay: &mut UiOverlay,
    rect: ScreenRect,
    text: &str,
    scale: f32,
    color: [f32; 4],
) {
    let text_width = text_pixel_width(text, scale);
    let text_height = 7.0 * scale;
    let x = rect.x + (rect.width - text_width) * 0.5;
    let y = rect.y + (rect.height - text_height) * 0.5;

    push_text(overlay, x, y, scale, text, color);
}

fn push_text(overlay: &mut UiOverlay, x: f32, y: f32, scale: f32, text: &str, color: [f32; 4]) {
    overlay
        .items
        .push(UiOverlayItem::Text(UiText::new(x, y, scale, color, text)));
}

fn text_pixel_width(text: &str, scale: f32) -> f32 {
    text.chars()
        .map(|character| if character == ' ' { 4.0 } else { 6.0 })
        .sum::<f32>()
        * scale
}

fn is_settings_character_allowed(field: SettingsField, character: char) -> bool {
    match field {
        SettingsField::MouseSensitivity => character.is_ascii_digit() || character == '.',
        SettingsField::ChunkViewDistance => character.is_ascii_digit(),
    }
}

fn settings_input_limit(field: SettingsField) -> usize {
    match field {
        SettingsField::MouseSensitivity => 8,
        SettingsField::ChunkViewDistance => 2,
    }
}

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

#[derive(Debug, Clone, Copy)]
struct PlayerController {
    center: [f32; 3],
    velocity: [f32; 3],
    on_ground: bool,
}

impl PlayerController {
    fn from_camera(camera: VoxelCamera) -> Self {
        Self {
            center: camera_position_to_player_center(camera.position),
            velocity: [0.0; 3],
            on_ground: false,
        }
    }

    fn camera_position(self) -> [f32; 3] {
        player_center_to_camera_position(self.center)
    }

    fn step(&mut self, world: &VoxelWorld, yaw_radians: f32, input: InputState, dt: f32) {
        let (mut horizontal_x, mut horizontal_z) = horizontal_movement(yaw_radians, input);
        let horizontal_length = (horizontal_x * horizontal_x + horizontal_z * horizontal_z).sqrt();

        if horizontal_length > 1.0 {
            horizontal_x /= horizontal_length;
            horizontal_z /= horizontal_length;
        }

        let movement_speed = if input.fast {
            PLAYER_SPRINT_SPEED
        } else {
            PLAYER_WALK_SPEED
        };

        self.velocity[0] = horizontal_x * movement_speed;
        self.velocity[2] = horizontal_z * movement_speed;

        if input.up && self.on_ground {
            self.velocity[1] = PLAYER_JUMP_SPEED;
            self.on_ground = false;
        }

        self.velocity[1] = (self.velocity[1] - PLAYER_GRAVITY * dt).max(-PLAYER_MAX_FALL_SPEED);

        let delta = [
            self.velocity[0] * dt,
            self.velocity[1] * dt,
            self.velocity[2] * dt,
        ];
        let collision = move_aabb_through_voxels(world, self.center, PLAYER_HALF_EXTENTS, delta);

        self.center = collision.center;
        self.on_ground = collision.on_ground;

        for axis in 0..3 {
            if collision.collided[axis] {
                self.velocity[axis] = 0.0;
            }
        }
    }
}

fn camera_position_to_player_center(camera_position: [f32; 3]) -> [f32; 3] {
    [
        camera_position[0],
        camera_position[1] - PLAYER_EYE_HEIGHT + PLAYER_HALF_EXTENTS[1],
        camera_position[2],
    ]
}

fn player_center_to_camera_position(center: [f32; 3]) -> [f32; 3] {
    [
        center[0],
        center[1] + PLAYER_EYE_HEIGHT - PLAYER_HALF_EXTENTS[1],
        center[2],
    ]
}

fn horizontal_movement(yaw_radians: f32, input: InputState) -> (f32, f32) {
    let forward_axis = input.forward_axis();
    let right_axis = input.right_axis();
    let forward = (yaw_radians.cos(), yaw_radians.sin());
    let right = (-yaw_radians.sin(), yaw_radians.cos());

    (
        forward.0 * forward_axis + right.0 * right_axis,
        forward.1 * forward_axis + right.1 * right_axis,
    )
}

#[derive(Debug, Default)]
struct DirtyMeshQueue {
    pending: VecDeque<ChunkCoord>,
}

impl DirtyMeshQueue {
    #[cfg(test)]
    fn enqueue_dirty(&mut self, world: &VoxelWorld) {
        for coord in dirty_window_chunk_coords(world) {
            self.enqueue(coord);
        }
    }

    fn enqueue_dirty_within(
        &mut self,
        world: &VoxelWorld,
        center: ChunkCoord,
        settings: WorldStreamingSettings,
    ) {
        for coord in dirty_window_chunk_coords(world) {
            if chunk_within_render_distance(center, coord, settings) {
                self.enqueue(coord);
            }
        }
    }

    fn enqueue(&mut self, coord: ChunkCoord) {
        if !self.pending.contains(&coord) {
            self.pending.push_back(coord);
        }
    }

    fn enqueue_priority<I>(&mut self, coords: I)
    where
        I: IntoIterator<Item = ChunkCoord>,
    {
        let coords: Vec<ChunkCoord> = coords.into_iter().collect();

        for coord in coords.into_iter().rev() {
            self.enqueue_front(coord);
        }
    }

    fn enqueue_front(&mut self, coord: ChunkCoord) {
        self.pending.retain(|pending| *pending != coord);
        self.pending.push_front(coord);
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

    fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

struct ChunkJobQueue {
    sender: Sender<ChunkJob>,
    priority_sender: Sender<ChunkJob>,
    receiver: Receiver<ChunkJobResult>,
    pending_generation: HashSet<ChunkCoord>,
    pending_meshes: HashSet<ChunkCoord>,
    pending_priority_meshes: HashSet<ChunkCoord>,
}

impl ChunkJobQueue {
    fn enqueue_generation(&mut self, seed: u64, coord: ChunkCoord) -> bool {
        if !self.pending_generation.insert(coord) {
            return false;
        }

        if self.sender.send(ChunkJob::Generate { seed, coord }).is_ok() {
            true
        } else {
            self.pending_generation.remove(&coord);
            false
        }
    }

    fn enqueue_mesh(&mut self, input: ChunkMeshInput) -> bool {
        let coord = input.coord;

        if !self.pending_meshes.insert(coord) {
            return false;
        }

        if self
            .sender
            .send(ChunkJob::Mesh {
                input,
                priority: MeshJobPriority::Normal,
            })
            .is_ok()
        {
            true
        } else {
            self.pending_meshes.remove(&coord);
            false
        }
    }

    fn enqueue_mesh_priority(&mut self, input: ChunkMeshInput) -> bool {
        let coord = input.coord;

        if !self.pending_priority_meshes.insert(coord) {
            return false;
        }

        if self
            .priority_sender
            .send(ChunkJob::Mesh {
                input,
                priority: MeshJobPriority::Priority,
            })
            .is_ok()
        {
            true
        } else {
            self.pending_priority_meshes.remove(&coord);
            false
        }
    }

    fn is_generation_pending(&self, coord: ChunkCoord) -> bool {
        self.pending_generation.contains(&coord)
    }

    fn is_mesh_pending(&self, coord: ChunkCoord) -> bool {
        self.pending_meshes.contains(&coord) || self.pending_priority_meshes.contains(&coord)
    }

    fn normal_mesh_pending_count(&self) -> usize {
        self.pending_meshes.len()
    }

    fn generation_pending_count(&self) -> usize {
        self.pending_generation.len()
    }

    fn drain_results(&mut self) -> Vec<ChunkJobResult> {
        let mut results = Vec::new();

        while let Ok(result) = self.receiver.try_recv() {
            match &result {
                ChunkJobResult::Generated { coord, .. } => {
                    self.pending_generation.remove(coord);
                }
                ChunkJobResult::Meshed {
                    coord, priority, ..
                } => {
                    match priority {
                        MeshJobPriority::Normal => self.pending_meshes.remove(coord),
                        MeshJobPriority::Priority => self.pending_priority_meshes.remove(coord),
                    };
                }
            }

            results.push(result);
        }

        results
    }
}

impl Default for ChunkJobQueue {
    fn default() -> Self {
        let (job_sender, job_receiver) = mpsc::channel();
        let (priority_job_sender, priority_job_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();

        thread::Builder::new()
            .name("aq-chunk-worker".to_string())
            .spawn({
                let result_sender = result_sender.clone();
                move || run_chunk_worker(job_receiver, result_sender)
            })
            .expect("chunk worker thread should start");
        thread::Builder::new()
            .name("aq-priority-chunk-worker".to_string())
            .spawn(move || run_chunk_worker(priority_job_receiver, result_sender))
            .expect("priority chunk worker thread should start");

        Self {
            sender: job_sender,
            priority_sender: priority_job_sender,
            receiver: result_receiver,
            pending_generation: HashSet::new(),
            pending_meshes: HashSet::new(),
            pending_priority_meshes: HashSet::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MeshJobPriority {
    Normal,
    Priority,
}

enum ChunkJob {
    Generate {
        seed: u64,
        coord: ChunkCoord,
    },
    Mesh {
        input: ChunkMeshInput,
        priority: MeshJobPriority,
    },
}

enum ChunkJobResult {
    Generated {
        coord: ChunkCoord,
        chunk: Chunk,
    },
    Meshed {
        coord: ChunkCoord,
        revision: u32,
        mesh: MeshData,
        priority: MeshJobPriority,
    },
}

fn run_chunk_worker(receiver: Receiver<ChunkJob>, sender: Sender<ChunkJobResult>) {
    while let Ok(job) = receiver.recv() {
        let result = match job {
            ChunkJob::Generate { seed, coord } => ChunkJobResult::Generated {
                coord,
                chunk: VoxelWorld::generate_chunk(seed, coord),
            },
            ChunkJob::Mesh { input, priority } => ChunkJobResult::Meshed {
                coord: input.coord,
                revision: input.revision,
                mesh: mesh_chunk_input(&input),
                priority,
            },
        };

        if sender.send(result).is_err() {
            break;
        }
    }
}

#[derive(Debug)]
struct ChunkStreamingState {
    settings: WorldStreamingSettings,
    center_chunk: Option<ChunkCoord>,
    pending_loads: VecDeque<ChunkCoord>,
}

impl ChunkStreamingState {
    fn new(settings: WorldStreamingSettings) -> Self {
        Self {
            settings,
            center_chunk: None,
            pending_loads: VecDeque::new(),
        }
    }

    fn set_settings(&mut self, settings: WorldStreamingSettings) {
        if self.settings == settings {
            return;
        }

        self.settings = settings;

        if let Some(center) = self.center_chunk {
            self.pending_loads = streaming_chunk_coords(center, self.settings);
        }
    }

    fn update_center(&mut self, center: ChunkCoord) -> bool {
        if self.center_chunk == Some(center) {
            return false;
        }

        self.center_chunk = Some(center);
        self.pending_loads = streaming_chunk_coords(center, self.settings).into();
        true
    }

    fn pop_missing(&mut self, world: &VoxelWorld) -> Option<ChunkCoord> {
        while let Some(coord) = self.pending_loads.pop_front() {
            if world.get_chunk(coord).is_none() {
                return Some(coord);
            }
        }

        None
    }

    #[cfg(test)]
    fn pending_count(&self) -> usize {
        self.pending_loads.len()
    }
}

impl Default for ChunkStreamingState {
    fn default() -> Self {
        Self::new(GameSettings::default().streaming_settings())
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

fn build_window_world(
    streaming_settings: WorldStreamingSettings,
    saved_chunks: &HashMap<ChunkCoord, Chunk>,
) -> WindowWorldState {
    let mut world = VoxelWorld::new(config::WORLD_SEED);
    let player_block = camera_block_position(VoxelCamera::looking_at_chunk_origin().position);
    let initial_radius =
        config::INITIAL_SYNC_CHUNK_RADIUS.min(streaming_settings.horizontal_load_radius);
    let initial_vertical_radius =
        config::INITIAL_SYNC_CHUNK_RADIUS.min(streaming_settings.vertical_load_radius);
    let initial_settings = WorldStreamingSettings::new(
        initial_radius,
        initial_vertical_radius,
        initial_radius,
        initial_radius,
        initial_radius,
        initial_radius + 1,
    );

    world.load_chunks_around_block(player_block, initial_settings);
    apply_saved_chunks_to_loaded_world(&mut world, saved_chunks);

    let meshes = build_window_meshes(&mut world);

    WindowWorldState { world, meshes }
}

fn apply_saved_chunks_to_loaded_world(
    world: &mut VoxelWorld,
    saved_chunks: &HashMap<ChunkCoord, Chunk>,
) {
    let loaded_coords: Vec<ChunkCoord> = world.chunks.keys().copied().collect();

    for coord in loaded_coords {
        let Some(chunk) = saved_chunks.get(&coord) else {
            continue;
        };

        world.chunks.insert(coord, chunk.clone());
    }
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

fn apply_mesh_job_result(
    world: &mut VoxelWorld,
    renderer: &mut ClearRenderer,
    dirty_mesh_queue: &mut DirtyMeshQueue,
    coord: ChunkCoord,
    revision: u32,
    mesh: MeshData,
) {
    let Some(current_revision) = world.get_chunk(coord).map(|chunk| chunk.revision) else {
        renderer.remove_chunk_mesh(coord);
        return;
    };

    if current_revision != revision {
        dirty_mesh_queue.enqueue(coord);
        return;
    }

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

    let mut world = VoxelWorld::new(config::WORLD_SEED);
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

    fn world_with_empty_chunks(coords: impl IntoIterator<Item = ChunkCoord>) -> VoxelWorld {
        let mut world = VoxelWorld::new(0);

        for coord in coords {
            world.chunks.insert(coord, Chunk::new_empty(coord));
        }

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

    #[test]
    fn main_menu_layout_has_quit_game_action() {
        let layout = build_menu_layout(
            AppScreen::MainMenu,
            &SettingsInputs::from_settings(GameSettings::default()),
            None,
            PhysicalSize::new(1280, 720),
        );

        assert!(layout
            .hits
            .iter()
            .any(|hit| hit.action == MenuAction::QuitGame));
    }

    #[test]
    fn pause_menu_layout_has_resume_settings_and_quit_to_menu() {
        let layout = build_menu_layout(
            AppScreen::PauseMenu,
            &SettingsInputs::from_settings(GameSettings::default()),
            None,
            PhysicalSize::new(1280, 720),
        );

        assert!(layout
            .hits
            .iter()
            .any(|hit| hit.action == MenuAction::Resume));
        assert!(layout
            .hits
            .iter()
            .any(|hit| hit.action == MenuAction::Settings));
        assert!(layout
            .hits
            .iter()
            .any(|hit| hit.action == MenuAction::QuitToMenu));
    }

    #[test]
    fn active_settings_textbox_adds_visual_highlight_geometry() {
        let settings_inputs = SettingsInputs::from_settings(GameSettings::default());
        let inactive = build_menu_layout(
            AppScreen::Settings,
            &settings_inputs,
            None,
            PhysicalSize::new(1280, 720),
        );
        let active = build_menu_layout(
            AppScreen::Settings,
            &settings_inputs,
            Some(SettingsField::MouseSensitivity),
            PhysicalSize::new(1280, 720),
        );

        assert!(active.overlay.items.len() > inactive.overlay.items.len());
    }

    #[test]
    fn camera_block_position_floors_negative_coordinates() {
        assert_eq!(
            camera_block_position([-0.1, 42.9, -32.0]),
            BlockPos::new(-1, 42, -32)
        );
    }

    #[test]
    fn streaming_chunk_coords_start_with_nearest_chunk() {
        let center = ChunkCoord::new(4, -2, 7);
        let coords = streaming_chunk_coords(center, WorldStreamingSettings::prototype());

        assert_eq!(coords.len(), 15);
        assert_eq!(coords.front().copied(), Some(center));
        assert!(coords.contains(&center.offset(1, 1, 0)));
        assert!(!coords.contains(&center.offset(1, 1, -1)));
    }

    #[test]
    fn chunk_streaming_state_skips_loaded_chunks() {
        let center = ChunkCoord::new(0, 0, 0);
        let mut state = ChunkStreamingState::new(WorldStreamingSettings::prototype());
        let world = world_with_empty_chunks([center]);

        assert!(state.update_center(center));

        let first_missing = state.pop_missing(&world);

        assert_ne!(first_missing, Some(center));
        assert_eq!(state.pending_count(), 13);
    }

    #[test]
    fn saved_render_distance_is_clamped_to_performance_cap() {
        let too_low = GameSettings::from_saved(SavedSettings {
            mouse_sensitivity: config::DEFAULT_MOUSE_SENSITIVITY,
            render_chunk_distance: 1,
        });
        let too_high = GameSettings::from_saved(SavedSettings {
            mouse_sensitivity: config::DEFAULT_MOUSE_SENSITIVITY,
            render_chunk_distance: 99,
        });

        assert_eq!(
            too_low.chunk_view_distance,
            config::MIN_RENDER_CHUNK_DISTANCE
        );
        assert_eq!(
            too_high.chunk_view_distance,
            config::MAX_RENDER_CHUNK_DISTANCE
        );
        assert!(too_high.streaming_settings().vertical_load_radius <= 2);
    }

    #[test]
    fn streamed_chunk_dirty_marking_updates_loaded_neighbors() {
        let coord = ChunkCoord::new(0, 0, 0);
        let neighbor = coord.offset(1, 0, 0);
        let missing_neighbor = coord.offset(-1, 0, 0);
        let mut world = world_with_empty_chunks([coord, neighbor]);

        let dirty = mark_streamed_chunk_dirty(&mut world, coord);

        assert!(dirty.contains(&coord));
        assert!(dirty.contains(&neighbor));
        assert!(!dirty.contains(&missing_neighbor));

        let expected_mask = ((1u16 << SUBCHUNK_COUNT) - 1) as u8;
        assert_eq!(
            world.get_chunk(coord).unwrap().subchunk_dirty_mask,
            expected_mask
        );
        assert_eq!(
            world.get_chunk(neighbor).unwrap().subchunk_dirty_mask,
            expected_mask
        );
    }

    #[test]
    fn player_camera_position_round_trips_through_body_center() {
        let camera_position = [12.0, 35.5, -8.0];
        let center = camera_position_to_player_center(camera_position);

        assert_eq!(player_center_to_camera_position(center), camera_position);
    }

    #[test]
    fn horizontal_movement_uses_camera_yaw() {
        let mut input = InputState::default();
        input.forward = true;

        assert_eq!(horizontal_movement(0.0, input), (1.0, 0.0));

        input.forward = false;
        input.right = true;

        assert_eq!(horizontal_movement(0.0, input), (0.0, 1.0));
    }

    #[test]
    fn player_controller_lands_on_voxel_floor() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunks([coord]);
        world.set_block(BlockPos::new(0, 0, 0), STONE_BLOCK);
        let mut player = PlayerController {
            center: [0.5, 2.5, 0.5],
            velocity: [0.0, -20.0, 0.0],
            on_ground: false,
        };

        player.step(&world, 0.0, InputState::default(), 0.05);

        assert!((player.center[1] - 1.901).abs() < 0.0001);
        assert_eq!(player.velocity[1], 0.0);
        assert!(player.on_ground);
    }

    #[test]
    fn chunk_worker_generates_chunks_off_thread() {
        let (job_sender, job_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();
        let handle = thread::spawn(move || run_chunk_worker(job_receiver, result_sender));
        let coord = ChunkCoord::new(2, 0, -1);

        job_sender
            .send(ChunkJob::Generate { seed: 99, coord })
            .unwrap();

        let result = result_receiver
            .recv_timeout(Duration::from_secs(2))
            .unwrap();

        match result {
            ChunkJobResult::Generated {
                coord: result_coord,
                chunk,
            } => {
                assert_eq!(result_coord, coord);
                assert_eq!(chunk.coord, coord);
            }
            ChunkJobResult::Meshed { .. } => panic!("expected generated chunk"),
        }

        drop(job_sender);
        handle.join().unwrap();
    }

    #[test]
    fn chunk_worker_meshes_snapshots_off_thread() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut world = world_with_empty_chunks([coord]);
        world.set_block(BlockPos::new(0, 0, 0), STONE_BLOCK);
        let input = ChunkMeshInput::from_world(&world, coord).expect("mesh input exists");
        let revision = input.revision;
        let (job_sender, job_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();
        let handle = thread::spawn(move || run_chunk_worker(job_receiver, result_sender));

        job_sender
            .send(ChunkJob::Mesh {
                input,
                priority: MeshJobPriority::Normal,
            })
            .unwrap();

        let result = result_receiver
            .recv_timeout(Duration::from_secs(2))
            .unwrap();

        match result {
            ChunkJobResult::Meshed {
                coord: result_coord,
                revision: result_revision,
                mesh,
                priority,
            } => {
                assert_eq!(result_coord, coord);
                assert_eq!(result_revision, revision);
                assert_eq!(mesh.visible_face_count, 6);
                assert_eq!(priority, MeshJobPriority::Normal);
            }
            ChunkJobResult::Generated { .. } => panic!("expected mesh result"),
        }

        drop(job_sender);
        handle.join().unwrap();
    }
}
