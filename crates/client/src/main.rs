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
mod tool_config;

use foundation::{BlockPos, ChunkCoord};
use gameplay::{
    break_hit_block, break_target_block, place_selected_block, BlockInteraction, Hotbar,
    HOTBAR_SLOT_COUNT,
};
use meshing::{mesh_chunk, mesh_chunk_input, mesh_dirty_subchunks, ChunkMeshInput, MeshData};
use physics::{aabb_has_ground_support, move_aabb_through_voxels, raycast_blocks};
use renderer::{
    ChunkMeshUpload, ChunkVisibility, ClearRenderer, RendererOptions, UiOverlay, UiOverlayItem,
    UiRect, UiText, UiTextureRect, VoxelCamera,
};
use voxels::{
    block_break_hp, block_color_rgba, block_hotbar_uvs, block_label, world_to_chunk_coord, BlockId,
    Chunk, AIR_BLOCK, CHUNK_SIZE, SUBCHUNK_COUNT,
};
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

    if options.no_window {
        run_voxel_prototype();
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum AppScreen {
    #[default]
    MainMenu,
    WorldList,
    LoadingWorld,
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
    Fov,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SettingsField {
    MouseSensitivity,
    ChunkViewDistance,
    Fov,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct GameSettings {
    mouse_sensitivity: f32,
    chunk_view_distance: i32,
    fov_degrees: f32,
}

impl GameSettings {
    fn streaming_settings(self) -> WorldStreamingSettings {
        let radius = self.chunk_view_distance.clamp(
            config::MIN_RENDER_CHUNK_DISTANCE,
            config::MAX_RENDER_CHUNK_DISTANCE,
        );

        WorldStreamingSettings::new(radius, radius, radius, radius, radius, radius + 1)
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
            fov_degrees: settings
                .fov_degrees
                .clamp(config::MIN_FOV_DEGREES, config::MAX_FOV_DEGREES),
        }
    }

    fn saved(self) -> SavedSettings {
        SavedSettings {
            mouse_sensitivity: self.mouse_sensitivity,
            render_chunk_distance: self.chunk_view_distance,
            fov_degrees: self.fov_degrees,
        }
    }
}

impl Default for GameSettings {
    fn default() -> Self {
        Self {
            mouse_sensitivity: config::DEFAULT_MOUSE_SENSITIVITY,
            chunk_view_distance: config::DEFAULT_RENDER_CHUNK_DISTANCE,
            fov_degrees: config::DEFAULT_FOV_DEGREES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SettingsInputs {
    mouse_sensitivity: String,
    chunk_view_distance: String,
    fov_degrees: String,
}

impl SettingsInputs {
    fn from_settings(settings: GameSettings) -> Self {
        Self {
            mouse_sensitivity: format!("{:.4}", settings.mouse_sensitivity),
            chunk_view_distance: settings.chunk_view_distance.to_string(),
            fov_degrees: format!("{:.0}", settings.fov_degrees),
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
    block_breaking: BlockBreakState,
    visible_break_progress: Option<u8>,
    visible_break_outline: Option<BlockPos>,
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
    world_loading: Option<WorldLoadState>,
}

impl AdventureQuestApp {
    fn new(options: ClientOptions) -> Self {
        let save_dir = persistence::default_save_dir();
        let settings = load_saved_game_settings(&save_dir);
        let saved_chunks = load_saved_world_chunks(&save_dir);
        let camera = camera_with_settings(VoxelCamera::looking_at_chunk_origin(), settings);

        Self {
            window: None,
            renderer: None,
            screen: AppScreen::MainMenu,
            input: InputState::default(),
            camera,
            last_frame: None,
            show_fps: options.show_fps,
            renderer_options: options.renderer,
            fps_counter: FpsCounter::default(),
            world: None,
            player: PlayerController::from_camera(camera),
            hotbar: Hotbar::starter(),
            block_breaking: BlockBreakState::default(),
            visible_break_progress: None,
            visible_break_outline: None,
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
            world_loading: None,
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
                    self.input.breaking = false;
                    self.reset_block_breaking();
                    self.release_mouse(&window);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_position = position;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if self.screen == AppScreen::LoadingWorld {
                    return;
                }

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
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;

                if self.screen == AppScreen::InGame {
                    if pressed {
                        self.capture_mouse(&window);
                    }

                    match button {
                        MouseButton::Left => {
                            self.input.breaking = pressed;

                            if !pressed {
                                self.reset_block_breaking();
                            }
                        }
                        MouseButton::Right if pressed => {
                            self.reset_block_breaking();
                            self.place_with_target();
                        }
                        _ => {}
                    }
                } else if self.screen == AppScreen::LoadingWorld {
                    return;
                } else if button == MouseButton::Left && pressed {
                    self.handle_menu_click(event_loop, &window);
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size);
                }
                self.refresh_active_overlay();
            }
            WindowEvent::RedrawRequested => {
                match self.screen {
                    AppScreen::InGame => {
                        let dt = self.advance_player();
                        self.advance_block_breaking(dt);
                        self.process_completed_chunk_jobs();
                        self.process_chunk_streaming();
                        self.schedule_dirty_mesh_jobs();
                    }
                    AppScreen::LoadingWorld => {
                        self.last_frame = None;
                        self.process_world_loading(&window);
                    }
                    _ => {
                        self.last_frame = None;
                    }
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
    fn advance_player(&mut self) -> f32 {
        let now = Instant::now();
        let dt = self
            .last_frame
            .replace(now)
            .map(|last_frame| now.duration_since(last_frame).as_secs_f32().min(0.05))
            .unwrap_or(0.0);

        if dt == 0.0 {
            return 0.0;
        }

        if self.noclip_enabled {
            self.advance_noclip_camera(dt);
            self.player = PlayerController::from_camera(self.camera);
            return dt;
        }

        let Some(world) = self.world.as_ref() else {
            return dt;
        };

        self.player
            .step(world, self.camera.yaw_radians, self.input, dt);
        self.camera.position = self.player.camera_position();

        dt
    }

    fn advance_noclip_camera(&mut self, dt: f32) {
        let movement_speed = if self.input.fast {
            config::NOCLIP_FAST_SPEED
        } else {
            config::NOCLIP_WALK_SPEED
        };

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
            MenuAction::Fov => {
                self.active_settings_field = Some(SettingsField::Fov);
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
            AppScreen::LoadingWorld => {}
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
                        Some(SettingsField::ChunkViewDistance) => SettingsField::Fov,
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

            if matches!(field, SettingsField::MouseSensitivity | SettingsField::Fov)
                && character == '.'
                && target.contains('.')
            {
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
            Some(SettingsField::Fov) => Some(&mut self.settings_inputs.fov_degrees),
            None => None,
        }
    }

    fn apply_settings_from_inputs(&mut self) {
        let mouse_sensitivity = parse_clamped_f32_setting(
            &self.settings_inputs.mouse_sensitivity,
            self.settings.mouse_sensitivity,
            config::MIN_MOUSE_SENSITIVITY,
            config::MAX_MOUSE_SENSITIVITY,
        );
        let chunk_view_distance = parse_clamped_i32_setting(
            &self.settings_inputs.chunk_view_distance,
            self.settings.chunk_view_distance,
            config::MIN_RENDER_CHUNK_DISTANCE,
            config::MAX_RENDER_CHUNK_DISTANCE,
        );
        let fov_degrees = parse_clamped_f32_setting(
            &self.settings_inputs.fov_degrees,
            self.settings.fov_degrees,
            config::MIN_FOV_DEGREES,
            config::MAX_FOV_DEGREES,
        );

        self.settings = GameSettings {
            mouse_sensitivity,
            chunk_view_distance,
            fov_degrees,
        };
        self.settings_inputs = SettingsInputs::from_settings(self.settings);
        self.streaming
            .set_settings(self.settings.streaming_settings());
        self.apply_camera_settings();
        self.save_settings();
    }

    fn apply_camera_settings(&mut self) {
        self.camera.fov_y_radians = self.settings.fov_degrees.to_radians();
        self.camera.far = camera_far_plane(self.settings);

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_camera(self.camera);
        }
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
        self.camera = camera_with_settings(VoxelCamera::looking_at_chunk_origin(), self.settings);
        self.player = PlayerController::from_camera(self.camera);
        self.input = InputState::default();
        self.reset_block_breaking();
        self.last_frame = None;
        self.noclip_enabled = false;
        self.world_save_dirty = false;
        self.modified_chunk_coords.clear();
        self.dirty_mesh_queue = DirtyMeshQueue::default();
        self.streaming = ChunkStreamingState::new(self.settings.streaming_settings());
        self.chunk_jobs = ChunkJobQueue::default();
        self.world_loading = Some(WorldLoadState::new(
            self.settings.streaming_settings(),
            self.camera,
            self.renderer
                .as_ref()
                .map(ClearRenderer::size)
                .unwrap_or_else(|| PhysicalSize::new(1280, 720)),
        ));

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.clear_chunk_meshes();
            renderer.set_camera(self.camera);
            renderer.set_crosshair_enabled(false);
            renderer.set_ui_overlay(build_loading_overlay(0.0, renderer.size()));
        }

        self.screen = AppScreen::LoadingWorld;
        window.request_redraw();
    }

    fn process_world_loading(&mut self, window: &Window) {
        let Some(loading) = self.world_loading.as_mut() else {
            return;
        };

        loading.generate_chunks(&self.saved_chunks);

        if loading.is_generation_complete() && !loading.meshing_started() {
            let visibility = self
                .renderer
                .as_ref()
                .map(|renderer| renderer.chunk_visibility(self.camera))
                .unwrap_or_else(ChunkVisibility::all);
            loading.start_meshing(self.camera, visibility, self.settings.streaming_settings());
        }

        loading.schedule_mesh_jobs(&mut self.chunk_jobs);

        if let Some(renderer) = self.renderer.as_mut() {
            for result in self
                .chunk_jobs
                .drain_results(config::LOADING_CHUNK_JOB_RESULTS_PROCESSED_PER_FRAME)
            {
                match result {
                    ChunkJobResult::Meshed {
                        coord,
                        revision,
                        mesh,
                        ..
                    } => {
                        apply_mesh_job_result(
                            &mut loading.world,
                            renderer,
                            &mut self.dirty_mesh_queue,
                            coord,
                            revision,
                            mesh,
                        );
                        loading.record_mesh_result();
                    }
                    ChunkJobResult::Generated { .. } => {}
                }
            }

            renderer.set_ui_overlay(build_loading_overlay(loading.progress(), renderer.size()));
        }

        if !loading.is_complete(&self.chunk_jobs) {
            return;
        }

        let Some(loading) = self.world_loading.take() else {
            return;
        };

        self.world = Some(loading.world);
        self.screen = AppScreen::InGame;
        self.streaming = ChunkStreamingState::new(self.settings.streaming_settings());
        self.last_frame = None;

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_crosshair_enabled(true);
        }

        self.refresh_gameplay_overlay();
        self.capture_mouse(window);
    }

    fn open_pause_menu(&mut self, window: &Window) {
        self.input = InputState::default();
        self.reset_block_breaking();
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
        self.reset_block_breaking();
        self.last_frame = None;
        self.active_settings_field = None;
        self.screen = AppScreen::InGame;

        self.refresh_gameplay_overlay();
        self.capture_mouse(window);
    }

    fn quit_to_main_menu(&mut self, window: &Window) {
        self.save_current_world_if_dirty();
        self.release_mouse(window);
        self.world = None;
        self.world_loading = None;
        self.input = InputState::default();
        self.reset_block_breaking();
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

    fn refresh_active_overlay(&mut self) {
        if self.screen == AppScreen::InGame {
            self.refresh_gameplay_overlay();
        } else if self.screen == AppScreen::LoadingWorld {
            self.refresh_loading_overlay();
        } else {
            self.refresh_menu_overlay();
        }
    }

    fn refresh_loading_overlay(&mut self) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let progress = self
            .world_loading
            .as_ref()
            .map(WorldLoadState::progress)
            .unwrap_or(0.0);

        renderer.set_crosshair_enabled(false);
        renderer.set_ui_overlay(build_loading_overlay(progress, renderer.size()));
    }

    fn refresh_menu_overlay(&mut self) {
        if matches!(self.screen, AppScreen::InGame | AppScreen::LoadingWorld) {
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

    fn refresh_gameplay_overlay(&mut self) {
        if self.screen != AppScreen::InGame {
            return;
        }

        let Some(size) = self.renderer.as_ref().map(ClearRenderer::size) else {
            return;
        };
        let overlay =
            build_gameplay_overlay(&self.hotbar, self.visible_break_progress_ratio(), size);

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_crosshair_enabled(true);
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

        self.refresh_gameplay_overlay();

        true
    }

    fn toggle_noclip(&mut self) {
        self.noclip_enabled = !self.noclip_enabled;
        self.player = PlayerController::from_camera(self.camera);
        self.reset_block_breaking();

        if self.noclip_enabled {
            println!("Noclip enabled");
        } else {
            println!("Noclip disabled");
        }
    }

    fn place_with_target(&mut self) {
        let Some(world) = self.world.as_mut() else {
            return;
        };

        let origin = self.camera.position;
        let direction = self.camera.forward_direction();
        let result = place_selected_block(world, &self.hotbar, origin, direction, PLAYER_REACH);

        self.handle_block_interaction_result(result, true);
    }

    fn reset_block_breaking(&mut self) {
        self.block_breaking.reset();
        self.set_visible_break_outline(None);
        self.set_visible_break_progress(None);
    }

    fn set_visible_break_outline(&mut self, block: Option<BlockPos>) {
        if self.visible_break_outline == block {
            return;
        }

        self.visible_break_outline = block;

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_block_outline(block);
        }
    }

    fn set_visible_break_progress(&mut self, progress: Option<f32>) {
        let visible_progress = progress.map(quantize_break_progress);

        if self.visible_break_progress == visible_progress {
            return;
        }

        self.visible_break_progress = visible_progress;
        self.refresh_gameplay_overlay();
    }

    fn visible_break_progress_ratio(&self) -> Option<f32> {
        self.visible_break_progress
            .map(|progress| progress as f32 / 100.0)
    }

    fn advance_block_breaking(&mut self, dt: f32) {
        if !self.input.breaking {
            self.reset_block_breaking();
            return;
        }

        if dt <= 0.0 {
            return;
        }

        let Some(world) = self.world.as_ref() else {
            self.reset_block_breaking();
            return;
        };

        let Some(hit) = raycast_blocks(
            world,
            self.camera.position,
            self.camera.forward_direction(),
            PLAYER_REACH,
        ) else {
            self.reset_block_breaking();
            return;
        };

        let block_hp = block_break_hp(hit.block_id);
        let damage = tool_config::UNARMED_BREAK_DAMAGE_PER_SECOND * dt;
        let broke_block = self.block_breaking.apply_damage(hit, block_hp, damage);

        self.set_visible_break_outline(Some(hit.world_block));

        if !broke_block {
            self.set_visible_break_progress(self.block_breaking.progress_ratio(block_hp));
            return;
        }

        let Some(world) = self.world.as_mut() else {
            self.reset_block_breaking();
            return;
        };
        let result = break_hit_block(world, hit);

        self.reset_block_breaking();
        self.handle_block_interaction_result(result, false);
    }

    fn handle_block_interaction_result(&mut self, result: BlockInteraction, print_miss: bool) {
        let changed_blocks = changed_block_count(result);
        let priority_chunks = interaction_priority_chunks(result);
        let modified_chunks = interaction_modified_chunks(result);

        if print_miss || changed_blocks > 0 {
            print_window_interaction(result);
        }

        if changed_blocks == 0 {
            return;
        }

        self.world_save_dirty = true;
        self.modified_chunk_coords.extend(modified_chunks);
        self.dirty_mesh_queue
            .enqueue_priority(priority_chunks.iter().copied());

        let Some(world) = self.world.as_ref() else {
            return;
        };

        for coord in priority_chunks {
            let Some(input) = ChunkMeshInput::from_world(world, coord) else {
                continue;
            };

            self.chunk_jobs.enqueue_mesh_priority(input);
        }
    }

    fn process_completed_chunk_jobs(&mut self) {
        let center = camera_chunk_coord(self.camera);
        let settings = self.settings.streaming_settings();
        let Some(world) = self.world.as_mut() else {
            return;
        };
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        for result in self
            .chunk_jobs
            .drain_results(config::MAX_CHUNK_JOB_RESULTS_PROCESSED_PER_FRAME)
        {
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
                    if !chunk_is_render_candidate(world, center, coord, settings) {
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
        let Some(world) = self.world.as_ref() else {
            return;
        };
        let Some(renderer) = self.renderer.as_ref() else {
            return;
        };
        let visibility = renderer.chunk_visibility(self.camera);
        let center = camera_chunk_coord(self.camera);
        let settings = self.settings.streaming_settings();

        self.dirty_mesh_queue
            .prioritize(world, self.camera, visibility, center, settings);

        for _ in 0..config::DIRTY_MESH_JOBS_PER_FRAME {
            if self.chunk_jobs.normal_mesh_pending_count() >= config::MAX_PENDING_NORMAL_MESH_JOBS {
                break;
            }

            let Some(coord) = self.dirty_mesh_queue.pop_dirty(world) else {
                break;
            };

            if !chunk_is_render_candidate(world, center, coord, settings) {
                continue;
            }

            if !chunk_is_visible_render_candidate(world, visibility, center, coord, settings) {
                self.dirty_mesh_queue.enqueue(coord);
                break;
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
        let viewport = self
            .renderer
            .as_ref()
            .map(ClearRenderer::size)
            .unwrap_or_else(|| PhysicalSize::new(1280, 720));
        let view_changed = self.streaming.update_view(center, self.camera, viewport);

        if view_changed {
            self.dirty_mesh_queue.request_prioritize();
        }

        self.prune_loaded_chunks();

        let render_mesh_over_budget = self.renderer.as_ref().is_some_and(|renderer| {
            renderer.chunk_mesh_count() > config::MAX_RENDERED_CHUNK_MESHES
        });

        if view_changed || render_mesh_over_budget {
            self.sync_render_radius();
        }

        for _ in 0..config::CHUNK_GENERATION_JOBS_PER_FRAME {
            if self.chunk_jobs.generation_pending_count() >= config::MAX_PENDING_GENERATION_JOBS {
                break;
            }

            let Some(world) = self.world.as_ref() else {
                break;
            };
            let Some(coord) = self.streaming.pop_missing_visible(world, self.camera) else {
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

        let mut retained_rendered_coords = Vec::new();
        let mut meshes_to_remove = Vec::new();

        {
            let Some(world) = self.world.as_ref() else {
                return;
            };
            let Some(renderer) = self.renderer.as_ref() else {
                return;
            };
            let visibility = renderer.chunk_visibility(self.camera);
            let rendered_coords: Vec<ChunkCoord> =
                renderer.chunk_mesh_info().map(|info| info.coord).collect();

            for coord in rendered_coords {
                if chunk_is_render_candidate(world, center, coord, settings) {
                    retained_rendered_coords.push(coord);
                } else {
                    meshes_to_remove.push(coord);
                }
            }

            retained_rendered_coords.sort_by_cached_key(|coord| {
                chunk_render_priority_key(world, self.camera, visibility, center, *coord, settings)
            });
        }

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

        let available_queue_slots = config::MAX_DIRTY_MESH_QUEUE_BACKLOG
            .saturating_sub(self.dirty_mesh_queue.pending_count());
        if available_queue_slots == 0 {
            return;
        }

        let mut chunks_to_mesh: Vec<ChunkCoord> = {
            let Some(world) = self.world.as_ref() else {
                return;
            };
            let Some(renderer) = self.renderer.as_ref() else {
                return;
            };
            let visibility = renderer.chunk_visibility(self.camera);

            let mut chunks: Vec<ChunkCoord> = world
                .chunks
                .keys()
                .copied()
                .filter(|coord| !rendered_lookup.contains(coord))
                .filter(|coord| {
                    chunk_is_visible_render_candidate(world, visibility, center, *coord, settings)
                })
                .collect();

            chunks.sort_by_cached_key(|coord| {
                chunk_render_priority_key(world, self.camera, visibility, center, *coord, settings)
            });
            chunks
        };
        chunks_to_mesh.truncate(
            config::MAX_NEW_RENDER_MESHES_QUEUED_PER_FRAME
                .min(available_queue_slots)
                .min(available_render_slots),
        );

        let mut queued_coords = Vec::new();
        let Some(world) = self.world.as_mut() else {
            return;
        };

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

fn camera_with_settings(mut camera: VoxelCamera, settings: GameSettings) -> VoxelCamera {
    camera.fov_y_radians = settings.fov_degrees.to_radians();
    camera.far = camera_far_plane(settings);
    camera
}

fn camera_far_plane(settings: GameSettings) -> f32 {
    let radius = settings.chunk_view_distance.max(1) as f32;

    (radius + 2.0) * CHUNK_SIZE as f32
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
    camera: VoxelCamera,
    viewport: PhysicalSize<u32>,
) -> VecDeque<ChunkCoord> {
    let radius = settings.render_radius.max(0);
    let mut coords = Vec::with_capacity(config::MAX_STREAMING_COORDS_PER_CENTER.min(4096));
    let mut seen = HashSet::new();

    push_near_streaming_coords(center, radius, &mut seen, &mut coords);
    push_visible_ray_streaming_coords(center, settings, camera, viewport, &mut seen, &mut coords);

    if coords.len() > config::MAX_STREAMING_COORDS_PER_CENTER {
        coords.select_nth_unstable_by_key(config::MAX_STREAMING_COORDS_PER_CENTER, |coord| {
            chunk_distance_key(center, *coord)
        });
        coords.truncate(config::MAX_STREAMING_COORDS_PER_CENTER);
    }

    coords.sort_by_cached_key(|coord| chunk_streaming_priority_key(camera, center, *coord));
    coords.into()
}

fn streaming_view_key(
    center: ChunkCoord,
    settings: WorldStreamingSettings,
    camera: VoxelCamera,
) -> StreamingViewKey {
    let bucket_size = config::STREAMING_VIEW_REBUILD_DEGREES.max(0.1).to_radians();

    StreamingViewKey {
        center,
        radius: settings.render_radius.max(0),
        yaw_bucket: (camera.yaw_radians / bucket_size).round() as i32,
        pitch_bucket: (camera.pitch_radians / bucket_size).round() as i32,
    }
}

fn push_near_streaming_coords(
    center: ChunkCoord,
    render_radius: i32,
    seen: &mut HashSet<ChunkCoord>,
    coords: &mut Vec<ChunkCoord>,
) {
    let radius = render_radius
        .min(config::STREAMING_NEAR_CHUNK_RADIUS)
        .max(0);

    for dy in -radius..=radius {
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy + dz * dz > radius * radius {
                    continue;
                }

                push_streaming_coord(center.offset(dx, dy, dz), seen, coords);
            }
        }
    }
}

fn push_visible_ray_streaming_coords(
    center: ChunkCoord,
    settings: WorldStreamingSettings,
    camera: VoxelCamera,
    viewport: PhysicalSize<u32>,
    seen: &mut HashSet<ChunkCoord>,
    coords: &mut Vec<ChunkCoord>,
) {
    let radius = settings.render_radius.max(0);
    let width = config::STREAMING_RAY_GRID_COLUMNS.max(1);
    let height = config::STREAMING_RAY_GRID_ROWS.max(1);
    let min_visible_y = visible_ray_min_chunk_y(center);
    let aspect = viewport.width.max(1) as f32 / viewport.height.max(1) as f32;
    let half_vertical_tan = (camera.fov_y_radians * 0.5).tan();
    let half_horizontal_tan = half_vertical_tan * aspect;
    let forward = normalize3(camera.forward_direction());
    let world_up = if dot3(forward, [0.0, 1.0, 0.0]).abs() > 0.98 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let right = normalize3(cross3(forward, world_up));
    let up = normalize3(cross3(right, forward));

    for row in 0..height {
        let v = sample_axis(row, height);

        for column in 0..width {
            let u = sample_axis(column, width);
            let direction = normalize3(add3(
                add3(forward, mul3(right, u * half_horizontal_tan)),
                mul3(up, v * half_vertical_tan),
            ));

            for distance in 1..=radius {
                let world_pos = add3(
                    camera.position,
                    mul3(direction, distance as f32 * CHUNK_SIZE as f32),
                );
                let coord = world_to_chunk_coord(
                    world_pos[0].floor() as i32,
                    world_pos[1].floor() as i32,
                    world_pos[2].floor() as i32,
                );

                if coord.y < min_visible_y {
                    break;
                }

                if chunk_within_render_distance(center, coord, settings) {
                    push_streaming_coord(coord, seen, coords);
                }
            }
        }
    }
}

fn visible_ray_min_chunk_y(center: ChunkCoord) -> i32 {
    if center.y >= 0 {
        -config::STREAMING_VISIBLE_RAY_MIN_CHUNKS_BELOW_SURFACE
    } else {
        center.y - config::STREAMING_NEAR_CHUNK_RADIUS
    }
}

fn sample_axis(index: usize, count: usize) -> f32 {
    if count <= 1 {
        return 0.0;
    }

    index as f32 / (count - 1) as f32 * 2.0 - 1.0
}

fn push_streaming_coord(
    coord: ChunkCoord,
    seen: &mut HashSet<ChunkCoord>,
    coords: &mut Vec<ChunkCoord>,
) {
    if seen.insert(coord) {
        coords.push(coord);
    }
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn mul3(vector: [f32; 3], scalar: f32) -> [f32; 3] {
    [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(vector: [f32; 3]) -> [f32; 3] {
    let length = dot3(vector, vector).sqrt();

    if length <= f32::EPSILON {
        return [0.0, 0.0, 0.0];
    }

    [vector[0] / length, vector[1] / length, vector[2] / length]
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
    let radius = settings.render_radius.max(0);

    chunk_distance_key(center, coord) <= radius * radius
}

fn chunk_is_render_candidate(
    world: &VoxelWorld,
    center: ChunkCoord,
    coord: ChunkCoord,
    settings: WorldStreamingSettings,
) -> bool {
    chunk_within_render_distance(center, coord, settings) && chunk_is_render_skin(world, coord)
}

fn chunk_is_visible_render_candidate(
    world: &VoxelWorld,
    visibility: ChunkVisibility,
    center: ChunkCoord,
    coord: ChunkCoord,
    settings: WorldStreamingSettings,
) -> bool {
    chunk_is_render_candidate(world, center, coord, settings) && visibility.contains(coord)
}

fn chunk_render_priority_key(
    world: &VoxelWorld,
    camera: VoxelCamera,
    visibility: ChunkVisibility,
    center: ChunkCoord,
    coord: ChunkCoord,
    settings: WorldStreamingSettings,
) -> (u8, i32, i32, i32, i32, i32) {
    let visibility_rank =
        if chunk_is_visible_render_candidate(world, visibility, center, coord, settings) {
            0
        } else if chunk_is_render_candidate(world, center, coord, settings) {
            1
        } else {
            2
        };

    (
        visibility_rank,
        chunk_camera_depth_key(camera, coord),
        chunk_distance_key(center, coord),
        coord.y,
        coord.z,
        coord.x,
    )
}

fn chunk_streaming_priority_key(
    camera: VoxelCamera,
    center: ChunkCoord,
    coord: ChunkCoord,
) -> (u8, i32, i32, i32, i32, i32) {
    (
        (camera.chunk_depth(coord) < -(CHUNK_SIZE as f32)) as u8,
        chunk_camera_depth_key(camera, coord),
        chunk_distance_key(center, coord),
        coord.y,
        coord.z,
        coord.x,
    )
}

fn chunk_camera_depth_key(camera: VoxelCamera, coord: ChunkCoord) -> i32 {
    let depth = camera.chunk_depth(coord);

    if !depth.is_finite() {
        return i32::MAX;
    }

    (depth.max(0.0) * 16.0).round() as i32
}

fn streaming_coord_occluded_by_loaded_world(
    world: &VoxelWorld,
    camera: VoxelCamera,
    target: ChunkCoord,
) -> bool {
    let center = camera_chunk_coord(camera);
    let near_radius_sq = config::STREAMING_NEAR_CHUNK_RADIUS * config::STREAMING_NEAR_CHUNK_RADIUS;

    if chunk_distance_key(center, target) <= near_radius_sq {
        return false;
    }

    let target_center = chunk_world_center(target);
    let to_target = [
        target_center[0] - camera.position[0],
        target_center[1] - camera.position[1],
        target_center[2] - camera.position[2],
    ];
    let distance = dot3(to_target, to_target).sqrt();

    if distance <= CHUNK_SIZE as f32 * 2.0 {
        return false;
    }

    let direction = normalize3(to_target);
    let steps = (distance / (CHUNK_SIZE as f32 * 0.5)).ceil().max(1.0) as i32;
    let mut last_coord = None;

    for step in 1..steps {
        let sample_distance = step as f32 / steps as f32 * distance;
        let position = add3(camera.position, mul3(direction, sample_distance));
        let coord = world_to_chunk_coord(
            position[0].floor() as i32,
            position[1].floor() as i32,
            position[2].floor() as i32,
        );

        if coord == center || coord == target || last_coord == Some(coord) {
            continue;
        }

        last_coord = Some(coord);

        if world.get_chunk(coord).is_some_and(|chunk| {
            !chunk.is_empty() && chunk_has_sky_exposed_surface(world, coord, chunk)
        }) {
            return true;
        }
    }

    false
}

fn chunk_world_center(coord: ChunkCoord) -> [f32; 3] {
    let size = CHUNK_SIZE as f32;

    [
        coord.x as f32 * size + size * 0.5,
        coord.y as f32 * size + size * 0.5,
        coord.z as f32 * size + size * 0.5,
    ]
}

fn chunk_is_render_skin(world: &VoxelWorld, coord: ChunkCoord) -> bool {
    let Some(chunk) = world.get_chunk(coord) else {
        return false;
    };

    if chunk.is_empty() {
        return false;
    }

    chunk_has_sky_exposed_surface(world, coord, chunk)
}

fn chunk_has_sky_exposed_surface(world: &VoxelWorld, coord: ChunkCoord, chunk: &Chunk) -> bool {
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            for y in (0..CHUNK_SIZE).rev() {
                if chunk.get_block(x, y, z) == AIR_BLOCK {
                    continue;
                }

                if block_has_loaded_air_above(world, coord, chunk, x, y, z) {
                    return true;
                }

                break;
            }
        }
    }

    false
}

fn block_has_loaded_air_above(
    world: &VoxelWorld,
    coord: ChunkCoord,
    chunk: &Chunk,
    x: usize,
    y: usize,
    z: usize,
) -> bool {
    if y + 1 < CHUNK_SIZE {
        return chunk.get_block(x, y + 1, z) == AIR_BLOCK;
    }

    world
        .get_chunk(coord.offset(0, 1, 0))
        .is_some_and(|above| above.get_block(x, 0, z) == AIR_BLOCK)
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

    for dirty_coord in [coord].into_iter().chain(neighbor_chunk_coords(coord)) {
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
            let third_y = height * 0.54;

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

            push_text(
                &mut overlay,
                label_x,
                third_y + 11.0,
                2.4,
                "FOV",
                [0.86, 0.91, 0.93, 1.0],
            );
            push_textbox(
                &mut overlay,
                &mut hits,
                MenuAction::Fov,
                ScreenRect::new(field_x, third_y, 260.0, 52.0),
                &settings_inputs.fov_degrees,
                active_field == Some(SettingsField::Fov),
            );

            push_button(
                &mut overlay,
                &mut hits,
                MenuAction::Back,
                ScreenRect::centered(width, height * 0.72, 240.0, 52.0),
                "BACK",
            );
        }
        AppScreen::LoadingWorld | AppScreen::InGame => {}
    }

    MenuLayout { overlay, hits }
}

fn build_loading_overlay(progress: f32, size: PhysicalSize<u32>) -> UiOverlay {
    const BAR_WIDTH: f32 = 340.0;
    const BAR_HEIGHT: f32 = 10.0;
    const BAR_INSET: f32 = 2.0;

    let width = size.width.max(1) as f32;
    let height = size.height.max(1) as f32;
    let progress = progress.clamp(0.0, 1.0);
    let mut overlay = UiOverlay::default();
    let background = ScreenRect::new(0.0, 0.0, width, height);
    let bar = ScreenRect::centered(width, height * 0.55, BAR_WIDTH, BAR_HEIGHT);
    let fill_width = (BAR_WIDTH - BAR_INSET * 2.0) * progress;

    push_ui_rect(&mut overlay, background, [0.0, 0.0, 0.0, 1.0]);
    push_ui_rect(&mut overlay, bar, [0.0, 0.0, 0.0, 1.0]);
    push_ui_border(&mut overlay, bar, 1.0, [1.0, 1.0, 1.0, 0.85]);

    if fill_width > 0.0 {
        push_ui_rect(
            &mut overlay,
            ScreenRect::new(
                bar.x + BAR_INSET,
                bar.y + BAR_INSET,
                fill_width,
                BAR_HEIGHT - BAR_INSET * 2.0,
            ),
            [1.0, 1.0, 1.0, 0.95],
        );
    }

    overlay
}

fn build_gameplay_overlay(
    hotbar: &Hotbar,
    break_progress: Option<f32>,
    size: PhysicalSize<u32>,
) -> UiOverlay {
    let width = size.width.max(1) as f32;
    let height = size.height.max(1) as f32;
    let mut overlay = UiOverlay::default();

    if let Some(progress) = break_progress {
        push_break_progress_bar(&mut overlay, width, height, progress);
    }

    let slot_size = 48.0;
    let slot_gap = 6.0;
    let slot_count = HOTBAR_SLOT_COUNT as f32;
    let bar_width = slot_size * slot_count + slot_gap * (slot_count - 1.0);
    let x_start = ((width - bar_width) * 0.5).max(8.0);
    let y = (height - slot_size - 26.0).max(8.0);
    let selected_block = hotbar.selected_block();
    let selected_label = block_ui_label(selected_block);

    push_centered_text(
        &mut overlay,
        selected_label,
        2.0,
        width,
        (y - 28.0).max(8.0),
        [1.0, 1.0, 1.0, 1.0],
    );

    for (slot, block) in hotbar.slots().iter().copied().enumerate() {
        let x = x_start + slot as f32 * (slot_size + slot_gap);
        let rect = ScreenRect::new(x, y, slot_size, slot_size);
        let is_selected = slot == hotbar.selected_slot();

        push_ui_rect(&mut overlay, rect, [0.035, 0.042, 0.047, 0.92]);
        push_ui_border(
            &mut overlay,
            rect,
            if is_selected { 3.0 } else { 2.0 },
            if is_selected {
                [1.0, 0.78, 0.28, 1.0]
            } else {
                [0.36, 0.43, 0.46, 1.0]
            },
        );

        let swatch = ScreenRect::new(rect.x + 11.0, rect.y + 12.0, 26.0, 26.0);
        if block == AIR_BLOCK {
            push_ui_rect(&mut overlay, swatch, [0.075, 0.085, 0.09, 0.86]);
            push_centered_text_in_rect(&mut overlay, swatch, "-", 2.2, [0.62, 0.68, 0.70, 1.0]);
        } else if let Some(uvs) = block_hotbar_uvs(block) {
            push_ui_texture_rect(&mut overlay, swatch, block_ui_color(block), uvs);
            push_ui_border(&mut overlay, swatch, 1.0, [0.0, 0.0, 0.0, 0.42]);
        } else {
            push_ui_rect(&mut overlay, swatch, block_ui_color(block));
            push_ui_border(&mut overlay, swatch, 1.0, [0.0, 0.0, 0.0, 0.42]);
        }

        push_text(
            &mut overlay,
            rect.x + 5.0,
            rect.y + 4.0,
            1.3,
            &(slot + 1).to_string(),
            [0.86, 0.90, 0.92, 1.0],
        );
    }

    overlay
}

fn push_break_progress_bar(overlay: &mut UiOverlay, width: f32, height: f32, progress: f32) {
    const BAR_WIDTH: f32 = 116.0;
    const BAR_HEIGHT: f32 = 8.0;
    const BAR_Y_OFFSET: f32 = 26.0;
    const BAR_INSET: f32 = 2.0;

    let progress = progress.clamp(0.0, 1.0);
    let background = ScreenRect::new(
        (width - BAR_WIDTH) * 0.5,
        height * 0.5 + BAR_Y_OFFSET,
        BAR_WIDTH,
        BAR_HEIGHT,
    );
    let fill_width = (BAR_WIDTH - BAR_INSET * 2.0) * progress;

    push_ui_rect(overlay, background, [0.0, 0.0, 0.0, 0.62]);
    push_ui_border(overlay, background, 1.0, [1.0, 1.0, 1.0, 0.78]);

    if fill_width > 0.0 {
        push_ui_rect(
            overlay,
            ScreenRect::new(
                background.x + BAR_INSET,
                background.y + BAR_INSET,
                fill_width,
                BAR_HEIGHT - BAR_INSET * 2.0,
            ),
            [1.0, 1.0, 1.0, 0.95],
        );
    }
}

fn block_ui_label(block: BlockId) -> &'static str {
    block_label(block)
}

fn block_ui_color(block: BlockId) -> [f32; 4] {
    block_color_rgba(block)
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

fn push_ui_texture_rect(
    overlay: &mut UiOverlay,
    rect: ScreenRect,
    color: [f32; 4],
    uvs: [[f32; 2]; 4],
) {
    overlay
        .items
        .push(UiOverlayItem::Texture(UiTextureRect::new(
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            color,
            uvs,
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
        SettingsField::MouseSensitivity | SettingsField::Fov => {
            character.is_ascii_digit() || character == '.'
        }
        SettingsField::ChunkViewDistance => character.is_ascii_digit(),
    }
}

fn parse_clamped_i32_setting(input: &str, fallback: i32, min: i32, max: i32) -> i32 {
    let input = input.trim();

    if input.is_empty() {
        return fallback;
    }

    input
        .parse::<i64>()
        .map(|value| value.clamp(min as i64, max as i64) as i32)
        .unwrap_or(max)
}

fn parse_clamped_f32_setting(input: &str, fallback: f32, min: f32, max: f32) -> f32 {
    let input = input.trim();

    if input.is_empty() {
        return fallback;
    }

    input
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| value.clamp(min as f64, max as f64) as f32)
        .unwrap_or(max)
}

fn quantize_break_progress(progress: f32) -> u8 {
    (progress.clamp(0.0, 1.0) * 100.0).round() as u8
}

#[derive(Debug, Default, Clone, Copy)]
struct BlockBreakState {
    target: Option<BlockBreakTarget>,
}

impl BlockBreakState {
    fn reset(&mut self) {
        self.target = None;
    }

    fn apply_damage(&mut self, hit: physics::VoxelRayHit, block_hp: f32, damage: f32) -> bool {
        if block_hp <= 0.0 {
            return true;
        }

        if damage <= 0.0 {
            return false;
        }

        let target = self.target.get_or_insert(BlockBreakTarget {
            block: hit.world_block,
            block_id: hit.block_id,
            accumulated_damage: 0.0,
        });

        if target.block != hit.world_block || target.block_id != hit.block_id {
            *target = BlockBreakTarget {
                block: hit.world_block,
                block_id: hit.block_id,
                accumulated_damage: 0.0,
            };
        }

        target.accumulated_damage += damage;
        target.accumulated_damage >= block_hp
    }

    fn progress_ratio(self, block_hp: f32) -> Option<f32> {
        if block_hp <= 0.0 {
            return None;
        }

        self.target
            .map(|target| (target.accumulated_damage / block_hp).clamp(0.0, 1.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BlockBreakTarget {
    block: BlockPos,
    block_id: BlockId,
    accumulated_damage: f32,
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
        self.on_ground = self.velocity[1] <= 0.0
            && aabb_has_ground_support(
                world,
                self.center,
                PLAYER_HALF_EXTENTS,
                config::PLAYER_GROUND_SUPPORT_DISTANCE,
            );

        let current_velocity = (self.velocity[0], self.velocity[2]);
        let target_velocity = horizontal_target_velocity(yaw_radians, input, self.on_ground);
        let (velocity_x, velocity_z) = if self.on_ground {
            integrate_horizontal_velocity(
                current_velocity,
                target_velocity,
                config::PLAYER_MOVE_ACCELERATION,
                dt,
            )
        } else {
            integrate_air_horizontal_velocity(
                current_velocity,
                target_velocity,
                config::PLAYER_MOVE_ACCELERATION,
                dt,
            )
        };

        self.velocity[0] = velocity_x;
        self.velocity[2] = velocity_z;

        if input.up && self.on_ground {
            self.velocity[1] = jump_velocity_from_height(config::PLAYER_JUMP_HEIGHT);
            self.on_ground = false;
        } else if self.on_ground {
            self.velocity[1] = 0.0;
        } else {
            self.velocity[1] = (self.velocity[1] - config::PLAYER_GRAVITY * dt)
                .max(-config::PLAYER_MAX_FALL_SPEED);
        }

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

fn horizontal_target_velocity(yaw_radians: f32, input: InputState, on_ground: bool) -> (f32, f32) {
    let forward_axis = input.forward_axis();
    let right_axis = input.right_axis();
    let forward = (yaw_radians.cos(), yaw_radians.sin());
    let right = (-yaw_radians.sin(), yaw_radians.cos());
    let (forward_speed, side_speed) = horizontal_axis_speeds(input, on_ground);

    (
        forward.0 * forward_axis * forward_speed + right.0 * right_axis * side_speed,
        forward.1 * forward_axis * forward_speed + right.1 * right_axis * side_speed,
    )
}

fn horizontal_axis_speeds(input: InputState, on_ground: bool) -> (f32, f32) {
    if !on_ground {
        return (
            config::PLAYER_IN_AIR_MOVE_SPEED,
            config::PLAYER_IN_AIR_MOVE_SPEED,
        );
    }

    if input.slow {
        return (config::PLAYER_CROUCH_SPEED, config::PLAYER_CROUCH_SPEED);
    }

    let forward_speed = if input.can_sprint_forward() {
        config::PLAYER_SPRINT_SPEED
    } else {
        config::PLAYER_WALK_SPEED
    };

    (forward_speed, config::PLAYER_WALK_SPEED)
}

fn integrate_horizontal_velocity(
    current: (f32, f32),
    target: (f32, f32),
    move_acceleration: f32,
    dt: f32,
) -> (f32, f32) {
    if move_acceleration <= 0.0 || dt <= 0.0 {
        return current;
    }

    let acceleration = (
        move_acceleration * (target.0 - current.0),
        move_acceleration * (target.1 - current.1),
    );
    let velocity_delta = (acceleration.0 * dt, acceleration.1 * dt);

    if move_acceleration * dt >= 1.0 {
        target
    } else {
        (current.0 + velocity_delta.0, current.1 + velocity_delta.1)
    }
}

fn integrate_air_horizontal_velocity(
    current: (f32, f32),
    target: (f32, f32),
    move_acceleration: f32,
    dt: f32,
) -> (f32, f32) {
    if horizontal_speed_squared(target) <= f32::EPSILON {
        return current;
    }

    let current_speed = horizontal_speed(current);
    let velocity = integrate_horizontal_velocity(current, target, move_acceleration, dt);
    let velocity_speed = horizontal_speed(velocity);

    if velocity_speed >= current_speed || current_speed <= f32::EPSILON {
        return velocity;
    }

    if velocity_speed <= f32::EPSILON {
        return current;
    }

    let scale = current_speed / velocity_speed;

    (velocity.0 * scale, velocity.1 * scale)
}

fn horizontal_speed(velocity: (f32, f32)) -> f32 {
    horizontal_speed_squared(velocity).sqrt()
}

fn horizontal_speed_squared(velocity: (f32, f32)) -> f32 {
    velocity.0 * velocity.0 + velocity.1 * velocity.1
}

fn jump_velocity_from_height(height: f32) -> f32 {
    (2.0 * config::PLAYER_GRAVITY * height.max(0.0)).sqrt()
}

#[derive(Debug, Default)]
struct DirtyMeshQueue {
    pending: VecDeque<ChunkCoord>,
    pending_set: HashSet<ChunkCoord>,
    needs_priority_sort: bool,
}

impl DirtyMeshQueue {
    #[cfg(test)]
    fn enqueue_dirty(&mut self, world: &VoxelWorld) {
        for coord in dirty_window_chunk_coords(world) {
            self.enqueue(coord);
        }
    }

    fn prioritize(
        &mut self,
        world: &VoxelWorld,
        camera: VoxelCamera,
        visibility: ChunkVisibility,
        center: ChunkCoord,
        settings: WorldStreamingSettings,
    ) {
        if !self.needs_priority_sort || self.pending.len() < 2 {
            self.needs_priority_sort = false;
            return;
        }

        self.pending.make_contiguous().sort_by_cached_key(|coord| {
            chunk_render_priority_key(world, camera, visibility, center, *coord, settings)
        });
        self.needs_priority_sort = false;
    }

    fn request_prioritize(&mut self) {
        self.needs_priority_sort = true;
    }

    fn enqueue(&mut self, coord: ChunkCoord) {
        if self.pending_set.insert(coord) {
            self.pending.push_back(coord);
            self.needs_priority_sort = true;
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
        if self.pending_set.remove(&coord) {
            self.pending.retain(|pending| *pending != coord);
        }

        self.pending_set.insert(coord);
        self.pending.push_front(coord);
        self.needs_priority_sort = true;
    }

    fn pop_dirty(&mut self, world: &VoxelWorld) -> Option<ChunkCoord> {
        while let Some(coord) = self.pending.pop_front() {
            self.pending_set.remove(&coord);

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

    fn mesh_pending_count(&self) -> usize {
        self.pending_meshes.len() + self.pending_priority_meshes.len()
    }

    fn generation_pending_count(&self) -> usize {
        self.pending_generation.len()
    }

    fn drain_results(&mut self, max_results: usize) -> Vec<ChunkJobResult> {
        let mut results = Vec::new();

        for _ in 0..max_results {
            let Ok(result) = self.receiver.try_recv() else {
                break;
            };

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

#[allow(clippy::large_enum_variant)]
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
    view_key: Option<StreamingViewKey>,
    pending_loads: VecDeque<ChunkCoord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StreamingViewKey {
    center: ChunkCoord,
    radius: i32,
    yaw_bucket: i32,
    pitch_bucket: i32,
}

impl ChunkStreamingState {
    fn new(settings: WorldStreamingSettings) -> Self {
        Self {
            settings,
            view_key: None,
            pending_loads: VecDeque::new(),
        }
    }

    fn set_settings(&mut self, settings: WorldStreamingSettings) {
        if self.settings == settings {
            return;
        }

        self.settings = settings;
        self.view_key = None;
        self.pending_loads.clear();
    }

    fn update_view(
        &mut self,
        center: ChunkCoord,
        camera: VoxelCamera,
        viewport: PhysicalSize<u32>,
    ) -> bool {
        let key = streaming_view_key(center, self.settings, camera);

        if self.view_key == Some(key) {
            return false;
        }

        self.view_key = Some(key);
        self.pending_loads = streaming_chunk_coords(center, self.settings, camera, viewport);
        true
    }

    #[cfg(test)]
    fn pop_missing(&mut self, world: &VoxelWorld) -> Option<ChunkCoord> {
        while let Some(coord) = self.pending_loads.pop_front() {
            if world.get_chunk(coord).is_none() {
                return Some(coord);
            }
        }

        None
    }

    fn pop_missing_visible(
        &mut self,
        world: &VoxelWorld,
        camera: VoxelCamera,
    ) -> Option<ChunkCoord> {
        while let Some(coord) = self.pending_loads.pop_front() {
            if world.get_chunk(coord).is_some() {
                continue;
            }

            if streaming_coord_occluded_by_loaded_world(world, camera, coord) {
                continue;
            }

            return Some(coord);
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
    slow: bool,
    fast: bool,
    breaking: bool,
}

impl InputState {
    fn set_key(&mut self, key: KeyCode, pressed: bool) {
        match key {
            KeyCode::KeyW => self.forward = pressed,
            KeyCode::KeyS => self.backward = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::AltLeft | KeyCode::AltRight => self.down = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.fast = pressed,
            KeyCode::ControlLeft | KeyCode::ControlRight => self.slow = pressed,
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

    fn can_sprint_forward(self) -> bool {
        self.fast && self.forward && !self.backward && !self.slow
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorldLoadPhase {
    Generating,
    Meshing,
}

struct WorldLoadState {
    world: VoxelWorld,
    coords: Vec<ChunkCoord>,
    next_generation: usize,
    generated_chunks: usize,
    mesh_queue: VecDeque<ChunkCoord>,
    mesh_target_chunks: usize,
    mesh_results: usize,
    mesh_jobs_queued: usize,
    phase: WorldLoadPhase,
}

impl WorldLoadState {
    fn new(
        streaming_settings: WorldStreamingSettings,
        camera: VoxelCamera,
        viewport: PhysicalSize<u32>,
    ) -> Self {
        Self {
            world: VoxelWorld::new(config::WORLD_SEED),
            coords: initial_world_preload_coords(streaming_settings, camera, viewport),
            next_generation: 0,
            generated_chunks: 0,
            mesh_queue: VecDeque::new(),
            mesh_target_chunks: 0,
            mesh_results: 0,
            mesh_jobs_queued: 0,
            phase: WorldLoadPhase::Generating,
        }
    }

    fn generate_chunks(&mut self, saved_chunks: &HashMap<ChunkCoord, Chunk>) {
        if self.phase != WorldLoadPhase::Generating {
            return;
        }

        for _ in 0..config::LOADING_CHUNKS_GENERATED_PER_FRAME {
            let Some(coord) = self.coords.get(self.next_generation).copied() else {
                break;
            };

            let chunk = saved_chunks
                .get(&coord)
                .cloned()
                .unwrap_or_else(|| VoxelWorld::generate_chunk(self.world.seed(), coord));
            self.world.insert_chunk(chunk);
            self.next_generation += 1;
            self.generated_chunks += 1;
        }
    }

    fn is_generation_complete(&self) -> bool {
        self.generated_chunks >= self.coords.len()
    }

    fn meshing_started(&self) -> bool {
        self.phase == WorldLoadPhase::Meshing
    }

    fn start_meshing(
        &mut self,
        camera: VoxelCamera,
        visibility: ChunkVisibility,
        settings: WorldStreamingSettings,
    ) {
        if self.phase == WorldLoadPhase::Meshing {
            return;
        }

        let center = camera_chunk_coord(camera);
        let mut coords: Vec<ChunkCoord> = self
            .coords
            .iter()
            .copied()
            .filter(|coord| {
                chunk_is_visible_render_candidate(&self.world, visibility, center, *coord, settings)
            })
            .collect();

        if coords.is_empty() {
            coords = self
                .coords
                .iter()
                .copied()
                .filter(|coord| chunk_is_render_candidate(&self.world, center, *coord, settings))
                .collect();
        }

        coords.sort_by_cached_key(|coord| {
            chunk_render_priority_key(&self.world, camera, visibility, center, *coord, settings)
        });
        coords.truncate(config::MAX_RENDERED_CHUNK_MESHES);

        self.mesh_target_chunks = coords.len();
        self.mesh_queue = coords.into();
        self.phase = WorldLoadPhase::Meshing;
    }

    fn schedule_mesh_jobs(&mut self, chunk_jobs: &mut ChunkJobQueue) {
        if self.phase != WorldLoadPhase::Meshing {
            return;
        }

        let mut queued_this_frame = 0;

        while queued_this_frame < config::LOADING_MESH_JOBS_QUEUED_PER_FRAME
            && chunk_jobs.mesh_pending_count() < config::MAX_LOADING_MESH_JOBS_PENDING
        {
            let Some(coord) = self.mesh_queue.pop_front() else {
                break;
            };

            if chunk_jobs.is_mesh_pending(coord) {
                continue;
            }

            let Some(input) = ChunkMeshInput::from_world(&self.world, coord) else {
                self.mesh_target_chunks = self.mesh_target_chunks.saturating_sub(1);
                continue;
            };

            let queued = if self.mesh_jobs_queued % 2 == 0 {
                chunk_jobs.enqueue_mesh(input)
            } else {
                chunk_jobs.enqueue_mesh_priority(input)
            };

            if queued {
                self.mesh_jobs_queued += 1;
                queued_this_frame += 1;
            } else {
                self.mesh_target_chunks = self.mesh_target_chunks.saturating_sub(1);
            }
        }
    }

    fn record_mesh_result(&mut self) {
        self.mesh_results += 1;
    }

    fn is_complete(&self, chunk_jobs: &ChunkJobQueue) -> bool {
        self.phase == WorldLoadPhase::Meshing
            && self.mesh_queue.is_empty()
            && self.mesh_results >= self.mesh_target_chunks
            && chunk_jobs.mesh_pending_count() == 0
    }

    fn progress(&self) -> f32 {
        let chunk_progress = progress_ratio(self.generated_chunks, self.coords.len());

        if self.phase == WorldLoadPhase::Generating {
            return chunk_progress * 0.75;
        }

        0.75 + progress_ratio(self.mesh_results, self.mesh_target_chunks) * 0.25
    }
}

fn initial_world_preload_coords(
    streaming_settings: WorldStreamingSettings,
    camera: VoxelCamera,
    viewport: PhysicalSize<u32>,
) -> Vec<ChunkCoord> {
    let center = camera_chunk_coord(camera);
    let mut coords: Vec<ChunkCoord> =
        streaming_chunk_coords(center, streaming_settings, camera, viewport)
            .into_iter()
            .collect();

    coords.truncate(config::INITIAL_WORLD_PRELOAD_CHUNK_LIMIT);

    if !coords.contains(&center) {
        coords.insert(0, center);
    }

    coords
}

fn progress_ratio(done: usize, total: usize) -> f32 {
    if total == 0 {
        return 1.0;
    }

    done.min(total) as f32 / total as f32
}

#[cfg(test)]
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
    use voxels::{
        Chunk, CHUNK_VOLUME, COAL_ORE_BLOCK, DIRT_BLOCK, GRASS_BLOCK, IRON_ORE_BLOCK, STONE_BLOCK,
    };

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

    fn full_stone_chunk(coord: ChunkCoord) -> Chunk {
        Chunk::from_blocks(coord, Box::new([STONE_BLOCK; CHUNK_VOLUME]), 0)
    }

    fn world_with_floor() -> VoxelWorld {
        let mut world = world_with_empty_chunks([ChunkCoord::new(0, 0, 0)]);

        for x in 0..4 {
            for z in 0..4 {
                world.set_block(BlockPos::new(x, 0, z), STONE_BLOCK);
            }
        }

        world
    }

    fn test_ray_hit(world_block: BlockPos, block_id: BlockId) -> physics::VoxelRayHit {
        physics::VoxelRayHit {
            world_block,
            chunk_coord: world_to_chunk_coord(world_block.x, world_block.y, world_block.z),
            local_block: (0, 0, 0),
            block_id,
            face_normal: [-1, 0, 0],
            distance: 1.0,
        }
    }

    fn assert_f32_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 0.0001,
            "expected {expected}, got {actual}"
        );
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
    fn settings_layout_has_fov_field() {
        let layout = build_menu_layout(
            AppScreen::Settings,
            &SettingsInputs::from_settings(GameSettings::default()),
            None,
            PhysicalSize::new(1280, 720),
        );

        assert!(layout.hits.iter().any(|hit| hit.action == MenuAction::Fov));
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
    fn settings_parsers_clamp_long_user_values() {
        assert_eq!(
            parse_clamped_i32_setting(
                "999999999999999999999999",
                config::DEFAULT_RENDER_CHUNK_DISTANCE,
                config::MIN_RENDER_CHUNK_DISTANCE,
                config::MAX_RENDER_CHUNK_DISTANCE,
            ),
            config::MAX_RENDER_CHUNK_DISTANCE
        );
        assert_eq!(
            parse_clamped_f32_setting(
                "999999999999999999999999.0",
                config::DEFAULT_FOV_DEGREES,
                config::MIN_FOV_DEGREES,
                config::MAX_FOV_DEGREES,
            ),
            config::MAX_FOV_DEGREES
        );
    }

    #[test]
    fn gameplay_overlay_draws_hotbar_slots() {
        let overlay =
            build_gameplay_overlay(&Hotbar::starter(), None, PhysicalSize::new(1280, 720));

        assert!(overlay.items.len() >= HOTBAR_SLOT_COUNT * 4);
    }

    #[test]
    fn gameplay_overlay_hides_break_progress_when_not_breaking() {
        let hidden = build_gameplay_overlay(&Hotbar::starter(), None, PhysicalSize::new(1280, 720));
        let visible =
            build_gameplay_overlay(&Hotbar::starter(), Some(0.45), PhysicalSize::new(1280, 720));

        assert!(visible.items.len() > hidden.items.len());
    }

    #[test]
    fn loading_overlay_draws_blank_screen_and_progress_bar() {
        let overlay = build_loading_overlay(0.5, PhysicalSize::new(1280, 720));

        assert!(overlay.items.len() >= 4);
        assert!(matches!(
            overlay.items.first(),
            Some(UiOverlayItem::Rect(rect)) if rect.color == [0.0, 0.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn initial_world_preload_coords_include_center_and_respect_cap() {
        let camera = VoxelCamera::looking_at_chunk_origin();
        let center = camera_chunk_coord(camera);
        let coords = initial_world_preload_coords(
            WorldStreamingSettings::new(99, 99, 99, 99, 99, 100),
            camera,
            PhysicalSize::new(1280, 720),
        );

        assert!(coords.contains(&center));
        assert!(coords.len() <= config::INITIAL_WORLD_PRELOAD_CHUNK_LIMIT + 1);
    }

    #[test]
    fn gameplay_overlay_uses_texture_rects_for_hotbar_blocks() {
        let overlay =
            build_gameplay_overlay(&Hotbar::starter(), None, PhysicalSize::new(1280, 720));

        assert!(overlay
            .items
            .iter()
            .any(|item| matches!(item, UiOverlayItem::Texture(_))));
    }

    #[test]
    fn break_progress_is_quantized_to_percent_steps() {
        assert_eq!(quantize_break_progress(-1.0), 0);
        assert_eq!(quantize_break_progress(0.456), 46);
        assert_eq!(quantize_break_progress(2.0), 100);
    }

    #[test]
    fn block_ui_labels_match_starter_hotbar_blocks() {
        assert_eq!(block_ui_label(DIRT_BLOCK), "DIRT");
        assert_eq!(block_ui_label(GRASS_BLOCK), "GRASS");
        assert_eq!(block_ui_label(STONE_BLOCK), "STONE");
        assert_eq!(block_ui_label(COAL_ORE_BLOCK), "COAL ORE");
        assert_eq!(block_ui_label(IRON_ORE_BLOCK), "IRON ORE");
        assert_eq!(block_ui_label(AIR_BLOCK), "EMPTY");
    }

    #[test]
    fn block_ui_colors_come_from_block_config() {
        assert_eq!(block_ui_color(DIRT_BLOCK), block_color_rgba(DIRT_BLOCK));
        assert_eq!(
            block_ui_color(IRON_ORE_BLOCK),
            block_color_rgba(IRON_ORE_BLOCK)
        );
    }

    #[test]
    fn block_break_state_accumulates_damage_until_block_hp_is_reached() {
        let mut breaking = BlockBreakState::default();
        let hit = test_ray_hit(BlockPos::new(3, 0, 0), STONE_BLOCK);
        let block_hp = block_break_hp(STONE_BLOCK);

        assert!(!breaking.apply_damage(hit, block_hp, block_hp * 0.5));
        assert!(!breaking.apply_damage(hit, block_hp, block_hp * 0.49));
        assert!(breaking.apply_damage(hit, block_hp, block_hp * 0.01));
    }

    #[test]
    fn block_break_state_retargeting_resets_accumulated_damage() {
        let mut breaking = BlockBreakState::default();
        let first_hit = test_ray_hit(BlockPos::new(3, 0, 0), DIRT_BLOCK);
        let second_hit = test_ray_hit(BlockPos::new(4, 0, 0), DIRT_BLOCK);
        let block_hp = block_break_hp(DIRT_BLOCK);

        assert!(!breaking.apply_damage(first_hit, block_hp, block_hp * 0.9));
        assert!(!breaking.apply_damage(second_hit, block_hp, block_hp * 0.2));
        assert!(breaking.apply_damage(second_hit, block_hp, block_hp * 0.8));
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
        let coords = streaming_chunk_coords(
            center,
            WorldStreamingSettings::prototype(),
            VoxelCamera::looking_at_chunk_origin(),
            PhysicalSize::new(1280, 720),
        );

        assert_eq!(coords.front().copied(), Some(center));
        assert!(coords.contains(&center.offset(0, 1, 0)));
        assert!(!coords.contains(&center.offset(1, 1, -1)));
    }

    #[test]
    fn streaming_chunk_coords_reach_high_visible_distances() {
        let center = ChunkCoord::new(0, 0, 0);
        let settings = WorldStreamingSettings::new(99, 99, 99, 99, 99, 100);
        let mut camera = VoxelCamera::new([16.0, 16.0, 16.0], 0.0, 0.0);
        camera.fov_y_radians = 70.0_f32.to_radians();
        let coords = streaming_chunk_coords(center, settings, camera, PhysicalSize::new(1280, 720));

        assert!(coords.contains(&center.offset(99, 0, 0)));
    }

    #[test]
    fn streaming_chunk_coords_prioritize_front_to_back_depth() {
        let center = ChunkCoord::new(0, 0, 0);
        let settings = WorldStreamingSettings::new(12, 12, 12, 12, 12, 13);
        let camera = VoxelCamera::new([16.0, 16.0, 16.0], 0.0, 0.0);
        let coords = streaming_chunk_coords(center, settings, camera, PhysicalSize::new(1280, 720));
        let near = center.offset(1, 0, 0);
        let far = center.offset(8, 0, 0);
        let near_index = coords
            .iter()
            .position(|coord| *coord == near)
            .expect("near forward chunk should be streamed");
        let far_index = coords
            .iter()
            .position(|coord| *coord == far)
            .expect("far forward chunk should be streamed");

        assert!(near_index < far_index);
    }

    #[test]
    fn visible_ray_streaming_does_not_walk_into_deep_underground_chunks() {
        let center = ChunkCoord::new(0, 1, 0);
        let settings = WorldStreamingSettings::new(32, 32, 32, 32, 32, 33);
        let mut camera = VoxelCamera::new([16.0, 48.0, 16.0], 0.0, -55.0_f32.to_radians());
        camera.fov_y_radians = 70.0_f32.to_radians();
        let coords = streaming_chunk_coords(center, settings, camera, PhysicalSize::new(1280, 720));

        let near_radius_sq =
            config::STREAMING_NEAR_CHUNK_RADIUS * config::STREAMING_NEAR_CHUNK_RADIUS;

        assert!(!coords
            .iter()
            .any(|coord| coord.y < 0 && chunk_distance_key(center, *coord) > near_radius_sq));
    }

    #[test]
    fn streaming_skips_far_chunks_hidden_behind_loaded_surface_chunks() {
        let blocker = ChunkCoord::new(1, 0, 0);
        let target = ChunkCoord::new(4, 0, 0);
        let mut world = world_with_empty_chunks([blocker]);
        let camera = VoxelCamera::new([16.0, 16.0, 16.0], 0.0, 0.0);

        world.set_block(BlockPos::new(32, 0, 0), STONE_BLOCK);

        assert!(streaming_coord_occluded_by_loaded_world(
            &world, camera, target
        ));
        assert!(!streaming_coord_occluded_by_loaded_world(
            &world,
            camera,
            ChunkCoord::new(2, 0, 0)
        ));
    }

    #[test]
    fn chunk_streaming_state_skips_loaded_chunks() {
        let center = ChunkCoord::new(0, 0, 0);
        let mut state = ChunkStreamingState::new(WorldStreamingSettings::prototype());
        let world = world_with_empty_chunks([center]);

        assert!(state.update_view(
            center,
            VoxelCamera::looking_at_chunk_origin(),
            PhysicalSize::new(1280, 720)
        ));

        let first_missing = state.pop_missing(&world);

        assert_ne!(first_missing, Some(center));
        assert!(state.pending_count() > 0);
    }

    #[test]
    fn chunk_render_distance_uses_full_3d_radius() {
        let center = ChunkCoord::new(0, 0, 0);
        let settings = WorldStreamingSettings::new(8, 8, 8, 8, 8, 9);

        assert!(chunk_within_render_distance(
            center,
            center.offset(0, 8, 0),
            settings
        ));
        assert!(!chunk_within_render_distance(
            center,
            center.offset(6, 6, 0),
            settings
        ));
    }

    #[test]
    fn render_skin_requires_sky_exposed_surface() {
        let surface = ChunkCoord::new(0, 0, 0);
        let empty = ChunkCoord::new(1, 0, 0);
        let buried = ChunkCoord::new(5, -1, 0);
        let above_buried = buried.offset(0, 1, 0);
        let mut world = VoxelWorld::new(0);
        let mut surface_chunk = Chunk::new_empty(surface);

        surface_chunk.set_block(0, 0, 0, STONE_BLOCK);
        surface_chunk.clear_dirty();

        world.chunks.insert(surface, surface_chunk);
        world.chunks.insert(empty, Chunk::new_empty(empty));
        world.chunks.insert(buried, full_stone_chunk(buried));
        world
            .chunks
            .insert(above_buried, full_stone_chunk(above_buried));

        assert!(chunk_is_render_skin(&world, surface));
        assert!(!chunk_is_render_skin(&world, empty));
        assert!(!chunk_is_render_skin(&world, buried));
    }

    #[test]
    fn render_candidate_retains_chunks_without_current_frustum_visibility() {
        let center = ChunkCoord::new(0, 0, 0);
        let behind = center.offset(-1, 0, 0);
        let world = world_with_dirty_chunk(behind);
        let settings = WorldStreamingSettings::new(8, 8, 8, 8, 8, 9);

        assert!(chunk_is_render_candidate(&world, center, behind, settings));
    }

    #[test]
    fn dirty_mesh_queue_prioritizes_nearer_forward_chunks() {
        let center = ChunkCoord::new(0, 0, 0);
        let near = center.offset(1, 0, 0);
        let far = center.offset(8, 0, 0);
        let mut world = VoxelWorld::new(0);
        let settings = WorldStreamingSettings::new(12, 12, 12, 12, 12, 13);
        let camera = VoxelCamera::new([16.0, 16.0, 16.0], 0.0, 0.0);
        let mut queue = DirtyMeshQueue::default();

        world.chunks.insert(near, {
            let mut chunk = Chunk::new_empty(near);
            chunk.set_block(0, 0, 0, STONE_BLOCK);
            chunk
        });
        world.chunks.insert(far, {
            let mut chunk = Chunk::new_empty(far);
            chunk.set_block(0, 0, 0, STONE_BLOCK);
            chunk
        });

        queue.enqueue(far);
        queue.enqueue(near);
        queue.prioritize(&world, camera, ChunkVisibility::all(), center, settings);

        assert_eq!(queue.pop_dirty(&world), Some(near));
        assert_eq!(queue.pop_dirty(&world), Some(far));
    }

    #[test]
    fn saved_render_distance_is_clamped_to_performance_cap() {
        let too_low = GameSettings::from_saved(SavedSettings {
            mouse_sensitivity: config::DEFAULT_MOUSE_SENSITIVITY,
            render_chunk_distance: 1,
            fov_degrees: 1.0,
        });
        let too_high = GameSettings::from_saved(SavedSettings {
            mouse_sensitivity: config::DEFAULT_MOUSE_SENSITIVITY,
            render_chunk_distance: 999,
            fov_degrees: 999.0,
        });

        assert_eq!(
            too_low.chunk_view_distance,
            config::MIN_RENDER_CHUNK_DISTANCE
        );
        assert_eq!(
            too_high.chunk_view_distance,
            config::MAX_RENDER_CHUNK_DISTANCE
        );
        assert_eq!(too_low.fov_degrees, config::MIN_FOV_DEGREES);
        assert_eq!(too_high.fov_degrees, config::MAX_FOV_DEGREES);
        assert_eq!(
            too_high.streaming_settings().vertical_load_radius,
            config::MAX_RENDER_CHUNK_DISTANCE
        );
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
    fn camera_settings_apply_fov_in_degrees() {
        let settings = GameSettings {
            fov_degrees: 95.0,
            ..GameSettings::default()
        };
        let camera = camera_with_settings(VoxelCamera::looking_at_chunk_origin(), settings);

        assert!((camera.fov_y_radians - 95.0_f32.to_radians()).abs() < f32::EPSILON);
    }

    #[test]
    fn player_accelerates_toward_air_move_speed_when_not_grounded() {
        let world = VoxelWorld::new(0);
        let mut player = PlayerController::from_camera(VoxelCamera::looking_at_chunk_origin());
        let input = InputState {
            forward: true,
            ..Default::default()
        };

        player.step(&world, 0.0, input, 1.0);

        assert_eq!(player.velocity[0], config::PLAYER_IN_AIR_MOVE_SPEED);
    }

    #[test]
    fn player_preserves_air_velocity_without_input() {
        let world = VoxelWorld::new(0);
        let mut player = PlayerController::from_camera(VoxelCamera::looking_at_chunk_origin());
        player.velocity = [7.0, 0.0, 0.0];

        player.step(&world, 0.0, InputState::default(), 0.01);

        assert_eq!(player.velocity[0], 7.0);
    }

    #[test]
    fn horizontal_velocity_integrates_acceleration_toward_target() {
        let velocity = integrate_horizontal_velocity((1.0, 0.0), (5.0, 0.0), 10.0, 0.05);

        assert_f32_close(velocity.0, 3.0);
        assert_eq!(velocity.1, 0.0);
    }

    #[test]
    fn air_velocity_does_not_decrease_toward_lower_target_speed() {
        let velocity = integrate_air_horizontal_velocity(
            (config::PLAYER_SPRINT_SPEED, 0.0),
            (config::PLAYER_IN_AIR_MOVE_SPEED, 0.0),
            config::PLAYER_MOVE_ACCELERATION,
            0.01,
        );

        assert_f32_close(horizontal_speed(velocity), config::PLAYER_SPRINT_SPEED);
    }

    #[test]
    fn air_velocity_can_steer_without_losing_speed() {
        let velocity = integrate_air_horizontal_velocity(
            (config::PLAYER_SPRINT_SPEED, 0.0),
            (0.0, config::PLAYER_IN_AIR_MOVE_SPEED),
            config::PLAYER_MOVE_ACCELERATION,
            0.01,
        );

        assert_f32_close(horizontal_speed(velocity), config::PLAYER_SPRINT_SPEED);
        assert!(velocity.0 < config::PLAYER_SPRINT_SPEED);
        assert!(velocity.1 > 0.0);
    }

    #[test]
    fn player_accelerates_toward_sprint_speed_when_supported_on_ground() {
        let world = world_with_floor();
        let mut player = PlayerController {
            center: [0.5, 1.901, 0.5],
            velocity: [0.0; 3],
            on_ground: false,
        };
        let input = InputState {
            forward: true,
            fast: true,
            ..Default::default()
        };

        let dt = 0.01;
        player.step(&world, 0.0, input, dt);

        assert_f32_close(
            player.velocity[0],
            config::PLAYER_SPRINT_SPEED * config::PLAYER_MOVE_ACCELERATION * dt,
        );
        assert!(player.velocity[0] < config::PLAYER_SPRINT_SPEED);
        assert!(player.on_ground);
    }

    #[test]
    fn player_uses_walk_target_when_sprinting_sideways() {
        let world = world_with_floor();
        let mut player = PlayerController {
            center: [0.5, 1.901, 0.5],
            velocity: [0.0; 3],
            on_ground: false,
        };
        let input = InputState {
            right: true,
            fast: true,
            ..Default::default()
        };

        player.step(&world, 0.0, input, 1.0);

        assert_eq!(player.velocity[2], config::PLAYER_WALK_SPEED);
        assert!(player.velocity[2] < config::PLAYER_SPRINT_SPEED);
    }

    #[test]
    fn player_sprints_forward_while_strafing_at_walk_speed() {
        let world = world_with_floor();
        let mut player = PlayerController {
            center: [0.5, 1.901, 0.5],
            velocity: [0.0; 3],
            on_ground: false,
        };
        let input = InputState {
            forward: true,
            right: true,
            fast: true,
            ..Default::default()
        };

        player.step(&world, 0.0, input, 1.0);

        assert_eq!(player.velocity[0], config::PLAYER_SPRINT_SPEED);
        assert_eq!(player.velocity[2], config::PLAYER_WALK_SPEED);
    }

    #[test]
    fn player_decelerates_toward_crouch_speed_when_supported_on_ground() {
        let world = world_with_floor();
        let mut player = PlayerController {
            center: [0.5, 1.901, 0.5],
            velocity: [config::PLAYER_WALK_SPEED, 0.0, 0.0],
            on_ground: false,
        };
        let input = InputState {
            forward: true,
            slow: true,
            ..Default::default()
        };

        player.step(&world, 0.0, input, 0.01);

        assert!(player.velocity[0] < config::PLAYER_WALK_SPEED);
        assert!(player.velocity[0] > config::PLAYER_CROUCH_SPEED);
        assert!(player.on_ground);
    }

    #[test]
    fn horizontal_target_velocity_uses_camera_yaw() {
        let mut input = InputState {
            forward: true,
            ..Default::default()
        };

        assert_eq!(
            horizontal_target_velocity(0.0, input, true),
            (config::PLAYER_WALK_SPEED, 0.0)
        );

        input.forward = false;
        input.right = true;

        assert_eq!(
            horizontal_target_velocity(0.0, input, true),
            (0.0, config::PLAYER_WALK_SPEED)
        );
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
                assert_eq!(mesh.visible_face_count, 3);
                assert_eq!(priority, MeshJobPriority::Normal);
            }
            ChunkJobResult::Generated { .. } => panic!("expected mesh result"),
        }

        drop(job_sender);
        handle.join().unwrap();
    }
}
