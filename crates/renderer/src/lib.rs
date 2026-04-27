use std::{collections::HashMap, error::Error, fmt, path::Path, sync::Arc};

use bytemuck::{Pod, Zeroable};
use foundation::{Aabb, BlockPos, ChunkCoord};
use glam::{Mat4, Vec3};
use meshing::MeshData;
use voxels::CHUNK_SIZE;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

pub type RendererResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RendererOptions {
    pub vsync: bool,
}

impl RendererOptions {
    pub const fn new(vsync: bool) -> Self {
        Self { vsync }
    }

    const fn present_mode(self) -> wgpu::PresentMode {
        if self.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        }
    }
}

impl Default for RendererOptions {
    fn default() -> Self {
        Self { vsync: true }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderFrameStatus {
    Rendered,
    Skipped,
    Reconfigured,
}

pub struct VoxelRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    clear_color: wgpu::Color,
    depth_texture: DepthTexture,
    block_atlas: BlockTextureAtlas,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera: VoxelCamera,
    render_pipeline: wgpu::RenderPipeline,
    outline_pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    ui_overlay: UiOverlay,
    overlay_text: Option<String>,
    crosshair_enabled: bool,
    outline_vertex_buffer: Option<wgpu::Buffer>,
    outline_vertex_count: u32,
    overlay_vertex_buffer: Option<wgpu::Buffer>,
    overlay_vertex_count: u32,
    terrain_batch: TerrainBatch,
    terrain_batch_key: Option<TerrainBatchKey>,
    chunk_mesh_revision: u64,
    chunk_meshes: Vec<GpuChunkMesh>,
    chunk_mesh_indices: HashMap<ChunkCoord, usize>,
    frustum_culling_enabled: bool,
    last_frame_stats: RenderFrameStats,
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkMeshUpload<'a> {
    pub coord: ChunkCoord,
    pub revision: u32,
    pub visible_mask: u8,
    pub mesh: &'a MeshData,
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkMeshInfo {
    pub coord: ChunkCoord,
    pub revision: u32,
    pub index_count: u32,
    pub visible_mask: u8,
    pub bounds: Aabb,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RenderFrameStats {
    pub uploaded_chunk_meshes: usize,
    pub drawn_chunk_meshes: usize,
    pub culled_chunk_meshes: usize,
    pub drawn_indices: u32,
    pub terrain_draw_calls: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkVisibility {
    frustum: Option<CameraFrustum>,
}

impl ChunkVisibility {
    pub const fn all() -> Self {
        Self { frustum: None }
    }

    pub fn contains(self, coord: ChunkCoord) -> bool {
        self.frustum
            .is_none_or(|frustum| frustum.intersects_aabb(chunk_bounds(coord)))
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct UiOverlay {
    pub items: Vec<UiOverlayItem>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UiOverlayItem {
    Rect(UiRect),
    Texture(UiTextureRect),
    Text(UiText),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub color: [f32; 4],
}

impl UiRect {
    pub const fn new(x: f32, y: f32, width: f32, height: f32, color: [f32; 4]) -> Self {
        Self {
            x,
            y,
            width,
            height,
            color,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiTextureRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub color: [f32; 4],
    pub uvs: [[f32; 2]; 4],
}

impl UiTextureRect {
    pub const fn new(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: [f32; 4],
        uvs: [[f32; 2]; 4],
    ) -> Self {
        Self {
            x,
            y,
            width,
            height,
            color,
            uvs,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UiText {
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub color: [f32; 4],
    pub text: String,
}

impl UiText {
    pub fn new(x: f32, y: f32, scale: f32, color: [f32; 4], text: impl Into<String>) -> Self {
        Self {
            x,
            y,
            scale,
            color,
            text: text.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelCamera {
    pub position: [f32; 3],
    pub yaw_radians: f32,
    pub pitch_radians: f32,
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
}

impl VoxelCamera {
    pub fn new(position: [f32; 3], yaw_radians: f32, pitch_radians: f32) -> Self {
        Self {
            position,
            yaw_radians,
            pitch_radians,
            fov_y_radians: 55.0_f32.to_radians(),
            near: 0.1,
            far: 500.0,
        }
    }

    pub fn looking_at_chunk_origin() -> Self {
        Self::new(
            [56.0, 44.0, 66.0],
            -135.0_f32.to_radians(),
            -24.0_f32.to_radians(),
        )
    }

    pub fn translate_local(&mut self, forward: f32, right: f32, up: f32) {
        let forward_vector = self.forward_vector();
        let right_vector = forward_vector.cross(Vec3::Y).normalize_or_zero();
        let position = Vec3::from_array(self.position)
            + forward_vector * forward
            + right_vector * right
            + Vec3::Y * up;

        self.position = position.to_array();
    }

    pub fn rotate(&mut self, yaw_delta: f32, pitch_delta: f32) {
        self.yaw_radians += yaw_delta;
        self.pitch_radians = (self.pitch_radians + pitch_delta).clamp(-1.45, 1.45);
    }

    pub fn forward_direction(self) -> [f32; 3] {
        self.forward_vector().to_array()
    }

    pub fn chunk_depth(self, coord: ChunkCoord) -> f32 {
        let center = chunk_center(coord);

        self.depth_sorter().depth_to_point(center)
    }

    fn view_projection(self, aspect: f32) -> Mat4 {
        let eye = Vec3::from_array(self.position);
        let view = Mat4::look_to_rh(eye, self.forward_vector(), Vec3::Y);
        let projection = Mat4::perspective_rh(self.fov_y_radians, aspect, self.near, self.far);

        projection * view
    }

    fn forward_vector(self) -> Vec3 {
        Vec3::new(
            self.yaw_radians.cos() * self.pitch_radians.cos(),
            self.pitch_radians.sin(),
            self.yaw_radians.sin() * self.pitch_radians.cos(),
        )
        .normalize_or_zero()
    }

    fn depth_sorter(self) -> CameraDepth {
        CameraDepth {
            position: Vec3::from_array(self.position),
            forward: self.forward_vector(),
        }
    }
}

impl VoxelRenderer {
    pub async fn new(window: Arc<Window>) -> RendererResult<Self> {
        Self::new_with_options(window, RendererOptions::default()).await
    }

    pub async fn new_with_options(
        window: Arc<Window>,
        options: RendererOptions,
    ) -> RendererResult<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Adventure Quest Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::default(),
            })
            .await?;

        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or(RendererInitError::NoSupportedSurfaceConfig)?;
        let config = wgpu::SurfaceConfiguration {
            present_mode: options.present_mode(),
            ..config
        };

        surface.configure(&device, &config);

        let depth_texture = DepthTexture::new(&device, config.width, config.height);
        let camera = VoxelCamera::looking_at_chunk_origin();
        let camera_uniform = CameraUniform::from_camera(camera, config.width, config.height);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Adventure Quest Camera Buffer"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Adventure Quest Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Adventure Quest Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let block_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Adventure Quest Block Texture Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let block_atlas =
            BlockTextureAtlas::load_or_fallback(&device, &queue, &block_texture_bind_group_layout);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Adventure Quest Voxel Shader"),
            source: wgpu::ShaderSource::Wgsl(VOXEL_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Adventure Quest Voxel Pipeline Layout"),
            bind_group_layouts: &[
                Some(&camera_bind_group_layout),
                Some(&block_texture_bind_group_layout),
            ],
            immediate_size: 0,
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Adventure Quest Voxel Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[GpuVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DepthTexture::FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let outline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Adventure Quest Block Outline Shader"),
            source: wgpu::ShaderSource::Wgsl(OUTLINE_SHADER.into()),
        });
        let outline_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Adventure Quest Block Outline Pipeline Layout"),
                bind_group_layouts: &[Some(&camera_bind_group_layout)],
                immediate_size: 0,
            });
        let outline_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Adventure Quest Block Outline Pipeline"),
            layout: Some(&outline_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &outline_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[OutlineVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &outline_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DepthTexture::FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Adventure Quest Overlay Shader"),
            source: wgpu::ShaderSource::Wgsl(OVERLAY_SHADER.into()),
        });
        let overlay_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Adventure Quest Overlay Pipeline Layout"),
                bind_group_layouts: &[Some(&block_texture_bind_group_layout)],
                immediate_size: 0,
            });
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Adventure Quest Overlay Pipeline"),
            layout: Some(&overlay_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[OverlayVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DepthTexture::FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            clear_color: wgpu::Color {
                r: 0.04,
                g: 0.07,
                b: 0.09,
                a: 1.0,
            },
            depth_texture,
            block_atlas,
            camera_buffer,
            camera_bind_group,
            camera,
            render_pipeline,
            outline_pipeline,
            overlay_pipeline,
            ui_overlay: UiOverlay::default(),
            overlay_text: None,
            crosshair_enabled: false,
            outline_vertex_buffer: None,
            outline_vertex_count: 0,
            overlay_vertex_buffer: None,
            overlay_vertex_count: 0,
            terrain_batch: TerrainBatch::default(),
            terrain_batch_key: None,
            chunk_mesh_revision: 0,
            chunk_meshes: Vec::new(),
            chunk_mesh_indices: HashMap::new(),
            frustum_culling_enabled: true,
            last_frame_stats: RenderFrameStats::default(),
        })
    }

    pub const fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    pub fn upload_mesh(&mut self, mesh: &MeshData) {
        self.chunk_meshes.clear();
        self.chunk_mesh_indices.clear();
        self.invalidate_terrain_batch();

        let upload = ChunkMeshUpload {
            coord: ChunkCoord::new(0, 0, 0),
            revision: 0,
            visible_mask: mesh.subchunk_visible_mask,
            mesh,
        };

        if let Some(mesh) =
            GpuChunkMesh::from_upload(&self.device, upload, self.block_atlas.available)
        {
            self.insert_chunk_mesh(mesh);
        }
    }

    pub fn upload_meshes<'a, I>(&mut self, meshes: I)
    where
        I: IntoIterator<Item = &'a MeshData>,
    {
        self.chunk_meshes.clear();
        self.chunk_mesh_indices.clear();
        self.invalidate_terrain_batch();

        for (index, mesh) in meshes.into_iter().enumerate() {
            let upload = ChunkMeshUpload {
                coord: ChunkCoord::new(index as i32, 0, 0),
                revision: 0,
                visible_mask: mesh.subchunk_visible_mask,
                mesh,
            };

            if let Some(mesh) =
                GpuChunkMesh::from_upload(&self.device, upload, self.block_atlas.available)
            {
                self.insert_chunk_mesh(mesh);
            }
        }
    }

    pub fn upload_chunk_mesh(&mut self, upload: ChunkMeshUpload<'_>) {
        let coord = upload.coord;

        let Some(mesh) =
            GpuChunkMesh::from_upload(&self.device, upload, self.block_atlas.available)
        else {
            self.remove_chunk_mesh(coord);
            return;
        };

        self.insert_chunk_mesh(mesh);
    }

    pub fn upload_chunk_meshes<'a, I>(&mut self, uploads: I)
    where
        I: IntoIterator<Item = ChunkMeshUpload<'a>>,
    {
        self.chunk_meshes.clear();
        self.chunk_mesh_indices.clear();
        self.invalidate_terrain_batch();

        for upload in uploads {
            if let Some(mesh) =
                GpuChunkMesh::from_upload(&self.device, upload, self.block_atlas.available)
            {
                self.insert_chunk_mesh(mesh);
            }
        }
    }

    pub fn clear_chunk_meshes(&mut self) {
        self.chunk_meshes.clear();
        self.chunk_mesh_indices.clear();
        self.invalidate_terrain_batch();
        self.set_block_outline(None);
    }

    pub fn remove_chunk_mesh(&mut self, coord: ChunkCoord) -> bool {
        let Some(index) = self.chunk_mesh_indices.remove(&coord) else {
            return false;
        };

        self.chunk_meshes.swap_remove(index);

        if let Some(swapped_mesh) = self.chunk_meshes.get(index) {
            self.chunk_mesh_indices.insert(swapped_mesh.coord, index);
        }

        self.invalidate_terrain_batch();
        true
    }

    fn insert_chunk_mesh(&mut self, mesh: GpuChunkMesh) {
        if let Some(index) = self.chunk_mesh_indices.get(&mesh.coord).copied() {
            self.chunk_meshes[index] = mesh;
            self.invalidate_terrain_batch();
            return;
        }

        let index = self.chunk_meshes.len();
        self.chunk_mesh_indices.insert(mesh.coord, index);
        self.chunk_meshes.push(mesh);
        self.invalidate_terrain_batch();
    }

    pub fn chunk_mesh_info(&self) -> impl Iterator<Item = ChunkMeshInfo> + '_ {
        self.chunk_meshes.iter().map(GpuChunkMesh::info)
    }

    pub fn chunk_mesh_count(&self) -> usize {
        self.chunk_meshes.len()
    }

    pub const fn last_frame_stats(&self) -> RenderFrameStats {
        self.last_frame_stats
    }

    pub const fn frustum_culling_enabled(&self) -> bool {
        self.frustum_culling_enabled
    }

    pub fn set_frustum_culling_enabled(&mut self, enabled: bool) {
        if self.frustum_culling_enabled != enabled {
            self.invalidate_terrain_batch();
        }

        self.frustum_culling_enabled = enabled;
    }

    pub fn chunk_visibility(&self, camera: VoxelCamera) -> ChunkVisibility {
        let aspect = self.config.width.max(1) as f32 / self.config.height.max(1) as f32;
        ChunkVisibility {
            frustum: self
                .frustum_culling_enabled
                .then(|| CameraFrustum::from_camera(camera, aspect)),
        }
    }

    pub fn chunk_visible_from_camera(&self, camera: VoxelCamera, coord: ChunkCoord) -> bool {
        self.chunk_visibility(camera).contains(coord)
    }

    pub fn set_fps_overlay(&mut self, fps: Option<u32>) {
        let text = fps.map(|fps| format!("FPS {fps}"));
        self.set_overlay_text(text);
    }

    pub fn set_overlay_text(&mut self, text: Option<String>) {
        self.overlay_text = text;
        self.rebuild_overlay_buffer();
    }

    pub fn set_ui_overlay(&mut self, overlay: UiOverlay) {
        self.ui_overlay = overlay;
        self.rebuild_overlay_buffer();
    }

    pub fn set_crosshair_enabled(&mut self, enabled: bool) {
        self.crosshair_enabled = enabled;
        self.rebuild_overlay_buffer();
    }

    pub fn set_block_outline(&mut self, block: Option<BlockPos>) {
        let Some(block) = block else {
            self.outline_vertex_buffer = None;
            self.outline_vertex_count = 0;
            return;
        };

        let vertices = block_outline_vertices(block);

        self.outline_vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Adventure Quest Block Outline Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));
        self.outline_vertex_count = vertices.len() as u32;
    }

    pub fn mesh_count(&self) -> usize {
        self.chunk_meshes.len()
    }

    pub fn index_count(&self) -> u32 {
        self.chunk_meshes.iter().map(|mesh| mesh.index_count).sum()
    }

    pub const fn camera(&self) -> VoxelCamera {
        self.camera
    }

    pub fn set_camera(&mut self, camera: VoxelCamera) {
        self.camera = camera;
        self.write_camera_uniform();
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.size = new_size;

        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_texture = DepthTexture::new(&self.device, new_size.width, new_size.height);
        self.invalidate_terrain_batch();

        self.write_camera_uniform();
        self.rebuild_overlay_buffer();
    }

    fn invalidate_terrain_batch(&mut self) {
        self.chunk_mesh_revision = self.chunk_mesh_revision.wrapping_add(1);
        self.terrain_batch_key = None;
    }

    fn write_camera_uniform(&self) {
        let camera_uniform =
            CameraUniform::from_camera(self.camera, self.config.width, self.config.height);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));
    }

    fn rebuild_overlay_buffer(&mut self) {
        let mut vertices = Vec::new();

        for item in &self.ui_overlay.items {
            match item {
                UiOverlayItem::Rect(rect) => push_overlay_rect(
                    &mut vertices,
                    rect.x,
                    rect.y,
                    rect.width,
                    rect.height,
                    rect.color,
                    self.config.width as f32,
                    self.config.height as f32,
                ),
                UiOverlayItem::Texture(rect) => push_overlay_textured_rect(
                    &mut vertices,
                    rect.x,
                    rect.y,
                    rect.width,
                    rect.height,
                    rect.color,
                    rect.uvs,
                    self.block_atlas.available,
                    self.config.width as f32,
                    self.config.height as f32,
                ),
                UiOverlayItem::Text(text) => push_text_vertices(
                    &mut vertices,
                    &text.text,
                    text.x,
                    text.y,
                    text.scale,
                    text.color,
                    self.config.width,
                    self.config.height,
                ),
            }
        }

        if let Some(text) = self.overlay_text.as_deref() {
            vertices.extend(build_text_vertices(
                text,
                self.config.width,
                self.config.height,
            ));
        }

        if self.crosshair_enabled {
            push_crosshair_vertices(&mut vertices, self.config.width, self.config.height);
        }

        if vertices.is_empty() {
            self.overlay_vertex_buffer = None;
            self.overlay_vertex_count = 0;
            return;
        }

        self.overlay_vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Adventure Quest Overlay Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));
        self.overlay_vertex_count = vertices.len() as u32;
    }

    pub fn render(&mut self) -> RenderFrameStatus {
        self.last_frame_stats = RenderFrameStats::default();

        if self.size.width == 0 || self.size.height == 0 {
            return RenderFrameStatus::Skipped;
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return RenderFrameStatus::Skipped;
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return RenderFrameStatus::Reconfigured;
            }
            wgpu::CurrentSurfaceTexture::Validation => return RenderFrameStatus::Skipped,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Adventure Quest Frame Encoder"),
            });
        let aspect = self.config.width.max(1) as f32 / self.config.height.max(1) as f32;
        let frustum = self
            .frustum_culling_enabled
            .then(|| CameraFrustum::from_camera(self.camera, aspect));
        let mut stats = RenderFrameStats {
            uploaded_chunk_meshes: self.chunk_meshes.len(),
            ..RenderFrameStats::default()
        };
        self.rebuild_terrain_batch_if_needed(aspect);

        {
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.clear_color),
                    store: wgpu::StoreOp::Store,
                },
            })];

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Adventure Quest Voxel Pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if let (Some(vertex_buffer), Some(index_buffer)) = (
                self.terrain_batch.vertex_buffer.as_ref(),
                self.terrain_batch.index_buffer.as_ref(),
            ) {
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_bind_group(1, &self.block_atlas.bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.terrain_batch.index_count, 0, 0..1);

                stats.drawn_chunk_meshes = self.terrain_batch.chunk_count;
                stats.culled_chunk_meshes = self
                    .chunk_meshes
                    .len()
                    .saturating_sub(self.terrain_batch.chunk_count);
                stats.drawn_indices = self.terrain_batch.index_count;
                stats.terrain_draw_calls = 1;
            } else if frustum.is_some() {
                stats.culled_chunk_meshes = self.chunk_meshes.len();
            }

            if let Some(vertex_buffer) = self.outline_vertex_buffer.as_ref() {
                render_pass.set_pipeline(&self.outline_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.draw(0..self.outline_vertex_count, 0..1);
            }

            if let Some(vertex_buffer) = self.overlay_vertex_buffer.as_ref() {
                render_pass.set_pipeline(&self.overlay_pipeline);
                render_pass.set_bind_group(0, &self.block_atlas.bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.draw(0..self.overlay_vertex_count, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        self.last_frame_stats = stats;

        RenderFrameStatus::Rendered
    }

    fn rebuild_terrain_batch_if_needed(&mut self, aspect: f32) {
        let key = TerrainBatchKey::from_camera(
            self.camera,
            self.config.width,
            self.config.height,
            self.chunk_mesh_revision,
            self.frustum_culling_enabled,
        );

        if self.terrain_batch_key == Some(key) {
            return;
        }

        self.terrain_batch.rebuild(
            &self.device,
            self.camera,
            aspect,
            self.frustum_culling_enabled,
            &self.chunk_meshes,
        );
        self.terrain_batch_key = Some(key);
    }
}

pub type ClearRenderer = VoxelRenderer;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OverlayVertex {
    position: [f32; 2],
    color: [f32; 4],
    uv: [f32; 2],
    texture_weight: f32,
}

impl OverlayVertex {
    fn new(x: f32, y: f32, width: f32, height: f32, color: [f32; 4]) -> Self {
        Self::with_uv(x, y, width, height, color, [0.0, 0.0], 0.0)
    }

    fn with_uv(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: [f32; 4],
        uv: [f32; 2],
        texture_weight: f32,
    ) -> Self {
        Self {
            position: [(x / width) * 2.0 - 1.0, 1.0 - (y / height) * 2.0],
            color,
            uv,
            texture_weight,
        }
    }

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 4]>())
                        as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 2]>()
                        + std::mem::size_of::<[f32; 4]>()
                        + std::mem::size_of::<[f32; 2]>())
                        as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

#[derive(Default)]
struct TerrainBatch {
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    index_count: u32,
    chunk_count: usize,
}

impl TerrainBatch {
    fn rebuild(
        &mut self,
        device: &wgpu::Device,
        camera: VoxelCamera,
        aspect: f32,
        frustum_culling_enabled: bool,
        meshes: &[GpuChunkMesh],
    ) {
        if meshes.is_empty() {
            self.clear();
            return;
        }

        let visible = visible_mesh_indices(camera, aspect, frustum_culling_enabled, meshes);

        if visible.is_empty() {
            self.clear();
            return;
        }

        let visible_count = visible.len();
        let total_vertices = visible
            .iter()
            .map(|index| meshes[*index].vertices.len())
            .sum();
        let total_indices = visible
            .iter()
            .map(|index| meshes[*index].indices.len())
            .sum();
        let mut vertices = Vec::with_capacity(total_vertices);
        let mut indices = Vec::with_capacity(total_indices);

        for index in visible {
            let mesh = &meshes[index];
            let vertex_offset = vertices.len() as u32;

            vertices.extend_from_slice(&mesh.vertices);
            indices.extend(mesh.indices.iter().map(|index| index + vertex_offset));
        }

        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Adventure Quest Terrain Batch Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        );
        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Adventure Quest Terrain Batch Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
        );
        self.index_count = indices.len() as u32;
        self.chunk_count = visible_count;
    }

    fn clear(&mut self) {
        self.vertex_buffer = None;
        self.index_buffer = None;
        self.index_count = 0;
        self.chunk_count = 0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TerrainBatchKey {
    position_bucket: [i32; 3],
    yaw_bucket: i32,
    pitch_bucket: i32,
    width: u32,
    height: u32,
    mesh_revision: u64,
    frustum_culling_enabled: bool,
}

impl TerrainBatchKey {
    fn from_camera(
        camera: VoxelCamera,
        width: u32,
        height: u32,
        mesh_revision: u64,
        frustum_culling_enabled: bool,
    ) -> Self {
        const ANGLE_BUCKET_RADIANS: f32 = 2.0_f32.to_radians();
        let position_bucket_size = CHUNK_SIZE as f32 * 0.5;

        Self {
            position_bucket: [
                (camera.position[0] / position_bucket_size).floor() as i32,
                (camera.position[1] / position_bucket_size).floor() as i32,
                (camera.position[2] / position_bucket_size).floor() as i32,
            ],
            yaw_bucket: (camera.yaw_radians / ANGLE_BUCKET_RADIANS).round() as i32,
            pitch_bucket: (camera.pitch_radians / ANGLE_BUCKET_RADIANS).round() as i32,
            width,
            height,
            mesh_revision,
            frustum_culling_enabled,
        }
    }
}

struct GpuChunkMesh {
    coord: ChunkCoord,
    revision: u32,
    bounds: Aabb,
    visible_mask: u8,
    vertices: Box<[GpuVertex]>,
    indices: Box<[u32]>,
    index_count: u32,
}

impl GpuChunkMesh {
    fn from_upload(
        _device: &wgpu::Device,
        upload: ChunkMeshUpload<'_>,
        texture_atlas_available: bool,
    ) -> Option<Self> {
        if upload.mesh.vertices.is_empty() || upload.mesh.indices.is_empty() {
            return None;
        }

        let mut vertices = Vec::with_capacity(upload.mesh.vertices.len());
        let mut cached_block_id = None;
        let mut cached_appearance = GpuBlockAppearance::from_block(0, texture_atlas_available);

        for vertex in &upload.mesh.vertices {
            if cached_block_id != Some(vertex.block_id) {
                cached_block_id = Some(vertex.block_id);
                cached_appearance =
                    GpuBlockAppearance::from_block(vertex.block_id, texture_atlas_available);
            }

            vertices.push(GpuVertex::from_mesh_vertex(vertex, cached_appearance));
        }
        Some(Self {
            coord: upload.coord,
            revision: upload.revision,
            bounds: chunk_visible_bounds(upload.coord, upload.visible_mask),
            visible_mask: upload.visible_mask,
            vertices: vertices.into_boxed_slice(),
            indices: upload.mesh.indices.clone().into_boxed_slice(),
            index_count: upload.mesh.indices.len() as u32,
        })
    }

    fn info(&self) -> ChunkMeshInfo {
        ChunkMeshInfo {
            coord: self.coord,
            revision: self.revision,
            index_count: self.index_count,
            visible_mask: self.visible_mask,
            bounds: self.bounds,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CameraDepth {
    position: Vec3,
    forward: Vec3,
}

impl CameraDepth {
    fn depth_to_bounds(self, bounds: Aabb) -> f32 {
        self.depth_to_point(bounds.center())
    }

    fn near_depth_to_bounds(self, bounds: Aabb) -> f32 {
        aabb_corners(bounds)
            .into_iter()
            .map(|corner| self.depth_to_point(corner))
            .fold(f32::INFINITY, f32::min)
    }

    fn depth_to_point(self, point: [f32; 3]) -> f32 {
        let to_point = Vec3::from_array(point) - self.position;

        to_point.dot(self.forward)
    }
}

#[derive(Debug, Clone, Copy)]
struct MeshVisibilityCandidate {
    index: usize,
    depth: f32,
    distance_key: i32,
}

fn visible_mesh_indices(
    camera: VoxelCamera,
    aspect: f32,
    frustum_culling_enabled: bool,
    meshes: &[GpuChunkMesh],
) -> Vec<usize> {
    let frustum = frustum_culling_enabled.then(|| CameraFrustum::from_camera(camera, aspect));
    let depth = camera.depth_sorter();
    let center = camera_chunk_coord(camera);
    let mut candidates = Vec::with_capacity(meshes.len());

    for (index, mesh) in meshes.iter().enumerate() {
        let distance_key = chunk_distance_key(center, mesh.coord);
        let protected = chunk_is_protected_near_camera(center, mesh.coord);

        if !protected
            && frustum
                .as_ref()
                .is_some_and(|frustum| !frustum.intersects_aabb(mesh.bounds))
        {
            continue;
        }

        let near_depth = depth.near_depth_to_bounds(mesh.bounds);

        if !protected && (near_depth > camera.far || near_depth < -(CHUNK_SIZE as f32)) {
            continue;
        }

        candidates.push(MeshVisibilityCandidate {
            index,
            depth: depth.depth_to_bounds(mesh.bounds),
            distance_key,
        });
    }

    candidates.sort_unstable_by(|left, right| {
        left.distance_key
            .cmp(&right.distance_key)
            .then_with(|| left.depth.total_cmp(&right.depth))
            .then_with(|| {
                meshes[left.index]
                    .index_count
                    .cmp(&meshes[right.index].index_count)
            })
    });

    candidates
        .into_iter()
        .map(|candidate| candidate.index)
        .collect()
}

const PROTECTED_NEAR_CAMERA_CHUNK_RADIUS: i32 = 2;

fn chunk_is_protected_near_camera(center: ChunkCoord, coord: ChunkCoord) -> bool {
    chunk_distance_key(center, coord)
        <= PROTECTED_NEAR_CAMERA_CHUNK_RADIUS * PROTECTED_NEAR_CAMERA_CHUNK_RADIUS
}

fn camera_chunk_coord(camera: VoxelCamera) -> ChunkCoord {
    ChunkCoord::new(
        (camera.position[0].floor() as i32).div_euclid(CHUNK_SIZE as i32),
        (camera.position[1].floor() as i32).div_euclid(CHUNK_SIZE as i32),
        (camera.position[2].floor() as i32).div_euclid(CHUNK_SIZE as i32),
    )
}

fn chunk_distance_key(center: ChunkCoord, coord: ChunkCoord) -> i32 {
    let dx = coord.x - center.x;
    let dy = coord.y - center.y;
    let dz = coord.z - center.z;

    dx * dx + dy * dy + dz * dz
}

fn aabb_corners(bounds: Aabb) -> [[f32; 3]; 8] {
    [
        [bounds.min[0], bounds.min[1], bounds.min[2]],
        [bounds.max[0], bounds.min[1], bounds.min[2]],
        [bounds.min[0], bounds.max[1], bounds.min[2]],
        [bounds.max[0], bounds.max[1], bounds.min[2]],
        [bounds.min[0], bounds.min[1], bounds.max[2]],
        [bounds.max[0], bounds.min[1], bounds.max[2]],
        [bounds.min[0], bounds.max[1], bounds.max[2]],
        [bounds.max[0], bounds.max[1], bounds.max[2]],
    ]
}

fn chunk_bounds(coord: ChunkCoord) -> Aabb {
    let size = CHUNK_SIZE as f32;
    let min = [
        coord.x as f32 * size,
        coord.y as f32 * size,
        coord.z as f32 * size,
    ];
    let max = [min[0] + size, min[1] + size, min[2] + size];

    Aabb::new(min, max)
}

fn chunk_visible_bounds(coord: ChunkCoord, visible_mask: u8) -> Aabb {
    if visible_mask == 0 {
        return chunk_bounds(coord);
    }

    let chunk_min = [
        coord.x as f32 * CHUNK_SIZE as f32,
        coord.y as f32 * CHUNK_SIZE as f32,
        coord.z as f32 * CHUNK_SIZE as f32,
    ];
    let subchunk_size = CHUNK_SIZE as f32 * 0.5;
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for subchunk in 0..8 {
        if visible_mask & (1u8 << subchunk) == 0 {
            continue;
        }

        let sx = (subchunk & 1) as f32;
        let sy = ((subchunk >> 1) & 1) as f32;
        let sz = ((subchunk >> 2) & 1) as f32;
        let sub_min = [
            chunk_min[0] + sx * subchunk_size,
            chunk_min[1] + sy * subchunk_size,
            chunk_min[2] + sz * subchunk_size,
        ];
        let sub_max = [
            sub_min[0] + subchunk_size,
            sub_min[1] + subchunk_size,
            sub_min[2] + subchunk_size,
        ];

        for axis in 0..3 {
            min[axis] = min[axis].min(sub_min[axis]);
            max[axis] = max[axis].max(sub_max[axis]);
        }
    }

    Aabb::new(min, max)
}

fn chunk_center(coord: ChunkCoord) -> [f32; 3] {
    let bounds = chunk_bounds(coord);

    bounds.center()
}

trait AabbCenter {
    fn center(self) -> [f32; 3];
}

impl AabbCenter for Aabb {
    fn center(self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }
}

#[derive(Debug, Clone, Copy)]
struct CameraFrustum {
    planes: [Plane; 6],
}

impl CameraFrustum {
    fn from_camera(camera: VoxelCamera, aspect: f32) -> Self {
        let position = Vec3::from_array(camera.position);
        let forward = camera.forward_vector();
        let world_up = if forward.dot(Vec3::Y).abs() > 0.98 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let right = forward.cross(world_up).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let half_vertical = camera.fov_y_radians * 0.5;
        let half_horizontal = (half_vertical.tan() * aspect.max(0.001)).atan();
        let sin_vertical = half_vertical.sin();
        let cos_vertical = half_vertical.cos();
        let sin_horizontal = half_horizontal.sin();
        let cos_horizontal = half_horizontal.cos();

        Self {
            planes: [
                Plane::from_point_normal(position + forward * camera.near, forward),
                Plane::from_point_normal(position + forward * camera.far, -forward),
                Plane::from_point_normal(
                    position,
                    (forward * sin_horizontal + right * cos_horizontal).normalize_or_zero(),
                ),
                Plane::from_point_normal(
                    position,
                    (forward * sin_horizontal - right * cos_horizontal).normalize_or_zero(),
                ),
                Plane::from_point_normal(
                    position,
                    (forward * sin_vertical + up * cos_vertical).normalize_or_zero(),
                ),
                Plane::from_point_normal(
                    position,
                    (forward * sin_vertical - up * cos_vertical).normalize_or_zero(),
                ),
            ],
        }
    }

    fn intersects_aabb(self, bounds: Aabb) -> bool {
        self.planes
            .iter()
            .all(|plane| plane.distance(aabb_positive_vertex(bounds, plane.normal)) >= 0.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct Plane {
    normal: Vec3,
    distance_from_origin: f32,
}

impl Plane {
    fn from_point_normal(point: Vec3, normal: Vec3) -> Self {
        Self {
            normal,
            distance_from_origin: -normal.dot(point),
        }
    }

    fn distance(self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.distance_from_origin
    }
}

fn aabb_positive_vertex(bounds: Aabb, normal: Vec3) -> Vec3 {
    Vec3::new(
        if normal.x >= 0.0 {
            bounds.max[0]
        } else {
            bounds.min[0]
        },
        if normal.y >= 0.0 {
            bounds.max[1]
        } else {
            bounds.min[1]
        },
        if normal.z >= 0.0 {
            bounds.max[2]
        } else {
            bounds.min[2]
        },
    )
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVertex {
    position: [f32; 3],
    uv: [f32; 2],
    color: [u8; 4],
    normal: [i8; 4],
    texture_weight: f32,
}

impl GpuVertex {
    fn from_mesh_vertex(vertex: &meshing::Vertex, appearance: GpuBlockAppearance) -> Self {
        Self {
            position: vertex.position,
            uv: vertex.uv,
            color: appearance.color,
            normal: pack_snorm4(vertex.normal),
            texture_weight: appearance.texture_weight,
        }
    }

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        const UV_OFFSET: wgpu::BufferAddress =
            std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress;
        const COLOR_OFFSET: wgpu::BufferAddress =
            UV_OFFSET + std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress;
        const NORMAL_OFFSET: wgpu::BufferAddress =
            COLOR_OFFSET + std::mem::size_of::<[u8; 4]>() as wgpu::BufferAddress;
        const TEXTURE_WEIGHT_OFFSET: wgpu::BufferAddress =
            NORMAL_OFFSET + std::mem::size_of::<[i8; 4]>() as wgpu::BufferAddress;

        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: UV_OFFSET,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: COLOR_OFFSET,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Unorm8x4,
                },
                wgpu::VertexAttribute {
                    offset: NORMAL_OFFSET,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Snorm8x4,
                },
                wgpu::VertexAttribute {
                    offset: TEXTURE_WEIGHT_OFFSET,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

#[derive(Clone, Copy)]
struct GpuBlockAppearance {
    color: [u8; 4],
    texture_weight: f32,
}

impl GpuBlockAppearance {
    fn from_block(block_id: u16, texture_atlas_available: bool) -> Self {
        Self {
            color: pack_unorm4(block_color(block_id)),
            texture_weight: if texture_atlas_available && voxels::block_has_texture(block_id) {
                1.0
            } else {
                0.0
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OutlineVertex {
    position: [f32; 3],
    color: [f32; 4],
}

impl OutlineVertex {
    const fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, color }
    }

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

fn block_outline_vertices(block: BlockPos) -> Vec<OutlineVertex> {
    const COLOR: [f32; 4] = [1.0, 1.0, 1.0, 0.96];
    const EXPAND: f32 = 0.002;
    const EDGES: [(usize, usize); 12] = [
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    let min_x = block.x as f32 - EXPAND;
    let min_y = block.y as f32 - EXPAND;
    let min_z = block.z as f32 - EXPAND;
    let max_x = block.x as f32 + 1.0 + EXPAND;
    let max_y = block.y as f32 + 1.0 + EXPAND;
    let max_z = block.z as f32 + 1.0 + EXPAND;
    let corners = [
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [min_x, max_y, min_z],
        [max_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [min_x, max_y, max_z],
        [max_x, max_y, max_z],
    ];
    let mut vertices = Vec::with_capacity(EDGES.len() * 2);

    for (start, end) in EDGES {
        vertices.push(OutlineVertex::new(corners[start], COLOR));
        vertices.push(OutlineVertex::new(corners[end], COLOR));
    }

    vertices
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn from_camera(camera: VoxelCamera, width: u32, height: u32) -> Self {
        let aspect = width.max(1) as f32 / height.max(1) as f32;

        Self {
            view_proj: camera.view_projection(aspect).to_cols_array_2d(),
        }
    }
}

struct DepthTexture {
    view: wgpu::TextureView,
}

impl DepthTexture {
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Adventure Quest Depth Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self { view }
    }
}

struct BlockTextureAtlas {
    bind_group: wgpu::BindGroup,
    available: bool,
}

impl BlockTextureAtlas {
    fn load_or_fallback(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let texture_data = load_block_texture_atlas().unwrap_or_else(|| TextureData {
            pixels: vec![0, 0, 0, 0],
            width: 1,
            height: 1,
            available: false,
        });
        let size = wgpu::Extent3d {
            width: texture_data.width.max(1),
            height: texture_data.height.max(1),
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Adventure Quest Block Texture Atlas"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &texture_data.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * size.width),
                rows_per_image: Some(size.height),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Adventure Quest Block Texture Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Adventure Quest Block Texture Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            bind_group,
            available: texture_data.available,
        }
    }
}

struct TextureData {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    available: bool,
}

fn load_block_texture_atlas() -> Option<TextureData> {
    let path = Path::new(voxels::BLOCK_TEXTURE_ATLAS_PATH);

    if !path.exists() {
        return None;
    }

    let image = image::open(path).ok()?.to_rgba8();
    let width = image.width();
    let height = image.height();

    if width == 0 || height == 0 {
        return None;
    }

    Some(TextureData {
        pixels: image.into_raw(),
        width,
        height,
        available: true,
    })
}

fn block_color(block_id: u16) -> [f32; 4] {
    voxels::block_color_rgba(block_id)
}

fn pack_unorm4(color: [f32; 4]) -> [u8; 4] {
    color.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn pack_snorm4(normal: [f32; 3]) -> [i8; 4] {
    [
        pack_snorm(normal[0]),
        pack_snorm(normal[1]),
        pack_snorm(normal[2]),
        0,
    ]
}

fn pack_snorm(value: f32) -> i8 {
    (value.clamp(-1.0, 1.0) * 127.0).round() as i8
}

fn build_text_vertices(text: &str, width: u32, height: u32) -> Vec<OverlayVertex> {
    const SCALE: f32 = 2.0;
    const ORIGIN_X: f32 = 12.0;
    const ORIGIN_Y: f32 = 12.0;
    const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

    let mut vertices = Vec::new();

    push_text_vertices(
        &mut vertices,
        text,
        ORIGIN_X,
        ORIGIN_Y,
        SCALE,
        WHITE,
        width,
        height,
    );

    vertices
}

#[allow(clippy::too_many_arguments)]
fn push_text_vertices(
    vertices: &mut Vec<OverlayVertex>,
    text: &str,
    x: f32,
    y: f32,
    scale: f32,
    color: [f32; 4],
    width: u32,
    height: u32,
) {
    if width == 0 || height == 0 || scale <= 0.0 {
        return;
    }

    const GLYPH_WIDTH: usize = 5;
    const GLYPH_HEIGHT: usize = 7;

    let width = width as f32;
    let height = height as f32;
    let mut cursor_x = x;

    for character in text.chars() {
        if character == ' ' {
            cursor_x += scale * 4.0;
            continue;
        }

        if let Some(pattern) = glyph_pattern(character.to_ascii_uppercase()) {
            for (row, line) in pattern.iter().enumerate().take(GLYPH_HEIGHT) {
                for (column, pixel) in line.bytes().enumerate().take(GLYPH_WIDTH) {
                    if pixel == b'#' {
                        let x0 = cursor_x + column as f32 * scale;
                        let y0 = y + row as f32 * scale;
                        push_overlay_square(vertices, x0, y0, scale, color, width, height);
                    }
                }
            }
        }

        cursor_x += scale * (GLYPH_WIDTH as f32 + 1.0);
    }
}

fn push_crosshair_vertices(vertices: &mut Vec<OverlayVertex>, width: u32, height: u32) {
    if width == 0 || height == 0 {
        return;
    }

    const LENGTH: f32 = 9.0;
    const GAP: f32 = 4.0;
    const THICKNESS: f32 = 2.0;

    let width = width as f32;
    let height = height as f32;
    let center_x = width * 0.5;
    let center_y = height * 0.5;
    let half_thickness = THICKNESS * 0.5;
    let color = [1.0, 1.0, 1.0, 1.0];

    push_overlay_rect(
        vertices,
        center_x - GAP - LENGTH,
        center_y - half_thickness,
        LENGTH,
        THICKNESS,
        color,
        width,
        height,
    );
    push_overlay_rect(
        vertices,
        center_x + GAP,
        center_y - half_thickness,
        LENGTH,
        THICKNESS,
        color,
        width,
        height,
    );
    push_overlay_rect(
        vertices,
        center_x - half_thickness,
        center_y - GAP - LENGTH,
        THICKNESS,
        LENGTH,
        color,
        width,
        height,
    );
    push_overlay_rect(
        vertices,
        center_x - half_thickness,
        center_y + GAP,
        THICKNESS,
        LENGTH,
        color,
        width,
        height,
    );
}

fn push_overlay_square(
    vertices: &mut Vec<OverlayVertex>,
    x: f32,
    y: f32,
    size: f32,
    color: [f32; 4],
    width: f32,
    height: f32,
) {
    push_overlay_rect(vertices, x, y, size, size, color, width, height);
}

#[allow(clippy::too_many_arguments)]
fn push_overlay_rect(
    vertices: &mut Vec<OverlayVertex>,
    x: f32,
    y: f32,
    rect_width: f32,
    rect_height: f32,
    color: [f32; 4],
    width: f32,
    height: f32,
) {
    if width <= 0.0 || height <= 0.0 || rect_width <= 0.0 || rect_height <= 0.0 {
        return;
    }

    let x1 = x + rect_width;
    let y1 = y + rect_height;

    vertices.extend_from_slice(&[
        OverlayVertex::new(x, y, width, height, color),
        OverlayVertex::new(x1, y, width, height, color),
        OverlayVertex::new(x1, y1, width, height, color),
        OverlayVertex::new(x, y, width, height, color),
        OverlayVertex::new(x1, y1, width, height, color),
        OverlayVertex::new(x, y1, width, height, color),
    ]);
}

#[allow(clippy::too_many_arguments)]
fn push_overlay_textured_rect(
    vertices: &mut Vec<OverlayVertex>,
    x: f32,
    y: f32,
    rect_width: f32,
    rect_height: f32,
    color: [f32; 4],
    uvs: [[f32; 2]; 4],
    texture_atlas_available: bool,
    width: f32,
    height: f32,
) {
    if width <= 0.0 || height <= 0.0 || rect_width <= 0.0 || rect_height <= 0.0 {
        return;
    }

    let x1 = x + rect_width;
    let y1 = y + rect_height;
    let texture_weight = if texture_atlas_available { 1.0 } else { 0.0 };

    vertices.extend_from_slice(&[
        OverlayVertex::with_uv(x, y, width, height, color, uvs[0], texture_weight),
        OverlayVertex::with_uv(x1, y, width, height, color, uvs[1], texture_weight),
        OverlayVertex::with_uv(x1, y1, width, height, color, uvs[2], texture_weight),
        OverlayVertex::with_uv(x, y, width, height, color, uvs[0], texture_weight),
        OverlayVertex::with_uv(x1, y1, width, height, color, uvs[2], texture_weight),
        OverlayVertex::with_uv(x, y1, width, height, color, uvs[3], texture_weight),
    ]);
}

fn glyph_pattern(character: char) -> Option<[&'static str; 7]> {
    match character {
        'A' => Some([
            ".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#",
        ]),
        'B' => Some([
            "####.", "#...#", "#...#", "####.", "#...#", "#...#", "####.",
        ]),
        'C' => Some([
            ".####", "#....", "#....", "#....", "#....", "#....", ".####",
        ]),
        'D' => Some([
            "####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####.",
        ]),
        'E' => Some([
            "#####", "#....", "#....", "####.", "#....", "#....", "#####",
        ]),
        'F' => Some([
            "#####", "#....", "#....", "####.", "#....", "#....", "#....",
        ]),
        'G' => Some([
            ".####", "#....", "#....", "#.###", "#...#", "#...#", ".####",
        ]),
        'H' => Some([
            "#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#",
        ]),
        'I' => Some([
            "#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####",
        ]),
        'J' => Some([
            "..###", "...#.", "...#.", "...#.", "#..#.", "#..#.", ".##..",
        ]),
        'K' => Some([
            "#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#",
        ]),
        'L' => Some([
            "#....", "#....", "#....", "#....", "#....", "#....", "#####",
        ]),
        'M' => Some([
            "#...#", "##.##", "#.#.#", "#...#", "#...#", "#...#", "#...#",
        ]),
        'N' => Some([
            "#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#",
        ]),
        'O' => Some([
            ".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###.",
        ]),
        'P' => Some([
            "####.", "#...#", "#...#", "####.", "#....", "#....", "#....",
        ]),
        'Q' => Some([
            ".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#",
        ]),
        'R' => Some([
            "####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#",
        ]),
        'S' => Some([
            "#####", "#....", "#....", "#####", "....#", "....#", "#####",
        ]),
        'T' => Some([
            "#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#..",
        ]),
        'U' => Some([
            "#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###.",
        ]),
        'V' => Some([
            "#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#..",
        ]),
        'W' => Some([
            "#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#",
        ]),
        'X' => Some([
            "#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#",
        ]),
        'Y' => Some([
            "#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#..",
        ]),
        'Z' => Some([
            "#####", "....#", "...#.", "..#..", ".#...", "#....", "#####",
        ]),
        '0' => Some([
            "#####", "#...#", "#..##", "#.#.#", "##..#", "#...#", "#####",
        ]),
        '1' => Some([
            "..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###.",
        ]),
        '2' => Some([
            "#####", "....#", "....#", "#####", "#....", "#....", "#####",
        ]),
        '3' => Some([
            "#####", "....#", "....#", "#####", "....#", "....#", "#####",
        ]),
        '4' => Some([
            "#...#", "#...#", "#...#", "#####", "....#", "....#", "....#",
        ]),
        '5' => Some([
            "#####", "#....", "#....", "#####", "....#", "....#", "#####",
        ]),
        '6' => Some([
            "#####", "#....", "#....", "#####", "#...#", "#...#", "#####",
        ]),
        '7' => Some([
            "#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#...",
        ]),
        '8' => Some([
            "#####", "#...#", "#...#", "#####", "#...#", "#...#", "#####",
        ]),
        '9' => Some([
            "#####", "#...#", "#...#", "#####", "....#", "....#", "#####",
        ]),
        '.' => Some([
            ".....", ".....", ".....", ".....", ".....", "..#..", "..#..",
        ]),
        _ => None,
    }
}

const VOXEL_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(1) @binding(0)
var block_texture: texture_2d<f32>;

@group(1) @binding(1)
var block_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) normal: vec4<f32>,
    @location(4) texture_weight: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) texture_weight: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = camera.view_proj * vec4<f32>(input.position, 1.0);
    output.color = input.color;
    output.normal = input.normal.xyz;
    output.uv = input.uv;
    output.texture_weight = input.texture_weight;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.45, 0.85, 0.25));
    let directional = max(dot(normalize(input.normal), light_dir), 0.0);
    let lighting = 0.38 + directional * 0.62;
    let texture_color = textureSample(block_texture, block_sampler, input.uv);
    let texture_mix = clamp(input.texture_weight * texture_color.a, 0.0, 1.0);
    let base_color = mix(input.color.rgb, texture_color.rgb, texture_mix);

    return vec4<f32>(base_color * lighting, input.color.a);
}
"#;

const OUTLINE_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = camera.view_proj * vec4<f32>(input.position, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

const OVERLAY_SHADER: &str = r#"
@group(0) @binding(0)
var block_texture: texture_2d<f32>;

@group(0) @binding(1)
var block_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) texture_weight: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) texture_weight: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = vec4<f32>(input.position, 0.0, 1.0);
    output.color = input.color;
    output.uv = input.uv;
    output.texture_weight = input.texture_weight;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texture_color = textureSample(block_texture, block_sampler, input.uv);
    let texture_mix = clamp(input.texture_weight * texture_color.a, 0.0, 1.0);
    let color = mix(input.color, texture_color, texture_mix);

    return vec4<f32>(color.rgb, max(input.color.a, color.a));
}
"#;

#[derive(Debug)]
enum RendererInitError {
    NoSupportedSurfaceConfig,
}

impl fmt::Display for RendererInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSupportedSurfaceConfig => {
                f.write_str("no supported surface configuration for this adapter/window")
            }
        }
    }
}

impl Error for RendererInitError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_bounds_match_chunk_world_space() {
        let bounds = chunk_bounds(ChunkCoord::new(-1, 2, 3));

        assert_eq!(bounds.min, [-32.0, 64.0, 96.0]);
        assert_eq!(bounds.max, [0.0, 96.0, 128.0]);
    }

    #[test]
    fn chunk_visible_bounds_follow_visible_subchunk_mask() {
        let bounds = chunk_visible_bounds(ChunkCoord::new(0, 0, 0), 0b1000_0000);

        assert_eq!(bounds.min, [16.0, 16.0, 16.0]);
        assert_eq!(bounds.max, [32.0, 32.0, 32.0]);
    }

    #[test]
    fn visible_mesh_indices_keep_near_chunks_before_far_chunks() {
        let mut camera = VoxelCamera::new([0.0, 16.0, 16.0], 0.0, 0.0);
        camera.fov_y_radians = 90.0_f32.to_radians();
        camera.far = 256.0;
        let meshes = vec![
            test_gpu_chunk_mesh(ChunkCoord::new(1, 0, 0), 0b1111_1111),
            test_gpu_chunk_mesh(ChunkCoord::new(4, 0, 0), 0b1111_1111),
        ];

        let visible = visible_mesh_indices(camera, 16.0 / 9.0, true, &meshes);

        assert_eq!(visible.first().copied(), Some(0));
        assert!(visible.contains(&1));
    }

    #[test]
    fn camera_frustum_accepts_visible_chunk_bounds() {
        let mut camera = VoxelCamera::new([0.0, 0.0, 0.0], 0.0, 0.0);
        camera.fov_y_radians = 90.0_f32.to_radians();
        camera.near = 0.1;
        camera.far = 96.0;

        let frustum = CameraFrustum::from_camera(camera, 16.0 / 9.0);

        assert!(frustum.intersects_aabb(chunk_bounds(ChunkCoord::new(1, 0, 0))));
    }

    #[test]
    fn camera_frustum_rejects_chunk_bounds_outside_view() {
        let mut camera = VoxelCamera::new([0.0, 0.0, 0.0], 0.0, 0.0);
        camera.fov_y_radians = 90.0_f32.to_radians();
        camera.near = 0.1;
        camera.far = 96.0;

        let frustum = CameraFrustum::from_camera(camera, 16.0 / 9.0);

        assert!(!frustum.intersects_aabb(chunk_bounds(ChunkCoord::new(-2, 0, 0))));
        assert!(!frustum.intersects_aabb(chunk_bounds(ChunkCoord::new(4, 0, 0))));
        assert!(!frustum.intersects_aabb(chunk_bounds(ChunkCoord::new(1, 0, 4))));
    }

    #[test]
    fn camera_chunk_depth_increases_along_forward_axis() {
        let camera = VoxelCamera::new([16.0, 16.0, 16.0], 0.0, 0.0);

        assert!(camera.chunk_depth(ChunkCoord::new(1, 0, 0)) > 0.0);
        assert!(
            camera.chunk_depth(ChunkCoord::new(5, 0, 0))
                > camera.chunk_depth(ChunkCoord::new(1, 0, 0))
        );
        assert!(camera.chunk_depth(ChunkCoord::new(-2, 0, 0)) < 0.0);
    }

    #[test]
    fn crosshair_builds_four_overlay_rectangles() {
        let mut vertices = Vec::new();

        push_crosshair_vertices(&mut vertices, 1280, 720);

        assert_eq!(vertices.len(), 24);
    }

    #[test]
    fn crosshair_ignores_zero_sized_surface() {
        let mut vertices = Vec::new();

        push_crosshair_vertices(&mut vertices, 0, 720);
        push_crosshair_vertices(&mut vertices, 1280, 0);

        assert!(vertices.is_empty());
    }

    #[test]
    fn block_outline_builds_cube_edges_without_diagonals() {
        let vertices = block_outline_vertices(BlockPos::new(3, 4, 5));

        assert_eq!(vertices.len(), 24);

        for edge in vertices.chunks_exact(2) {
            let start = edge[0].position;
            let end = edge[1].position;
            let changed_axes = [
                (start[0] - end[0]).abs() > f32::EPSILON,
                (start[1] - end[1]).abs() > f32::EPSILON,
                (start[2] - end[2]).abs() > f32::EPSILON,
            ]
            .into_iter()
            .filter(|changed| *changed)
            .count();

            assert_eq!(changed_axes, 1);
        }
    }

    #[test]
    fn gpu_chunk_vertex_uses_compact_layout() {
        assert_eq!(std::mem::size_of::<GpuVertex>(), 32);
    }

    #[test]
    fn renderer_options_select_expected_present_mode() {
        assert_eq!(
            RendererOptions::default().present_mode(),
            wgpu::PresentMode::AutoVsync
        );
        assert_eq!(
            RendererOptions::new(false).present_mode(),
            wgpu::PresentMode::AutoNoVsync
        );
    }

    fn test_gpu_chunk_mesh(coord: ChunkCoord, visible_mask: u8) -> GpuChunkMesh {
        GpuChunkMesh {
            coord,
            revision: 0,
            bounds: chunk_visible_bounds(coord, visible_mask),
            visible_mask,
            vertices: vec![GpuVertex {
                position: [0.0; 3],
                uv: [0.0; 2],
                color: [255; 4],
                normal: [0, 0, 127, 0],
                texture_weight: 0.0,
            }]
            .into_boxed_slice(),
            indices: vec![0].into_boxed_slice(),
            index_count: 1,
        }
    }
}
