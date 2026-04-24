use std::{error::Error, fmt, sync::Arc};

use bytemuck::{Pod, Zeroable};
use foundation::{Aabb, ChunkCoord};
use glam::{Mat4, Vec3};
use meshing::MeshData;
use voxels::CHUNK_SIZE;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

pub type RendererResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

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
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera: VoxelCamera,
    render_pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    overlay_text: Option<String>,
    overlay_vertex_buffer: Option<wgpu::Buffer>,
    overlay_vertex_count: u32,
    chunk_meshes: Vec<GpuChunkMesh>,
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
}

impl VoxelRenderer {
    pub async fn new(window: Arc<Window>) -> RendererResult<Self> {
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Adventure Quest Voxel Shader"),
            source: wgpu::ShaderSource::Wgsl(VOXEL_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Adventure Quest Voxel Pipeline Layout"),
            bind_group_layouts: &[Some(&camera_bind_group_layout)],
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
                cull_mode: None,
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

        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Adventure Quest Overlay Shader"),
            source: wgpu::ShaderSource::Wgsl(OVERLAY_SHADER.into()),
        });
        let overlay_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Adventure Quest Overlay Pipeline Layout"),
                bind_group_layouts: &[],
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
            camera_buffer,
            camera_bind_group,
            camera,
            render_pipeline,
            overlay_pipeline,
            overlay_text: None,
            overlay_vertex_buffer: None,
            overlay_vertex_count: 0,
            chunk_meshes: Vec::new(),
        })
    }

    pub const fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    pub fn upload_mesh(&mut self, mesh: &MeshData) {
        self.chunk_meshes.clear();

        let upload = ChunkMeshUpload {
            coord: ChunkCoord::new(0, 0, 0),
            revision: 0,
            visible_mask: mesh.subchunk_visible_mask,
            mesh,
        };

        if let Some(mesh) = GpuChunkMesh::from_upload(&self.device, upload) {
            self.chunk_meshes.push(mesh);
        }
    }

    pub fn upload_meshes<'a, I>(&mut self, meshes: I)
    where
        I: IntoIterator<Item = &'a MeshData>,
    {
        self.chunk_meshes = meshes
            .into_iter()
            .enumerate()
            .filter_map(|(index, mesh)| {
                let upload = ChunkMeshUpload {
                    coord: ChunkCoord::new(index as i32, 0, 0),
                    revision: 0,
                    visible_mask: mesh.subchunk_visible_mask,
                    mesh,
                };

                GpuChunkMesh::from_upload(&self.device, upload)
            })
            .collect();
    }

    pub fn upload_chunk_mesh(&mut self, upload: ChunkMeshUpload<'_>) {
        let existing = self
            .chunk_meshes
            .iter()
            .position(|mesh| mesh.coord == upload.coord);

        let Some(mesh) = GpuChunkMesh::from_upload(&self.device, upload) else {
            if let Some(index) = existing {
                self.chunk_meshes.remove(index);
            }
            return;
        };

        if let Some(index) = existing {
            self.chunk_meshes[index] = mesh;
        } else {
            self.chunk_meshes.push(mesh);
        }
    }

    pub fn upload_chunk_meshes<'a, I>(&mut self, uploads: I)
    where
        I: IntoIterator<Item = ChunkMeshUpload<'a>>,
    {
        self.chunk_meshes = uploads
            .into_iter()
            .filter_map(|upload| GpuChunkMesh::from_upload(&self.device, upload))
            .collect();
    }

    pub fn remove_chunk_mesh(&mut self, coord: ChunkCoord) -> bool {
        let Some(index) = self
            .chunk_meshes
            .iter()
            .position(|mesh| mesh.coord == coord)
        else {
            return false;
        };

        self.chunk_meshes.remove(index);
        true
    }

    pub fn chunk_mesh_info(&self) -> impl Iterator<Item = ChunkMeshInfo> + '_ {
        self.chunk_meshes.iter().map(GpuChunkMesh::info)
    }

    pub fn set_fps_overlay(&mut self, fps: Option<u32>) {
        let text = fps.map(|fps| format!("FPS {fps}"));
        self.set_overlay_text(text);
    }

    pub fn set_overlay_text(&mut self, text: Option<String>) {
        self.overlay_text = text;
        self.rebuild_overlay_buffer();
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

        self.write_camera_uniform();
        self.rebuild_overlay_buffer();
    }

    fn write_camera_uniform(&self) {
        let camera_uniform =
            CameraUniform::from_camera(self.camera, self.config.width, self.config.height);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));
    }

    fn rebuild_overlay_buffer(&mut self) {
        let Some(text) = self.overlay_text.as_deref() else {
            self.overlay_vertex_buffer = None;
            self.overlay_vertex_count = 0;
            return;
        };

        let vertices = build_text_vertices(text, self.config.width, self.config.height);

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

            if !self.chunk_meshes.is_empty() {
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

                for mesh in &self.chunk_meshes {
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }

            if let Some(vertex_buffer) = self.overlay_vertex_buffer.as_ref() {
                render_pass.set_pipeline(&self.overlay_pipeline);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.draw(0..self.overlay_vertex_count, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        RenderFrameStatus::Rendered
    }
}

pub type ClearRenderer = VoxelRenderer;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OverlayVertex {
    position: [f32; 2],
    color: [f32; 4],
}

impl OverlayVertex {
    fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            position: [(x / width) * 2.0 - 1.0, 1.0 - (y / height) * 2.0],
            color: [1.0, 1.0, 1.0, 1.0],
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
            ],
        }
    }
}

struct GpuChunkMesh {
    coord: ChunkCoord,
    revision: u32,
    bounds: Aabb,
    visible_mask: u8,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl GpuChunkMesh {
    fn from_upload(device: &wgpu::Device, upload: ChunkMeshUpload<'_>) -> Option<Self> {
        if upload.mesh.vertices.is_empty() || upload.mesh.indices.is_empty() {
            return None;
        }

        let vertices: Vec<GpuVertex> = upload
            .mesh
            .vertices
            .iter()
            .map(GpuVertex::from_mesh_vertex)
            .collect();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Adventure Quest Chunk Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Adventure Quest Chunk Index Buffer"),
            contents: bytemuck::cast_slice(&upload.mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Some(Self {
            coord: upload.coord,
            revision: upload.revision,
            bounds: chunk_bounds(upload.coord),
            visible_mask: upload.visible_mask,
            vertex_buffer,
            index_buffer,
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

impl GpuVertex {
    fn from_mesh_vertex(vertex: &meshing::Vertex) -> Self {
        Self {
            position: vertex.position,
            normal: vertex.normal,
            color: block_color(vertex.block_id),
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
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() * 2) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
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

fn block_color(block_id: u16) -> [f32; 3] {
    match block_id {
        1 => [0.45, 0.27, 0.13],
        2 => [0.18, 0.58, 0.18],
        3 => [0.48, 0.48, 0.50],
        4 => [0.12, 0.12, 0.12],
        5 => [0.72, 0.48, 0.32],
        _ => [0.95, 0.1, 0.85],
    }
}

fn build_text_vertices(text: &str, width: u32, height: u32) -> Vec<OverlayVertex> {
    if width == 0 || height == 0 {
        return Vec::new();
    }

    const GLYPH_WIDTH: usize = 5;
    const GLYPH_HEIGHT: usize = 7;
    const SCALE: f32 = 2.0;
    const ORIGIN_X: f32 = 12.0;
    const ORIGIN_Y: f32 = 12.0;

    let width = width as f32;
    let height = height as f32;
    let mut vertices = Vec::new();
    let mut cursor_x = ORIGIN_X;

    for character in text.chars() {
        if character == ' ' {
            cursor_x += SCALE * 4.0;
            continue;
        }

        if let Some(pattern) = glyph_pattern(character) {
            for (row, line) in pattern.iter().enumerate().take(GLYPH_HEIGHT) {
                for (column, pixel) in line.bytes().enumerate().take(GLYPH_WIDTH) {
                    if pixel == b'#' {
                        let x0 = cursor_x + column as f32 * SCALE;
                        let y0 = ORIGIN_Y + row as f32 * SCALE;
                        push_overlay_quad(&mut vertices, x0, y0, SCALE, width, height);
                    }
                }
            }
        }

        cursor_x += SCALE * (GLYPH_WIDTH as f32 + 1.0);
    }

    vertices
}

fn push_overlay_quad(
    vertices: &mut Vec<OverlayVertex>,
    x: f32,
    y: f32,
    size: f32,
    width: f32,
    height: f32,
) {
    let x1 = x + size;
    let y1 = y + size;

    vertices.extend_from_slice(&[
        OverlayVertex::new(x, y, width, height),
        OverlayVertex::new(x1, y, width, height),
        OverlayVertex::new(x1, y1, width, height),
        OverlayVertex::new(x, y, width, height),
        OverlayVertex::new(x1, y1, width, height),
        OverlayVertex::new(x, y1, width, height),
    ]);
}

fn glyph_pattern(character: char) -> Option<[&'static str; 7]> {
    match character {
        'F' => Some([
            "#####", "#....", "#....", "####.", "#....", "#....", "#....",
        ]),
        'P' => Some([
            "####.", "#...#", "#...#", "####.", "#....", "#....", "#....",
        ]),
        'S' => Some([
            "#####", "#....", "#....", "#####", "....#", "....#", "#####",
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
        _ => None,
    }
}

const VOXEL_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = camera.view_proj * vec4<f32>(input.position, 1.0);
    output.color = input.color;
    output.normal = input.normal;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.45, 0.85, 0.25));
    let directional = max(dot(normalize(input.normal), light_dir), 0.0);
    let lighting = 0.38 + directional * 0.62;

    return vec4<f32>(input.color * lighting, 1.0);
}
"#;

const OVERLAY_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = vec4<f32>(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
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
