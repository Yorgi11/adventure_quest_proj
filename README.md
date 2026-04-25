# Adventure Quest - Rust Starter

This is the first working codebase for **Adventure Quest**, starting with the voxel-core architecture from the technical report.

- Rust workspace layout
- `aq_core` crate for shared core types
- `aq_voxel` crate for chunk/block storage
- `aq_app` executable crate
- 32x32x32 flat chunk storage
- signed infinite chunk coordinates
- Euclidean world-to-chunk conversion for negative coordinates
- 8 subchunks per chunk
- occupancy, dirty, visible, and full-solid masks
- safe block editing with revision tracking
- VS Code settings and tasks
- unit tests for indexing, conversion, and mask behavior

## Required tools

Install:

1. Microsoft C++ Build Tools with **Desktop development with C++** workload.
2. Rust via `rustup`.
3. Visual Studio Code.
4. VS Code extensions:
   - `rust-analyzer`
   - `CodeLLDB` or Microsoft C++ tools for debugging
   - `crates` or `Even Better TOML` optionally

## Run

```text
cargo run -p client -- --fps-window
cargo run -p client -- --fps-window --no-vsync
cargo run -p client -- --no-window
```

## Window controls

```text
WASD        Move
Mouse       Look
Space       Jump
Shift       Sprint
Left click  Break block
Right click Place selected block
1-9         Select hotbar slot
V           Toggle noclip debug movement
Esc         Exit
```
