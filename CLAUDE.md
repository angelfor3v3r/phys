# phys

2D physics sandbox. Single-file C++23 application (`src/main.cpp`, ~3300 lines).
Spawn rigid bodies, draw collision geometry, link bodies with ropes, cut/erase ropes, pin bodies in place, and apply formula-driven force zones.

## Project layout

```
src/main.cpp            -- Entire application (single translation unit)
src/imconfig.hpp        -- Dear ImGui compile-time config
deploy/windows/         -- phys.ico, phys.png, phys.manifest, phys.rc
deploy/linux/           -- Flatpak manifest, .desktop file, phys.svg
third_party/cmake/      -- get_cpm.cmake (CPM v0.42.1 bootstrap)
.clang-format           -- clang-format 20 config
.gitattributes          -- LF endings, UTF-8, vendored third_party
```

## Build

CMake 3.28+, C++23. GCC, Clang (GNU frontend, LLVM 20), or MSVC. AppleClang is NOT supported (missing `std::jthread` / `std::stop_token`). 64-bit only.

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

All dependencies fetched via CPM — no manual installs. Set `CPM_SOURCE_CACHE` to avoid re-downloading.

### CMake options

| Option | Default | Purpose |
|--------|---------|---------|
| `PHYS_AVX2` | ON (x86_64), OFF (ARM) | Enables Box2D AVX2 SIMD + runtime check |

### Compile definitions

`PHYS_DEBUG` (0/1), `PHYS_AVX2` (0/1), `SDL_MAIN_USE_CALLBACKS`, and on Windows: `NOMINMAX`, `UNICODE`, `_UNICODE`, `WINVER=0x0A00`.

### Platform specifics

- **Windows**: `WIN32` subsystem (no console), static MSVC runtime (`/MT`), `.rc` resource embeds icon + manifest, links `dwmapi` for `DwmFlush()` drag-stutter fix. Clang-on-Windows: `/MANIFEST:EMBED` stripped to avoid conflict with RC-embedded manifest.
- **Linux**: Static libgcc/libstdc++. Flatpak packaging (Freedesktop 25.08 runtime, GCC 15). CMake install rules for binary, `.desktop`, icon.
- **macOS**: LLVM Clang 20 via Homebrew. `CMAKE_OSX_SYSROOT` from `xcrun --show-sdk-path`.

## Stack

| Library | Version | Role |
|---------|---------|------|
| SDL3 | 3.4.2 | Window, input, SDL_Renderer (static, most subsystems disabled) |
| Dear ImGui | 1.92.6-docking | UI + all rendering via draw lists |
| Box2D | 3.1.1 | Rigid body physics (v3 C API, optional AVX2) |
| FreeType | 2.14.2 | Font rasterization (all optional deps disabled) |
| dp::thread-pool | 0.7.0 | Worker threads shared with Box2D task system |
| scope_guard | 1.1.0 | RAII scope guard (`read_binary_file` cleanup) |
| TinyExpr++ | 1.1.0 | Runtime math expression parser for custom force zone formulas |

## Architecture

- **SDL3 callback model** (`SDL_MAIN_USE_CALLBACKS`): `SDL_AppInit`, `SDL_AppIterate`, `SDL_AppEvent`, `SDL_AppQuit`. No explicit main loop.
- **All state is global.** No classes, no OOP. Flat structs + free functions. Globals prefixed `g_`.
- **Box2D multithreading**: `box2d_enqueue_task` / `box2d_finish_task` using `std::latch` over `dp::thread_pool`. Task storage is a fixed-size placement-new array (`g_task_storage`), no dynamic allocation per task.
- **Rendering**: All drawing goes through `ImGui::GetBackgroundDrawList()` / `GetForegroundDrawList()`. No raw SDL draw calls.
- **Fixed timestep**: Accumulator pattern with previous-transform interpolation. `PHYSICS_DELTA_TIME = 1/60`. `g_physics_alpha` is the lerp factor. Sleeping bodies skip interpolation.
- **P-core detection at init**: Pins thread pool workers + main thread to performance cores. Windows (`GetSystemCpuSetInformation`), Linux (sysfs `cpu_type` / `base_frequency`), macOS (no-op fallback).
- **Cross-platform file I/O**: `read_binary_file()` with scope_guard RAII. Windows `fopen_s` / `_fseeki64` / `_ftelli64`, POSIX `fopen` / `fseeko` / `ftello`, chunked fallback.

### Rendering pipeline (SDL_AppIterate)

1. Skip if minimized.
2. Camera clamp to playground bounds.
3. `step_physics()` — accumulator loop, previous transforms saved inside loop, `b2World_Step`. EMA-smoothed timing.
4. Kill out-of-bounds bodies (kill bounds = `AREA ± 5m`).
5. Tick emitters and force zones (if not paused).
6. ImGui new frame + dockspace.
7. Toolbox window (Spawn, Selection, Emitters, Force Zones, World, Info, Graphics, Controls).
8. Drag-drop ghost preview.
9. FPS/renderer overlay.
10. Visual overlays: box-select rectangle, rope cut line (red), rope erase circle (red).
11. Render bodies to background draw list — shape queries hoisted once per body, frustum culled, fill AA disabled (~2x vertex savings), outer outline retains AA.
12. Render drawn lines (Catmull-Rom smoothed).
13. Render ropes (Catmull-Rom, tangent kill at >90° bends, outline + fill).
14. Render pins (Phillips-head screw, `sqrt(zoom_ratio)` scaling).
15. Render force zones (boundary + arrow grid).
16. Render emitters (arrow + circle).
17. `ImGui::Render()` + `SDL_RenderPresent`.
18. Frame timing (EMA-smoothed, 0.01 factor). FPS counter (snapshot once/sec).
19. FPS cap via `SDL_DelayPrecise` (skipped when <1ms slack).

### Input (SDL_AppEvent)

| Input | Action |
|-------|--------|
| Left-click drag | Grab body (mouse joint) or drag emitter/zone |
| Right-click while dragging | Pin body at current position |
| Ctrl+Right-click on pin | Unpin body |
| Ctrl+Left drag | Draw freehand line |
| Ctrl+Right drag | Erase lines and ropes (red circle indicator) |
| Shift+Left-click | Link rope between 2 bodies |
| Shift+Right drag | Cut ropes (red line indicator, one cut per drag) |
| Right-click | Toggle body selection |
| Right-drag | Box select |
| Delete / Backspace | Delete selected bodies |
| Escape | Clear selection / cancel pending rope |
| Middle-click drag | Pan camera |
| Scroll wheel | Zoom to cursor |

## Types

### Enums

- `SpawnShape` — Box, Circle, Triangle, Count
- `ZoneShape` — Rectangle, Circle, Count

### Structs

| Struct | Purpose |
|--------|---------|
| `BodyState` | `b2Vec2 position` + `b2Rot rotation` — interpolation snapshot |
| `PhysBody` | `b2BodyId` + `b2ShapeId` + `b2ShapeType` + `BodyState previous` + `damping_offset` + `selected` |
| `DrawnLine` | Static body + polyline points for freehand collision geometry |
| `Emitter` | Position, angle, speed, rate, timer, shape type, active flag |
| `ForceZone` | Position, shape, radius/half_size, angle, strength, max_speed, formula strings, tinyexpr++ parsers, bound variables, preset index |
| `Rope` | Capsule segment bodies + revolute joints + anchor body/local references |
| `Pin` | `b2JointId` + `b2BodyId` — revolute joint pinning body to ground |
| `TaskHandle` | `std::latch` wrapper for Box2D parallel task completion |
| `CoreTopology` | Platform-conditional: P-core IDs + total logical CPUs |
| `Context` | Nested in `tick_force_zones`: center, angle rotation, strength, max_speed, radius, shape, zone pointer |
| `FormulaPreset` | `name` + `formula_x` + `formula_y` — entry in `ZONE_PRESETS` array |

## Globals

### State

| Global | Type | Purpose |
|--------|------|---------|
| `g_world` | `b2WorldId` | Box2D world |
| `g_ground_id` | `b2BodyId` | Static ground body |
| `g_bodies` | `vector<PhysBody>` | All dynamic bodies |
| `g_drawn_lines` | `vector<DrawnLine>` | Freehand static collision geometry |
| `g_current_stroke` | `vector<b2Vec2>` | In-progress drawing points |
| `g_emitters` | `vector<Emitter>` | Body emitters |
| `g_force_zones` | `vector<ForceZone>` | Force zone regions |
| `g_ropes` | `vector<Rope>` | All ropes (intact and halves) |
| `g_pins` | `vector<Pin>` | Pinned bodies |

### Window / Renderer

| Global | Type | Purpose |
|--------|------|---------|
| `g_window` | `SDL_Window*` | Application window |
| `g_renderer` | `SDL_Renderer*` | SDL GPU renderer |
| `g_renderer_name` | `const char*` | Active renderer backend name |
| `g_dpi_scaling` | `float` | OS display scale factor |
| `g_window_w` / `g_window_h` | `int` | Window dimensions in pixels |

### Threading

| Global | Type | Purpose |
|--------|------|---------|
| `g_thread_pool` | `unique_ptr<dp::thread_pool<>>` | Shared worker pool |
| `g_worker_count` | `uint32_t` | Thread pool worker count |
| `g_perf_core_count` | `uint32_t` | P-cores detected (0 = homogeneous) |
| `g_total_core_count` | `uint32_t` | Total logical CPUs |
| `g_next_worker` | `atomic<uint32_t>` | Round-robin worker index |
| `g_task_count` | `int32_t` | Active Box2D tasks |
| `g_task_storage` | `byte[][sizeof(TaskHandle)]` | Placement-new task array |

### Physics

| Global | Type | Purpose |
|--------|------|---------|
| `g_physics_accumulator` | `float` | Leftover time for fixed-step integration |
| `g_physics_alpha` | `float` | Interpolation factor (0..1) |
| `g_paused` | `bool` | Simulation paused |
| `g_single_step` | `bool` | Advance one step then re-pause |
| `g_step_count` | `int32_t` | Physics steps per frame (default 1) |
| `g_sub_steps` | `int32_t` | Box2D solver sub-steps (default 4) |
| `g_linear_damping` | `float` | Base linear drag on all bodies (default 0.1) |
| `g_random_damping` | `float` | Max random drag offset per body (default 0.02) |
| `g_culled_count` | `size_t` | Bodies frustum-culled this frame |

### Camera

| Global | Type | Purpose |
|--------|------|---------|
| `g_camera_center` | `b2Vec2` | Camera center in world coords (default 0, 10) |
| `g_camera_zoom` | `float` | Pixels per world unit (default `ZOOM_DEFAULT` = 30) |
| `g_camera_zoom_min` | `float` | Minimum zoom (clamped to keep viewport in bounds) |
| `g_camera_dragging` | `bool` | Middle-mouse pan active |

### Graphics / Timing

| Global | Type | Purpose |
|--------|------|---------|
| `g_vsync` | `int` | VSync mode (off/on/adaptive) |
| `g_fps_cap` | `int` | Target FPS (0 = off, defaults to monitor refresh) |
| `g_physics_ms` / `g_render_ms` / `g_present_ms` / `g_frame_ms` | `float` | EMA-smoothed frame timers |
| `g_display_fps` | `int` | FPS counter (updated once/sec) |
| `g_total_vertices` / `g_total_indices` | `int` | Last frame's ImGui draw data counts |

### Interaction

| Global | Type | Purpose |
|--------|------|---------|
| `g_mouse_joint` | `b2JointId` | Mouse-drag joint |
| `g_mouse_body` | `b2BodyId` | Body being dragged |
| `g_box_selecting` | `bool` | Box-selection active |
| `g_box_select_start` | `ImVec2` | Box-select screen-space origin |
| `g_dragged_emitter` | `optional<size_t>` | Emitter being repositioned |
| `g_dragging_zone` | `optional<size_t>` | Force zone being repositioned |
| `g_rope_start_body` | `b2BodyId` | First body for pending rope link |
| `g_rope_start_anchor` | `b2Vec2` | Local-space anchor on rope start body |
| `g_erasing` | `bool` | Ctrl+Right-drag eraser active |
| `g_cutting` | `bool` | Shift+Right-drag cutter active |
| `g_cut_start` | `ImVec2` | Cut line screen-space origin |
| `g_just_pinned` | `bool` | Suppresses right-click selection after pin/unpin |
| `g_just_cut` | `bool` | Limits to one rope cut per drag |

### Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `PHYSICS_DELTA_TIME` | 1/60 | Fixed physics timestep |
| `ZOOM_DEFAULT` | 30 | Default camera zoom (px/world unit) |
| `ZOOM_MAX` | 500 | Maximum camera zoom |
| `ZOOM_FACTOR` | 1.1 | Zoom per scroll tick |
| `MAX_TASKS` | 64 | Max concurrent Box2D tasks |
| `MAX_FRAME_TIME` | 0.25 | Spiral-of-death clamp (seconds) |
| `MIN_STROKE_DIST` | 0.15 | Min distance between stroke points |
| `HIT_RADIUS` | 0.15 | Rope cut/erase hit-test radius |
| `SEGMENT_RADIUS` | 0.06 | Rope capsule collision radius |
| `SEGMENT_SPACING` | 0.15 | Distance between rope segment centers |
| `SEGMENT_HALF_LENGTH` | spacing × 0.6 | Capsule half-length (overlap) |
| `SEGMENT_TIP_OFFSET` | half_length + radius | Center to capsule tip |
| `AREA_MIN/MAX_X` | -100 / 100 | Playground X bounds |
| `AREA_MIN/MAX_Y` | -40 / 120 | Playground Y bounds |

## Code style

Run `clang-format` (v20, config in `.clang-format`) before committing.

- Allman braces, 4-space indent, no tabs, 150-column limit, LF endings.
- Right-aligned pointers/references: `int *p`, `float &r`.
- `auto` for locals. `auto` with suffix for float globals (e.g. `auto g_foo = 0.1f`).
- `int` only for `g_window_w` / `g_window_h` (SDL API match). Otherwise `std::int32_t` / `std::uint32_t` / `std::size_t`.
- Globals: `g_snake_case`. Constants/constexpr: `UPPER_SNAKE_CASE`. Functions: `snake_case`.
- Includes: third-party first, then stdlib sorted alphabetically. Manual sort (`SortIncludes: Never`).
- `emplace_back` over `push_back`. `= {}` over `.reset()` for optionals.
- C-style casts preferred. Windows `BOOL`: check `== FALSE` / `!= FALSE`, not `!`.
- `noexcept` on pure-computation functions/lambdas. NOT on allocating or Box2D-create functions.
- `[[nodiscard]]` on non-void free functions.
- `auto &&` in all range-for loops.
- `slider()` / `combo()` wrappers instead of raw ImGui widgets (they combine widget + scroll_adjust).
- `std::numbers::pi_v<float>` for pi. Box2D math builtins (`b2Dot`, `b2Lerp`, `b2Normalize`, etc.) over custom helpers.
- `b2Vec2` has no `operator/` — use `* (1.0f / x)`. Factor out `inverse_distance` when reused.
- `1e-6f` for physics epsilon (division-by-zero), `1e-3f` for render epsilon (skip degenerate visuals).
- **Descriptive names.** No abbreviations except `pos`, `io`, `id`, `def` suffix. Full words: `distance`, `direction`, `previous`, `segment`, `polygon`, `background`, `foreground`, `camera`, `frequency`, `format`, `delta_time`, `context`, `control`, `vertices`, `inverse`, `_squared`.

## Patterns and gotchas

- **New shape type** touches 6+ places: `SpawnShape` enum, `add_*()`, `tick_emitters` switch, drag-drop preview, drag-drop spawn, render loop.
- **Body creation** always appends to `g_bodies` with a `BodyState` for interpolation.
- **Body deletion** goes through `delete_selected()` — both ImGui button and Delete key call it.
- **Coordinate transforms**: `screen_to_world()` function and `to_screen` lambda in render block.
- **Ground**: static body at y=-1, 160m wide, 1m tall.
- **Polygon rendering**: expand vertices by shape radius along bisectors, inset for fill (edge rim). Shrink by `B2_LINEAR_SLOP` so shapes look flush. Outer outline gets AA fill; inner fill skips AA.
- **Rope segments**: capsules with `linearDamping=0.5`, `angularDamping=2.0`. Revolute joints at tips. Filter joints prevent anchor-segment collision.
- **Rope pruning**: checks anchor body validity each frame. Dead anchors → destroy orphan segments + joints.
- **Rope cutting** (`wire_half` lambda): splits segments into two half-ropes, rewires revolute joints. Guards null anchors. One cut per drag (`g_just_cut`).
- **Rope erasing**: inline hit-test with `previous_point` tracking (zero allocation). Destroys entire rope.
- **Force zones are formula-driven**: All zone types (Vortex, Radial, Uniform, Turbulence, etc.) are presets that fill `formula_x`/`formula_y` strings. Evaluated per body via TinyExpr++ with bound variables `x`, `y` (relative to center), `r` (distance), `angle` (atan2). Preset dropdown auto-matches when user edits formula text to a known preset. `Context` struct passes zone pointer for formula evaluation in overlap callback.
- **Force zone arrow rendering**: Grid-sampled formula evaluation, normalized for direction. Arrows flip for negative strength. Bodies/arrows within `1e-3f` of center skipped (singularity). `#if PHYS_DEBUG` shows grid lines and evaluated values at each sample point.
- **Shape queries hoisted**: `b2Shape_GetPolygon` / `b2Shape_GetCircle` called once per body per frame, reused for selection AABB + color + render.
- **Single-step transform bug**: previous transforms saved inside the accumulator loop, not before.
- **Catmull-Rom tangent kill**: when `b2Dot(tangent, segment_direction) < 0`, tangent zeroed to prevent vertex explosion.
- **Adaptive vsync**: SDL does NOT auto-fallback. Code explicitly retries with vsync=1 on failure.
- **Camera pixel-snapped**: `std::round()` on camera offset avoids subpixel blurriness.
- **No `imgui.ini`**: `io.IniFilename = nullptr`. Layout not persisted.
- **Renderer preference**: `direct3d12,direct3d11,direct3d,metal,vulkan,opengl,software`.
- `maxContactPushSpeed` = `6.0f * b2GetLengthUnitsPerMeter()`.
