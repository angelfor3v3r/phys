# phys

2D physics sandbox. Single-translation-unit C++23 app.

## Build

```bash
# GCC, Clang (GNU frontend), or MSVC. CMake 3.28+.
cmake -B cmake-build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug
# Release
cmake -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

```bash
# Non-AVX2 build (e.g. ARM or older x86):
cmake -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DPHYS_AVX2=OFF
cmake --build cmake-build-release
```

Dependencies are fetched automatically via CPM (no manual installs).

## Project layout

```
src/main.cpp          -- Entire application (single TU)
src/imconfig.hpp      -- Dear ImGui compile-time config (custom, not upstream)
deploy/windows/       -- phys.ico, phys.manifest, phys.rc (icon + manifest resource)
deploy/linux/         -- Flatpak manifest, .desktop, phys.svg (icon)
third_party/cmake/    -- get_cpm.cmake (CPM bootstrap)
.clang-format         -- clang-format 20 config
```

## Stack

| Library | Version | Role |
|---------|---------|------|
| SDL3 | 3.4.2 | Window, input, SDL_Renderer |
| Dear ImGui | 1.92.6-docking | UI, draw lists (all rendering goes through ImGui) |
| Box2D | 3.1.1 | Rigid body physics (v3 C API, AVX2 optional via PHYS_AVX2) |
| FreeType | 2.14.2 | Font rasterization for ImGui |
| dp::thread-pool | 0.7.0 | Worker threads shared with Box2D task system |
| scope_guard | 1.1.0 | RAII scope guard for file cleanup in read_binary_file |

## Architecture

- **SDL3 callback model** (`SDL_MAIN_USE_CALLBACKS`): `SDL_AppInit`, `SDL_AppIterate`, `SDL_AppEvent`, `SDL_AppQuit`. No explicit main loop.
- **All state is global.** No classes, no OOP hierarchy. Flat structs + free functions.
- **Box2D multithreading** wired via `box2d_enqueue_task`/`box2d_finish_task` using `std::latch` over a thread pool.
- **Rendering** uses ImGui background/foreground draw lists exclusively (no raw SDL draw calls).
- **Fixed timestep** with accumulator + interpolation (`g_physics_alpha`). Sleeping bodies skip interpolation.
- **P-core detection** at init: pins thread pool workers + main thread to performance cores. Windows (`GetSystemCpuSetInformation`), Linux (sysfs `cpu_type` / `base_frequency`), macOS (no-op).
- **Cross-platform file I/O** via `read_binary_file()`: RAII scope guard, Windows (`fopen_s`/`_fseeki64`/`_ftelli64`) and POSIX (`fopen`/`fseeko`/`ftello`) paths, chunked fallback for unseekable files.

## Code style

- clang-format 20 (config in `.clang-format`). Run before committing.
- Allman braces, 4-space indent, no tabs, 150-column limit.
- Pointer/reference right-aligned (`char *p`, `int &r`).
- Consecutive assignments/declarations aligned.
- `auto` used liberally for local variables.
- `std::int32_t` / `std::uint32_t` / `std::size_t` — no bare `int` for sizes or counts.
- Globals prefixed `g_`. Constants are `UPPER_SNAKE_CASE`.
- Includes: third-party first, then stdlib sorted alphabetically. (`SortIncludes: Never` in clang-format — manual sort.)
- `noexcept` on pure-computation functions/lambdas. `[[nodiscard]]` on non-void free functions.
- `slider()` / `combo()` wrappers (with `scroll_adjust`) instead of raw ImGui widget calls.
- **Descriptive variable names.** No abbreviations: `distance` not `dist`, `direction` not `dir`, `previous` not `prev`, `segment` not `seg`, `polygon` not `poly`, `background`/`foreground` not `bg`/`fg`, `camera` not `cam`, `frequency` not `freq`, `format` not `fmt`, `delta_time` not `dt`, `context` not `ctx`, `control` not `ctrl`, `vertices` not `verts`. Only `pos` is acceptable as abbreviation.
- `b2Vec2` has **no `operator/`** — use `delta * (1.0f / distance)` pattern. Factor out `inverse_distance = 1.0f / distance` when used multiple times.
- C-style casts preferred. Windows `BOOL`: check `== FALSE` or `!= FALSE`, never `!`.
- `emplace_back` preferred over `push_back`.
- `= {}` to clear optionals, not `.reset()`.
- `1e-6f` for physics epsilon (division-by-zero guard), `1e-3f` for render epsilon (skip degenerate visuals).

## Key globals

| Global | Purpose |
|--------|---------|
| `g_world` | Box2D world |
| `g_bodies` | `vector<PhysBody>` — all dynamic bodies |
| `g_drawn_lines` | `vector<DrawnLine>` — user-drawn static line geometry |
| `g_emitters` | `vector<Emitter>` — particle emitters |
| `g_camera_center`, `g_camera_zoom` | Camera state |
| `g_ropes` | `vector<Rope>` — chain links between bodies |
| `g_pins` | `vector<Pin>` — revolute-joint pins to ground |
| `g_force_zones` | `vector<ForceZone>` — areas applying forces to bodies |
| `g_rope_start_body` | Pending rope endpoint (null when idle) |
| `g_thread_pool` | Shared worker pool |
| `g_window`, `g_renderer` | SDL window and renderer |
| `g_perf_core_count`, `g_total_core_count` | CPU topology (P-cores vs total) |

## Platform notes

- Windows, Linux, and macOS. GCC, Clang (GNU frontend, LLVM 20), or MSVC. AppleClang is NOT supported (missing `std::jthread`/`std::stop_token`).
- AVX2 is a CMake option (`PHYS_AVX2`): ON by default for x86_64, OFF for ARM. Non-AVX2 builds use scalar fallbacks.
- Windows: hooks `WndProc` for `DwmFlush()` on `WM_MOVING` (drag stutter fix), links `dwmapi`.
- Static MSVC runtime on Windows (`/MT` / `/MTd`).
- `WIN32` subsystem on Windows (no console window).
- Windows manifest: DPI awareness (PerMonitorV2 + fallback), supportedOS (Win10/11 GUID). No UTF-8 codepage (risky on oldest Win10), no Common Controls v6 (unused by SDL+ImGui).
- Clang-on-Windows: `/MANIFEST:EMBED` stripped from link command to avoid conflict with RC-embedded manifest.
- Renderer preference: D3D12 > D3D11 > Metal > Vulkan > OpenGL > software.
- Camera pixel-snapped via `std::round()` to avoid subpixel blurriness.
- Adaptive vsync by default (`SDL_RENDERER_VSYNC_ADAPTIVE`), falls back to regular vsync if unsupported. SDL does NOT auto-fallback — code explicitly retries with vsync=1.
- FPS cap defaults to monitor refresh rate, uses `SDL_DelayPrecise` (skipped when <1ms slack remains — avoids fighting driver vsync).
- Graphics section in UI: VSync dropdown (Off/On/Adaptive), FPS cap slider (Off or 10-1000).
- Frame timers in Info: Physics, Render, Present, Frame (EMA-smoothed, ~100 frame window).
- Fill anti-aliasing disabled for body rendering (2x vertex savings). Outer outline fill retains AA for smooth silhouette.
- Kill bounds: `AREA_MIN/MAX ± 5m` margin, tied to world bounds (not viewport-derived).

## Conventions

- When adding a new shape type, update: `SpawnShape` enum, `add_*()` function, `tick_emitters` switch, drag-drop preview switch, drag-drop spawn switch, and rendering in `SDL_AppIterate`.
- Body creation always appends to `g_bodies` with a `BodyState` for interpolation.
- Body deletion goes through `delete_selected()` — both the ImGui button and Delete key call it.
- All coordinate transforms go through `screen_to_world()` or the `to_screen` lambda in the render block.
- Kill bounds: `AREA_MIN/MAX ± 5m` — tied to world bounds, not viewport. Bodies destroyed just past the world edge.
- Rope segments are capsules (`SEGMENT_RADIUS`, `SEGMENT_SPACING`, `SEGMENT_HALF_LENGTH` constants) connected by revolute joints at capsule tips. Capsules extend to 60% of spacing for overlap.
- Rope cleanup: prune ropes with dead anchors each frame, destroy orphaned segment bodies + joints.
- Rope cutting (Shift+Right drag): destroys all joints, splits segments into two half-ropes re-wired with revolute joints. `wire_half` guards null anchors (dangling ends get no anchor joint or filter joints). Visual: red line from drag start to cursor.
- Rope erasing (Ctrl+Right drag): inline hit-tests cursor against rope links (no allocation), destroys entire rope (all joints + segment bodies). Visual: red circle at cursor scaled to `HIT_RADIUS * g_camera_zoom`.
- Pin cleanup: prune invalid joints each frame.
- Pin/unpin actions set `g_just_pinned` to suppress right-click-up selection toggle.
- All range-for loops use `auto &&`.
- Use Box2D math builtins (`b2Dot`, `b2Lerp`, `b2NLerp`, `b2Normalize`, etc.) over custom wrappers.
- Force zone overlap callback: `Context` struct passed via `void*`, precomputes `angle_rotation` (`b2Rot`) and `radius` to avoid per-body trig/sqrt. Dead fields removed — if a field isn't read in the callback, don't store it.
- Shape queries hoisted: call `b2Shape_GetPolygon`/`b2Shape_GetCircle` once per body per frame, reuse for selection AABB + color + render.
- Catmull-Rom rope rendering: tangents killed (zeroed) when they oppose the segment direction to prevent vertex explosion at extreme bend angles.
