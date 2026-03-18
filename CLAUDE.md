# phys

2D physics sandbox. Single-translation-unit C++23 app.

## Build

```bash
# Clang (GNU frontend) or MSVC. CMake 3.28+.
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
deploy/linux/         -- phys.desktop, phys.svg (AppImage assets)
third_party/cmake/    -- get_cpm.cmake (CPM bootstrap)
.clang-format         -- clang-format 20 config
```

## Stack

| Library | Version | Role |
|---------|---------|------|
| SDL3 | 3.4.2 | Window, input, SDL_Renderer |
| Dear ImGui | 1.92.6-docking | UI, draw lists (all rendering goes through ImGui) |
| Box2D | 3.1.1 | Rigid body physics (v3 C API, AVX2 optional via PHYS_AVX2) |
| FreeType | 2.14.1 | Font rasterization for ImGui |
| dp::thread-pool | 0.7.0 | Worker threads shared with Box2D task system |

## Architecture

- **SDL3 callback model** (`SDL_MAIN_USE_CALLBACKS`): `SDL_AppInit`, `SDL_AppIterate`, `SDL_AppEvent`, `SDL_AppQuit`. No explicit main loop.
- **All state is global.** No classes, no OOP hierarchy. Flat structs + free functions.
- **Box2D multithreading** wired via `box2d_enqueue_task`/`box2d_finish_task` using `std::latch` over a thread pool.
- **Rendering** uses ImGui background/foreground draw lists exclusively (no raw SDL draw calls).
- **Fixed timestep** with accumulator + interpolation (`g_physics_alpha`). Sleeping bodies skip interpolation.

## Code style

- clang-format 20 (config in `.clang-format`). Run before committing.
- Allman braces, 4-space indent, no tabs, 150-column limit.
- Pointer/reference right-aligned (`char *p`, `int &r`).
- Consecutive assignments/declarations aligned.
- `auto` used liberally for local variables.
- `std::int32_t` / `std::uint32_t` / `std::size_t` — no bare `int` for sizes or counts.
- Globals prefixed `g_`. Constants are `UPPER_SNAKE_CASE`.
- Includes: third-party first, then stdlib, not sorted by clang-format (`SortIncludes: Never`).

## Key globals

| Global | Purpose |
|--------|---------|
| `g_world` | Box2D world |
| `g_bodies` | `vector<PhysBody>` — all dynamic bodies |
| `g_drawn_lines` | `vector<DrawnLine>` — user-drawn static line geometry |
| `g_emitters` | `vector<Emitter>` — particle emitters |
| `g_cam_center`, `g_cam_zoom` | Camera state |
| `g_ropes` | `vector<Rope>` — chain links between bodies |
| `g_pins` | `vector<Pin>` — revolute-joint pins to ground |
| `g_rope_start_body` | Pending rope endpoint (null when idle) |
| `g_thread_pool` | Shared worker pool |
| `g_window`, `g_renderer` | SDL window and renderer |

## Platform notes

- Windows, Linux, and macOS. Clang (GNU frontend, LLVM 20) or MSVC. AppleClang is NOT supported (missing `std::jthread`/`std::stop_token`).
- AVX2 is a CMake option (`PHYS_AVX2`): ON by default for x86_64, OFF for ARM. Non-AVX2 builds use scalar fallbacks.
- Windows: hooks `WndProc` for `DwmFlush()` on `WM_MOVING` (drag stutter fix), links `dwmapi`.
- Static MSVC runtime on Windows (`/MT` / `/MTd`).
- `WIN32` subsystem on Windows (no console window).
- Windows icon, manifest (DPI + UTF-8 + common controls v6), and RC resource in `deploy/windows/`.
- Clang-on-Windows: `/MANIFEST:EMBED` stripped from link command to avoid conflict with RC-embedded manifest.
- Renderer preference: D3D12 > D3D11 > Metal > Vulkan > OpenGL > software.
- Camera pixel-snapped via `std::round()` to avoid subpixel blurriness.

## Conventions

- When adding a new shape type, update: `SpawnShape` enum, `add_*()` function, `tick_emitters` switch, drag-drop preview switch, drag-drop spawn switch, and rendering in `SDL_AppIterate`.
- Body creation always appends to `g_bodies` with a `BodyState` for interpolation.
- Body deletion goes through `delete_selected()` — both the ImGui button and Delete key call it.
- All coordinate transforms go through `screen_to_world()` or the `to_screen` lambda in the render block.
- Out-of-bounds body cleanup happens every frame at the top of `SDL_AppIterate`.

- Rope cleanup: prune ropes with dead anchors each frame, destroy orphaned segment bodies + joints.
- Rope cutting (Shift+Right drag): destroys all joints, splits segments into two half-ropes re-wired at current separations. `wire_half` guards null anchors (dangling ends get no anchor joint or filter joints).
- Rope erasing (Ctrl+Right drag): hit-tests cursor against rope links, destroys entire rope (all joints + segment bodies).
- Pin cleanup: prune invalid joints each frame.
- Pin/unpin actions set `g_just_pinned` to suppress right-click-up selection toggle.
- All range-for loops use `auto &&`.
- Use Box2D math builtins (`b2Dot`, `b2Lerp`, `b2NLerp`, `b2Normalize`, etc.) over custom wrappers.
- Variable naming: `dist_def`, `rev_def`, `mouse_def`, `filter_def_a/b` for joint defs. No single-letter names except standard math (`t`, `r`, `d`).