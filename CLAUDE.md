# phys

2D physics sandbox. Single-translation-unit C++23 app.

## Build

```bash
# Clang (primary) or MSVC. CMake 3.28+.
cmake -B cmake-build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug
# Release
cmake -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

Dependencies are fetched automatically via CPM (no manual installs).

## Project layout

```
src/main.cpp       -- Entire application (single TU)
src/imconfig.hpp   -- Dear ImGui compile-time config (custom, not upstream)
third_party/cmake/ -- get_cpm.cmake (CPM bootstrap)
.clang-format      -- clang-format 20 config
```

## Stack

| Library | Version | Role |
|---------|---------|------|
| SDL3 | 3.4.2 | Window, input, SDL_Renderer |
| Dear ImGui | 1.92.6-docking | UI, draw lists (all rendering goes through ImGui) |
| Box2D | 3.1.1 | Rigid body physics (v3 C API, AVX2 enabled) |
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
| `g_thread_pool` | Shared worker pool |
| `g_window`, `g_renderer` | SDL window and renderer |

## Platform notes

- Windows and Linux only. Clang or MSVC.
- Windows: hooks `WndProc` for `DwmFlush()` on `WM_MOVING` (drag stutter fix), links `dwmapi`.
- Static MSVC runtime on Windows (`/MT` / `/MTd`).
- `WIN32` subsystem (no console window).

## Conventions

- When adding a new shape type, update: `SpawnShape` enum, `add_*()` function, `tick_emitters` switch, drag-drop preview switch, drag-drop spawn switch, and rendering in `SDL_AppIterate`.
- Body creation always appends to `g_bodies` with a `BodyState` for interpolation.
- All coordinate transforms go through `screen_to_world()` or the `to_screen` lambda in the render block.
- Out-of-bounds body cleanup happens every frame at the top of `SDL_AppIterate`.
