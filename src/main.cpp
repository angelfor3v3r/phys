#include <tinyexpr.h>
#include <scope_guard.hpp>
#include <thread_pool/thread_pool.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>
#include <imgui_stdlib.h>
#include <box2d/box2d.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <format>
#include <latch>
#include <memory>
#include <new>
#include <numbers>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

enum class SpawnShape : std::uint8_t
{
    Box = 0,
    Circle,
    Triangle,
    Count
};

// Constants.
constexpr auto         PI                  = std::numbers::pi_v<float>; // Pi constant (float precision).
constexpr auto         RAD_TO_DEG          = 180.0f / PI;               // Multiply radians to get degrees.
constexpr auto         DEG_TO_RAD          = PI / 180.0f;               // Multiply degrees to get radians.
constexpr std::int32_t MAX_TASKS           = 64;                        // Maximum concurrent Box2D tasks in flight.
constexpr auto         MAX_FRAME_TIME      = 0.25f;                     // Clamp for frame delta to prevent spiral of death (seconds).
constexpr auto         PHYSICS_DELTA_TIME  = 1.0f / 60.0f;              // Fixed physics timestep (seconds).
constexpr auto         ZOOM_DEFAULT        = 30.0f;                     // Default camera zoom (pixels per world unit).
constexpr auto         ZOOM_MAX            = 500.0f;                    // Maximum camera zoom.
constexpr auto         ZOOM_FACTOR         = 1.1f;                      // Multiplicative zoom per scroll tick.
constexpr auto         MIN_STROKE_DIST     = 0.15f;                     // Minimum distance between stroke points (world units).
constexpr auto         HIT_RADIUS          = 0.15f;                     // Hit-test radius for rope cut/erase (world units).
constexpr auto         HIT_RADIUS_SQUARED  = HIT_RADIUS * HIT_RADIUS;   // Squared hit-test radius (avoids sqrt in distance checks).
constexpr auto         SEGMENT_RADIUS      = 0.06f;                     // Rope capsule segment collision radius.
constexpr auto         SEGMENT_SPACING     = 0.15f;                     // Distance between rope segment centers.
constexpr auto         SEGMENT_HALF_LENGTH = SEGMENT_SPACING * 0.6f;    // Capsule half-length (>0.5x spacing so adjacent capsules overlap).
constexpr auto         SEGMENT_TIP_OFFSET  = SEGMENT_HALF_LENGTH + SEGMENT_RADIUS; // Distance from segment center to capsule tip.

// Playground bounds (world units).
constexpr auto AREA_MIN_X = -100.0f;
constexpr auto AREA_MAX_X = 100.0f;
constexpr auto AREA_MIN_Y = -40.0f;
constexpr auto AREA_MAX_Y = 120.0f;

// Window / renderer.
SDL_Window   *g_window{};        // Main application window.
float         g_dpi_scaling{};   // OS display scale factor.
SDL_Renderer *g_renderer{};      // SDL GPU renderer.
const char   *g_renderer_name{}; // Name of the active GPU renderer backend.
int           g_window_w{};      // Window width in pixels.
int           g_window_h{};      // Window height in pixels.

// Graphics.
int    g_fps_cap = 0;                           // 0 = off, 10-1000 = target FPS. Defaults to monitor refresh rate at init.
int    g_vsync   = SDL_RENDERER_VSYNC_ADAPTIVE; // Falls back to regular vsync if unsupported.
Uint64 g_frame_start_ns{};                      // Counter at start of frame (Nanoseconds since SDL initialization).
float  g_physics_ms{};                          // Last frame's physics time.
float  g_render_ms{};                           // Last frame's render time (draw list build).
float  g_present_ms{};                          // Last frame's GPU submit + present time.
float  g_frame_ms{};                            // Last frame's total time.
int    g_total_vertices{};                      // Last frame's total vertex count.
int    g_total_indices{};                       // Last frame's total index count.
int    g_frame_count{};                         // Frames counted in current second.
Uint64 g_fps_timer_ns{};                        // Timestamp of last FPS update.
int    g_display_fps{};                         // FPS shown on screen, updated once per second.

// Camera.
auto   g_camera_zoom_min = 1.0f;         // Minimum zoom level (clamped to keep viewport inside world bounds).
auto   g_camera_zoom     = ZOOM_DEFAULT; // Pixels per world unit.
b2Vec2 g_camera_center{0.0f, 10.0f};     // Camera center in world coordinates.
bool   g_camera_dragging{};              // Middle-mouse camera pan in progress.

// Threading.
std::uint32_t                      g_total_core_count{}; // Total logical CPUs on the system.
std::uint32_t                      g_perf_core_count{};  // P-cores detected (0 = homogeneous/unknown).
std::uint32_t                      g_worker_count{};     // Thread pool worker count.
std::unique_ptr<dp::thread_pool<>> g_thread_pool{};      // Box2D task thread pool.
std::int32_t                       g_task_count{};       // Active Box2D tasks in `g_task_storage`.

struct TaskHandle
{
    explicit TaskHandle(std::int32_t count) noexcept : done{count} {}

    std::latch done;
};

alignas(TaskHandle) std::byte g_task_storage[MAX_TASKS][sizeof(TaskHandle)]; // Placement-new storage for `TaskHandle` instances.
std::atomic<std::uint32_t> g_next_worker{};                                  // Round-robin worker index for thread-local storage.

// Physics.
b2WorldId    g_world = b2_nullWorldId; // Box2D physics world.
bool         g_paused{};               // Simulation paused.
bool         g_single_step{};          // Advance one physics step then re-pause.
std::int32_t g_step_count = 1;         // Physics steps per frame.
std::int32_t g_sub_steps  = 4;         // Box2D solver sub-steps per step.
float        g_physics_alpha{};        // Interpolation factor between previous and current physics state.
float        g_physics_accumulator{};  // Leftover time from previous frame for fixed-step integration.
auto         g_linear_damping = 0.1f;  // Base linear drag applied to every dynamic body.
auto         g_random_damping = 0.02f; // Max random offset added to base drag per body.
std::size_t  g_culled_count{};         // Bodies culled from rendering this frame (outside viewport).

// Bodies.
b2BodyId g_ground_id = b2_nullBodyId; // Static ground body (world boundary).

struct BodyState
{
    b2Vec2 position{};
    b2Rot  rotation{};
};

struct PhysBody
{
    b2BodyId    body       = b2_nullBodyId;
    b2ShapeId   shape      = b2_nullShapeId;
    b2ShapeType shape_type = b2_shapeTypeCount;
    BodyState   previous{};
    float       damping_offset{}; // Random addition to `g_linear_damping`.
    bool        selected{};
};

std::vector<PhysBody> g_bodies{}; // All spawned dynamic bodies.

// Drawing.
struct DrawnLine
{
    b2BodyId            body = b2_nullBodyId;
    std::vector<b2Vec2> points{};
};

std::vector<DrawnLine> g_drawn_lines{};    // Committed freehand-drawn ground segments.
std::vector<b2Vec2>    g_current_stroke{}; // Points in the in-progress drawing stroke.

// Emitters.
struct Emitter
{
    b2Vec2     position{};
    float      angle{};       // Radians, 0 = right.
    float      speed = 15.0f; // m/s.
    float      rate  = 3.0f;  // Bodies per second.
    float      timer{};
    SpawnShape shape = SpawnShape::Box;
    bool       active{};
};

std::vector<Emitter> g_emitters{}; // All body emitters.

// Force zones: Areas that apply forces to dynamic bodies via user-defined formulas.
// Formulas are evaluated per body with bound variables: x, y (relative to center), r (distance), angle (atan2).
enum class ZoneShape : std::uint8_t
{
    Rectangle = 0, // Axis-aligned box boundary.
    Circle,        // Circular boundary with radius.
    Count,
};

// Preset formula definitions for force zones.
struct FormulaPreset
{
    const char *name{};
    const char *formula_x{};
    const char *formula_y{};
};

constexpr std::array ZONE_PRESETS = {
    FormulaPreset{"Vortex", "-y / r", "x / r"},
    FormulaPreset{"Radial In", "-x / r", "-y / r"},
    FormulaPreset{"Radial Out", "x / r", "y / r"},
    FormulaPreset{"Uniform", "1", "0"},
    FormulaPreset{"Spiral In", "-y/r - x/r", "x/r - y/r"},
    FormulaPreset{"Spiral Out", "-y/r + x/r", "x/r + y/r"},
    FormulaPreset{"Turbulence", "sin(y * 3)", "cos(x * 3)"},
    FormulaPreset{"Saddle", "x", "-y"},
    FormulaPreset{"Dipole", "2*x*y / (r*r)", "(x*x - y*y) / (r*r)"},
};

constexpr int ZONE_PRESET_COUNT  = (int)ZONE_PRESETS.size();
constexpr int ZONE_PRESET_CUSTOM = ZONE_PRESET_COUNT; // Index used when formula text is manually edited.

struct ForceZone
{
    b2Vec2       position{};
    b2Vec2       half_size{2.0f, 2.0f};   // Half extents (Rectangle).
    float        angle{};                 // Radians. Rotates the evaluated force vector.
    float        strength        = 20.0f; // Acceleration (m/s^2). Scales the evaluated formula.
    float        max_speed       = 30.0f; // Terminal velocity (m/s). Force scales to zero at this speed. 0 = unlimited.
    float        radius          = 5.0f;  // Area-of-effect (Circle).
    std::int32_t grid_resolution = 5;     // Arrow grid NxN resolution.
    int          preset          = 0;     // Index into `ZONE_PRESETS`, or `ZONE_PRESET_CUSTOM` for user-edited.
    ZoneShape    shape           = ZoneShape::Circle;
    bool         active          = true;

    // Formula fields evaluated per body via tinyexpr++.
    std::string formula_x = ZONE_PRESETS[0].formula_x; // X component of force direction.
    std::string formula_y = ZONE_PRESETS[0].formula_y; // Y component of force direction.
    te_parser   parser_x{};                            // Compiled X expression.
    te_parser   parser_y{};                            // Compiled Y expression.
    double      bound_x{};                             // Bound variable: Body X relative to zone center.
    double      bound_y{};                             // Bound variable: Body Y relative to zone center.
    double      bound_distance{};                      // Bound variable: Distance from zone center (formula name: `r`).
    double      bound_angle{};                         // Bound variable: `atan2(y, x)` angle to zone center.
    bool        formula_valid{};                       // True if both formulas compiled successfully.
    std::string formula_error{};                       // Error message from last failed compile.
};

// Compile formula expressions. Call when formula text changes.
void compile_formulas(ForceZone &zone)
{
    zone.parser_x.set_variables_and_functions(
        {{"x", &zone.bound_x}, {"y", &zone.bound_y}, {"r", &zone.bound_distance}, {"angle", &zone.bound_angle}});
    zone.parser_y.set_variables_and_functions(
        {{"x", &zone.bound_x}, {"y", &zone.bound_y}, {"r", &zone.bound_distance}, {"angle", &zone.bound_angle}});

    auto ok_x          = zone.parser_x.compile(zone.formula_x);
    auto ok_y          = zone.parser_y.compile(zone.formula_y);
    zone.formula_valid = ok_x && ok_y;
    if (!zone.formula_valid)
    {
        zone.formula_error = !ok_x ? zone.parser_x.get_last_error_message() : zone.parser_y.get_last_error_message();
    }
}

// Apply a preset by index. Fills formula strings and recompiles.
void apply_zone_preset(ForceZone &zone, int index)
{
    zone.preset    = index;
    zone.formula_x = ZONE_PRESETS[(std::size_t)index].formula_x;
    zone.formula_y = ZONE_PRESETS[(std::size_t)index].formula_y;
    compile_formulas(zone);
}

std::vector<ForceZone>     g_force_zones{};   // All active force zone regions.
std::optional<std::size_t> g_dragging_zone{}; // Index of force zone being dragged, if any.

// Ropes: Chain of capsule segments connected by revolute joints.
struct Rope
{
    std::vector<b2BodyId>  segments{};             // Intermediate chain links.
    std::vector<b2JointId> joints{};               // All joints (revolute + filter).
    b2BodyId               body_a = b2_nullBodyId; // Anchor body at start (null if dangling).
    b2BodyId               body_b = b2_nullBodyId; // Anchor body at end   (null if dangling).
    b2Vec2                 local_a{};              // Attach point in `body_a`'s local space.
    b2Vec2                 local_b{};              // Attach point in `body_b`'s local space.
};

std::vector<Rope> g_ropes{};                         // All ropes (intact and cut halves).
b2BodyId          g_rope_start_body = b2_nullBodyId; // First body clicked when creating a rope.
b2Vec2            g_rope_start_anchor{};             // Local-space attach point on rope start body.

// Pins: Revolute joint fixing a body to the static ground.
struct Pin
{
    b2JointId joint = b2_nullJointId; // Revolute joint anchoring body to ground.
    b2BodyId  body  = b2_nullBodyId;  // The pinned dynamic body.
};

std::vector<Pin> g_pins{}; // All pinned bodies (revolute joint to ground).

// Interaction.
b2JointId                  g_mouse_joint = b2_nullJointId; // Joint for mouse-drag interaction.
b2BodyId                   g_mouse_body  = b2_nullBodyId;  // Body currently being mouse-dragged.
bool                       g_box_selecting{};              // Box-selection drag in progress.
ImVec2                     g_box_select_start{};           // Screen-space start point of box selection.
bool                       g_cutting{};                    // Shift+Right-drag rope cutter active.
bool                       g_just_cut{};                   // Limits rope cutting to one cut per drag.
ImVec2                     g_cut_start{};                  // Screen-space start point of rope cut drag.
bool                       g_erasing{};                    // Ctrl+Right-drag rope eraser active.
std::optional<std::size_t> g_dragged_emitter{};            // Index of emitter being repositioned.
bool                       g_just_pinned{};                // Suppresses right-click selection after pinning.

[[nodiscard]] std::vector<std::byte> read_binary_file(std::string_view filename, std::size_t read_size = 0) noexcept
{
    if (filename.empty())
    {
        return {};
    }

#if defined(_WIN32)
    FILE *file;
    if (fopen_s(&file, filename.data(), "rb") != 0)
#else
    auto *file = std::fopen(filename.data(), "rb");
    if (file == nullptr)
#endif
    {
        return {};
    }

    auto guard = sg::make_scope_guard([&file]() noexcept { std::fclose(file); });

    // If a specific read size was requested, use it directly.
    if (read_size > 0)
    {
        std::vector<std::byte> result{};
        try
        {
            result.resize(read_size);
        }
        catch (...)
        {
            return {};
        }

        if (std::fread(result.data(), 1, read_size, file) != read_size)
        {
            return {};
        }

        return result;
    }

    // Try to get the file size via seeking.
#if defined(_WIN32)
    long long file_size = -1;
    if (_fseeki64(file, 0, SEEK_END) == 0)
    {
        file_size = _ftelli64(file);
        _fseeki64(file, 0, SEEK_SET);
    }
#else
    off_t file_size = -1;
    if (fseeko(file, 0, SEEK_END) == 0)
    {
        file_size = ftello(file);
        fseeko(file, 0, SEEK_SET);
    }
#endif

    // We have a valid size, read it all in one go.
    if (file_size > 0)
    {
        std::vector<std::byte> result{};
        if ((std::size_t)file_size > result.max_size())
        {
            return {};
        }

        try
        {
            result.resize((std::size_t)file_size);
        }
        catch (...)
        {
            return {};
        }

        if (std::fread(result.data(), 1, result.size(), file) != result.size())
        {
            return {};
        }

        return result;
    }

    // Size unknown or zero. Try to read in chunks as a last resort.
    constexpr std::size_t CHUNK_SIZE = 4096;

    std::vector<std::byte> result{};
    try
    {
        result.reserve(CHUNK_SIZE);
    }
    catch (...)
    {
        return {};
    }

    for (;;)
    {
        std::array<std::byte, CHUNK_SIZE> chunk{};
        auto                              read_amount = std::fread(chunk.data(), 1, CHUNK_SIZE, file);
        if (read_amount > 0)
        {
            try
            {
                result.insert(result.end(), chunk.begin(), chunk.begin() + (std::ptrdiff_t)read_amount);
            }
            catch (...)
            {
                return {};
            }
        }

        // Read error.
        if (std::ferror(file) != 0)
        {
            return {};
        }

        // Reached end of file.
        if (std::feof(file) != 0)
        {
            break;
        }
    }

    return result;
}

// Platform: P-core detection and thread affinity.
// Box2D performs best on performance cores accessing a single L2 cache.
// We detect P-cores (highest `EfficiencyClass` on Windows, highest `base_frequency` on Linux) and pin thread-pool workers to them. On homogeneous
// CPUs all cores qualify.
// TODO: On multi-CCD AMD chips (7950X3D, 9950X, Threadripper), further filter P-cores to the largest subset sharing an L3 cache.
//       Linux: `/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list`
//       Windows: `GetLogicalProcessorInformationEx(RelationCache)` with `CacheLevel` == 3
#if defined(_WIN32)
#include <dwmapi.h>

struct CoreTopology
{
    std::vector<DWORD> perf_cpu_set_ids{}; // CPU Set IDs for `SetThreadSelectedCpuSets`.
    std::uint32_t      total_logical{};
};

[[nodiscard]] CoreTopology detect_performance_cores()
{
    // First call gets required buffer size. Returns `FALSE` with `ERROR_INSUFFICIENT_BUFFER`.
    ULONG buffer_size;
    if (GetSystemCpuSetInformation(nullptr, 0, &buffer_size, GetCurrentProcess(), 0) != FALSE || buffer_size == 0)
    {
        return {};
    }

    std::vector<std::byte> buffer(buffer_size);
    ULONG                  returned_size;
    if (GetSystemCpuSetInformation((SYSTEM_CPU_SET_INFORMATION *)buffer.data(), buffer_size, &returned_size, GetCurrentProcess(), 0) == FALSE)
    {
        return {};
    }

    CoreTopology result{};
    BYTE         max_efficiency{};

    // Single pass: Count CPUs, find max `EfficiencyClass`, collect all entries.
    struct CPUEntry
    {
        DWORD id{};
        BYTE  efficiency{};
    };

    std::vector<CPUEntry> entries{};
    auto                 *cursor = buffer.data();
    auto                 *end    = buffer.data() + returned_size;
    while (cursor < end)
    {
        auto *info = (SYSTEM_CPU_SET_INFORMATION *)cursor;
        if (info->Type == CpuSetInformation)
        {
            ++result.total_logical;

            max_efficiency = std::max(max_efficiency, info->CpuSet.EfficiencyClass);

            entries.emplace_back(info->CpuSet.Id, info->CpuSet.EfficiencyClass);
        }

        cursor += info->Size;
    }

    // Keep only CPUs at the highest `EfficiencyClass` (= P-cores).
    for (auto &&entry : entries)
    {
        if (entry.efficiency == max_efficiency)
        {
            result.perf_cpu_set_ids.emplace_back(entry.id);
        }
    }

    return result;
}

void bind_thread_to_perf_cores(const std::vector<DWORD> &cpu_set_ids)
{
    if (cpu_set_ids.empty())
    {
        return;
    }

    SetThreadSelectedCpuSets(GetCurrentThread(), cpu_set_ids.data(), (ULONG)cpu_set_ids.size());
}
#elif defined(__linux__)
#include <sched.h>

struct CoreTopology
{
    std::vector<int> perf_cpu_indices{}; // Logical CPU indices for `cpu_set_t`.
    std::uint32_t    total_logical{};
};

// Parse a small sysfs text file as a string. Returns empty on failure.
[[nodiscard]] std::string read_sysfs_text(const std::string &path)
{
    auto bytes = read_binary_file(path);
    if (bytes.empty())
    {
        return {};
    }

    // Trim trailing newline.
    std::string text((const char *)bytes.data(), bytes.size());
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
    {
        text.pop_back();
    }

    return text;
}

[[nodiscard]] CoreTopology detect_performance_cores()
{
    long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu_count <= 0)
    {
        return {};
    }

    int num_cpus = (int)cpu_count; // Narrowing is safe; CPU count will never exceed INT_MAX.

    CoreTopology result{};
    result.total_logical = (std::uint32_t)num_cpus;

    // Strategy 1: Try sysfs cpu_type (modern kernels).
    // P-cores report "intel_core", E-cores report "intel_atom".
    // Non-hybrid or non-Intel CPUs may not have this file at all.
    std::vector<int> core_indices{};
    for (int i{}; i < num_cpus; ++i)
    {
        auto type = read_sysfs_text(std::format("/sys/devices/system/cpu/cpu{}/cpu_type", i));
        if (!type.empty() && !type.contains("atom"))
        {
            core_indices.emplace_back(i);
        }
    }

    if (!core_indices.empty())
    {
        result.perf_cpu_indices = std::move(core_indices);

        return result;
    }

    // Strategy 2: Compare base_frequency across CPUs.
    //   P-cores have higher base frequency than E-cores.
    std::vector<std::pair<int, long>> cpu_frequencies{};
    cpu_frequencies.reserve(num_cpus);

    for (int i{}; i < num_cpus; ++i)
    {
        auto text = read_sysfs_text(std::format("/sys/devices/system/cpu/cpu{}/cpufreq/base_frequency", i));
        if (text.empty())
        {
            continue;
        }

        long frequency{};
        auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), frequency);
        if (ec == std::errc{} && frequency > 0)
        {
            cpu_frequencies.emplace_back(i, frequency);
        }
    }

    if (!cpu_frequencies.empty())
    {
        long max_frequency{};
        for (auto &&[cpu, frequency] : cpu_frequencies)
        {
            max_frequency = std::max(max_frequency, frequency);
        }

        // Only filter if there's a meaningful frequency gap (>10%).
        long threshold = max_frequency * 9 / 10;
        for (auto &&[cpu, frequency] : cpu_frequencies)
        {
            if (frequency >= threshold)
            {
                result.perf_cpu_indices.emplace_back(cpu);
            }
        }

        if (!result.perf_cpu_indices.empty())
        {
            return result;
        }
    }

    // Fallback: All CPUs are performance cores (homogeneous or unknown).
    result.perf_cpu_indices.resize(num_cpus);
    std::iota(result.perf_cpu_indices.begin(), result.perf_cpu_indices.end(), 0);

    return result;
}

void bind_thread_to_perf_cores(const std::vector<int> &cpu_indices)
{
    if (cpu_indices.empty())
    {
        return;
    }

    cpu_set_t set;
    CPU_ZERO(&set);

    for (auto &&cpu : cpu_indices)
    {
        CPU_SET(cpu, &set);
    }

    sched_setaffinity(0, sizeof(cpu_set_t), &set);
}
#else
// macOS / Other: No affinity control.
// Apple Silicon hardcodes `ml_get_max_affinity_sets` to 0 (XNU arm64) - the Intel-era THREAD_AFFINITY_POLICY API` is non-functional.
// QoS classes handle P/E core placement instead; default-QoS threads already land on P-cores under normal load.
struct CoreTopology
{
    std::uint32_t total_logical{};
};

[[nodiscard]] CoreTopology detect_performance_cores() { return {(std::uint32_t)std::thread::hardware_concurrency()}; }

void bind_thread_to_perf_cores() {}
#endif

// Silly fix for window dragging stutter.
#if defined(_WIN32)
WNDPROC g_original_wndproc{};

LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    auto result = CallWindowProc(g_original_wndproc, hwnd, msg, wp, lp);

    if (msg == WM_MOVING)
    {
        DwmFlush();
    }

    return result;
}
#endif

// Brighten an `ImU32` color by adding `amount` to each RGB channel, clamped to 255.
[[nodiscard]] constexpr ImU32 brighten(ImU32 color, int amount) noexcept
{
    int red   = std::min(255, ((int)(color >> IM_COL32_R_SHIFT) & 0xFF) + amount);
    int green = std::min(255, ((int)(color >> IM_COL32_G_SHIFT) & 0xFF) + amount);
    int blue  = std::min(255, ((int)(color >> IM_COL32_B_SHIFT) & 0xFF) + amount);

    return IM_COL32(red, green, blue, 255);
}

[[nodiscard]] b2Vec2 screen_to_world(float screen_x, float screen_y) noexcept
{
    return {
        g_camera_center.x + (screen_x - (float)g_window_w / 2.0f) / g_camera_zoom,
        g_camera_center.y - (screen_y - (float)g_window_h / 2.0f) / g_camera_zoom,
    };
}

[[nodiscard]] float segment_distance_squared(b2Vec2 point, b2Vec2 segment_start, b2Vec2 segment_end) noexcept
{
    auto  edge        = segment_end - segment_start;
    auto  len_squared = b2Dot(edge, edge);
    float projection{};
    if (len_squared > 0.0f)
    {
        projection = std::clamp(b2Dot(point - segment_start, edge) / len_squared, 0.0f, 1.0f);
    }

    auto closest = segment_start + edge * projection;
    auto offset  = point - closest;

    return b2Dot(offset, offset);
}

[[nodiscard]] b2BodyId find_body_at(b2Vec2 world_point) noexcept
{
    auto     proxy  = b2MakeProxy(&world_point, 1, 0.01f);
    b2BodyId result = b2_nullBodyId;
    b2World_OverlapShape(
        g_world, &proxy, b2DefaultQueryFilter(),
        [](b2ShapeId shapeId, void *context) noexcept
        {
            auto body = b2Shape_GetBody(shapeId);
            if (b2Body_GetType(body) != b2_dynamicBody)
            {
                return true;
            }

            *(b2BodyId *)context = body;

            return false;
        },
        &result);

    return result;
}

void add_box(b2Vec2 position, float half_width, float half_height)
{
    auto random_offset     = std::round(SDL_randf() * g_random_damping * 1000.0f) * 0.001f;
    auto body_def          = b2DefaultBodyDef();
    body_def.type          = b2_dynamicBody;
    body_def.position      = position;
    body_def.linearDamping = g_linear_damping + random_offset;

    auto body = b2CreateBody(g_world, &body_def);

    auto shape_def              = b2DefaultShapeDef();
    shape_def.density           = 1.0f;
    shape_def.material.friction = 0.3f;

    auto box      = b2MakeBox(half_width, half_height);
    auto shape_id = b2CreatePolygonShape(body, &shape_def, &box);

    g_bodies.emplace_back(body, shape_id, b2_polygonShape, BodyState{position, b2Rot_identity}, random_offset);
}

void add_circle(b2Vec2 position, float radius)
{
    auto random_offset     = std::round(SDL_randf() * g_random_damping * 1000.0f) * 0.001f;
    auto body_def          = b2DefaultBodyDef();
    body_def.type          = b2_dynamicBody;
    body_def.position      = position;
    body_def.linearDamping = g_linear_damping + random_offset;

    auto body = b2CreateBody(g_world, &body_def);

    auto shape_def              = b2DefaultShapeDef();
    shape_def.density           = 1.0f;
    shape_def.material.friction = 0.3f;

    b2Circle circle{{}, radius};
    auto     shape_id = b2CreateCircleShape(body, &shape_def, &circle);

    g_bodies.emplace_back(body, shape_id, b2_circleShape, BodyState{position, b2Rot_identity}, random_offset);
}

void add_triangle(b2Vec2 position, float height)
{
    auto random_offset     = std::round(SDL_randf() * g_random_damping * 1000.0f) * 0.001f;
    auto body_def          = b2DefaultBodyDef();
    body_def.type          = b2_dynamicBody;
    body_def.position      = position;
    body_def.linearDamping = g_linear_damping + random_offset;

    auto body = b2CreateBody(g_world, &body_def);

    auto shape_def              = b2DefaultShapeDef();
    shape_def.density           = 1.0f;
    shape_def.material.friction = 0.3f;

    // Equilateral triangle, centered at centroid.
    auto   half_base = height / std::sqrt(3.0f);
    b2Vec2 vertices[3]{
        {0.0f, height * 2.0f / 3.0f},
        {-half_base, -height / 3.0f},
        {half_base, -height / 3.0f},
    };
    auto hull     = b2ComputeHull(vertices, 3);
    auto polygon  = b2MakePolygon(&hull, 0.0f);
    auto shape_id = b2CreatePolygonShape(body, &shape_def, &polygon);

    g_bodies.emplace_back(body, shape_id, b2_polygonShape, BodyState{position, b2Rot_identity}, random_offset);
}

[[nodiscard]] DrawnLine create_drawn_line(std::vector<b2Vec2> points)
{
    auto body_def = b2DefaultBodyDef();
    auto body     = b2CreateBody(g_world, &body_def);

    auto shape_def              = b2DefaultShapeDef();
    shape_def.material.friction = 0.5f;

    for (std::size_t i{}; i < points.size() - 1; ++i)
    {
        b2Segment segment{points[i], points[i + 1]};
        b2CreateSegmentShape(body, &shape_def, &segment);
    }

    return {body, std::move(points)};
}

void finish_stroke()
{
    if (g_current_stroke.size() < 2)
    {
        g_current_stroke.clear();

        return;
    }

    g_drawn_lines.emplace_back(create_drawn_line(std::move(g_current_stroke)));

    g_current_stroke.clear();
}

void delete_selected()
{
    for (auto &&body : g_bodies)
    {
        if (body.selected)
        {
            b2DestroyBody(body.body);
        }
    }

    std::erase_if(g_bodies, [](const PhysBody &body) noexcept { return body.selected; });
}

// Adjust a value by mouse wheel or +/- keys when the last ImGui item is hovered.
template <class T>
requires (std::integral<T> || std::floating_point<T>) void scroll_adjust(T &value, T step, T min_value, T max_value) noexcept
{
    if (!ImGui::IsItemHovered())
    {
        return;
    }

    auto wheel = ImGui::GetIO().MouseWheel;
    if (wheel != 0.0f)
    {
        value = std::clamp(value + (T)(wheel * (float)step), min_value, max_value);

        ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Equal) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd))
    {
        value = std::clamp(value + step, min_value, max_value);
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Minus) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract))
    {
        value = std::clamp(value - step, min_value, max_value);
    }
}

// `SliderFloat`/`SliderInt` + scroll_adjust in one call.
bool slider(const char *label, float &value, float min_value, float max_value, float scroll_step, const char *format = "%.3f") noexcept
{
    auto result = ImGui::SliderFloat(label, &value, min_value, max_value, format);
    scroll_adjust(value, scroll_step, min_value, max_value);

    return result;
}

bool slider(const char *label, int &value, int min_value, int max_value, int scroll_step, const char *format = "%d") noexcept
{
    auto result = ImGui::SliderInt(label, &value, min_value, max_value, format);
    scroll_adjust(value, scroll_step, min_value, max_value);

    return result;
}

// Combo (null-separated items) + scroll_adjust in one call.
bool combo(const char *label, int &index, const char *items_separated_by_zeros, int item_count) noexcept
{
    auto result = ImGui::Combo(label, &index, items_separated_by_zeros);
    scroll_adjust(index, 1, 0, item_count - 1);

    return result;
}

void tick_emitters(float delta_time)
{
    for (auto &&emitter : g_emitters)
    {
        if (!emitter.active || emitter.rate <= 0.0f)
        {
            continue;
        }

        emitter.timer += delta_time;

        auto interval = 1.0f / emitter.rate;
        while (emitter.timer >= interval)
        {
            emitter.timer -= interval;

            switch (emitter.shape)
            {
            case SpawnShape::Box:
            {
                add_box(emitter.position, 0.5f, 0.5f);

                break;
            }

            case SpawnShape::Circle:
            {
                add_circle(emitter.position, 0.5f);

                break;
            }

            case SpawnShape::Triangle:
            {
                add_triangle(emitter.position, 1.0f);

                break;
            }

            default:
            {
                return;
            }
            }

            // Apply velocity to the just-spawned body.
            auto  &spawned = g_bodies.back();
            b2Vec2 direction{std::cos(emitter.angle), std::sin(emitter.angle)};
            b2Body_SetLinearVelocity(spawned.body, direction * emitter.speed);
        }
    }
}

void tick_force_zones()
{
    for (auto &&zone : g_force_zones)
    {
        if (!zone.active || !zone.formula_valid)
        {
            continue;
        }

        // Build AABB for the overlap query.
        b2AABB aabb;
        if (zone.shape == ZoneShape::Rectangle)
        {
            auto lower = zone.position - zone.half_size;
            auto upper = zone.position + zone.half_size;
            aabb       = {lower, upper};
        }
        else
        {
            b2Vec2 extent{zone.radius, zone.radius};
            aabb = {zone.position - extent, zone.position + extent};
        }

        // Context passed to the overlap callback via `void*`.
        struct Context
        {
            b2Vec2     center{};
            b2Rot      angle_rotation{}; // Precomputed cos/sin of the zone angle.
            float      strength{};
            float      max_speed{};
            float      radius{}; // For edge fade (Circle shape).
            float      radius_squared{};
            ZoneShape  shape{};
            ForceZone *zone{}; // Pointer to zone for formula evaluation.
        };

        Context context{zone.position, b2MakeRot(zone.angle),     zone.strength, zone.max_speed,
                        zone.radius,   zone.radius * zone.radius, zone.shape,    &zone};

        b2World_OverlapAABB(
            g_world, aabb, b2DefaultQueryFilter(),
            [](b2ShapeId shape_id, void *user_data) noexcept
            {
                auto body = b2Shape_GetBody(shape_id);
                if (b2Body_GetType(body) != b2_dynamicBody)
                {
                    return true;
                }

                auto &context  = *(Context *)user_data;
                auto &zone_ref = *context.zone;
                auto  body_pos = b2Body_GetPosition(body);
                auto  delta    = body_pos - context.center;

                // For circular zones, skip bodies outside the radius.
                if (context.shape == ZoneShape::Circle)
                {
                    if (b2LengthSquared(delta) > context.radius_squared)
                    {
                        return true;
                    }
                }

                // Skip bodies at zone center where r-dependent formulas are singular.
                auto distance = b2Length(delta);
                if (distance < 1e-3f)
                {
                    return true;
                }

                // Bind variables for tinyexpr++ evaluation.
                zone_ref.bound_x        = (double)delta.x;
                zone_ref.bound_y        = (double)delta.y;
                zone_ref.bound_distance = (double)distance;
                zone_ref.bound_angle    = std::atan2(zone_ref.bound_y, zone_ref.bound_x);

                auto fx    = (float)zone_ref.parser_x.evaluate();
                auto fy    = (float)zone_ref.parser_y.evaluate();
                auto mass  = b2Body_GetMass(body);
                auto force = b2RotateVector(context.angle_rotation, {fx, fy}) * (context.strength * mass);

                // Edge fade for circular boundary.
                if (context.shape == ZoneShape::Circle && context.radius > 0.0f)
                {
                    auto edge_fade = std::max(0.0f, 1.0f - distance / context.radius);
                    force          = force * edge_fade;
                }

                // Terminal velocity: Scale force to zero as body approaches `max_speed`.
                if (context.max_speed > 0.0f)
                {
                    auto force_len_squared = b2LengthSquared(force);
                    if (force_len_squared > 1e-6f)
                    {
                        auto force_direction = force * (1.0f / std::sqrt(force_len_squared));
                        auto velocity        = b2Body_GetLinearVelocity(body);
                        auto speed_along     = b2Dot(velocity, force_direction);
                        auto speed_fraction  = std::max(0.0f, speed_along) / context.max_speed;
                        force                = force * std::max(0.0f, 1.0f - speed_fraction);
                    }
                }

                b2Body_ApplyForceToCenter(body, force, true);

                return true;
            },
            &context);
    }
}

void reset_tasks() noexcept
{
    for (std::int32_t i{}; i < g_task_count; ++i)
    {
        auto *handle = std::launder((TaskHandle *)&g_task_storage[i]);
        handle->~TaskHandle();
    }

    g_task_count = 0;
}

void *box2d_enqueue_task(b2TaskCallback *task, std::int32_t itemCount, std::int32_t minRange, void *taskContext, void *userContext)
{
    auto &pool        = *(dp::thread_pool<> *)userContext;
    auto  chunk_size  = std::max(minRange, itemCount / (std::int32_t)g_worker_count);
    auto  chunk_count = (itemCount + chunk_size - 1) / chunk_size;

    // Fallback: Run serially. `nullptr` tells Box2D to skip finishTask.
    if (g_task_count >= MAX_TASKS)
    {
        task(0, itemCount, 0, taskContext);

        return nullptr;
    }

    auto *handle = new(&g_task_storage[g_task_count++]) TaskHandle(chunk_count);

    for (std::int32_t i{}; i < itemCount; i += chunk_size)
    {
        auto end = std::min(i + chunk_size, itemCount);
        pool.enqueue_detach(
            [task, i, end, taskContext, handle]
            {
                thread_local std::uint32_t worker = g_next_worker++;
                task(i, end, worker, taskContext);

                handle->done.count_down();
            });
    }

    return handle;
}

void box2d_finish_task(void *userTask, [[maybe_unused]] void *userContext) noexcept
{
    if (userTask == nullptr)
    {
        return;
    }

    ((TaskHandle *)userTask)->done.wait();
}

void step_physics(float delta_time)
{
    if (g_paused && !g_single_step)
    {
        return;
    }

    if (g_single_step)
    {
        for (std::int32_t i{}; i < g_step_count; ++i)
        {
            // Save previous transforms for interpolation.
            for (auto &&body : g_bodies)
            {
                auto transform         = b2Body_GetTransform(body.body);
                body.previous.position = transform.p;
                body.previous.rotation = transform.q;
            }

            reset_tasks();
            b2World_Step(g_world, PHYSICS_DELTA_TIME, g_sub_steps);
        }

        tick_emitters(PHYSICS_DELTA_TIME * (float)g_step_count);
        tick_force_zones();

        g_single_step   = false;
        g_physics_alpha = 1.0f;

        return;
    }

    g_physics_accumulator += std::min(delta_time, MAX_FRAME_TIME);
    while (g_physics_accumulator >= PHYSICS_DELTA_TIME)
    {
        // Save previous transforms for interpolation.
        for (auto &&body : g_bodies)
        {
            auto transform         = b2Body_GetTransform(body.body);
            body.previous.position = transform.p;
            body.previous.rotation = transform.q;
        }

        reset_tasks();
        b2World_Step(g_world, PHYSICS_DELTA_TIME, g_sub_steps);

        g_physics_accumulator -= PHYSICS_DELTA_TIME;
    }

    g_physics_alpha = g_physics_accumulator / PHYSICS_DELTA_TIME;
}

SDL_AppResult SDL_AppInit([[maybe_unused]] void **appstate, [[maybe_unused]] std::int32_t argc, [[maybe_unused]] char *argv[])
{
#if PHYS_AVX2
    // Box2D is compiled with AVX2. Bail early on unsupported CPUs.
    if (!SDL_HasAVX2())
    {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "phys", "This application requires a CPU with AVX2 support.", nullptr);

        return SDL_APP_FAILURE;
    }
#endif

    // Set up window and renderer.
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "phys", std::format("Error `SDL_Init()`: {}", SDL_GetError()).c_str(), nullptr);

        return SDL_APP_FAILURE;
    }

    g_window = SDL_CreateWindow("phys", 1280, 800, SDL_WINDOW_HIDDEN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (g_window == nullptr)
    {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "phys", std::format("Error `SDL_CreateWindow()`: {}", SDL_GetError()).c_str(), nullptr);

        return SDL_APP_FAILURE;
    }

    // Silly fix for window dragging stutter.
#if defined(_WIN32)
    auto *hwnd = (HWND)SDL_GetPointerProperty(SDL_GetWindowProperties(g_window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    if (hwnd != nullptr)
    {
        g_original_wndproc = (WNDPROC)SetWindowLongPtr(hwnd, GWLP_WNDPROC, (LONG_PTR)wndproc);
    }
#endif

    // Default FPS cap to monitor refresh rate. Falls back to 0 (off) if unavailable - vsync handles pacing.
    auto display = SDL_GetDisplayForWindow(g_window);
    if (display != 0)
    {
        if (auto *mode = SDL_GetCurrentDisplayMode(display); mode != nullptr && mode->refresh_rate > 0.0f)
        {
            g_fps_cap = std::clamp((int)std::round(mode->refresh_rate), 10, 1000);
        }
    }

    g_dpi_scaling = SDL_GetWindowDisplayScale(g_window);
    if (g_dpi_scaling == 0.0f)
    {
        SDL_ShowSimpleMessageBox(
            SDL_MESSAGEBOX_ERROR, "phys", std::format("Error `SDL_GetWindowDisplayScale()`: {}", SDL_GetError()).c_str(), nullptr);

        return SDL_APP_FAILURE;
    }

    SDL_SetHint(SDL_HINT_RENDER_DRIVER, "direct3d12,direct3d11,direct3d,metal,vulkan,opengl,software");

    g_renderer = SDL_CreateRenderer(g_window, nullptr);
    if (g_renderer == nullptr)
    {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "phys", std::format("Error `SDL_CreateRenderer()`: {}", SDL_GetError()).c_str(), nullptr);

        return SDL_APP_FAILURE;
    }

    // Adaptive vsync preferred. Falls back to regular vsync if unsupported.
    if (!SDL_SetRenderVSync(g_renderer, g_vsync) && g_vsync == SDL_RENDERER_VSYNC_ADAPTIVE)
    {
        g_vsync = 1;
        SDL_SetRenderVSync(g_renderer, g_vsync);
    }

    g_renderer_name = SDL_GetRendererName(g_renderer);
    if (g_renderer_name == nullptr)
    {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "phys", std::format("Error `SDL_GetRendererName()`: {}", SDL_GetError()).c_str(), nullptr);

        return SDL_APP_FAILURE;
    }

    ImGui::CreateContext();

    auto &io        = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.IniFilename  = nullptr; // TODO: This is probably nice to have.
    io.LogFilename  = nullptr;

    ImGui::StyleColorsDark();

    auto &style = ImGui::GetStyle();
    style.ScaleAllSizes(g_dpi_scaling);
    style.FontScaleDpi = g_dpi_scaling;

    ImGui_ImplSDL3_InitForSDLRenderer(g_window, g_renderer);
    ImGui_ImplSDLRenderer3_Init(g_renderer);

    // Detect performance cores and create thread pool pinned to them.
    auto topology = detect_performance_cores();

    g_total_core_count = topology.total_logical;

#if defined(_WIN32)
    g_perf_core_count = (std::uint32_t)topology.perf_cpu_set_ids.size();
#elif defined(__linux__)
    g_perf_core_count = (std::uint32_t)topology.perf_cpu_indices.size();
#else
    g_perf_core_count = topology.total_logical;
#endif

    auto num_threads = std::max(1u, g_perf_core_count > 0 ? g_perf_core_count : g_total_core_count);

    g_worker_count = num_threads <= 2 ? num_threads : num_threads - 1;

#if defined(_WIN32)
    g_thread_pool =
        std::make_unique<dp::thread_pool<>>(g_worker_count, [ids = topology.perf_cpu_set_ids](std::size_t) { bind_thread_to_perf_cores(ids); });
#elif defined(__linux__)
    g_thread_pool = std::make_unique<dp::thread_pool<>>(
        g_worker_count, [indices = topology.perf_cpu_indices](std::size_t) { bind_thread_to_perf_cores(indices); });
#else
    g_thread_pool = std::make_unique<dp::thread_pool<>>(g_worker_count);
#endif

    // Pin the main (render) thread to P-cores as well.
#if defined(_WIN32)
    bind_thread_to_perf_cores(topology.perf_cpu_set_ids);
#elif defined(__linux__)
    bind_thread_to_perf_cores(topology.perf_cpu_indices);
#endif

    // Create world.
    auto world_def            = b2DefaultWorldDef();
    world_def.workerCount     = (int)g_worker_count;
    world_def.enqueueTask     = box2d_enqueue_task;
    world_def.finishTask      = box2d_finish_task;
    world_def.userTaskContext = g_thread_pool.get();

    // Raise overlap recovery speed cap (default 3 m/s). Reduces visible sinking under stacks.
    world_def.maxContactPushSpeed = 6.0f * b2GetLengthUnitsPerMeter();

    g_world = b2CreateWorld(&world_def);

    // Ground: Static body with a wide thin box.
    {
        auto body_def     = b2DefaultBodyDef();
        body_def.position = {0.0f, -1.0f};

        g_ground_id = b2CreateBody(g_world, &body_def);

        auto shape_def  = b2DefaultShapeDef();
        auto ground_box = b2MakeBox(80.0f, 0.5f); // 160m wide, 1m tall.
        b2CreatePolygonShape(g_ground_id, &shape_def, &ground_box);
    }

    g_bodies.reserve(4096);

    // Spawn initial boxes stacked vertically, zigzagging left/right every 10.
    for (std::int32_t i{}; i < 50; ++i)
    {
        auto group     = i / 10;
        auto direction = group % 2 == 0 ? 1.0f : -1.0f;
        auto x         = direction * 2.0f;
        auto y         = 4.0f + (float)i * 1.1f;
        add_box({x, y}, 0.5f, 0.5f);
    }

    // Center and show window.
    SDL_SetWindowPosition(g_window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_SyncWindow(g_window);
    SDL_ShowWindow(g_window);

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate([[maybe_unused]] void *appstate)
{
    if ((SDL_GetWindowFlags(g_window) & SDL_WINDOW_MINIMIZED) != 0)
    {
        SDL_Delay(50);

        return SDL_APP_CONTINUE;
    }

    g_frame_start_ns = SDL_GetTicksNS();

    SDL_GetWindowSize(g_window, &g_window_w, &g_window_h);

    // Clamp camera so the viewport never extends past the world boundary.
    auto area_w       = AREA_MAX_X - AREA_MIN_X;
    auto area_h       = AREA_MAX_Y - AREA_MIN_Y;
    g_camera_zoom_min = std::max((float)g_window_w / area_w, (float)g_window_h / area_h);
    g_camera_zoom     = std::clamp(g_camera_zoom, g_camera_zoom_min, ZOOM_MAX);

    auto half_view_width  = (float)g_window_w / 2.0f / g_camera_zoom;
    auto half_view_height = (float)g_window_h / 2.0f / g_camera_zoom;
    auto center_x         = (AREA_MIN_X + AREA_MAX_X) / 2.0f;
    auto center_y         = (AREA_MIN_Y + AREA_MAX_Y) / 2.0f;
    auto clamp_x          = area_w / 2.0f - half_view_width;
    auto clamp_y          = area_h / 2.0f - half_view_height;
    g_camera_center.x     = std::clamp(g_camera_center.x, center_x - clamp_x, center_x + clamp_x);
    g_camera_center.y     = std::clamp(g_camera_center.y, center_y - clamp_y, center_y + clamp_y);

    auto &io = ImGui::GetIO();

    auto timer_frequency = (double)SDL_GetPerformanceFrequency();
    auto physics_timer   = SDL_GetPerformanceCounter();

    step_physics(io.DeltaTime);

    // Exponential moving average (smoothing factor 0.01 = ~100 frame window).
    auto physics_sample = (float)((double)(SDL_GetPerformanceCounter() - physics_timer) / timer_frequency * 1000.0);
    g_physics_ms        = g_physics_ms + 0.01f * (physics_sample - g_physics_ms);

    // Kill bounds: World area + margin so bodies don't pop out of existence at the edge.
    constexpr auto KILL_MARGIN = 5.0f;

    // Destroy bodies that fall outside the playground.
    std::erase_if(
        g_bodies,
        [](const PhysBody &body)
        {
            auto position = b2Body_GetPosition(body.body);
            if (position.x < AREA_MIN_X - KILL_MARGIN
                || position.x > AREA_MAX_X + KILL_MARGIN
                || position.y < AREA_MIN_Y - KILL_MARGIN
                || position.y > AREA_MAX_Y + KILL_MARGIN)
            {
                // Release mouse joint if we're about to destroy the dragged body.
                if (B2_IS_NON_NULL(g_mouse_joint) && body.body.index1 == g_mouse_body.index1)
                {
                    b2DestroyJoint(g_mouse_joint);

                    g_mouse_joint = b2_nullJointId;
                    g_mouse_body  = b2_nullBodyId;
                }

                b2DestroyBody(body.body);

                return true;
            }

            return false;
        });

    if (!g_paused)
    {
        tick_emitters(io.DeltaTime);
        tick_force_zones();
    }

    // Start the Dear ImGui frame.
    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui_ImplSDL3_NewFrame();

    auto render_timer = SDL_GetPerformanceCounter();

    ImGui::NewFrame();
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

    auto *foreground = ImGui::GetForegroundDrawList();
    auto *background = ImGui::GetBackgroundDrawList();
    auto  mouse_pos  = ImGui::GetMousePos();

    // Toolbox.
    if (ImGui::Begin("phys"))
    {
        ImGui::SeparatorText("Spawn");

        if (ImGui::Button("Box"))
        {
            add_box(g_camera_center, 0.5f, 0.5f);
        }

        if (ImGui::BeginDragDropSource())
        {
            auto type = SpawnShape::Box;
            ImGui::SetDragDropPayload("SPAWN", &type, sizeof(type));
            ImGui::TextUnformatted("Box");
            ImGui::EndDragDropSource();
        }

        if (ImGui::Button("Circle"))
        {
            add_circle(g_camera_center, 0.5f);
        }

        if (ImGui::BeginDragDropSource())
        {
            auto type = SpawnShape::Circle;
            ImGui::SetDragDropPayload("SPAWN", &type, sizeof(type));
            ImGui::TextUnformatted("Circle");
            ImGui::EndDragDropSource();
        }

        if (ImGui::Button("Triangle"))
        {
            add_triangle(g_camera_center, 1.0f);
        }

        if (ImGui::BeginDragDropSource())
        {
            auto type = SpawnShape::Triangle;
            ImGui::SetDragDropPayload("SPAWN", &type, sizeof(type));
            ImGui::TextUnformatted("Triangle");
            ImGui::EndDragDropSource();
        }

        ImGui::SeparatorText("Selection");

        int selected_count{};
        for (auto &&body : g_bodies)
        {
            if (body.selected)
            {
                ++selected_count;
            }
        }

        ImGui::Text("%d selected", selected_count);

        ImGui::BeginDisabled(selected_count == 0);

        if (ImGui::Button("Delete Selected"))
        {
            delete_selected();
        }

        ImGui::EndDisabled();

        if (ImGui::CollapsingHeader("Emitters"))
        {
            if (ImGui::Button("Add Emitter"))
            {
                g_emitters.emplace_back(g_camera_center);
            }

            for (std::size_t i{}; i < g_emitters.size(); ++i)
            {
                ImGui::PushID((int)i);

                auto &emitter = g_emitters[i];

                ImGui::Checkbox("##active", &emitter.active);
                ImGui::SameLine();
                ImGui::Text("Emitter (e%zu)", i);

                auto angle_deg = emitter.angle * RAD_TO_DEG;
                slider("Angle", angle_deg, -180.0f, 180.0f, 5.0f, "%.0f deg");

                emitter.angle = angle_deg * DEG_TO_RAD;

                slider("Speed", emitter.speed, 1.0f, 50.0f, 1.0f, "%.1f m/s");
                slider("Rate", emitter.rate, 0.5f, 20.0f, 0.5f, "%.1f /s");

                int shape_idx = (int)emitter.shape;
                combo("Shape", shape_idx, "Box\0Circle\0Triangle\0", (int)SpawnShape::Count);

                emitter.shape = (SpawnShape)shape_idx;

                if (ImGui::Button("Remove"))
                {
                    g_emitters.erase(g_emitters.begin() + (std::ptrdiff_t)i);

                    --i;

                    ImGui::PopID();

                    continue;
                }

                ImGui::Separator();
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("Force Zones"))
        {
            if (ImGui::Button("Add Zone"))
            {
                auto &zone = g_force_zones.emplace_back(g_camera_center);
                apply_zone_preset(zone, 0); // Default preset (Vortex).
            }

            // Build null-separated preset combo string once.
            static std::string preset_combo_items = []
            {
                std::string items{};
                for (std::size_t j{}; j < ZONE_PRESETS.size(); ++j)
                {
                    items += ZONE_PRESETS[j].name;
                    items += '\0';
                }

                items += "Custom";
                items += '\0';

                return items;
            }();

            for (std::size_t i{}; i < g_force_zones.size(); ++i)
            {
                ImGui::PushID((int)i);

                auto &zone = g_force_zones[i];

                ImGui::Checkbox("##active", &zone.active);
                ImGui::SameLine();
                ImGui::Text("Zone (z%zu)", i);

                int shape_idx = (int)zone.shape;
                combo("Shape", shape_idx, "Rectangle\0Circle\0", (int)ZoneShape::Count);

                zone.shape = (ZoneShape)shape_idx;

                // Preset dropdown: selecting a preset fills formula fields.
                int previous_preset = zone.preset;
                combo("Preset", zone.preset, preset_combo_items.c_str(), ZONE_PRESET_COUNT + 1);

                if (zone.preset != previous_preset && zone.preset < ZONE_PRESET_COUNT)
                {
                    apply_zone_preset(zone, zone.preset);
                }

                // Formula text inputs - editing switches preset to Custom.
                auto fx_changed = ImGui::InputText("Fx", &zone.formula_x, ImGuiInputTextFlags_EnterReturnsTrue);
                auto fy_changed = ImGui::InputText("Fy", &zone.formula_y, ImGuiInputTextFlags_EnterReturnsTrue);

                ImGui::TextDisabled("Variables: x, y, r, angle");

                if (fx_changed || fy_changed)
                {
                    compile_formulas(zone);

                    // Match against known presets; Fall back to Custom.
                    zone.preset = ZONE_PRESET_CUSTOM;
                    for (std::size_t j{}; j < ZONE_PRESETS.size(); ++j)
                    {
                        if (zone.formula_x == ZONE_PRESETS[j].formula_x && zone.formula_y == ZONE_PRESETS[j].formula_y)
                        {
                            zone.preset = (int)j;

                            break;
                        }
                    }
                }
                if (!zone.formula_valid)
                {
                    ImGui::TextColored({1.0f, 0.3f, 0.3f, 1.0f}, "%s", zone.formula_error.c_str());
                }

                auto angle_deg = zone.angle * RAD_TO_DEG;
                slider("Angle", angle_deg, -180.0f, 180.0f, 5.0f, "%.0f deg");

                zone.angle = angle_deg * DEG_TO_RAD;

                if (zone.shape == ZoneShape::Rectangle)
                {
                    slider("Width", zone.half_size.x, 0.5f, 20.0f, 0.5f, "%.1f m");
                    slider("Height", zone.half_size.y, 0.5f, 20.0f, 0.5f, "%.1f m");
                }
                else
                {
                    slider("Radius", zone.radius, 0.5f, 30.0f, 0.5f, "%.1f m");
                }

                slider("Strength", zone.strength, -500.0f, 500.0f, 5.0f, "%.1f m/s^2");
                slider("Max Speed", zone.max_speed, 0.0f, 200.0f, 5.0f, "%.0f m/s");
                slider("Arrows", zone.grid_resolution, 2, 15, 1);
                slider("X", zone.position.x, -100.0f, 100.0f, 0.5f, "%.1f m");
                slider("Y", zone.position.y, -40.0f, 120.0f, 0.5f, "%.1f m");

                if (ImGui::Button("Remove"))
                {
                    g_force_zones.erase(g_force_zones.begin() + (std::ptrdiff_t)i);

                    --i;

                    ImGui::PopID();

                    continue;
                }

                ImGui::Separator();
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("World"))
        {
            auto gravity = b2World_GetGravity(g_world);
            slider("Gravity", gravity.y, -50.0f, 50.0f, 1.0f, "%.1f m/s^2");

            b2World_SetGravity(g_world, gravity);

            auto sleeping = b2World_IsSleepingEnabled(g_world);
            if (ImGui::Checkbox("Sleeping", &sleeping))
            {
                b2World_EnableSleeping(g_world, sleeping);
            }

            ImGui::Checkbox("Paused", &g_paused);
            ImGui::BeginDisabled(!g_paused);
            ImGui::SameLine();

            if (ImGui::Button("Step"))
            {
                g_single_step = true;
            }

            slider("Step count", g_step_count, 1, 100, 1);

            ImGui::EndDisabled();

            slider("Sub-steps", g_sub_steps, 1, 8, 1);

            // Update all existing dynamic bodies when drag settings change.
            auto drag_changed   = slider("Drag", g_linear_damping, 0.0f, 5.0f, 0.1f, "%.2f");
            auto spread_changed = slider("Drag Spread", g_random_damping, 0.0f, 0.5f, 0.01f, "%.2f");
            if (drag_changed || spread_changed)
            {
                for (auto &&body : g_bodies)
                {
                    if (b2Body_GetType(body.body) == b2_dynamicBody)
                    {
                        b2Body_SetLinearDamping(body.body, g_linear_damping + body.damping_offset);
                    }
                }
            }
        }

        if (ImGui::CollapsingHeader("Info", ImGuiTreeNodeFlags_DefaultOpen))
        {
            // `+ 0.0f` canonicalizes IEEE 754 negative zero to positive zero (`-0.0 + 0.0 = +0.0`, §6.3).
            ImGui::Text("Camera: (%.1f, %.1f) zoom %.1f", g_camera_center.x + 0.0f, g_camera_center.y + 0.0f, g_camera_zoom);

            auto counters = b2World_GetCounters(g_world);
            ImGui::Text("Bodies: %zu (%d awake, %zu culled)", g_bodies.size(), b2World_GetAwakeBodyCount(g_world), g_culled_count);
            ImGui::Text("Workers: %u (P-cores: %u / %u)", g_worker_count, g_perf_core_count, g_total_core_count);
            ImGui::Text("Islands: %d", counters.islandCount);
            ImGui::Text("Contacts: %d", counters.contactCount);
            ImGui::Text("Ropes: %zu", g_ropes.size());

            ImGui::Text("Physics: %.2f ms", g_physics_ms);
            ImGui::Text("Render:  %.2f ms", g_render_ms);
            ImGui::Text("Present: %.2f ms", g_present_ms);
            ImGui::Text("Frame:   %.2f ms", g_frame_ms);

            ImGui::Text("Vertices: %d | Indices: %d", g_total_vertices, g_total_indices);
        }

        if (ImGui::CollapsingHeader("Graphics"))
        {
            // VSync mode: Maps combo index to SDL vsync values.
            constexpr int VSYNC_VALUES[] = {SDL_RENDERER_VSYNC_DISABLED, 1, SDL_RENDERER_VSYNC_ADAPTIVE};

            int vsync_index = g_vsync == SDL_RENDERER_VSYNC_ADAPTIVE ? 2 : g_vsync;
            combo("VSync", vsync_index, "Off\0On\0Adaptive\0", 3);

            g_vsync = VSYNC_VALUES[vsync_index];
            if (!SDL_SetRenderVSync(g_renderer, g_vsync) && g_vsync == SDL_RENDERER_VSYNC_ADAPTIVE)
            {
                g_vsync     = 1;
                vsync_index = 1;
                SDL_SetRenderVSync(g_renderer, g_vsync);
            }

            slider("FPS Cap", g_fps_cap, 0, 1000, 10, g_fps_cap == 0 ? "Off" : "%d");

            if (g_fps_cap > 0)
            {
                g_fps_cap = std::clamp(g_fps_cap, 10, 1000);
            }
        }

        if (ImGui::CollapsingHeader("Controls"))
        {
            auto hint = [](const char *key, const char *action) noexcept
            {
                ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "%s", key);
                ImGui::SameLine();
                ImGui::TextUnformatted(action);
            };

            hint("Left-click drag", "Move body");
            hint("Right-click while dragging", "Pin body at position");
            hint("Ctrl+Right-click on pin", "Unpin body");
            hint("Ctrl+Left drag", "Draw line");
            hint("Ctrl+Right drag", "Erase lines / ropes");
            hint("Shift+Left-click", "Link rope (2 bodies)");
            hint("Shift+Right drag", "Cut ropes");
            hint("Right-click drag", "Box select");
            hint("Delete / Backspace", "Delete selected");
            hint("Escape", "Clear selection / cancel");
            hint("Middle-click drag", "Pan camera");
            hint("Scroll wheel", "Zoom to cursor");
            hint("Drag spawn button", "Place shape at cursor");
        }
    }

    ImGui::End();

    // Handle drag-drop onto viewport.
    if (auto *payload = ImGui::GetDragDropPayload(); payload != nullptr && payload->IsDataType("SPAWN"))
    {
        // Ghost preview.
        auto world_point = screen_to_world(mouse_pos.x, mouse_pos.y);
        auto type        = *(SpawnShape *)payload->Data;
        switch (type)
        {
        case SpawnShape::Box:
        {
            ImVec2 half{g_camera_zoom / 2.0f, g_camera_zoom / 2.0f};
            foreground->AddRectFilled(mouse_pos - half, mouse_pos + half, IM_COL32(51, 153, 230, 80));
            foreground->AddRect(mouse_pos - half, mouse_pos + half, IM_COL32(255, 255, 255, 100));

            break;
        }

        case SpawnShape::Circle:
        {
            auto screen_radius = g_camera_zoom / 2.0f;
            foreground->AddCircleFilled(mouse_pos, screen_radius, IM_COL32(230, 153, 51, 80));
            foreground->AddCircle(mouse_pos, screen_radius, IM_COL32(255, 255, 255, 100));

            break;
        }

        case SpawnShape::Triangle:
        {
            auto   half_extent = g_camera_zoom / 2.0f;
            ImVec2 triangle_vertices[3]{
                {mouse_pos.x, mouse_pos.y - half_extent},
                {mouse_pos.x - half_extent, mouse_pos.y + half_extent},
                {mouse_pos.x + half_extent, mouse_pos.y + half_extent},
            };
            foreground->AddTriangleFilled(triangle_vertices[0], triangle_vertices[1], triangle_vertices[2], IM_COL32(50, 200, 50, 80));
            foreground->AddTriangle(triangle_vertices[0], triangle_vertices[1], triangle_vertices[2], IM_COL32(255, 255, 255, 100));

            break;
        }

        default:
        {
            break;
        }
        }

        // Drop on release outside ImGui windows.
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow))
        {
            switch (type)
            {
            case SpawnShape::Box:
            {
                add_box(world_point, 0.5f, 0.5f);

                break;
            }

            case SpawnShape::Circle:
            {
                add_circle(world_point, 0.5f);

                break;
            }

            case SpawnShape::Triangle:
            {
                add_triangle(world_point, 1.0f);

                break;
            }

            default:
            {
                break;
            }
            }
        }
    }

    foreground->AddText({4, 4}, IM_COL32_WHITE, g_renderer_name);

    ++g_frame_count;

    auto now_ns = SDL_GetTicksNS();
    if (now_ns - g_fps_timer_ns >= SDL_NS_PER_SECOND)
    {
        g_display_fps  = g_frame_count;
        g_frame_count  = 0;
        g_fps_timer_ns = now_ns;
    }

    char fps_buffer[32];
    std::snprintf(fps_buffer, sizeof(fps_buffer), "%d FPS", g_display_fps);
    foreground->AddText({4, 20}, IM_COL32_WHITE, fps_buffer);

    // Box select preview.
    if (g_box_selecting)
    {
        foreground->AddRectFilled(g_box_select_start, mouse_pos, IM_COL32(255, 80, 80, 30));
        foreground->AddRect(g_box_select_start, mouse_pos, IM_COL32(255, 80, 80, 150), 0.0f, 0, 1.5f);
    }

    // Rope cut preview: Red line from drag start to cursor.
    if (g_cutting || g_just_cut)
    {
        foreground->AddLine(g_cut_start, mouse_pos, IM_COL32(255, 60, 60, 200), 2.0f);
    }

    // Rope erase preview: Red circle at cursor, matching the hit-test radius.
    if (g_erasing)
    {
        auto erase_screen_radius = HIT_RADIUS * g_camera_zoom;
        foreground->AddCircle(mouse_pos, erase_screen_radius, IM_COL32(255, 60, 60, 200), 0, 2.0f);
    }

    // Render physics bodies to background draw list.
    {
        // Disable fill anti-aliasing for body rendering - saves ~2x vertices per filled shape.
        // Outlines keep AA via `ImDrawListFlags_AntiAliasedLines` (unchanged).
        auto old_flags     = background->Flags;
        background->Flags &= ~ImDrawListFlags_AntiAliasedFill;

        // Camera transform - snap to pixel grid to avoid subpixel blurriness.
        auto camera_x = std::round((float)g_window_w / 2.0f - g_camera_center.x * g_camera_zoom);
        auto camera_y = std::round((float)g_window_h / 2.0f + g_camera_center.y * g_camera_zoom);

        auto to_screen = [camera_x, camera_y](b2Vec2 point) noexcept -> ImVec2
        { return {camera_x + point.x * g_camera_zoom, camera_y - point.y * g_camera_zoom}; };

        // Render a polygon shape with its radius expansion.
        auto render_polygon =
            [to_screen, background](b2Vec2 position, b2Rot rotation, const b2Polygon &polygon, ImU32 color, bool show_edge = true) noexcept
        {
            constexpr auto BORDER_PX = 1.5f;

            ImVec2 screen[B2_MAX_POLYGON_VERTICES]{};
            ImVec2 inset[B2_MAX_POLYGON_VERTICES]{};
            auto   border_world = BORDER_PX / g_camera_zoom;
            for (std::int32_t i{}; i < polygon.count; ++i)
            {
                // Corner bisector from Box2D's precomputed unit normals.
                auto previous_index = (i + polygon.count - 1) % polygon.count;
                auto bisect         = b2Normalize(polygon.normals[previous_index] + polygon.normals[i]);
                auto bisect_dot     = b2Dot(bisect, polygon.normals[i]);

                // Shrink by linear slop so shapes appear flush when the solver allows slight overlap - see `B2_LINEAR_SLOP`.
                auto radius       = std::max(0.0f, polygon.radius - 0.005f * b2GetLengthUnitsPerMeter());
                auto outer_offset = bisect_dot > 0.0f ? radius / bisect_dot : radius;
                auto outer_local  = polygon.vertices[i] + bisect * outer_offset;

                screen[i] = to_screen(b2RotateVector(rotation, outer_local) + position);

                // Inset vertex: Same bisector, reduced offset. Gives uniform border_world inset per edge.
                if (show_edge)
                {
                    auto inner_radius = radius - border_world;
                    auto inner_offset = bisect_dot > 0.0f ? inner_radius / bisect_dot : inner_radius;
                    auto inner_local  = polygon.vertices[i] + bisect * inner_offset;

                    inset[i] = to_screen(b2RotateVector(rotation, inner_local) + position);
                }
            }

            auto draw_filled = [background](ImVec2 *vertices, std::int32_t count, ImU32 fill_color) noexcept
            {
                if (count == 4)
                {
                    background->AddQuadFilled(vertices[0], vertices[1], vertices[2], vertices[3], fill_color);
                }
                else if (count >= 3)
                {
                    for (std::int32_t i = 1; i < count - 1; ++i)
                    {
                        background->AddTriangleFilled(vertices[0], vertices[i], vertices[i + 1], fill_color);
                    }
                }
            };

            if (show_edge)
            {
                // Edge rim: Outer fill with AA for smooth silhouette, inner fill without.
                auto edge_color = brighten(color, 60);

                background->Flags |= ImDrawListFlags_AntiAliasedFill;

                draw_filled(screen, polygon.count, edge_color);

                background->Flags &= ~ImDrawListFlags_AntiAliasedFill;

                draw_filled(inset, polygon.count, color);
            }
            else
            {
                draw_filled(screen, polygon.count, color);
            }
        };

        // Helper: Get the single polygon shape from a body.
        auto get_polygon = [](b2BodyId body) noexcept
        {
            b2ShapeId shape;
            b2Body_GetShapes(body, &shape, 1);

            return b2Shape_GetPolygon(shape);
        };

        // Ground (static, no interpolation).
        render_polygon(b2Body_GetPosition(g_ground_id), b2Body_GetRotation(g_ground_id), get_polygon(g_ground_id), IM_COL32(102, 102, 102, 255));

        // Compute live box-select AABB for hover highlighting.
        std::optional<b2AABB> select_aabb{};
        if (g_box_selecting)
        {
            auto world_corner_min = screen_to_world(g_box_select_start.x, g_box_select_start.y);
            auto world_corner_max = screen_to_world(mouse_pos.x, mouse_pos.y);
            select_aabb           = b2AABB{b2Min(world_corner_min, world_corner_max), b2Max(world_corner_min, world_corner_max)};
        }

        // Viewport culling AABB in world space.
        // Margin accounts for shape radius so partially-visible bodies aren't culled.
        constexpr auto CULL_MARGIN_WORLD = 2.0f; // Meters (covers shapes up to 4m diameter).

        auto view_min  = screen_to_world(0.0f, (float)g_window_h);
        auto view_max  = screen_to_world((float)g_window_w, 0.0f);
        view_min.x    -= CULL_MARGIN_WORLD;
        view_min.y    -= CULL_MARGIN_WORLD;
        view_max.x    += CULL_MARGIN_WORLD;
        view_max.y    += CULL_MARGIN_WORLD;

        // Dynamic bodies (interpolated).
        auto alpha = g_physics_alpha;

        g_culled_count = 0;

        auto outline_enabled = g_camera_zoom >= 15.0f;

        for (auto &&body : g_bodies)
        {
            // Interpolate awake bodies; sleeping bodies use current transform directly.
            b2Vec2 position;
            b2Rot  rotation;
            if (b2Body_IsAwake(body.body))
            {
                auto transform = b2Body_GetTransform(body.body);
                position       = b2Lerp(body.previous.position, transform.p, alpha);
                rotation       = b2NLerp(body.previous.rotation, transform.q, alpha);
            }
            // Sleeping: Current transform is stable, no interpolation needed.
            else
            {
                auto transform = b2Body_GetTransform(body.body);
                position       = transform.p;
                rotation       = transform.q;
            }

            // Frustum cull.
            if (position.x < view_min.x || position.x > view_max.x || position.y < view_min.y || position.y > view_max.y)
            {
                ++g_culled_count;

                continue;
            }

            auto type = body.shape_type;

            // Query shape once (Used for both selection AABB and rendering).
            b2Polygon polygon{};
            b2Circle  circle{};
            if (type == b2_polygonShape)
            {
                polygon = b2Shape_GetPolygon(body.shape);
            }
            else if (type == b2_circleShape)
            {
                circle = b2Shape_GetCircle(body.shape);
            }

            // Compute selection state from shape AABB vs selection AABB.
            auto is_selected = body.selected;
            if (!is_selected && select_aabb)
            {
                b2AABB body_aabb{};
                if (type == b2_polygonShape)
                {
                    auto min_x = position.x, min_y = position.y, max_x = position.x, max_y = position.y;
                    for (std::int32_t j{}; j < polygon.count; ++j)
                    {
                        auto world = b2RotateVector(rotation, polygon.vertices[j]) + position;
                        min_x      = std::min(min_x, world.x);
                        min_y      = std::min(min_y, world.y);
                        max_x      = std::max(max_x, world.x);
                        max_y      = std::max(max_y, world.y);
                    }

                    body_aabb = {{min_x - polygon.radius, min_y - polygon.radius}, {max_x + polygon.radius, max_y + polygon.radius}};
                }
                else if (type == b2_circleShape)
                {
                    auto center = b2RotateVector(rotation, circle.center) + position;
                    body_aabb   = {{center.x - circle.radius, center.y - circle.radius}, {center.x + circle.radius, center.y + circle.radius}};
                }

                is_selected = body_aabb.lowerBound.x <= select_aabb->upperBound.x
                           && body_aabb.upperBound.x >= select_aabb->lowerBound.x
                           && body_aabb.lowerBound.y <= select_aabb->upperBound.y
                           && body_aabb.upperBound.y >= select_aabb->lowerBound.y;
            }

            auto fill_color = is_selected ? IM_COL32(230, 50, 50, 255) : ImU32{};
            if (type == b2_polygonShape)
            {
                if (fill_color == 0)
                {
                    fill_color = polygon.count == 3 ? IM_COL32(50, 200, 50, 255) : IM_COL32(51, 153, 230, 255);
                }

                render_polygon(position, rotation, polygon, fill_color, outline_enabled);
            }
            else if (type == b2_circleShape)
            {
                if (fill_color == 0)
                {
                    fill_color = IM_COL32(230, 153, 51, 255);
                }

                auto world_center  = b2RotateVector(rotation, circle.center) + position;
                auto screen_center = to_screen(world_center);
                auto screen_radius = circle.radius * g_camera_zoom;
                if (outline_enabled)
                {
                    auto edge_color = brighten(fill_color, 60);
                    background->AddCircleFilled(screen_center, screen_radius, edge_color);
                    background->AddCircleFilled(screen_center, screen_radius - 1.5f, fill_color);

                    auto   inset_radius = screen_radius - 1.5f;
                    ImVec2 edge{screen_center.x + rotation.c * inset_radius, screen_center.y - rotation.s * inset_radius};
                    background->AddLine(screen_center, edge, edge_color, std::max(2.0f, 3.0f * g_camera_zoom / ZOOM_DEFAULT));
                }
                else
                {
                    background->AddCircleFilled(screen_center, screen_radius, fill_color);
                }
            }
        }

        // Drawn lines (on top of bodies).
        auto render_smooth_line = [to_screen, background](const std::vector<b2Vec2> &points, ImU32 color, float thickness) noexcept
        {
            if (points.size() < 2)
            {
                return;
            }

            if (points.size() == 2)
            {
                background->AddLine(to_screen(points[0]), to_screen(points[1]), color, thickness);

                return;
            }

            background->PathLineTo(to_screen(points[0]));

            for (std::size_t i{}; i < points.size() - 1; ++i)
            {
                auto previous = points[i > 0 ? i - 1 : 0];
                auto current  = points[i];
                auto next     = points[i + 1];
                auto after    = points[i + 1 < points.size() - 1 ? i + 2 : points.size() - 1];

                // Catmull-Rom tangents, clamped to prevent overshoot at endpoints and sharp angles.
                auto segment_direction = next - current;
                auto segment_length    = b2Length(segment_direction);

                // Start tangent.
                auto tangent_start        = (next - previous) * (1.0f / 6.0f);
                auto tangent_start_length = b2Length(tangent_start);
                auto previous_length      = i > 0 ? b2Length(current - previous) : segment_length;
                auto max_tangent_start    = std::min(segment_length, previous_length) * 0.5f;
                if (tangent_start_length > max_tangent_start)
                {
                    tangent_start = tangent_start * (max_tangent_start / tangent_start_length);
                }

                // Kill tangent if it opposes the segment direction (>90 degree bend).
                if (segment_length > 1e-6f && b2Dot(tangent_start, segment_direction) < 0.0f)
                {
                    tangent_start = b2Vec2_zero;
                }

                // End tangent.
                auto tangent_end        = (after - current) * (1.0f / 6.0f);
                auto tangent_end_length = b2Length(tangent_end);
                auto next_length        = i + 1 < points.size() - 1 ? b2Length(after - next) : segment_length;
                auto max_tangent_end    = std::min(segment_length, next_length) * 0.5f;
                if (tangent_end_length > max_tangent_end)
                {
                    tangent_end = tangent_end * (max_tangent_end / tangent_end_length);
                }

                // Kill tangent if it opposes the segment direction.
                if (segment_length > 1e-6f && b2Dot(tangent_end, segment_direction) < 0.0f)
                {
                    tangent_end = b2Vec2_zero;
                }

                auto control_start = to_screen(current + tangent_start);
                auto control_end   = to_screen(next - tangent_end);
                auto end           = to_screen(next);
                background->PathBezierCubicCurveTo(control_start, control_end, end);
            }

            background->PathStroke(color, 0, thickness);
        };

        auto line_thickness = std::max(2.0f, 3.0f * g_camera_zoom / ZOOM_DEFAULT);
        for (auto &&line : g_drawn_lines)
        {
            render_smooth_line(line.points, IM_COL32(200, 200, 200, 255), line_thickness);
        }

        if (!g_current_stroke.empty())
        {
            render_smooth_line(g_current_stroke, IM_COL32(100, 255, 100, 200), line_thickness);
        }

        // Ropes.
        {
            constexpr auto ROPE_COLOR   = IM_COL32(194, 154, 108, 255);
            constexpr auto ROPE_OUTLINE = IM_COL32(120, 90, 60, 255);

            auto rope_thickness = std::max(2.0f, 4.0f * g_camera_zoom / ZOOM_DEFAULT);

            // Prune ropes whose anchor body was destroyed.
            std::erase_if(
                g_ropes,
                [](const Rope &rope)
                {
                    auto start_dead = B2_IS_NON_NULL(rope.body_a) && !b2Body_IsValid(rope.body_a);
                    auto end_dead   = B2_IS_NON_NULL(rope.body_b) && !b2Body_IsValid(rope.body_b);
                    if (!start_dead && !end_dead)
                    {
                        return false;
                    }

                    for (auto &&joint : rope.joints)
                    {
                        if (b2Joint_IsValid(joint))
                        {
                            b2DestroyJoint(joint);
                        }
                    }

                    for (auto &&segment : rope.segments)
                    {
                        if (b2Body_IsValid(segment))
                        {
                            b2DestroyBody(segment);
                        }
                    }

                    return true;
                });

            for (auto &&rope : g_ropes)
            {
                // Build point list: Anchor/tip A, segment centers, anchor/tip B.
                // At dangling ends (cut rope), use the capsule tip instead of an anchor so the visual rope covers the full collision extent.
                // `SEGMENT_HALF_LENGTH` (0.09) + `SEGMENT_RADIUS` (0.06) = `SEGMENT_TIP_OFFSET` (0.15).
                std::vector<b2Vec2> points{};
                points.reserve(rope.segments.size() + 2);

                if (B2_IS_NON_NULL(rope.body_a))
                {
                    points.emplace_back(b2Body_GetWorldPoint(rope.body_a, rope.local_a));
                }
                else if (!rope.segments.empty())
                {
                    // Dangling start: Extend to the capsule tip.
                    auto rotation = b2Body_GetRotation(rope.segments.front());
                    points.emplace_back(b2Body_GetPosition(rope.segments.front()) + b2RotateVector(rotation, {0.0f, -SEGMENT_TIP_OFFSET}));
                }

                for (auto &&segment : rope.segments)
                {
                    points.emplace_back(b2Body_GetPosition(segment));
                }

                if (B2_IS_NON_NULL(rope.body_b))
                {
                    points.emplace_back(b2Body_GetWorldPoint(rope.body_b, rope.local_b));
                }
                else if (!rope.segments.empty())
                {
                    // Dangling end: Extend to the capsule tip.
                    auto rotation = b2Body_GetRotation(rope.segments.back());
                    points.emplace_back(b2Body_GetPosition(rope.segments.back()) + b2RotateVector(rotation, {0.0f, SEGMENT_TIP_OFFSET}));
                }

                if (points.size() < 2)
                {
                    continue;
                }

                // Draw outline then fill using the Catmull-Rom path renderer.
                render_smooth_line(points, ROPE_OUTLINE, rope_thickness + std::max(2.0f, 3.0f * std::sqrt(g_camera_zoom / ZOOM_DEFAULT)));
                render_smooth_line(points, ROPE_COLOR, rope_thickness);
            }

            // Pending rope: Line from start anchor to cursor.
            if (B2_IS_NON_NULL(g_rope_start_body))
            {
                auto world_start  = b2Body_GetWorldPoint(g_rope_start_body, g_rope_start_anchor);
                auto screen_start = to_screen(world_start);
                auto cursor       = ImGui::GetMousePos();
                background->AddLine(screen_start, cursor, IM_COL32(194, 154, 108, 128), rope_thickness);
            }
        }

        // Pins.
        {
            std::erase_if(g_pins, [](const Pin &pin) noexcept { return !b2Joint_IsValid(pin.joint); });

            auto zoom_ratio = g_camera_zoom / ZOOM_DEFAULT;
            auto radius     = std::clamp(8.0f * std::sqrt(zoom_ratio), 4.0f, 20.0f);
            auto slot_size  = radius * 0.55f;
            auto thickness  = std::clamp(1.5f * std::sqrt(zoom_ratio), 1.0f, 4.0f);
            for (auto &&pin : g_pins)
            {
                auto world_pos  = b2Body_GetWorldPoint(pin.body, b2Vec2_zero);
                auto screen_pos = to_screen(world_pos);

                // Head.
                background->AddCircleFilled(screen_pos, radius, IM_COL32(140, 140, 140, 255));
                background->AddCircle(screen_pos, radius, IM_COL32(90, 90, 90, 255), 0, thickness);

                // Phillips cross.
                background->AddLine(
                    {screen_pos.x - slot_size, screen_pos.y}, {screen_pos.x + slot_size, screen_pos.y}, IM_COL32(60, 60, 60, 200), thickness);
                background->AddLine(
                    {screen_pos.x, screen_pos.y - slot_size}, {screen_pos.x, screen_pos.y + slot_size}, IM_COL32(60, 60, 60, 200), thickness);
            }
        }

        // Force zones.
        for (std::size_t i{}; i < g_force_zones.size(); ++i)
        {
            auto &zone          = g_force_zones[i];
            auto  screen_center = to_screen(zone.position);
            auto  fill_color    = zone.active ? IM_COL32(80, 140, 255, 30) : IM_COL32(255, 80, 80, 20);
            auto  border_color  = zone.active ? IM_COL32(80, 140, 255, 120) : IM_COL32(255, 80, 80, 80);
            auto  arrow_color   = zone.active ? IM_COL32(80, 140, 255, 160) : IM_COL32(255, 80, 80, 100);
            auto  thickness     = std::max(1.0f, 1.5f * g_camera_zoom / ZOOM_DEFAULT);

            // Draw boundary.
            if (zone.shape == ZoneShape::Rectangle)
            {
                auto screen_min = to_screen(zone.position - zone.half_size);
                auto screen_max = to_screen(zone.position + zone.half_size);
                auto rect_min   = ImVec2{std::min(screen_min.x, screen_max.x), std::min(screen_min.y, screen_max.y)};
                auto rect_max   = ImVec2{std::max(screen_min.x, screen_max.x), std::max(screen_min.y, screen_max.y)};
                background->AddRectFilled(rect_min, rect_max, fill_color);
                background->AddRect(rect_min, rect_max, border_color, 0.0f, 0, thickness);
            }
            else
            {
                auto screen_radius = zone.radius * g_camera_zoom;
                background->AddCircleFilled(screen_center, screen_radius, fill_color, 48);
                background->AddCircle(screen_center, screen_radius, border_color, 48, thickness);
            }

            // Arrow grid: Sample field vectors on a grid and draw small arrows.
            constexpr float ARROW_SCALE = 0.3f;

            auto grid_cells     = zone.grid_resolution;
            auto extent_x       = zone.shape == ZoneShape::Rectangle ? zone.half_size.x : zone.radius;
            auto extent_y       = zone.shape == ZoneShape::Rectangle ? zone.half_size.y : zone.radius;
            auto cell_x         = extent_x * 2.0f / (float)grid_cells;
            auto cell_y         = extent_y * 2.0f / (float)grid_cells;
            auto arrow_max_len  = std::min(cell_x, cell_y) * ARROW_SCALE * g_camera_zoom;
            auto radius_squared = zone.radius * zone.radius;
            for (std::int32_t grid_y{}; grid_y < grid_cells; ++grid_y)
            {
                for (std::int32_t grid_x{}; grid_x < grid_cells; ++grid_x)
                {
                    // Sample at cell center.
                    auto   sample_x = zone.position.x - extent_x + cell_x * ((float)grid_x + 0.5f);
                    auto   sample_y = zone.position.y - extent_y + cell_y * ((float)grid_y + 0.5f);
                    b2Vec2 sample{sample_x, sample_y};

                    // For circular zones, skip samples whose arrow would extend past the boundary.
                    if (zone.shape == ZoneShape::Circle)
                    {
                        auto delta_cull       = sample - zone.position;
                        auto distance_squared = b2LengthSquared(delta_cull);
                        if (distance_squared > radius_squared)
                        {
                            continue;
                        }
                    }

                    // Evaluate formula to get field direction.
                    b2Vec2 field_direction{};
                    if (zone.formula_valid)
                    {
                        // Skip arrows near center where formulas using r are singular.
                        auto delta    = sample - zone.position;
                        auto distance = b2Length(delta);
                        if (distance < 1e-3f)
                        {
                            continue;
                        }

                        zone.bound_x        = (double)delta.x;
                        zone.bound_y        = (double)delta.y;
                        zone.bound_distance = (double)distance;
                        zone.bound_angle    = std::atan2(zone.bound_y, zone.bound_x);

                        auto fx         = (float)zone.parser_x.evaluate();
                        auto fy         = (float)zone.parser_y.evaluate();
                        field_direction = {fx, fy};

                        // Normalize for arrow direction.
                        auto length = b2Length(field_direction);
                        if (length > 1e-3f)
                        {
                            field_direction = field_direction * (1.0f / length);
                        }

                        // Rotate by zone angle.
                        field_direction = b2RotateVector(b2MakeRot(zone.angle), field_direction);

                        // Flip arrow if strength is negative.
                        if (zone.strength < 0.0f)
                        {
                            field_direction = field_direction * -1.0f;
                        }
                    }

                    // Skip near-zero arrows.
                    if (b2LengthSquared(field_direction) < 1e-3f)
                    {
                        continue;
                    }
                    // Draw arrow centered on sample: Tail behind, tip ahead.
                    auto   screen_sample = to_screen(sample);
                    ImVec2 screen_direction{field_direction.x * arrow_max_len, -field_direction.y * arrow_max_len};
                    auto   half_direction = screen_direction * 0.5f;
                    auto   arrow_tail     = screen_sample - half_direction;
                    auto   arrow_tip      = screen_sample + half_direction;
                    ImVec2 arrow_perpendicular{-screen_direction.y * 0.3f, screen_direction.x * 0.3f};
                    auto   head_base = screen_sample + half_direction * 0.1f;
                    background->AddLine(arrow_tail, head_base, arrow_color, thickness);
                    background->AddTriangleFilled(arrow_tip, head_base + arrow_perpendicular, head_base - arrow_perpendicular, arrow_color);
                }
            }

            // Label.
            char label[24];
            std::snprintf(label, sizeof(label), "z%zu", i);
            auto label_pos = to_screen({zone.position.x - extent_x, zone.position.y + extent_y}) + ImVec2{4.0f, 2.0f};
            foreground->AddText(label_pos, IM_COL32(255, 255, 255, 200), label);
        }

        // Emitters (on top of everything).
        for (std::size_t i{}; i < g_emitters.size(); ++i)
        {
            auto  &emitter       = g_emitters[i];
            auto   screen_center = to_screen(emitter.position);
            auto   arrow_length  = 0.8f * g_camera_zoom;
            auto   thickness     = std::max(1.5f, 2.0f * g_camera_zoom / ZOOM_DEFAULT);
            ImVec2 arrow_direction{std::cos(emitter.angle) * arrow_length, -std::sin(emitter.angle) * arrow_length};
            auto   arrow_tip = screen_center + arrow_direction;
            ImVec2 perpendicular{-arrow_direction.y * 0.2f, arrow_direction.x * 0.2f};
            auto   arrow_base = screen_center + arrow_direction * 0.65f;
            auto   color      = emitter.active ? IM_COL32(100, 255, 100, 255) : IM_COL32(255, 100, 100, 255);
            background->AddLine(screen_center, arrow_base, color, thickness);
            background->AddTriangleFilled(arrow_tip, arrow_base + perpendicular, arrow_base - perpendicular, color);
            background->AddCircleFilled(screen_center, std::max(3.0f, 0.1f * g_camera_zoom), color);

            // Label on foreground so it's always visible.
            char label[24];
            std::snprintf(label, sizeof(label), "e%zu", i);
            auto label_pos = screen_center + ImVec2{-8, -24};
            foreground->AddText(label_pos, IM_COL32(255, 255, 255, 200), label);
        }

        background->Flags = old_flags;
    }

    auto render_sample = (float)((double)(SDL_GetPerformanceCounter() - render_timer) / timer_frequency * 1000.0);
    g_render_ms        = g_render_ms + 0.01f * (render_sample - g_render_ms);

    auto present_timer = SDL_GetPerformanceCounter();

    ImGui::Render();

    auto *draw_data  = ImGui::GetDrawData();
    g_total_vertices = draw_data->TotalVtxCount;
    g_total_indices  = draw_data->TotalIdxCount;

    SDL_SetRenderScale(g_renderer, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);

    constexpr ImVec4 CLEAR_COLOR(0.10f, 0.10f, 0.10f, 1.00f);
    SDL_SetRenderDrawColorFloat(g_renderer, CLEAR_COLOR.x, CLEAR_COLOR.y, CLEAR_COLOR.z, CLEAR_COLOR.w);

    SDL_RenderClear(g_renderer);
    ImGui_ImplSDLRenderer3_RenderDrawData(draw_data, g_renderer);
    SDL_RenderPresent(g_renderer);

    auto present_sample = (float)((double)(SDL_GetPerformanceCounter() - present_timer) / timer_frequency * 1000.0);
    g_present_ms        = g_present_ms + 0.01f * (present_sample - g_present_ms);

    auto frame_sample = (float)((double)(SDL_GetPerformanceCounter() - physics_timer) / timer_frequency * 1000.0);
    g_frame_ms        = g_frame_ms + 0.01f * (frame_sample - g_frame_ms);

    // FPS cap: Delay to hit target frame time.
    if (g_fps_cap > 0)
    {
        auto target_ns = SDL_NS_PER_SECOND / (Uint64)g_fps_cap;
        auto elapsed   = SDL_GetTicksNS() - g_frame_start_ns;
        if (elapsed + SDL_NS_PER_MS < target_ns)
        {
            SDL_DelayPrecise(target_ns - elapsed);
        }
    }

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent([[maybe_unused]] void *appstate, SDL_Event *event)
{
    ImGui_ImplSDL3_ProcessEvent(event);

    switch (event->type)
    {
    case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
    {
        g_dpi_scaling = SDL_GetWindowDisplayScale(g_window);

        // Reset style.
        auto &style     = ImGui::GetStyle();
        auto  old_style = style;
        style           = {};

        ImGui::StyleColorsDark();

        // Apply new scaling.
        style.ScaleAllSizes(g_dpi_scaling);
        style.FontSizeBase = old_style.FontSizeBase;
        style.FontScaleDpi = g_dpi_scaling;

        break;
    }

    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    {
        if (ImGui::GetIO().WantCaptureMouse)
        {
            break;
        }

        // Middle-click.
        if (event->button.button == SDL_BUTTON_MIDDLE)
        {
            g_camera_dragging = true;
        }
        // Left-click.
        else if (event->button.button == SDL_BUTTON_LEFT)
        {
            auto world_point = screen_to_world(event->button.x, event->button.y);

            // Shift+Left-click: Rope linking.
            if ((SDL_GetModState() & SDL_KMOD_SHIFT) != 0)
            {
                auto target_body = find_body_at(world_point);
                if (B2_IS_NON_NULL(target_body))
                {
                    if (B2_IS_NON_NULL(g_rope_start_body))
                    {
                        // Second click: Create rope chain between the two bodies.
                        if (target_body.index1 != g_rope_start_body.index1)
                        {
                            auto anchor_end    = b2Body_GetLocalPoint(target_body, world_point);
                            auto world_start   = b2Body_GetWorldPoint(g_rope_start_body, g_rope_start_anchor);
                            auto world_end     = world_point;
                            auto distance      = b2Distance(world_start, world_end);
                            auto segment_count = std::max(2, (std::int32_t)(distance / SEGMENT_SPACING));

                            // Create segment bodies along the line.
                            auto body_def           = b2DefaultBodyDef();
                            body_def.type           = b2_dynamicBody;
                            body_def.gravityScale   = 1.0f;
                            body_def.linearDamping  = 0.5f;
                            body_def.angularDamping = 2.0f;

                            auto shape_def              = b2DefaultShapeDef();
                            shape_def.density           = 0.5f;
                            shape_def.material.friction = 0.6f;

                            // Capsule: Pill shape aligned vertically, tips at +/- `SEGMENT_HALF_LENGTH`.
                            b2Capsule capsule{{0.0f, -SEGMENT_HALF_LENGTH}, {0.0f, SEGMENT_HALF_LENGTH}, SEGMENT_RADIUS};

                            // Orient all capsules along the rope direction.
                            auto rope_direction = b2Normalize(world_end - world_start);
                            body_def.rotation   = b2MakeRot(std::atan2(rope_direction.y, rope_direction.x) - PI / 2.0f);

                            Rope rope{};
                            for (std::int32_t i{}; i < segment_count; ++i)
                            {
                                auto fraction     = (float)(i + 1) / (float)(segment_count + 1);
                                body_def.position = b2Lerp(world_start, world_end, fraction);

                                auto segment = b2CreateBody(g_world, &body_def);
                                b2CreateCapsuleShape(segment, &shape_def, &capsule);

                                rope.segments.emplace_back(segment);
                            }

                            // Revolute joints: Connect capsule tips end-to-end.
                            // Anchor A -> First segment bottom tip.
                            {
                                auto revolute_def         = b2DefaultRevoluteJointDef();
                                revolute_def.bodyIdA      = g_rope_start_body;
                                revolute_def.bodyIdB      = rope.segments.front();
                                revolute_def.localAnchorA = g_rope_start_anchor;
                                revolute_def.localAnchorB = {0.0f, -SEGMENT_HALF_LENGTH};

                                rope.joints.emplace_back(b2CreateRevoluteJoint(g_world, &revolute_def));
                            }

                            // Segment-to-segment: Top tip of [i] -> bottom tip of [i+1].
                            for (std::int32_t i{}; i < segment_count - 1; ++i)
                            {
                                auto revolute_def         = b2DefaultRevoluteJointDef();
                                revolute_def.bodyIdA      = rope.segments[i];
                                revolute_def.bodyIdB      = rope.segments[i + 1];
                                revolute_def.localAnchorA = {0.0f, SEGMENT_HALF_LENGTH};
                                revolute_def.localAnchorB = {0.0f, -SEGMENT_HALF_LENGTH};

                                rope.joints.emplace_back(b2CreateRevoluteJoint(g_world, &revolute_def));
                            }

                            // Last segment top tip -> Anchor B.
                            {
                                auto revolute_def         = b2DefaultRevoluteJointDef();
                                revolute_def.bodyIdA      = rope.segments.back();
                                revolute_def.bodyIdB      = target_body;
                                revolute_def.localAnchorA = {0.0f, SEGMENT_HALF_LENGTH};
                                revolute_def.localAnchorB = anchor_end;

                                rope.joints.emplace_back(b2CreateRevoluteJoint(g_world, &revolute_def));
                            }

                            // Disable collision between rope segments and their anchor bodies.
                            for (auto &&segment : rope.segments)
                            {
                                auto filter_def_start    = b2DefaultFilterJointDef();
                                filter_def_start.bodyIdA = g_rope_start_body;
                                filter_def_start.bodyIdB = segment;

                                rope.joints.emplace_back(b2CreateFilterJoint(g_world, &filter_def_start));

                                auto filter_def_end    = b2DefaultFilterJointDef();
                                filter_def_end.bodyIdA = target_body;
                                filter_def_end.bodyIdB = segment;

                                rope.joints.emplace_back(b2CreateFilterJoint(g_world, &filter_def_end));
                            }

                            rope.body_a  = g_rope_start_body;
                            rope.body_b  = target_body;
                            rope.local_a = g_rope_start_anchor;
                            rope.local_b = anchor_end;

                            g_ropes.emplace_back(std::move(rope));
                        }

                        g_rope_start_body = b2_nullBodyId;
                    }
                    else
                    {
                        // First click: Start linking.
                        g_rope_start_body   = target_body;
                        g_rope_start_anchor = b2Body_GetLocalPoint(target_body, world_point);
                    }
                }
            }
            // Ctrl+Left-click: Start drawing a line.
            else if ((SDL_GetModState() & SDL_KMOD_CTRL) != 0)
            {
                g_current_stroke.clear();
                g_current_stroke.emplace_back(world_point);
            }
            // Just Left-click: Drag emitter or body.
            else
            {
                // Check emitters first (small grab radius in world units).
                constexpr auto EMITTER_GRAB_RADIUS = 0.5f;
                for (std::size_t i{}; i < g_emitters.size(); ++i)
                {
                    if (b2LengthSquared(world_point - g_emitters[i].position) <= EMITTER_GRAB_RADIUS * EMITTER_GRAB_RADIUS)
                    {
                        g_dragged_emitter = i;

                        break;
                    }
                }

                // Check force zones next.
                if (!g_dragged_emitter && !g_dragging_zone)
                {
                    for (std::size_t i{}; i < g_force_zones.size(); ++i)
                    {
                        auto &zone             = g_force_zones[i];
                        auto  distance_squared = b2LengthSquared(world_point - zone.position);
                        auto  grab_radius      = zone.shape == ZoneShape::Rectangle ? std::max(zone.half_size.x, zone.half_size.y) : zone.radius;
                        if (distance_squared <= grab_radius * grab_radius)
                        {
                            g_dragging_zone = i;

                            break;
                        }
                    }
                }

                // If no emitter or zone hit, try grabbing a body.
                if (!g_dragged_emitter && !g_dragging_zone)
                {
                    g_mouse_body = find_body_at(world_point);
                    if (B2_IS_NON_NULL(g_mouse_body))
                    {
                        b2Body_SetAwake(g_mouse_body, true);

                        auto mouse_joint_def         = b2DefaultMouseJointDef();
                        mouse_joint_def.bodyIdA      = g_ground_id;
                        mouse_joint_def.bodyIdB      = g_mouse_body;
                        mouse_joint_def.target       = world_point;
                        mouse_joint_def.maxForce     = 10000.0f * b2Body_GetMass(g_mouse_body);
                        mouse_joint_def.hertz        = 60.0f;
                        mouse_joint_def.dampingRatio = 1.0f;

                        g_mouse_joint = b2CreateMouseJoint(g_world, &mouse_joint_def);

                        b2Body_SetFixedRotation(g_mouse_body, true);
                        b2Body_EnableSleep(g_mouse_body, false);
                    }
                }
            }
        }
        // Right-click.
        else if (event->button.button == SDL_BUTTON_RIGHT)
        {
            // Right-click while dragging: Pin body at current position.
            if (B2_IS_NON_NULL(g_mouse_joint))
            {
                auto position             = b2Body_GetPosition(g_mouse_body);
                auto revolute_def         = b2DefaultRevoluteJointDef();
                revolute_def.bodyIdA      = g_ground_id;
                revolute_def.bodyIdB      = g_mouse_body;
                revolute_def.localAnchorA = b2Body_GetLocalPoint(g_ground_id, position);
                revolute_def.localAnchorB = b2Vec2_zero;

                g_pins.emplace_back(b2CreateRevoluteJoint(g_world, &revolute_def), g_mouse_body);

                // Release drag.
                b2DestroyJoint(g_mouse_joint);
                b2Body_SetFixedRotation(g_mouse_body, false);
                b2Body_EnableSleep(g_mouse_body, true);

                g_mouse_joint = b2_nullJointId;
                g_mouse_body  = b2_nullBodyId;
                g_just_pinned = true;
            }
            else if ((SDL_GetModState() & SDL_KMOD_SHIFT) != 0)
            {
                g_cutting   = true;
                g_cut_start = {event->button.x, event->button.y};
            }
            else if ((SDL_GetModState() & SDL_KMOD_CTRL) != 0)
            {
                // Ctrl+Right-click on a pinned body: Unpin it.
                auto world_point = screen_to_world(event->button.x, event->button.y);
                auto hit_body    = find_body_at(world_point);
                auto pin_iter =
                    std::find_if(g_pins.begin(), g_pins.end(), [hit_body](const Pin &pin) noexcept { return B2_ID_EQUALS(pin.body, hit_body); });
                if (B2_IS_NON_NULL(hit_body) && pin_iter != g_pins.end())
                {
                    b2DestroyJoint(pin_iter->joint);

                    g_pins.erase(pin_iter);

                    // Suppress right-click-up selection.
                    g_just_pinned = true;
                }
                else
                {
                    // No pin under cursor: Start erasing.
                    g_erasing = true;
                }
            }
            else
            {
                g_box_selecting    = true;
                g_box_select_start = {event->button.x, event->button.y};
            }
        }

        break;
    }

    case SDL_EVENT_MOUSE_BUTTON_UP:
    {
        // Middle-click.
        if (event->button.button == SDL_BUTTON_MIDDLE)
        {
            g_camera_dragging = false;
        }
        // Left-click.
        else if (event->button.button == SDL_BUTTON_LEFT)
        {
            if (!g_current_stroke.empty())
            {
                finish_stroke();
            }
            else if (g_dragged_emitter)
            {
                g_dragged_emitter = {};
            }
            else if (g_dragging_zone)
            {
                g_dragging_zone = {};
            }
            else if (B2_IS_NON_NULL(g_mouse_joint))
            {
                b2DestroyJoint(g_mouse_joint);
                b2Body_SetFixedRotation(g_mouse_body, false);
                b2Body_EnableSleep(g_mouse_body, true);

                g_mouse_joint = b2_nullJointId;
                g_mouse_body  = b2_nullBodyId;
            }
        }
        // Right-click.
        else if (event->button.button == SDL_BUTTON_RIGHT)
        {
            g_box_selecting = false;

            auto drag_delta = ImVec2{event->button.x, event->button.y} - g_box_select_start;

            // Pin was handled on right-click down; suppress the up action.
            if (g_just_pinned)
            {
                g_just_pinned = false;
            }
            // Ctrl+Right was erase mode - already handled in motion.
            else if (g_erasing)
            {
                g_erasing = false;
            }
            // Shift+Right was rope cut mode - already handled in motion.
            else if (g_cutting || g_just_cut)
            {
                g_cutting  = false;
                g_just_cut = false;
            }
            // Click, not drag: Toggle selection on body under cursor.
            else if (drag_delta.x * drag_delta.x + drag_delta.y * drag_delta.y < 25.0f)
            {
                auto world_point = screen_to_world(event->button.x, event->button.y);
                auto hit_body    = find_body_at(world_point);
                if (B2_IS_NON_NULL(hit_body))
                {
                    for (auto &&body : g_bodies)
                    {
                        if (body.body.index1 == hit_body.index1)
                        {
                            body.selected = !body.selected;

                            break;
                        }
                    }
                }
                else
                {
                    // Clicked empty space: Clear selection.
                    for (auto &&body : g_bodies)
                    {
                        body.selected = false;
                    }
                }
            }
            // Drag: Box select.
            else
            {
                // Clear previous selection unless Shift is held.
                if ((SDL_GetModState() & SDL_KMOD_SHIFT) == 0)
                {
                    for (auto &&body : g_bodies)
                    {
                        body.selected = false;
                    }
                }

                auto                  world_corner_min = screen_to_world(g_box_select_start.x, g_box_select_start.y);
                auto                  world_corner_max = screen_to_world(g_box_select_start.x + drag_delta.x, g_box_select_start.y + drag_delta.y);
                b2AABB                aabb{b2Min(world_corner_min, world_corner_max), b2Max(world_corner_min, world_corner_max)};
                std::vector<b2BodyId> in_box{};
                b2World_OverlapAABB(
                    g_world, aabb, b2DefaultQueryFilter(),
                    [](b2ShapeId shapeId, void *context)
                    {
                        auto body = b2Shape_GetBody(shapeId);
                        if (b2Body_GetType(body) == b2_dynamicBody)
                        {
                            auto &selected_bodies = *(std::vector<b2BodyId> *)context;
                            selected_bodies.emplace_back(body);
                        }

                        return true;
                    },
                    &in_box);

                for (auto &&id : in_box)
                {
                    for (auto &&body : g_bodies)
                    {
                        if (body.body.index1 == id.index1)
                        {
                            body.selected = true;

                            break;
                        }
                    }
                }
            }
        }

        break;
    }

    case SDL_EVENT_MOUSE_MOTION:
    {
        if (g_camera_dragging)
        {
            g_camera_center.x -= event->motion.xrel / g_camera_zoom;
            g_camera_center.y += event->motion.yrel / g_camera_zoom;
        }
        else if (g_dragged_emitter)
        {
            auto world_point                        = screen_to_world(event->motion.x, event->motion.y);
            g_emitters[*g_dragged_emitter].position = world_point;
        }
        else if (g_dragging_zone)
        {
            auto world_point                         = screen_to_world(event->motion.x, event->motion.y);
            g_force_zones[*g_dragging_zone].position = world_point;
        }
        else if (B2_IS_NON_NULL(g_mouse_joint))
        {
            auto world_point = screen_to_world(event->motion.x, event->motion.y);
            b2MouseJoint_SetTarget(g_mouse_joint, world_point);
        }
        else if (g_erasing)
        {
            auto world_point = screen_to_world(event->motion.x, event->motion.y);
            for (std::size_t i{}; i < g_drawn_lines.size(); ++i)
            {
                auto                      &line = g_drawn_lines[i];
                std::optional<std::size_t> hit_segment{};
                for (std::size_t j{}; j < line.points.size() - 1; ++j)
                {
                    if (segment_distance_squared(world_point, line.points[j], line.points[j + 1]) <= HIT_RADIUS_SQUARED)
                    {
                        hit_segment = j + 1;

                        break;
                    }
                }

                if (!hit_segment)
                {
                    continue;
                }

                // Destroy old body.
                b2DestroyBody(line.body);

                // Split into left [0..hit_segment-1] and right [hit_segment..end].
                auto       &points = line.points;
                std::vector left(points.begin(), points.begin() + (std::ptrdiff_t)*hit_segment);
                std::vector right(points.begin() + (std::ptrdiff_t)*hit_segment, points.end());

                // Remove original line.
                g_drawn_lines.erase(g_drawn_lines.begin() + (std::ptrdiff_t)i);

                // Re-create parts with >= 2 points.
                if (left.size() >= 2)
                {
                    g_drawn_lines.emplace_back(create_drawn_line(std::move(left)));
                }

                if (right.size() >= 2)
                {
                    g_drawn_lines.emplace_back(create_drawn_line(std::move(right)));
                }

                --i;
            }

            // Erase ropes: Hit-test cursor against rope links, destroy entire rope on hit.
            for (std::size_t i{}; i < g_ropes.size(); ++i)
            {
                auto &rope = g_ropes[i];

                // Inline hit-test: Walk the chain [anchor_a?, segments..., anchor_b?] without allocating.
                bool   rope_hit{};
                b2Vec2 previous_point{};
                bool   has_previous{};

                auto test_link = [&](b2Vec2 point)
                {
                    if (has_previous && segment_distance_squared(world_point, previous_point, point) <= HIT_RADIUS_SQUARED)
                    {
                        rope_hit = true;
                    }

                    previous_point = point;
                    has_previous   = true;
                };

                if (B2_IS_NON_NULL(rope.body_a))
                {
                    test_link(b2Body_GetWorldPoint(rope.body_a, rope.local_a));
                }

                for (auto &&segment : rope.segments)
                {
                    if (rope_hit)
                    {
                        break;
                    }

                    test_link(b2Body_GetPosition(segment));
                }

                if (!rope_hit && B2_IS_NON_NULL(rope.body_b))
                {
                    test_link(b2Body_GetWorldPoint(rope.body_b, rope.local_b));
                }

                if (!rope_hit)
                {
                    continue;
                }

                // Destroy all joints and segment bodies.
                for (auto &&joint : rope.joints)
                {
                    if (b2Joint_IsValid(joint))
                    {
                        b2DestroyJoint(joint);
                    }
                }

                for (auto &&segment : rope.segments)
                {
                    if (b2Body_IsValid(segment))
                    {
                        b2DestroyBody(segment);
                    }
                }

                g_ropes.erase(g_ropes.begin() + (std::ptrdiff_t)i);

                --i;
            }
        }
        else if (g_cutting)
        {
            auto world_point = screen_to_world(event->motion.x, event->motion.y);
            auto rope_count  = g_ropes.size(); // Snapshot - don't iterate newly appended halves.
            for (std::size_t i{}; i < rope_count; ++i)
            {
                auto &rope          = g_ropes[i];
                auto  segment_count = (std::int32_t)rope.segments.size();
                auto  body_a        = rope.body_a;
                auto  body_b        = rope.body_b;
                auto  anchor_start  = rope.local_a;
                auto  anchor_end    = rope.local_b;
                auto  has_start     = B2_IS_NON_NULL(body_a);
                auto  has_end       = B2_IS_NON_NULL(body_b);

                // Inline hit-test: Walk the chain [anchor_a?, segments..., anchor_b?] without allocating.
                std::optional<std::int32_t> hit_link{};
                b2Vec2                      previous_point{};
                std::int32_t                link_index{};
                bool                        has_previous{};

                auto test_link = [&](b2Vec2 point)
                {
                    if (has_previous && segment_distance_squared(world_point, previous_point, point) <= HIT_RADIUS_SQUARED)
                    {
                        hit_link = link_index - 1; // Index of the segment start in the conceptual point list.
                    }

                    previous_point = point;
                    has_previous   = true;

                    ++link_index;
                };

                if (has_start)
                {
                    test_link(b2Body_GetWorldPoint(body_a, anchor_start));
                }

                for (auto &&segment : rope.segments)
                {
                    if (hit_link)
                    {
                        break;
                    }

                    test_link(b2Body_GetPosition(segment));
                }

                if (!hit_link && has_end)
                {
                    test_link(b2Body_GetWorldPoint(body_b, anchor_end));
                }

                if (!hit_link)
                {
                    continue;
                }

                // Destroy all joints.
                for (auto &&joint : rope.joints)
                {
                    if (b2Joint_IsValid(joint))
                    {
                        b2DestroyJoint(joint);
                    }
                }

                // Map hit_link in points[] to a segment-space cut index.
                // points: [anchor_start?] [seg_0 .. seg_(N-1)] [anchor_end?]
                // `cut_index` = First segment on the right side of the cut.
                auto cut_index   = has_start ? *hit_link : *hit_link + 1;
                auto left_count  = cut_index;
                auto right_count = segment_count - cut_index;

                // Helper: Create a revolute joint between two bodies.
                auto make_rev = [](b2BodyId body_from, b2Vec2 local_from, b2BodyId body_to, b2Vec2 local_to)
                {
                    auto revolute_def         = b2DefaultRevoluteJointDef();
                    revolute_def.bodyIdA      = body_from;
                    revolute_def.bodyIdB      = body_to;
                    revolute_def.localAnchorA = local_from;
                    revolute_def.localAnchorB = local_to;

                    return b2CreateRevoluteJoint(g_world, &revolute_def);
                };

                // Wire a sub-rope from existing segment bodies and push to `g_ropes`.
                // `anchor_first`: true = `anchor` is `body_a` (left half), false = `body_b` (right half).
                auto wire_half = [&make_rev](b2BodyId anchor, b2Vec2 anchor_local, b2BodyId *segment_data, std::int32_t count, bool anchor_first)
                {
                    Rope half{};
                    half.segments.assign(segment_data, segment_data + count);

                    auto has_anchor = B2_IS_NON_NULL(anchor);

                    if (anchor_first)
                    {
                        if (has_anchor)
                        {
                            half.body_a  = anchor;
                            half.local_a = anchor_local;
                            half.joints.emplace_back(make_rev(anchor, anchor_local, half.segments.front(), {0.0f, -SEGMENT_HALF_LENGTH}));
                        }

                        for (std::int32_t j{}; j < count - 1; ++j)
                        {
                            half.joints.emplace_back(
                                make_rev(half.segments[j], {0.0f, SEGMENT_HALF_LENGTH}, half.segments[j + 1], {0.0f, -SEGMENT_HALF_LENGTH}));
                        }
                    }
                    else
                    {
                        for (std::int32_t j{}; j < count - 1; ++j)
                        {
                            half.joints.emplace_back(
                                make_rev(half.segments[j], {0.0f, SEGMENT_HALF_LENGTH}, half.segments[j + 1], {0.0f, -SEGMENT_HALF_LENGTH}));
                        }

                        if (has_anchor)
                        {
                            half.body_b  = anchor;
                            half.local_b = anchor_local;
                            half.joints.emplace_back(make_rev(half.segments.back(), {0.0f, SEGMENT_HALF_LENGTH}, anchor, anchor_local));
                        }
                    }

                    if (has_anchor)
                    {
                        for (std::int32_t j{}; j < count; ++j)
                        {
                            auto filter_joint_def    = b2DefaultFilterJointDef();
                            filter_joint_def.bodyIdA = anchor;
                            filter_joint_def.bodyIdB = half.segments[j];

                            half.joints.emplace_back(b2CreateFilterJoint(g_world, &filter_joint_def));
                        }
                    }

                    g_ropes.emplace_back(std::move(half));
                };

                // Move segments out - `wire_half` pushes to `g_ropes` which may invalidate `rope`.
                auto all_segments = std::move(rope.segments);

                if (left_count > 0)
                {
                    wire_half(body_a, anchor_start, all_segments.data(), left_count, true);
                }

                if (right_count > 0)
                {
                    wire_half(body_b, anchor_end, all_segments.data() + cut_index, right_count, false);
                }

                // Remove the original rope (joints already destroyed, segments moved to halves).
                g_ropes.erase(g_ropes.begin() + (std::ptrdiff_t)i);

                --i;
                --rope_count;

                // One cut per drag. `g_just_cut` suppresses selection on release.
                g_cutting  = false;
                g_just_cut = true;

                break;
            }
        }
        else if (!g_current_stroke.empty())
        {
            auto  world_point = screen_to_world(event->motion.x, event->motion.y);
            auto &last        = g_current_stroke.back();
            if (b2LengthSquared(world_point - last) >= MIN_STROKE_DIST * MIN_STROKE_DIST)
            {
                g_current_stroke.emplace_back(world_point);
            }
        }

        break;
    }

    case SDL_EVENT_MOUSE_WHEEL:
    {
        if (!ImGui::GetIO().WantCaptureMouse && event->wheel.y != 0.0f)
        {
            // Screen-space mouse position.
            auto mouse_x = event->wheel.mouse_x;
            auto mouse_y = event->wheel.mouse_y;

            // World position under cursor before zoom.
            auto world_x = g_camera_center.x + (mouse_x - (float)g_window_w / 2.0f) / g_camera_zoom;
            auto world_y = g_camera_center.y - (mouse_y - (float)g_window_h / 2.0f) / g_camera_zoom;

            // Apply zoom.
            auto factor   = event->wheel.y > 0.0f ? ZOOM_FACTOR : 1.0f / ZOOM_FACTOR;
            g_camera_zoom = std::clamp(g_camera_zoom * factor, g_camera_zoom_min, ZOOM_MAX);

            // Adjust center so world point stays under cursor.
            g_camera_center.x = world_x - (mouse_x - (float)g_window_w / 2.0f) / g_camera_zoom;
            g_camera_center.y = world_y + (mouse_y - (float)g_window_h / 2.0f) / g_camera_zoom;
        }

        break;
    }

    case SDL_EVENT_KEY_DOWN:
    {
        if (event->key.key == SDLK_DELETE || event->key.key == SDLK_BACKSPACE)
        {
            delete_selected();
        }
        else if (event->key.key == SDLK_ESCAPE)
        {
            // Clear selection.
            for (auto &&body : g_bodies)
            {
                body.selected = false;
            }

            // Cancel rope linking.
            g_rope_start_body = b2_nullBodyId;
        }

        break;
    }

    case SDL_EVENT_QUIT:
    {
        return SDL_APP_SUCCESS;
    }

    default:
    {
        break;
    }
    }

    return SDL_APP_CONTINUE;
}

void SDL_AppQuit([[maybe_unused]] void *appstate, [[maybe_unused]] SDL_AppResult result)
{
    reset_tasks();
    b2DestroyWorld(g_world);

    g_thread_pool = {};

    if (ImGui::GetCurrentContext() != nullptr)
    {
        ImGui_ImplSDLRenderer3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
    }

    SDL_DestroyRenderer(g_renderer);
    SDL_DestroyWindow(g_window);

    // `SDL_Quit()` will get called for us after this returns.
}
