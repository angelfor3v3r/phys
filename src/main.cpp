#include <thread_pool/thread_pool.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>
#include <box2d/box2d.h>

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <format>
#include <latch>
#include <memory>
#include <new>
#include <numbers>
#include <optional>
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
constexpr auto         PI              = std::numbers::pi_v<float>;
constexpr auto         RAD_TO_DEG      = 180.0f / PI;
constexpr auto         DEG_TO_RAD      = PI / 180.0f;
constexpr std::int32_t MAX_TASKS       = 64;
constexpr auto         MAX_FRAME_TIME  = 0.25f;
constexpr auto         PHYSICS_DT      = 1.0f / 60.0f;
constexpr auto         ZOOM_DEFAULT    = 50.0f;
constexpr auto         ZOOM_MAX        = 500.0f;
constexpr auto         ZOOM_FACTOR     = 1.1f;
constexpr auto         MIN_STROKE_DIST = 0.15f;

// Playground bounds (world units).
constexpr auto AREA_MIN_X = -100.0f;
constexpr auto AREA_MAX_X = 100.0f;
constexpr auto AREA_MIN_Y = -40.0f;
constexpr auto AREA_MAX_Y = 120.0f;

// Window / renderer.
float         g_dpi_scaling{};
int           g_window_w{};
int           g_window_h{};
SDL_Window   *g_window{};
SDL_Renderer *g_renderer{};
const char   *g_renderer_name{};

// Threading.
std::uint32_t                      g_worker_count{};
std::unique_ptr<dp::thread_pool<>> g_thread_pool{};
std::atomic<std::uint32_t>         g_next_worker{};
std::int32_t                       g_task_count{};

// Physics.
b2WorldId    g_world = b2_nullWorldId;
float        g_physics_accumulator{};
float        g_physics_alpha{};
bool         g_paused{};
bool         g_single_step{};
std::int32_t g_step_count = 1;
std::int32_t g_sub_steps  = 4;
std::size_t  g_culled_count{};

// Camera.
b2Vec2 g_cam_center{0.0f, 2.0f};
auto   g_cam_zoom     = ZOOM_DEFAULT;
auto   g_cam_zoom_min = 1.0f;
bool   g_cam_dragging{};

// Graphics.
int    g_vsync   = SDL_RENDERER_VSYNC_ADAPTIVE; // Falls back to regular vsync if unsupported.
int    g_fps_cap = 0;                           // 0 = off, 10-1000 = target FPS. Defaults to monitor refresh rate at init.
Uint64 g_frame_start_ns{};
float  g_phys_ms{};      // Last frame's physics time.
float  g_render_ms{};    // Last frame's render time (draw list build).
float  g_present_ms{};   // Last frame's GPU submit + present time.
float  g_frame_ms{};     // Last frame's total time.
int    g_display_fps{};  // FPS shown on screen, updated once per second.
int    g_frame_count{};  // Frames counted in current second.
Uint64 g_fps_timer_ns{}; // Timestamp of last FPS update.
int    g_total_vtx{};    // Last frame's total vertex count.
int    g_total_idx{};    // Last frame's total index count.

// Bodies.
b2BodyId g_ground_id = b2_nullBodyId;

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
    BodyState   prev{};
    bool        selected{};
};

std::vector<PhysBody> g_bodies{};

// Drawing.
struct DrawnLine
{
    b2BodyId            body = b2_nullBodyId;
    std::vector<b2Vec2> points{};
};

std::vector<DrawnLine> g_drawn_lines{};
std::vector<b2Vec2>    g_current_stroke{};

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

std::vector<Emitter> g_emitters{};

// Ropes: Chain of small circle bodies connected by distance joints.
struct Rope
{
    std::vector<b2BodyId>  segments{};             // Intermediate chain links.
    std::vector<b2JointId> joints{};               // All joints (distance + filter).
    b2BodyId               body_a = b2_nullBodyId; // Anchor body at start (null if dangling).
    b2BodyId               body_b = b2_nullBodyId; // Anchor body at end   (null if dangling).
    b2Vec2                 local_a{};              // Attach point in body_a's local space.
    b2Vec2                 local_b{};              // Attach point in body_b's local space.
};

std::vector<Rope> g_ropes{};
b2BodyId          g_rope_start_body = b2_nullBodyId;
b2Vec2            g_rope_start_anchor{};

struct Pin
{
    b2JointId joint = b2_nullJointId;
    b2BodyId  body  = b2_nullBodyId;
};

std::vector<Pin> g_pins{};

// Interaction.
b2JointId                  g_mouse_joint = b2_nullJointId;
b2BodyId                   g_mouse_body  = b2_nullBodyId;
bool                       g_box_selecting{};
ImVec2                     g_box_select_start{};
std::optional<std::size_t> g_dragged_emitter{};
bool                       g_erasing{};
bool                       g_cutting{};
bool                       g_just_pinned{};

struct TaskHandle
{
    explicit TaskHandle(std::int32_t count) noexcept : done{count} {}

    std::latch done;
};

alignas(TaskHandle) std::byte g_task_storage[MAX_TASKS][sizeof(TaskHandle)];

// Silly fix for window dragging stutter.
#if defined(_WIN32)
#include <dwmapi.h>

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
    auto r = std::min(255, (int)(color >> IM_COL32_R_SHIFT & 0xFF) + amount);
    auto g = std::min(255, (int)(color >> IM_COL32_G_SHIFT & 0xFF) + amount);
    auto b = std::min(255, (int)(color >> IM_COL32_B_SHIFT & 0xFF) + amount);

    return IM_COL32(r, g, b, 255);
}

[[nodiscard]] b2Vec2 screen_to_world(float screen_x, float screen_y) noexcept
{
    return {
        g_cam_center.x + (screen_x - (float)g_window_w / 2.0f) / g_cam_zoom,
        g_cam_center.y - (screen_y - (float)g_window_h / 2.0f) / g_cam_zoom,
    };
}

[[nodiscard]] float segment_distance_squared(b2Vec2 point, b2Vec2 seg_a, b2Vec2 seg_b) noexcept
{
    auto d      = seg_b - seg_a;
    auto len_sq = b2Dot(d, d);

    float t{};
    if (len_sq > 0.0f)
    {
        t = std::clamp(b2Dot(point - seg_a, d) / len_sq, 0.0f, 1.0f);
    }

    auto closest = seg_a + d * t;
    auto diff    = point - closest;

    return b2Dot(diff, diff);
}

[[nodiscard]] b2BodyId find_body_at(b2Vec2 world_point) noexcept
{
    auto     proxy  = b2MakeProxy(&world_point, 1, 0.01f);
    b2BodyId result = b2_nullBodyId;
    b2World_OverlapShape(
        g_world, &proxy, b2DefaultQueryFilter(),
        [](b2ShapeId shapeId, void *context)
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
    auto body_def     = b2DefaultBodyDef();
    body_def.type     = b2_dynamicBody;
    body_def.position = position;

    auto body = b2CreateBody(g_world, &body_def);

    auto shape_def              = b2DefaultShapeDef();
    shape_def.density           = 1.0f;
    shape_def.material.friction = 0.3f;

    auto box      = b2MakeBox(half_width, half_height);
    auto shape_id = b2CreatePolygonShape(body, &shape_def, &box);

    g_bodies.emplace_back(body, shape_id, b2_polygonShape, BodyState{position, b2Rot_identity});
}

void add_circle(b2Vec2 position, float radius)
{
    auto body_def     = b2DefaultBodyDef();
    body_def.type     = b2_dynamicBody;
    body_def.position = position;

    auto body = b2CreateBody(g_world, &body_def);

    auto shape_def              = b2DefaultShapeDef();
    shape_def.density           = 1.0f;
    shape_def.material.friction = 0.3f;

    b2Circle circle{{0.0f, 0.0f}, radius};
    auto     shape_id = b2CreateCircleShape(body, &shape_def, &circle);

    g_bodies.emplace_back(body, shape_id, b2_circleShape, BodyState{position, b2Rot_identity});
}

void add_triangle(b2Vec2 position, float height)
{
    auto body_def     = b2DefaultBodyDef();
    body_def.type     = b2_dynamicBody;
    body_def.position = position;

    auto body = b2CreateBody(g_world, &body_def);

    auto shape_def              = b2DefaultShapeDef();
    shape_def.density           = 1.0f;
    shape_def.material.friction = 0.3f;

    // Equilateral triangle, centered at centroid.
    auto   half_base = height / std::sqrt(3.0f);
    b2Vec2 verts[3]{
        {0.0f, height * 2.0f / 3.0f},
        {-half_base, -height / 3.0f},
        {half_base, -height / 3.0f},
    };
    auto hull     = b2ComputeHull(verts, 3);
    auto poly     = b2MakePolygon(&hull, 0.0f);
    auto shape_id = b2CreatePolygonShape(body, &shape_def, &poly);

    g_bodies.emplace_back(body, shape_id, b2_polygonShape, BodyState{position, b2Rot_identity});
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
requires (std::integral<T> || std::floating_point<T>) void scroll_adjust(T &value, T step, T min_val, T max_val) noexcept
{
    if (!ImGui::IsItemHovered())
    {
        return;
    }

    auto wheel = ImGui::GetIO().MouseWheel;
    if (wheel != 0.0f)
    {
        value = std::clamp(value + (T)(wheel * (float)step), min_val, max_val);

        ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Equal) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd))
    {
        value = std::clamp(value + step, min_val, max_val);
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Minus) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract))
    {
        value = std::clamp(value - step, min_val, max_val);
    }
}

// `SliderFloat`/`SliderInt` + scroll_adjust in one call.
bool slider(const char *label, float &value, float min_val, float max_val, float scroll_step, const char *fmt = "%.3f") noexcept
{
    auto result = ImGui::SliderFloat(label, &value, min_val, max_val, fmt);
    scroll_adjust(value, scroll_step, min_val, max_val);

    return result;
}

bool slider(const char *label, int &value, int min_val, int max_val, int scroll_step, const char *fmt = "%d") noexcept
{
    auto result = ImGui::SliderInt(label, &value, min_val, max_val, fmt);
    scroll_adjust(value, scroll_step, min_val, max_val);

    return result;
}

// Combo (null-separated items) + scroll_adjust in one call.
bool combo(const char *label, int &index, const char *items_separated_by_zeros, int item_count) noexcept
{
    auto result = ImGui::Combo(label, &index, items_separated_by_zeros);
    scroll_adjust(index, 1, 0, item_count - 1);

    return result;
}

void tick_emitters(float dt)
{
    for (auto &&emitter : g_emitters)
    {
        if (!emitter.active || emitter.rate <= 0.0f)
        {
            continue;
        }

        emitter.timer += dt;

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
            b2Vec2 dir{std::cos(emitter.angle), std::sin(emitter.angle)};
            b2Body_SetLinearVelocity(spawned.body, dir * emitter.speed);
        }
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
            [=]
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

void step_physics(float dt)
{
    if (g_paused && !g_single_step)
    {
        return;
    }

    if (g_single_step)
    {
        // Save previous transforms for interpolation.
        for (auto &&body : g_bodies)
        {
            auto transform     = b2Body_GetTransform(body.body);
            body.prev.position = transform.p;
            body.prev.rotation = transform.q;
        }

        for (std::int32_t i{}; i < g_step_count; ++i)
        {
            reset_tasks();
            b2World_Step(g_world, PHYSICS_DT, g_sub_steps);
        }

        tick_emitters(PHYSICS_DT * (float)g_step_count);

        g_single_step   = false;
        g_physics_alpha = 1.0f;

        return;
    }

    g_physics_accumulator += std::min(dt, MAX_FRAME_TIME);
    while (g_physics_accumulator >= PHYSICS_DT)
    {
        // Save previous transforms for interpolation.
        for (auto &&body : g_bodies)
        {
            auto transform     = b2Body_GetTransform(body.body);
            body.prev.position = transform.p;
            body.prev.rotation = transform.q;
        }

        reset_tasks();
        b2World_Step(g_world, PHYSICS_DT, g_sub_steps);

        g_physics_accumulator -= PHYSICS_DT;
    }

    g_physics_alpha = g_physics_accumulator / PHYSICS_DT;
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

    // VSync defaults to on. Non-fatal if unsupported.
    SDL_SetRenderVSync(g_renderer, g_vsync);

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

    // Create thread pool.
    auto num_threads = std::clamp<std::uint32_t>(std::thread::hardware_concurrency(), 1, 16);
    g_worker_count   = num_threads <= 2 ? num_threads : num_threads - 1;
    g_thread_pool    = std::make_unique<dp::thread_pool<>>(g_worker_count);

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

    g_bodies.reserve(2048);

    // Spawn some initial boxes.
    for (std::int32_t i{}; i < 25; ++i)
    {
        add_box({-2.0f + (float)i * 0.5f, 4.0f + (float)i * 1.5f}, 0.5f, 0.5f);
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
    auto area_w    = AREA_MAX_X - AREA_MIN_X;
    auto area_h    = AREA_MAX_Y - AREA_MIN_Y;
    g_cam_zoom_min = std::max((float)g_window_w / area_w, (float)g_window_h / area_h);
    g_cam_zoom     = std::clamp(g_cam_zoom, g_cam_zoom_min, ZOOM_MAX);

    auto half_vw   = (float)g_window_w / 2.0f / g_cam_zoom;
    auto half_vh   = (float)g_window_h / 2.0f / g_cam_zoom;
    auto center_x  = (AREA_MIN_X + AREA_MAX_X) / 2.0f;
    auto center_y  = (AREA_MIN_Y + AREA_MAX_Y) / 2.0f;
    auto clamp_x   = area_w / 2.0f - half_vw;
    auto clamp_y   = area_h / 2.0f - half_vh;
    g_cam_center.x = std::clamp(g_cam_center.x, center_x - clamp_x, center_x + clamp_x);
    g_cam_center.y = std::clamp(g_cam_center.y, center_y - clamp_y, center_y + clamp_y);

    auto &io = ImGui::GetIO();

    auto freq    = (double)SDL_GetPerformanceFrequency();
    auto phys_t0 = SDL_GetPerformanceCounter();

    step_physics(io.DeltaTime);

    // Exponential moving average (smoothing factor 0.01 = ~100 frame window).
    auto phys_sample = (float)((double)(SDL_GetPerformanceCounter() - phys_t0) / freq * 1000.0);
    g_phys_ms        = g_phys_ms + 0.01f * (phys_sample - g_phys_ms);

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
    }

    // Start the Dear ImGui frame.
    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui_ImplSDL3_NewFrame();

    auto render_t0 = SDL_GetPerformanceCounter();

    ImGui::NewFrame();
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

    auto *fg        = ImGui::GetForegroundDrawList();
    auto *bg        = ImGui::GetBackgroundDrawList();
    auto  mouse_pos = ImGui::GetMousePos();

    // Toolbox.
    if (ImGui::Begin("phys"))
    {
        ImGui::SeparatorText("Spawn");

        if (ImGui::Button("Box"))
        {
            add_box(g_cam_center, 0.5f, 0.5f);
        }

        if (ImGui::BeginDragDropSource())
        {
            auto type = SpawnShape::Box;
            ImGui::SetDragDropPayload("SPAWN", &type, sizeof(type));
            ImGui::Text("Box");
            ImGui::EndDragDropSource();
        }

        if (ImGui::Button("Circle"))
        {
            add_circle(g_cam_center, 0.5f);
        }

        if (ImGui::BeginDragDropSource())
        {
            auto type = SpawnShape::Circle;
            ImGui::SetDragDropPayload("SPAWN", &type, sizeof(type));
            ImGui::Text("Circle");
            ImGui::EndDragDropSource();
        }

        if (ImGui::Button("Triangle"))
        {
            add_triangle(g_cam_center, 1.0f);
        }

        if (ImGui::BeginDragDropSource())
        {
            auto type = SpawnShape::Triangle;
            ImGui::SetDragDropPayload("SPAWN", &type, sizeof(type));
            ImGui::Text("Triangle");
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
                g_emitters.emplace_back(g_cam_center);
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
        }

        if (ImGui::CollapsingHeader("Info", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("Bodies: %zu (%d awake, %zu culled)", g_bodies.size(), b2World_GetAwakeBodyCount(g_world), g_culled_count);
            ImGui::Text("Workers: %u", g_worker_count);

            auto counters = b2World_GetCounters(g_world);
            ImGui::Text("Islands: %d", counters.islandCount);
            ImGui::Text("Contacts: %d", counters.contactCount);
            ImGui::Text("Camera: (%.1f, %.1f) zoom %.1f", g_cam_center.x + 0.0f, g_cam_center.y + 0.0f, g_cam_zoom);
            ImGui::Text("Ropes: %zu", g_ropes.size());
            ImGui::Text("Physics: %.2f ms", g_phys_ms);
            ImGui::Text("Render:  %.2f ms", g_render_ms);
            ImGui::Text("Present: %.2f ms", g_present_ms);
            ImGui::Text("Frame:   %.2f ms", g_frame_ms);
            ImGui::Text("Vertices: %d | Indices: %d", g_total_vtx, g_total_idx);
        }

        if (ImGui::CollapsingHeader("Graphics"))
        {
            // VSync mode: Maps combo index to SDL vsync values.
            constexpr int VSYNC_VALUES[] = {SDL_RENDERER_VSYNC_DISABLED, 1, SDL_RENDERER_VSYNC_ADAPTIVE};

            int vsync_index = g_vsync == SDL_RENDERER_VSYNC_ADAPTIVE ? 2 : g_vsync;
            combo("VSync", vsync_index, "Off\0On\0Adaptive\0", 3);

            g_vsync = VSYNC_VALUES[vsync_index];
            SDL_SetRenderVSync(g_renderer, g_vsync);

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
                ImGui::Text("%s", action);
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
            ImVec2 half{g_cam_zoom / 2.0f, g_cam_zoom / 2.0f};
            fg->AddRectFilled(mouse_pos - half, mouse_pos + half, IM_COL32(51, 153, 230, 80));
            fg->AddRect(mouse_pos - half, mouse_pos + half, IM_COL32(255, 255, 255, 100));

            break;
        }

        case SpawnShape::Circle:
        {
            auto screen_radius = g_cam_zoom / 2.0f;
            fg->AddCircleFilled(mouse_pos, screen_radius, IM_COL32(230, 153, 51, 80));
            fg->AddCircle(mouse_pos, screen_radius, IM_COL32(255, 255, 255, 100));

            break;
        }

        case SpawnShape::Triangle:
        {
            auto   half = g_cam_zoom / 2.0f;
            ImVec2 tri_verts[3]{
                {mouse_pos.x, mouse_pos.y - half},
                {mouse_pos.x - half, mouse_pos.y + half},
                {mouse_pos.x + half, mouse_pos.y + half},
            };
            fg->AddTriangleFilled(tri_verts[0], tri_verts[1], tri_verts[2], IM_COL32(50, 200, 50, 80));
            fg->AddTriangle(tri_verts[0], tri_verts[1], tri_verts[2], IM_COL32(255, 255, 255, 100));

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

    fg->AddText({4, 4}, IM_COL32_WHITE, g_renderer_name);

    ++g_frame_count;

    auto now_ns = SDL_GetTicksNS();
    if (now_ns - g_fps_timer_ns >= SDL_NS_PER_SECOND)
    {
        g_display_fps  = g_frame_count;
        g_frame_count  = 0;
        g_fps_timer_ns = now_ns;
    }

    char fps_buf[32];
    std::snprintf(fps_buf, sizeof(fps_buf), "%d FPS", g_display_fps);
    fg->AddText({4, 20}, IM_COL32_WHITE, fps_buf);

    // Box select preview.
    if (g_box_selecting)
    {
        fg->AddRectFilled(g_box_select_start, mouse_pos, IM_COL32(255, 80, 80, 30));
        fg->AddRect(g_box_select_start, mouse_pos, IM_COL32(255, 80, 80, 150), 0.0f, 0, 1.5f);
    }

    // Render physics bodies to background draw list.
    {
        // Disable fill anti-aliasing for body rendering - saves ~2x vertices per filled shape.
        // Outlines keep AA via `ImDrawListFlags_AntiAliasedLines` (unchanged).
        auto old_flags  = bg->Flags;
        bg->Flags      &= ~ImDrawListFlags_AntiAliasedFill;

        // Camera transform - snap to pixel grid to avoid subpixel blurriness.
        auto cam_x = std::round((float)g_window_w / 2.0f - g_cam_center.x * g_cam_zoom);
        auto cam_y = std::round((float)g_window_h / 2.0f + g_cam_center.y * g_cam_zoom);

        auto to_screen = [cam_x, cam_y](b2Vec2 point) noexcept -> ImVec2 { return {cam_x + point.x * g_cam_zoom, cam_y - point.y * g_cam_zoom}; };

        // Render a polygon shape with its radius expansion.
        auto render_poly = [to_screen, bg](b2Vec2 position, b2Rot rotation, const b2Polygon &poly, ImU32 color, bool show_edge = true) noexcept
        {
            constexpr auto BORDER_PX = 1.5f;

            ImVec2 screen[B2_MAX_POLYGON_VERTICES]{};
            ImVec2 inset[B2_MAX_POLYGON_VERTICES]{};
            auto   border_world = BORDER_PX / g_cam_zoom;
            for (std::int32_t i{}; i < poly.count; ++i)
            {
                // Corner bisector from Box2D's precomputed unit normals.
                auto prev_idx   = (i + poly.count - 1) % poly.count;
                auto bisect     = b2Normalize(poly.normals[prev_idx] + poly.normals[i]);
                auto bisect_dot = b2Dot(bisect, poly.normals[i]);

                // Shrink by linear slop so shapes appear flush when the solver allows slight overlap - see `B2_LINEAR_SLOP`.
                auto radius       = std::max(0.0f, poly.radius - 0.005f * b2GetLengthUnitsPerMeter());
                auto outer_offset = bisect_dot > 0.0f ? radius / bisect_dot : radius;
                auto outer_local  = poly.vertices[i] + bisect * outer_offset;

                screen[i] = to_screen(b2RotateVector(rotation, outer_local) + position);

                // Inset vertex: Same bisector, reduced offset. Gives uniform border_world inset per edge.
                if (show_edge)
                {
                    auto inner_radius = radius - border_world;
                    auto inner_offset = bisect_dot > 0.0f ? inner_radius / bisect_dot : inner_radius;
                    auto inner_local  = poly.vertices[i] + bisect * inner_offset;

                    inset[i] = to_screen(b2RotateVector(rotation, inner_local) + position);
                }
            }

            auto draw_filled = [bg](ImVec2 *verts, std::int32_t count, ImU32 fill_color) noexcept
            {
                if (count == 4)
                {
                    bg->AddQuadFilled(verts[0], verts[1], verts[2], verts[3], fill_color);
                }
                else if (count >= 3)
                {
                    for (std::int32_t i = 1; i < count - 1; ++i)
                    {
                        bg->AddTriangleFilled(verts[0], verts[i], verts[i + 1], fill_color);
                    }
                }
            };

            if (show_edge)
            {
                // Edge rim: Outer fill with AA for smooth silhouette, inner fill without.
                auto edge_color = brighten(color, 60);

                bg->Flags |= ImDrawListFlags_AntiAliasedFill;

                draw_filled(screen, poly.count, edge_color);

                bg->Flags &= ~ImDrawListFlags_AntiAliasedFill;

                draw_filled(inset, poly.count, color);
            }
            else
            {
                draw_filled(screen, poly.count, color);
            }
        };

        auto outline_enabled = g_cam_zoom >= 15.0f;

        // Helper: Get the single polygon shape from a body.
        auto get_poly = [](b2BodyId body) noexcept
        {
            b2ShapeId shape;
            b2Body_GetShapes(body, &shape, 1);

            return b2Shape_GetPolygon(shape);
        };

        // Ground (static, no interpolation).
        render_poly(b2Body_GetPosition(g_ground_id), b2Body_GetRotation(g_ground_id), get_poly(g_ground_id), IM_COL32(102, 102, 102, 255));

        // Compute live box-select AABB for hover highlighting.
        std::optional<b2AABB> select_aabb{};
        if (g_box_selecting)
        {
            auto world_corner_a = screen_to_world(g_box_select_start.x, g_box_select_start.y);
            auto world_corner_b = screen_to_world(mouse_pos.x, mouse_pos.y);
            select_aabb         = b2AABB{b2Min(world_corner_a, world_corner_b), b2Max(world_corner_a, world_corner_b)};
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

        for (auto &&body : g_bodies)
        {
            // Not sleeping.
            b2Vec2 position;
            b2Rot  rotation;
            if (b2Body_IsAwake(body.body))
            {
                auto transform = b2Body_GetTransform(body.body);
                position       = b2Lerp(body.prev.position, transform.p, alpha);
                rotation       = b2NLerp(body.prev.rotation, transform.q, alpha);
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

            // Compute selection state from shape AABB vs selection AABB.
            auto is_selected = body.selected;
            if (!is_selected && select_aabb)
            {
                b2AABB body_aabb{};
                if (type == b2_polygonShape)
                {
                    auto poly  = b2Shape_GetPolygon(body.shape);
                    auto min_x = position.x, min_y = position.y, max_x = position.x, max_y = position.y;
                    for (std::int32_t j{}; j < poly.count; ++j)
                    {
                        auto world = b2RotateVector(rotation, poly.vertices[j]) + position;
                        min_x      = std::min(min_x, world.x);
                        min_y      = std::min(min_y, world.y);
                        max_x      = std::max(max_x, world.x);
                        max_y      = std::max(max_y, world.y);
                    }

                    body_aabb = {{min_x - poly.radius, min_y - poly.radius}, {max_x + poly.radius, max_y + poly.radius}};
                }
                else if (type == b2_circleShape)
                {
                    auto circle = b2Shape_GetCircle(body.shape);
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
                    auto poly  = b2Shape_GetPolygon(body.shape);
                    fill_color = poly.count == 3 ? IM_COL32(50, 200, 50, 255) : IM_COL32(51, 153, 230, 255);
                }

                render_poly(position, rotation, b2Shape_GetPolygon(body.shape), fill_color, outline_enabled);
            }
            else if (type == b2_circleShape)
            {
                if (fill_color == 0)
                {
                    fill_color = IM_COL32(230, 153, 51, 255);
                }

                auto circle        = b2Shape_GetCircle(body.shape);
                auto world_center  = b2RotateVector(rotation, circle.center) + position;
                auto screen_center = to_screen(world_center);
                auto screen_radius = circle.radius * g_cam_zoom;
                if (outline_enabled)
                {
                    auto edge_color = brighten(fill_color, 60);
                    bg->AddCircleFilled(screen_center, screen_radius, edge_color);
                    bg->AddCircleFilled(screen_center, screen_radius - 1.5f, fill_color);

                    auto   inset_radius = screen_radius - 1.5f;
                    ImVec2 edge{screen_center.x + rotation.c * inset_radius, screen_center.y - rotation.s * inset_radius};
                    bg->AddLine(screen_center, edge, edge_color, std::max(2.0f, 3.0f * g_cam_zoom / ZOOM_DEFAULT));
                }
                else
                {
                    bg->AddCircleFilled(screen_center, screen_radius, fill_color);
                }
            }
        }

        // Drawn lines (on top of bodies).
        auto render_smooth_line = [to_screen, bg](const std::vector<b2Vec2> &points, ImU32 color, float thickness) noexcept
        {
            if (points.size() < 2)
            {
                return;
            }

            if (points.size() == 2)
            {
                bg->AddLine(to_screen(points[0]), to_screen(points[1]), color, thickness);

                return;
            }

            bg->PathLineTo(to_screen(points[0]));

            for (std::size_t i{}; i < points.size() - 1; ++i)
            {
                auto prev  = points[i > 0 ? i - 1 : 0];
                auto curr  = points[i];
                auto next  = points[i + 1];
                auto after = points[i + 1 < points.size() - 1 ? i + 2 : points.size() - 1];

                // Catmull-Rom tangents, clamped to prevent overshoot at endpoints and sharp angles.
                auto segment_length = b2Length(next - curr);
                auto prev_length    = i > 0 ? b2Length(curr - prev) : segment_length;
                auto max_tangent    = std::min(segment_length, prev_length) * 0.5f;
                auto tangent_start  = (next - prev) * (1.0f / 6.0f);
                auto tangent_end    = (after - curr) * (1.0f / 6.0f);

                auto tangent_start_length = b2Length(tangent_start);
                if (tangent_start_length > max_tangent)
                {
                    tangent_start = tangent_start * (max_tangent / tangent_start_length);
                }

                auto next_length        = i + 1 < points.size() - 1 ? b2Length(after - next) : segment_length;
                auto max_tangent_end    = std::min(segment_length, next_length) * 0.5f;
                auto tangent_end_length = b2Length(tangent_end);
                if (tangent_end_length > max_tangent_end)
                {
                    tangent_end = tangent_end * (max_tangent_end / tangent_end_length);
                }

                auto ctrl1 = to_screen(curr + tangent_start);
                auto ctrl2 = to_screen(next - tangent_end);
                auto end   = to_screen(next);

                bg->PathBezierCubicCurveTo(ctrl1, ctrl2, end);
            }

            bg->PathStroke(color, 0, thickness);
        };

        auto line_thickness = std::max(2.0f, 3.0f * g_cam_zoom / ZOOM_DEFAULT);
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
            constexpr auto ROPE_COLOR     = IM_COL32(194, 154, 108, 255);
            constexpr auto ROPE_OUTLINE   = IM_COL32(120, 90, 60, 255);
            auto           rope_thickness = std::max(2.0f, 4.0f * g_cam_zoom / ZOOM_DEFAULT);

            // Prune ropes whose anchor body was destroyed.
            std::erase_if(
                g_ropes,
                [](const Rope &rope)
                {
                    auto a_dead = B2_IS_NON_NULL(rope.body_a) && !b2Body_IsValid(rope.body_a);
                    auto b_dead = B2_IS_NON_NULL(rope.body_b) && !b2Body_IsValid(rope.body_b);
                    if (!a_dead && !b_dead)
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
                // Build point list: Anchor A, segment centers, anchor B.
                std::vector<b2Vec2> points{};
                points.reserve(rope.segments.size() + 2);

                if (B2_IS_NON_NULL(rope.body_a))
                {
                    points.emplace_back(b2Body_GetWorldPoint(rope.body_a, rope.local_a));
                }

                for (auto &&segment : rope.segments)
                {
                    points.emplace_back(b2Body_GetPosition(segment));
                }

                if (B2_IS_NON_NULL(rope.body_b))
                {
                    points.emplace_back(b2Body_GetWorldPoint(rope.body_b, rope.local_b));
                }

                if (points.size() < 2)
                {
                    continue;
                }

                // Draw outline then fill using the Catmull-Rom path renderer.
                render_smooth_line(points, ROPE_OUTLINE, rope_thickness + std::max(2.0f, 3.0f * std::sqrt(g_cam_zoom / ZOOM_DEFAULT)));
                render_smooth_line(points, ROPE_COLOR, rope_thickness);
            }

            // Pending rope: Line from start anchor to cursor.
            if (B2_IS_NON_NULL(g_rope_start_body))
            {
                auto world_a      = b2Body_GetWorldPoint(g_rope_start_body, g_rope_start_anchor);
                auto screen_start = to_screen(world_a);
                auto cursor       = ImGui::GetMousePos();
                bg->AddLine(screen_start, cursor, IM_COL32(194, 154, 108, 128), rope_thickness);
            }
        }

        // Pins.
        {
            std::erase_if(g_pins, [](const Pin &pin) noexcept { return !b2Joint_IsValid(pin.joint); });

            auto zoom_ratio = g_cam_zoom / ZOOM_DEFAULT;
            auto radius     = std::clamp(8.0f * std::sqrt(zoom_ratio), 4.0f, 20.0f);
            auto slot       = radius * 0.55f;
            auto thickness  = std::clamp(1.5f * std::sqrt(zoom_ratio), 1.0f, 4.0f);
            for (auto &&pin : g_pins)
            {
                auto world_pos  = b2Body_GetWorldPoint(pin.body, b2Vec2_zero);
                auto screen_pos = to_screen(world_pos);

                // Head.
                bg->AddCircleFilled(screen_pos, radius, IM_COL32(140, 140, 140, 255));
                bg->AddCircle(screen_pos, radius, IM_COL32(90, 90, 90, 255), 0, thickness);

                // Phillips cross.
                bg->AddLine({screen_pos.x - slot, screen_pos.y}, {screen_pos.x + slot, screen_pos.y}, IM_COL32(60, 60, 60, 200), thickness);
                bg->AddLine({screen_pos.x, screen_pos.y - slot}, {screen_pos.x, screen_pos.y + slot}, IM_COL32(60, 60, 60, 200), thickness);
            }
        }

        // Emitters (on top of everything).
        for (std::size_t i{}; i < g_emitters.size(); ++i)
        {
            auto &&emitter       = g_emitters[i];
            auto   screen_center = to_screen(emitter.position);
            auto   arrow_length  = 0.8f * g_cam_zoom;
            auto   thickness     = std::max(1.5f, 2.0f * g_cam_zoom / ZOOM_DEFAULT);
            ImVec2 arrow_dir{std::cos(emitter.angle) * arrow_length, -std::sin(emitter.angle) * arrow_length};
            auto   tip = screen_center + arrow_dir;
            ImVec2 perp{-arrow_dir.y * 0.2f, arrow_dir.x * 0.2f};
            auto   base = screen_center + arrow_dir * 0.65f;

            auto color = emitter.active ? IM_COL32(100, 255, 100, 255) : IM_COL32(255, 100, 100, 255);
            bg->AddLine(screen_center, base, color, thickness);
            bg->AddTriangleFilled(tip, base + perp, base - perp, color);
            bg->AddCircleFilled(screen_center, std::max(3.0f, 0.1f * g_cam_zoom), color);

            // Label on foreground so it's always visible.
            char label[24];
            std::snprintf(label, sizeof(label), "e%zu", i);
            auto label_pos = screen_center + ImVec2{-8, -24};
            fg->AddText(label_pos, IM_COL32(255, 255, 255, 200), label);
        }

        bg->Flags = old_flags;
    }

    auto render_sample = (float)((double)(SDL_GetPerformanceCounter() - render_t0) / freq * 1000.0);
    g_render_ms        = g_render_ms + 0.01f * (render_sample - g_render_ms);

    auto present_t0 = SDL_GetPerformanceCounter();

    ImGui::Render();

    auto *draw_data = ImGui::GetDrawData();
    g_total_vtx     = draw_data->TotalVtxCount;
    g_total_idx     = draw_data->TotalIdxCount;

    SDL_SetRenderScale(g_renderer, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);

    constexpr ImVec4 CLEAR_COLOR(0.10f, 0.10f, 0.10f, 1.00f);
    SDL_SetRenderDrawColorFloat(g_renderer, CLEAR_COLOR.x, CLEAR_COLOR.y, CLEAR_COLOR.z, CLEAR_COLOR.w);

    SDL_RenderClear(g_renderer);
    ImGui_ImplSDLRenderer3_RenderDrawData(draw_data, g_renderer);
    SDL_RenderPresent(g_renderer);

    auto present_sample = (float)((double)(SDL_GetPerformanceCounter() - present_t0) / freq * 1000.0);
    g_present_ms        = g_present_ms + 0.01f * (present_sample - g_present_ms);

    auto frame_sample = (float)((double)(SDL_GetPerformanceCounter() - phys_t0) / freq * 1000.0);
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

        // Apply new style stuff.
        style.FontSizeBase = old_style.FontSizeBase;
        style.ScaleAllSizes(g_dpi_scaling);
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
            g_cam_dragging = true;
        }
        // Left-click.
        else if (event->button.button == SDL_BUTTON_LEFT)
        {
            auto world_point = screen_to_world(event->button.x, event->button.y);

            // Shift+Left-click: Rope linking.
            if ((SDL_GetModState() & SDL_KMOD_SHIFT) != 0)
            {
                auto hit = find_body_at(world_point);
                if (B2_IS_NON_NULL(hit))
                {
                    if (B2_IS_NON_NULL(g_rope_start_body))
                    {
                        // Second click: Create rope chain between the two bodies.
                        if (hit.index1 != g_rope_start_body.index1)
                        {
                            constexpr auto SEG_RADIUS  = 0.06f;
                            constexpr auto SEG_SPACING = 0.15f;
                            auto           half_len    = SEG_SPACING * 0.6f; // > 0.5 so adjacent capsules overlap.

                            auto anchor_b  = b2Body_GetLocalPoint(hit, world_point);
                            auto world_a   = b2Body_GetWorldPoint(g_rope_start_body, g_rope_start_anchor);
                            auto world_b   = world_point;
                            auto dist      = b2Distance(world_a, world_b);
                            auto seg_count = std::max(2, (std::int32_t)(dist / SEG_SPACING));

                            // Create segment bodies along the line.
                            auto body_def           = b2DefaultBodyDef();
                            body_def.type           = b2_dynamicBody;
                            body_def.gravityScale   = 1.0f;
                            body_def.linearDamping  = 0.5f;
                            body_def.angularDamping = 2.0f;

                            auto shape_def              = b2DefaultShapeDef();
                            shape_def.density           = 0.5f;
                            shape_def.material.friction = 0.6f;

                            // Capsule: Pill shape aligned vertically, tips at +/- half_len.
                            b2Capsule capsule{{0.0f, -half_len}, {0.0f, half_len}, SEG_RADIUS};

                            // Orient all capsules along the rope direction.
                            auto rope_dir     = b2Normalize(world_b - world_a);
                            body_def.rotation = b2MakeRot(std::atan2(rope_dir.y, rope_dir.x) - PI / 2.0f);

                            Rope rope{};
                            for (std::int32_t i{}; i < seg_count; ++i)
                            {
                                auto t            = (float)(i + 1) / (float)(seg_count + 1);
                                body_def.position = b2Lerp(world_a, world_b, t);

                                auto segment = b2CreateBody(g_world, &body_def);
                                b2CreateCapsuleShape(segment, &shape_def, &capsule);

                                rope.segments.emplace_back(segment);
                            }

                            // Revolute joints: Connect capsule tips end-to-end.
                            // Anchor A -> First segment bottom tip.
                            {
                                auto rev_def         = b2DefaultRevoluteJointDef();
                                rev_def.bodyIdA      = g_rope_start_body;
                                rev_def.bodyIdB      = rope.segments.front();
                                rev_def.localAnchorA = g_rope_start_anchor;
                                rev_def.localAnchorB = {0.0f, -half_len};

                                rope.joints.emplace_back(b2CreateRevoluteJoint(g_world, &rev_def));
                            }

                            // Segment-to-segment: Top tip of [i] -> bottom tip of [i+1].
                            for (std::int32_t i{}; i < seg_count - 1; ++i)
                            {
                                auto rev_def         = b2DefaultRevoluteJointDef();
                                rev_def.bodyIdA      = rope.segments[i];
                                rev_def.bodyIdB      = rope.segments[i + 1];
                                rev_def.localAnchorA = {0.0f, half_len};
                                rev_def.localAnchorB = {0.0f, -half_len};

                                rope.joints.emplace_back(b2CreateRevoluteJoint(g_world, &rev_def));
                            }

                            // Last segment top tip -> Anchor B.
                            {
                                auto rev_def         = b2DefaultRevoluteJointDef();
                                rev_def.bodyIdA      = rope.segments.back();
                                rev_def.bodyIdB      = hit;
                                rev_def.localAnchorA = {0.0f, half_len};
                                rev_def.localAnchorB = anchor_b;

                                rope.joints.emplace_back(b2CreateRevoluteJoint(g_world, &rev_def));
                            }

                            // Disable collision between rope segments and their anchor bodies.
                            for (auto &&segment : rope.segments)
                            {
                                auto filter_def_a    = b2DefaultFilterJointDef();
                                filter_def_a.bodyIdA = g_rope_start_body;
                                filter_def_a.bodyIdB = segment;

                                rope.joints.emplace_back(b2CreateFilterJoint(g_world, &filter_def_a));

                                auto filter_def_b    = b2DefaultFilterJointDef();
                                filter_def_b.bodyIdA = hit;
                                filter_def_b.bodyIdB = segment;

                                rope.joints.emplace_back(b2CreateFilterJoint(g_world, &filter_def_b));
                            }

                            rope.body_a  = g_rope_start_body;
                            rope.body_b  = hit;
                            rope.local_a = g_rope_start_anchor;
                            rope.local_b = anchor_b;

                            g_ropes.emplace_back(std::move(rope));
                        }

                        g_rope_start_body = b2_nullBodyId;
                    }
                    else
                    {
                        // First click: Start linking.
                        g_rope_start_body   = hit;
                        g_rope_start_anchor = b2Body_GetLocalPoint(hit, world_point);
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
                for (std::int32_t i{}; i < (std::int32_t)g_emitters.size(); ++i)
                {
                    if (b2LengthSquared(world_point - g_emitters[i].position) <= EMITTER_GRAB_RADIUS * EMITTER_GRAB_RADIUS)
                    {
                        g_dragged_emitter = i;

                        break;
                    }
                }

                // If no emitter hit, try grabbing a body.
                if (!g_dragged_emitter)
                {
                    g_mouse_body = find_body_at(world_point);
                    if (B2_IS_NON_NULL(g_mouse_body))
                    {
                        b2Body_SetAwake(g_mouse_body, true);

                        auto mouse_def         = b2DefaultMouseJointDef();
                        mouse_def.bodyIdA      = g_ground_id;
                        mouse_def.bodyIdB      = g_mouse_body;
                        mouse_def.target       = world_point;
                        mouse_def.maxForce     = 10000.0f * b2Body_GetMass(g_mouse_body);
                        mouse_def.hertz        = 60.0f;
                        mouse_def.dampingRatio = 1.0f;

                        g_mouse_joint = b2CreateMouseJoint(g_world, &mouse_def);

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
                auto position        = b2Body_GetPosition(g_mouse_body);
                auto rev_def         = b2DefaultRevoluteJointDef();
                rev_def.bodyIdA      = g_ground_id;
                rev_def.bodyIdB      = g_mouse_body;
                rev_def.localAnchorA = b2Body_GetLocalPoint(g_ground_id, position);
                rev_def.localAnchorB = b2Vec2_zero;

                g_pins.emplace_back(b2CreateRevoluteJoint(g_world, &rev_def), g_mouse_body);

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
                g_cutting = true;
            }
            else if ((SDL_GetModState() & SDL_KMOD_CTRL) != 0)
            {
                // Ctrl+Right-click on a pinned body: Unpin it.
                auto world_point = screen_to_world(event->button.x, event->button.y);
                auto hit_body    = find_body_at(world_point);
                auto pin_it =
                    std::find_if(g_pins.begin(), g_pins.end(), [hit_body](const Pin &pin) noexcept { return B2_ID_EQUALS(pin.body, hit_body); });
                if (B2_IS_NON_NULL(hit_body) && pin_it != g_pins.end())
                {
                    b2DestroyJoint(pin_it->joint);

                    g_pins.erase(pin_it);

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
            g_cam_dragging = false;
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
                g_dragged_emitter.reset();
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

            auto drag = ImVec2{event->button.x, event->button.y} - g_box_select_start;

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
            else if (g_cutting)
            {
                g_cutting = false;
            }
            // Click, not drag: Toggle selection on body under cursor.
            else if (drag.x * drag.x + drag.y * drag.y < 25.0f)
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

                auto                  world_corner_a = screen_to_world(g_box_select_start.x, g_box_select_start.y);
                auto                  world_corner_b = screen_to_world(g_box_select_start.x + drag.x, g_box_select_start.y + drag.y);
                b2AABB                aabb{b2Min(world_corner_a, world_corner_b), b2Max(world_corner_a, world_corner_b)};
                std::vector<b2BodyId> in_box{};
                b2World_OverlapAABB(
                    g_world, aabb, b2DefaultQueryFilter(),
                    [](b2ShapeId shapeId, void *context)
                    {
                        auto body = b2Shape_GetBody(shapeId);
                        if (b2Body_GetType(body) == b2_dynamicBody)
                        {
                            auto &vec = *(std::vector<b2BodyId> *)context;
                            vec.emplace_back(body);
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
        if (g_cam_dragging)
        {
            g_cam_center.x -= event->motion.xrel / g_cam_zoom;
            g_cam_center.y += event->motion.yrel / g_cam_zoom;
        }
        else if (g_dragged_emitter)
        {
            auto world_point                        = screen_to_world(event->motion.x, event->motion.y);
            g_emitters[*g_dragged_emitter].position = world_point;
        }
        else if (B2_IS_NON_NULL(g_mouse_joint))
        {
            auto world_point = screen_to_world(event->motion.x, event->motion.y);
            b2MouseJoint_SetTarget(g_mouse_joint, world_point);
        }
        else if (g_erasing)
        {
            constexpr auto HIT_DIST_SQUARED = 0.15f * 0.15f;

            auto world_point = screen_to_world(event->motion.x, event->motion.y);
            for (std::size_t i{}; i < g_drawn_lines.size(); ++i)
            {
                auto                      &line = g_drawn_lines[i];
                std::optional<std::size_t> hit_seg{};
                for (std::size_t j{}; j < line.points.size() - 1; ++j)
                {
                    if (segment_distance_squared(world_point, line.points[j], line.points[j + 1]) <= HIT_DIST_SQUARED)
                    {
                        hit_seg = j + 1;

                        break;
                    }
                }

                if (!hit_seg)
                {
                    continue;
                }

                // Destroy old body.
                b2DestroyBody(line.body);

                // Split into left [0..hit_seg-1] and right [hit_seg..end].
                auto       &pts = line.points;
                std::vector left(pts.begin(), pts.begin() + (std::ptrdiff_t)*hit_seg);
                std::vector right(pts.begin() + (std::ptrdiff_t)*hit_seg, pts.end());

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
                auto &&rope = g_ropes[i];

                std::vector<b2Vec2> points{};
                points.reserve(rope.segments.size() + 2);

                if (B2_IS_NON_NULL(rope.body_a))
                {
                    points.emplace_back(b2Body_GetWorldPoint(rope.body_a, rope.local_a));
                }

                for (auto &&segment : rope.segments)
                {
                    points.emplace_back(b2Body_GetPosition(segment));
                }

                if (B2_IS_NON_NULL(rope.body_b))
                {
                    points.emplace_back(b2Body_GetWorldPoint(rope.body_b, rope.local_b));
                }

                if (points.size() < 2)
                {
                    continue;
                }

                bool hit{};
                for (std::size_t j{}; j < points.size() - 1; ++j)
                {
                    if (segment_distance_squared(world_point, points[j], points[j + 1]) <= HIT_DIST_SQUARED)
                    {
                        hit = true;

                        break;
                    }
                }

                if (!hit)
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
            constexpr auto HIT_DIST_SQUARED = 0.15f * 0.15f;

            auto world_point = screen_to_world(event->motion.x, event->motion.y);
            auto rope_count  = g_ropes.size(); // Snapshot - don't iterate newly appended halves.
            for (std::size_t i{}; i < rope_count; ++i)
            {
                auto &rope      = g_ropes[i];
                auto  seg_count = (std::int32_t)rope.segments.size();
                auto  body_a    = rope.body_a;
                auto  body_b    = rope.body_b;
                auto  anchor_a  = rope.local_a;
                auto  anchor_b  = rope.local_b;
                auto  has_a     = B2_IS_NON_NULL(body_a);
                auto  has_b     = B2_IS_NON_NULL(body_b);

                // Build point list matching render layout.
                std::vector<b2Vec2> points{};
                points.reserve(rope.segments.size() + 2);

                if (has_a)
                {
                    points.emplace_back(b2Body_GetWorldPoint(body_a, anchor_a));
                }

                for (auto &&segment : rope.segments)
                {
                    points.emplace_back(b2Body_GetPosition(segment));
                }

                if (has_b)
                {
                    points.emplace_back(b2Body_GetWorldPoint(body_b, anchor_b));
                }

                if (points.size() < 2)
                {
                    continue;
                }

                // Test cursor against each link.
                std::optional<std::int32_t> hit_link{};
                for (std::size_t j{}; j < points.size() - 1; ++j)
                {
                    if (segment_distance_squared(world_point, points[j], points[j + 1]) <= HIT_DIST_SQUARED)
                    {
                        hit_link = (std::int32_t)j;

                        break;
                    }
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
                // points: [anchor_a?] [seg_0 .. seg_(N-1)] [anchor_b?]
                // `cut_seg` = First segment on the right side of the cut.
                auto cut_seg     = has_a ? *hit_link : *hit_link + 1;
                auto left_count  = cut_seg;
                auto right_count = seg_count - cut_seg;

                constexpr auto HALF_LEN = 0.15f * 0.5f;

                // Helper: Create a revolute joint between two bodies.
                auto make_rev = [](b2BodyId body_from, b2Vec2 local_from, b2BodyId body_to, b2Vec2 local_to)
                {
                    auto rev_def         = b2DefaultRevoluteJointDef();
                    rev_def.bodyIdA      = body_from;
                    rev_def.bodyIdB      = body_to;
                    rev_def.localAnchorA = local_from;
                    rev_def.localAnchorB = local_to;

                    return b2CreateRevoluteJoint(g_world, &rev_def);
                };

                // Wire a sub-rope from existing segment bodies and push to g_ropes.
                // `anchor_first`: true = `anchor` is `body_a` (left half), false = `body_b` (right half).
                auto wire_half = [&make_rev](b2BodyId anchor, b2Vec2 anchor_local, b2BodyId *segs, std::int32_t count, bool anchor_first)
                {
                    Rope half{};
                    half.segments.assign(segs, segs + count);

                    auto has_anchor = B2_IS_NON_NULL(anchor);

                    if (anchor_first)
                    {
                        if (has_anchor)
                        {
                            half.body_a  = anchor;
                            half.local_a = anchor_local;
                            half.joints.emplace_back(make_rev(anchor, anchor_local, half.segments.front(), {0.0f, -HALF_LEN}));
                        }

                        for (std::int32_t j{}; j < count - 1; ++j)
                        {
                            half.joints.emplace_back(make_rev(half.segments[j], {0.0f, HALF_LEN}, half.segments[j + 1], {0.0f, -HALF_LEN}));
                        }
                    }
                    else
                    {
                        for (std::int32_t j{}; j < count - 1; ++j)
                        {
                            half.joints.emplace_back(make_rev(half.segments[j], {0.0f, HALF_LEN}, half.segments[j + 1], {0.0f, -HALF_LEN}));
                        }

                        if (has_anchor)
                        {
                            half.body_b  = anchor;
                            half.local_b = anchor_local;
                            half.joints.emplace_back(make_rev(half.segments.back(), {0.0f, HALF_LEN}, anchor, anchor_local));
                        }
                    }

                    if (has_anchor)
                    {
                        for (std::int32_t j{}; j < count; ++j)
                        {
                            auto filter_def    = b2DefaultFilterJointDef();
                            filter_def.bodyIdA = anchor;
                            filter_def.bodyIdB = half.segments[j];

                            half.joints.emplace_back(b2CreateFilterJoint(g_world, &filter_def));
                        }
                    }

                    g_ropes.emplace_back(std::move(half));
                };

                // Move segments out - wire_half pushes to g_ropes which may invalidate `rope`.
                auto all_segments = std::move(rope.segments);

                if (left_count > 0)
                {
                    wire_half(body_a, anchor_a, all_segments.data(), left_count, true);
                }

                if (right_count > 0)
                {
                    wire_half(body_b, anchor_b, all_segments.data() + cut_seg, right_count, false);
                }

                // Remove the original rope (joints already destroyed, segments moved to halves).
                g_ropes.erase(g_ropes.begin() + (std::ptrdiff_t)i);

                --i;
                --rope_count;
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
            auto mx = event->wheel.mouse_x;
            auto my = event->wheel.mouse_y;

            // World position under cursor before zoom.
            auto world_x = g_cam_center.x + (mx - (float)g_window_w / 2.0f) / g_cam_zoom;
            auto world_y = g_cam_center.y - (my - (float)g_window_h / 2.0f) / g_cam_zoom;

            // Apply zoom.
            auto factor = event->wheel.y > 0.0f ? ZOOM_FACTOR : 1.0f / ZOOM_FACTOR;
            g_cam_zoom  = std::clamp(g_cam_zoom * factor, g_cam_zoom_min, ZOOM_MAX);

            // Adjust center so world point stays under cursor.
            g_cam_center.x = world_x - (mx - (float)g_window_w / 2.0f) / g_cam_zoom;
            g_cam_center.y = world_y + (my - (float)g_window_h / 2.0f) / g_cam_zoom;
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

    if (auto *bd = ImGui::GetCurrentContext(); bd != nullptr)
    {
        ImGui_ImplSDLRenderer3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
    }

    SDL_DestroyRenderer(g_renderer);
    SDL_DestroyWindow(g_window);

    // `SDL_Quit()` will get called for us after this returns.
}
