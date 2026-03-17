#include <thread_pool/thread_pool.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>
#include <box2d/box2d.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <format>
#include <latch>
#include <memory>
#include <new>
#include <thread>
#include <utility>
#include <vector>

enum class SpawnShape : std::uint8_t
{
    Box = 0,
    Circle,
    Triangle
};

// Constants.
constexpr auto         PI              = 3.141592653589793f;
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
std::int32_t  g_window_w{};
std::int32_t  g_window_h{};
SDL_Window   *g_window{};
SDL_Renderer *g_renderer{};
const char   *g_renderer_name{};

// Threading.
std::uint32_t                      g_worker_count{};
std::unique_ptr<dp::thread_pool<>> g_thread_pool{};
std::atomic<std::uint32_t>         g_next_worker{};
std::int32_t                       g_task_count{};

// Physics.
b2WorldId    g_world{};
float        g_physics_accumulator{};
float        g_physics_alpha{};
bool         g_paused{};
bool         g_single_step{};
std::int32_t g_step_count = 1;
std::int32_t g_sub_steps  = 4;
std::size_t  g_culled_count{};

// Camera.
b2Vec2 g_cam_center{0.0f, 2.0f};
float  g_cam_zoom{ZOOM_DEFAULT};
auto   g_cam_zoom_min = 1.0f;
bool   g_cam_dragging{};

// Bodies.
b2BodyId g_ground_id{};

struct BodyState
{
    b2Vec2 position{};
    b2Rot  rotation{};
};

struct PhysBody
{
    b2BodyId    body{};
    b2ShapeId   shape{};
    b2ShapeType shape_type{};
    BodyState   prev{};
    bool        selected{};
};

std::vector<PhysBody> g_bodies{};

// Drawing.
struct DrawnLine
{
    b2BodyId            body{};
    std::vector<b2Vec2> points{};
};

std::vector<DrawnLine> g_drawn_lines{};
std::vector<b2Vec2>    g_current_stroke{};

// Emitters.
struct Emitter
{
    b2Vec2     position{};
    float      angle{};      // Radians, 0 = right.
    float      speed{15.0f}; // m/s.
    float      rate{3.0f};   // Bodies per second.
    float      timer{};
    SpawnShape shape{SpawnShape::Box};
    bool       active{};
};

std::vector<Emitter> g_emitters{};

// Interaction.
b2JointId    g_mouse_joint = b2_nullJointId;
b2BodyId     g_mouse_body  = b2_nullBodyId;
bool         g_box_selecting{};
ImVec2       g_box_select_start{};
std::int32_t g_dragged_emitter = -1;
bool         g_erasing{};

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

void reset_tasks() noexcept
{
    for (std::int32_t i{}; i < g_task_count; ++i)
    {
        auto *handle = std::launder((TaskHandle *)&g_task_storage[i]);
        handle->~TaskHandle();
    }

    g_task_count = 0;
}

b2Vec2 screen_to_world(float sx, float sy) noexcept
{
    return {
        g_cam_center.x + (sx - (float)g_window_w / 2.0f) / g_cam_zoom,
        g_cam_center.y - (sy - (float)g_window_h / 2.0f) / g_cam_zoom,
    };
}

float point_to_segment_dist_sq(b2Vec2 point, b2Vec2 seg_a, b2Vec2 seg_b) noexcept
{
    auto dx     = seg_b.x - seg_a.x;
    auto dy     = seg_b.y - seg_a.y;
    auto len_sq = dx * dx + dy * dy;

    float t{};
    if (len_sq > 0.0f)
    {
        t = std::clamp(((point.x - seg_a.x) * dx + (point.y - seg_a.y) * dy) / len_sq, 0.0f, 1.0f);
    }

    auto closest_x = seg_a.x + t * dx;
    auto closest_y = seg_a.y + t * dy;
    auto px        = point.x - closest_x;
    auto py        = point.y - closest_y;

    return px * px + py * py;
}

b2BodyId find_body_at(b2Vec2 world_pt)
{
    auto proxy = b2MakeProxy(&world_pt, 1, 0.01f);

    b2BodyId result{};
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

void add_box(b2Vec2 position, float hw, float hh)
{
    auto body_def     = b2DefaultBodyDef();
    body_def.type     = b2_dynamicBody;
    body_def.position = position;

    auto body = b2CreateBody(g_world, &body_def);

    auto shape_def              = b2DefaultShapeDef();
    shape_def.density           = 1.0f;
    shape_def.material.friction = 0.3f;

    auto box      = b2MakeBox(hw, hh);
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

DrawnLine create_drawn_line(std::vector<b2Vec2> points)
{
    auto body_def = b2DefaultBodyDef();
    auto body     = b2CreateBody(g_world, &body_def);

    auto shape_def              = b2DefaultShapeDef();
    shape_def.material.friction = 0.5f;

    for (std::size_t i = 1; i < points.size(); ++i)
    {
        b2Segment segment{points[i - 1], points[i]};
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
            }

            // Apply velocity to the just-spawned body.
            b2Vec2 dir{std::cos(emitter.angle), std::sin(emitter.angle)};
            auto  &spawned = g_bodies.back();
            b2Body_SetLinearVelocity(spawned.body, {dir.x * emitter.speed, dir.y * emitter.speed});
        }
    }
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

        tick_emitters(PHYSICS_DT * g_step_count);

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

void *box2d_enqueue_task(b2TaskCallback *task, std::int32_t itemCount, std::int32_t minRange, void *taskContext, void *userContext)
{
    auto &pool = *(dp::thread_pool<> *)userContext;

    auto chunk_size  = std::max(minRange, itemCount / (std::int32_t)g_worker_count);
    auto chunk_count = (itemCount + chunk_size - 1) / chunk_size;

    // Fallback: Run serially. `nullptr` tells Box2D to skip finishTask.
    if (g_task_count >= MAX_TASKS)
    {
        task(0, itemCount, 0, taskContext);

        return nullptr;
    }

    auto *handle = new(&g_task_storage[g_task_count]) TaskHandle(chunk_count);
    ++g_task_count;

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

void box2d_finish_task(void *userTask, void *userContext) noexcept
{
    if (userTask == nullptr)
    {
        return;
    }

    ((TaskHandle *)userTask)->done.wait();
}

SDL_AppResult SDL_AppInit(void **appstate, std::int32_t argc, char *argv[])
{
    // Box2D is compiled with AVX2; bail early on unsupported CPUs.
    if (!SDL_HasAVX2())
    {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "phys", "This application requires a CPU with AVX2 support.", nullptr);

        return SDL_APP_FAILURE;
    }

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

    g_dpi_scaling = SDL_GetWindowDisplayScale(g_window);
    if (g_dpi_scaling == 0.0f)
    {
        SDL_ShowSimpleMessageBox(
            SDL_MESSAGEBOX_ERROR, "phys", std::format("Error `SDL_GetWindowDisplayScale()`: {}", SDL_GetError()).c_str(), nullptr);

        return SDL_APP_FAILURE;
    }

    SDL_SetHint(SDL_HINT_RENDER_DRIVER, "direct3d12,direct3d11,vulkan,opengl,software");

    g_renderer = SDL_CreateRenderer(g_window, nullptr);
    if (g_renderer == nullptr)
    {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "phys", std::format("Error `SDL_CreateRenderer()`: {}", SDL_GetError()).c_str(), nullptr);

        return SDL_APP_FAILURE;
    }

    // SDL_SetRenderVSync(g_renderer, SDL_RENDERER_VSYNC_ADAPTIVE)
    if (!SDL_SetRenderVSync(g_renderer, 1))
    {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "phys", std::format("Error `SDL_SetRenderVSync()`: {}", SDL_GetError()).c_str(), nullptr);

        return SDL_APP_FAILURE;
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

    // Create thread pool.
    auto num_threads = std::clamp<std::uint32_t>(std::thread::hardware_concurrency(), 1, 16);
    g_worker_count   = num_threads <= 2 ? num_threads : num_threads - 1;
    g_thread_pool    = std::make_unique<dp::thread_pool<>>(g_worker_count);

    // Create world.
    auto world_def            = b2DefaultWorldDef();
    world_def.workerCount     = g_worker_count;
    world_def.enqueueTask     = box2d_enqueue_task;
    world_def.finishTask      = box2d_finish_task;
    world_def.userTaskContext = g_thread_pool.get();

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

SDL_AppResult SDL_AppIterate(void *appstate)
{
    if ((SDL_GetWindowFlags(g_window) & SDL_WINDOW_MINIMIZED) != 0)
    {
        SDL_Delay(50);

        return SDL_APP_CONTINUE;
    }

    SDL_GetWindowSize(g_window, &g_window_w, &g_window_h);

    // Clamp camera to playground bounds.
    auto area_w    = AREA_MAX_X - AREA_MIN_X;
    auto area_h    = AREA_MAX_Y - AREA_MIN_Y;
    g_cam_zoom_min = std::min((float)g_window_w / area_w, (float)g_window_h / area_h);
    g_cam_zoom     = std::clamp(g_cam_zoom, g_cam_zoom_min, ZOOM_MAX);

    auto half_vw   = (float)g_window_w / 2.0f / g_cam_zoom;
    auto half_vh   = (float)g_window_h / 2.0f / g_cam_zoom;
    auto center_x  = (AREA_MIN_X + AREA_MAX_X) / 2.0f;
    auto center_y  = (AREA_MIN_Y + AREA_MAX_Y) / 2.0f;
    auto clamp_x   = std::max(0.0f, area_w / 2.0f - half_vw);
    auto clamp_y   = std::max(0.0f, area_h / 2.0f - half_vh);
    g_cam_center.x = std::clamp(g_cam_center.x, center_x - clamp_x, center_x + clamp_x);
    g_cam_center.y = std::clamp(g_cam_center.y, center_y - clamp_y, center_y + clamp_y);

    auto &io = ImGui::GetIO();
    step_physics(io.DeltaTime);

    // Kill bounds: Actual visible area at max zoom-out + margin.
    auto kill_half_w = (float)g_window_w / 2.0f / g_cam_zoom_min + 5.0f;
    auto kill_half_h = (float)g_window_h / 2.0f / g_cam_zoom_min + 5.0f;
    auto kill_cx     = (AREA_MIN_X + AREA_MAX_X) / 2.0f;
    auto kill_cy     = (AREA_MIN_Y + AREA_MAX_Y) / 2.0f;

    // Destroy bodies that fall outside the playground.
    std::erase_if(
        g_bodies,
        [=](const PhysBody &body)
        {
            auto pos = b2Body_GetPosition(body.body);
            if (pos.x < kill_cx - kill_half_w || pos.x > kill_cx + kill_half_w || pos.y < kill_cy - kill_half_h || pos.y > kill_cy + kill_half_h)
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
        std::int64_t selected_count{};
        for (auto &&body : g_bodies)
        {
            if (body.selected)
            {
                ++selected_count;
            }
        }

        ImGui::Text("%lld selected", selected_count);

        ImGui::BeginDisabled(selected_count == 0);
        if (ImGui::Button("Delete Selected"))
        {
            for (auto &&body : g_bodies)
            {
                if (body.selected)
                {
                    b2DestroyBody(body.body);
                }
            }

            std::erase_if(g_bodies, [](const PhysBody &body) { return body.selected; });
        }
        ImGui::EndDisabled();

        ImGui::SeparatorText("Emitters");
        if (ImGui::Button("Add Emitter"))
        {
            g_emitters.emplace_back(g_cam_center);
        }

        for (std::size_t i{}; i < g_emitters.size(); ++i)
        {
            ImGui::PushID((std::int32_t)i);
            auto &emitter = g_emitters[i];

            ImGui::Checkbox("##active", &emitter.active);
            ImGui::SameLine();
            ImGui::Text("Emitter %zu", i);

            auto angle_deg = emitter.angle * RAD_TO_DEG;
            if (ImGui::SliderFloat("Angle", &angle_deg, -180.0f, 180.0f, "%.0f deg"))
            {
                emitter.angle = angle_deg * DEG_TO_RAD;
            }

            ImGui::SliderFloat("Speed", &emitter.speed, 1.0f, 50.0f, "%.1f m/s");
            ImGui::SliderFloat("Rate", &emitter.rate, 0.5f, 20.0f, "%.1f /s");

            auto shape_idx = (std::int32_t)emitter.shape;
            if (ImGui::Combo("Shape", &shape_idx, "Box\0Circle\0Triangle\0"))
            {
                emitter.shape = (SpawnShape)shape_idx;
            }

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

        ImGui::SeparatorText("World");
        auto gravity = b2World_GetGravity(g_world);
        if (ImGui::SliderFloat("Gravity", &gravity.y, -50.0f, 50.0f, "%.1f m/s^2"))
        {
            b2World_SetGravity(g_world, gravity);
        }

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

        ImGui::SliderInt("Step count", &g_step_count, 1, 100);
        ImGui::EndDisabled();
        ImGui::SliderInt("Sub-steps", &g_sub_steps, 1, 8);

        ImGui::SeparatorText("Info");
        ImGui::Text("Bodies: %zu (%d awake, %zu culled)", g_bodies.size(), b2World_GetAwakeBodyCount(g_world), g_culled_count);
        ImGui::Text("Workers: %u", g_worker_count);

        auto counters = b2World_GetCounters(g_world);
        ImGui::Text("Islands: %d", counters.islandCount);
        ImGui::Text("Contacts: %d", counters.contactCount);

        ImGui::SeparatorText("Controls");
        ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "Left-click drag");
        ImGui::SameLine();
        ImGui::Text("Move body");
        ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "Ctrl+Left-click drag");
        ImGui::SameLine();
        ImGui::Text("Draw line");
        ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "Ctrl+Right-click drag");
        ImGui::SameLine();
        ImGui::Text("Erase lines");
        ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "Right-click");
        ImGui::SameLine();
        ImGui::Text("Select/deselect body");
        ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "Right-click drag");
        ImGui::SameLine();
        ImGui::Text("Box select");
        ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "Shift+Right drag");
        ImGui::SameLine();
        ImGui::Text("Add to selection");
        ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "Delete / Backspace");
        ImGui::SameLine();
        ImGui::Text("Delete selected");
        ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "Escape");
        ImGui::SameLine();
        ImGui::Text("Clear selection");
        ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "Middle-click drag");
        ImGui::SameLine();
        ImGui::Text("Pan camera");
        ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "Scroll wheel");
        ImGui::SameLine();
        ImGui::Text("Zoom to cursor");
        ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "Drag spawn button");
        ImGui::SameLine();
        ImGui::Text("Place shape at cursor");
    }

    ImGui::End();

    // Handle drag-drop onto viewport.
    if (auto *payload = ImGui::GetDragDropPayload(); payload != nullptr && payload->IsDataType("SPAWN"))
    {
        auto world_point = screen_to_world(mouse_pos.x, mouse_pos.y);
        auto type        = *(SpawnShape *)payload->Data;

        // Ghost preview.
        switch (type)
        {
        case SpawnShape::Box:
        {
            auto half_width = g_cam_zoom / 2.0f;
            fg->AddRectFilled(
                {mouse_pos.x - half_width, mouse_pos.y - half_width}, {mouse_pos.x + half_width, mouse_pos.y + half_width},
                IM_COL32(51, 153, 230, 80));
            fg->AddRect(
                {mouse_pos.x - half_width, mouse_pos.y - half_width}, {mouse_pos.x + half_width, mouse_pos.y + half_width},
                IM_COL32(255, 255, 255, 100));

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
            }
        }
    }

    fg->AddText({4, 4}, IM_COL32_WHITE, g_renderer_name);

    char fps_buf[32];
    std::snprintf(fps_buf, sizeof(fps_buf), "%ld FPS", std::lround(io.Framerate));
    fg->AddText({4, 20}, IM_COL32_WHITE, fps_buf);

    // Box select preview.
    if (g_box_selecting)
    {
        fg->AddRectFilled(g_box_select_start, mouse_pos, IM_COL32(255, 80, 80, 30));
        fg->AddRect(g_box_select_start, mouse_pos, IM_COL32(255, 80, 80, 150), 0.0f, 0, 1.5f);
    }

    // Render physics bodies to background draw list.
    {
        // Camera transform.
        auto cam_x = (float)g_window_w / 2.0f - g_cam_center.x * g_cam_zoom;
        auto cam_y = (float)g_window_h / 2.0f + g_cam_center.y * g_cam_zoom;

        auto to_screen = [cam_x, cam_y](b2Vec2 point) noexcept -> ImVec2 { return {cam_x + point.x * g_cam_zoom, cam_y - point.y * g_cam_zoom}; };

        // Render a polygon shape with its radius expansion.
        auto render_poly = [to_screen, bg](b2Vec2 pos, b2Rot rot, const b2Polygon &poly, ImU32 color, bool show_edge = true)
        {
            constexpr auto BORDER_PX = 1.5f;

            ImVec2 screen[B2_MAX_POLYGON_VERTICES]{};
            ImVec2 inset[B2_MAX_POLYGON_VERTICES]{};
            auto   border_world = BORDER_PX / g_cam_zoom;
            for (std::int32_t j{}; j < poly.count; ++j)
            {
                // Corner bisector from Box2D's precomputed unit normals.
                auto prev = (j + poly.count - 1) % poly.count;
                auto n0   = poly.normals[prev];
                auto n1   = poly.normals[j];
                auto bx   = n0.x + n1.x;
                auto by   = n0.y + n1.y;
                auto bln  = std::sqrt(bx * bx + by * by);
                if (bln > 0.0f)
                {
                    bx /= bln;
                    by /= bln;
                }

                auto dot = bx * n1.x + by * n1.y;

                // Shrink by linear slop so shapes appear flush when the solver allows slight overlap - See: `B2_LINEAR_SLOP`.
                auto r       = std::max(0.0f, poly.radius - 0.005f * b2GetLengthUnitsPerMeter());
                auto offset  = dot > 0.0f ? r / dot : r;
                auto local_x = poly.vertices[j].x + bx * offset;
                auto local_y = poly.vertices[j].y + by * offset;
                auto world_x = local_x * rot.c - local_y * rot.s + pos.x;
                auto world_y = local_x * rot.s + local_y * rot.c + pos.y;

                screen[j] = to_screen({world_x, world_y});

                // Inset vertex: same bisector, reduced offset. Gives uniform border_world inset per edge.
                if (show_edge)
                {
                    auto inset_r   = r - border_world;
                    auto inset_off = dot > 0.0f ? inset_r / dot : inset_r;
                    auto il_x      = poly.vertices[j].x + bx * inset_off;
                    auto il_y      = poly.vertices[j].y + by * inset_off;
                    auto iw_x      = il_x * rot.c - il_y * rot.s + pos.x;
                    auto iw_y      = il_x * rot.s + il_y * rot.c + pos.y;

                    inset[j] = to_screen({iw_x, iw_y});
                }
            }

            auto draw_filled = [bg](ImVec2 *verts, std::int32_t count, ImU32 c)
            {
                if (count == 4)
                {
                    bg->AddQuadFilled(verts[0], verts[1], verts[2], verts[3], c);
                }
                else if (count >= 3)
                {
                    for (std::int32_t j = 1; j < count - 1; ++j)
                    {
                        bg->AddTriangleFilled(verts[0], verts[j], verts[j + 1], c);
                    }
                }
            };

            if (show_edge)
            {
                // Edge rim: fill outer with brighter color, fill inset with body color.
                auto edge_color = IM_COL32(
                    (ImU32)std::min(255, (std::int32_t)((color >> IM_COL32_R_SHIFT) & 0xFF) + 60),
                    (ImU32)std::min(255, (std::int32_t)((color >> IM_COL32_G_SHIFT) & 0xFF) + 60),
                    (ImU32)std::min(255, (std::int32_t)((color >> IM_COL32_B_SHIFT) & 0xFF) + 60), 255);
                draw_filled(screen, poly.count, edge_color);
                draw_filled(inset, poly.count, color);
            }
            else
            {
                draw_filled(screen, poly.count, color);
            }
        };

        auto outline_enabled = g_cam_zoom >= 10.0f;

        // Helper: Get the single polygon shape from a body.
        auto get_poly = [](b2BodyId body)
        {
            b2ShapeId shape;
            b2Body_GetShapes(body, &shape, 1);

            return b2Shape_GetPolygon(shape);
        };

        // Ground (static, no interpolation).
        render_poly(b2Body_GetPosition(g_ground_id), b2Body_GetRotation(g_ground_id), get_poly(g_ground_id), IM_COL32(102, 102, 102, 255));

        // Compute live box-select AABB for hover highlighting.
        b2AABB select_aabb{};
        bool   has_select_aabb{};
        if (g_box_selecting)
        {
            auto w0 = screen_to_world(g_box_select_start.x, g_box_select_start.y);
            auto w1 = screen_to_world(mouse_pos.x, mouse_pos.y);

            select_aabb = {
                {std::min(w0.x, w1.x), std::min(w0.y, w1.y)},
                {std::max(w0.x, w1.x), std::max(w0.y, w1.y)},
            };

            has_select_aabb = true;
        }

        // Viewport culling AABB in world space.
        // Margin accounts for shape radius so partially-visible bodies aren't culled.
        constexpr float CULL_MARGIN_WORLD  = 2.0f; // Meters (covers shapes up to 4m diameter).
        auto            view_min           = screen_to_world(0.0f, (float)g_window_h);
        auto            view_max           = screen_to_world((float)g_window_w, 0.0f);
        view_min.x                        -= CULL_MARGIN_WORLD;
        view_min.y                        -= CULL_MARGIN_WORLD;
        view_max.x                        += CULL_MARGIN_WORLD;
        view_max.y                        += CULL_MARGIN_WORLD;

        // Dynamic bodies (interpolated).
        auto alpha = g_physics_alpha;

        g_culled_count = 0;

        for (auto &&body : g_bodies)
        {
            // Skip interpolation for sleeping bodies.
            b2Vec2 pos;
            b2Rot  rot;
            if (b2Body_IsAwake(body.body))
            {
                auto transform = b2Body_GetTransform(body.body);

                pos = {
                    body.prev.position.x + alpha * (transform.p.x - body.prev.position.x),
                    body.prev.position.y + alpha * (transform.p.y - body.prev.position.y),
                };

                rot = {
                    body.prev.rotation.c + alpha * (transform.q.c - body.prev.rotation.c),
                    body.prev.rotation.s + alpha * (transform.q.s - body.prev.rotation.s),
                };

                auto len  = std::sqrt(rot.c * rot.c + rot.s * rot.s);
                rot.c    /= len;
                rot.s    /= len;
            }
            else
            {
                // Sleeping: Current transform is stable, no interpolation needed.
                auto transform = b2Body_GetTransform(body.body);
                pos            = transform.p;
                rot            = transform.q;
            }

            // Frustum cull.
            if (pos.x < view_min.x || pos.x > view_max.x || pos.y < view_min.y || pos.y > view_max.y)
            {
                ++g_culled_count;

                continue;
            }

            auto type = body.shape_type;

            // Compute selection state from shape AABB vs selection AABB.
            auto is_selected = body.selected;
            if (!is_selected && has_select_aabb)
            {
                b2AABB body_aabb{};
                if (type == b2_polygonShape)
                {
                    auto poly  = b2Shape_GetPolygon(body.shape);
                    auto min_x = pos.x, min_y = pos.y, max_x = pos.x, max_y = pos.y;
                    for (std::int32_t j{}; j < poly.count; ++j)
                    {
                        auto world_x = poly.vertices[j].x * rot.c - poly.vertices[j].y * rot.s + pos.x;
                        auto world_y = poly.vertices[j].x * rot.s + poly.vertices[j].y * rot.c + pos.y;
                        min_x        = std::min(min_x, world_x);
                        min_y        = std::min(min_y, world_y);
                        max_x        = std::max(max_x, world_x);
                        max_y        = std::max(max_y, world_y);
                    }

                    body_aabb = {{min_x - poly.radius, min_y - poly.radius}, {max_x + poly.radius, max_y + poly.radius}};
                }
                else if (type == b2_circleShape)
                {
                    auto circle = b2Shape_GetCircle(body.shape);
                    auto cx     = circle.center.x * rot.c - circle.center.y * rot.s + pos.x;
                    auto cy     = circle.center.x * rot.s + circle.center.y * rot.c + pos.y;
                    body_aabb   = {{cx - circle.radius, cy - circle.radius}, {cx + circle.radius, cy + circle.radius}};
                }

                is_selected = body_aabb.lowerBound.x <= select_aabb.upperBound.x
                           && body_aabb.upperBound.x >= select_aabb.lowerBound.x
                           && body_aabb.lowerBound.y <= select_aabb.upperBound.y
                           && body_aabb.upperBound.y >= select_aabb.lowerBound.y;
            }

            auto fill_color = is_selected ? IM_COL32(230, 50, 50, 255) : ImU32{};
            if (type == b2_polygonShape)
            {
                if (fill_color == 0)
                {
                    auto poly  = b2Shape_GetPolygon(body.shape);
                    fill_color = poly.count == 3 ? IM_COL32(50, 200, 50, 255) : IM_COL32(51, 153, 230, 255);
                }

                render_poly(pos, rot, b2Shape_GetPolygon(body.shape), fill_color, outline_enabled);
            }
            else if (type == b2_circleShape)
            {
                if (fill_color == 0)
                {
                    fill_color = IM_COL32(230, 153, 51, 255);
                }

                auto circle        = b2Shape_GetCircle(body.shape);
                auto world_x       = circle.center.x * rot.c - circle.center.y * rot.s + pos.x;
                auto world_y       = circle.center.x * rot.s + circle.center.y * rot.c + pos.y;
                auto screen_center = to_screen({world_x, world_y});
                auto screen_radius = circle.radius * g_cam_zoom;

                if (outline_enabled)
                {
                    auto edge_color = IM_COL32(
                        (ImU32)std::min(255, (std::int32_t)((fill_color >> IM_COL32_R_SHIFT) & 0xFF) + 60),
                        (ImU32)std::min(255, (std::int32_t)((fill_color >> IM_COL32_G_SHIFT) & 0xFF) + 60),
                        (ImU32)std::min(255, (std::int32_t)((fill_color >> IM_COL32_B_SHIFT) & 0xFF) + 60), 255);
                    bg->AddCircleFilled(screen_center, screen_radius, edge_color);
                    bg->AddCircleFilled(screen_center, screen_radius - 1.5f, fill_color);

                    auto   inset_radius = screen_radius - 1.5f;
                    ImVec2 edge{screen_center.x + rot.c * inset_radius, screen_center.y - rot.s * inset_radius};
                    bg->AddLine(screen_center, edge, edge_color, std::max(2.0f, 3.0f * g_cam_zoom / ZOOM_DEFAULT));
                }
                else
                {
                    bg->AddCircleFilled(screen_center, screen_radius, fill_color);
                }
            }
        }

        // Drawn lines (on top of bodies).
        auto render_smooth_line = [to_screen, bg](const std::vector<b2Vec2> &points, ImU32 color, float thick)
        {
            if (points.size() < 2)
            {
                return;
            }

            if (points.size() == 2)
            {
                bg->AddLine(to_screen(points[0]), to_screen(points[1]), color, thick);

                return;
            }

            bg->PathLineTo(to_screen(points[0]));

            for (std::size_t i{}; i < points.size() - 1; ++i)
            {
                auto p0 = points[i > 0 ? i - 1 : 0];
                auto p1 = points[i];
                auto p2 = points[i + 1];
                auto p3 = points[i + 1 < points.size() - 1 ? i + 2 : points.size() - 1];
                auto b1 = to_screen({p1.x + (p2.x - p0.x) / 6.0f, p1.y + (p2.y - p0.y) / 6.0f});
                auto b2 = to_screen({p2.x - (p3.x - p1.x) / 6.0f, p2.y - (p3.y - p1.y) / 6.0f});
                auto b3 = to_screen(p2);

                bg->PathBezierCubicCurveTo(b1, b2, b3);
            }

            bg->PathStroke(color, 0, thick);
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

        // Emitters (on top of everything).
        for (std::size_t i{}; i < g_emitters.size(); ++i)
        {
            auto &&emitter       = g_emitters[i];
            auto   screen_center = to_screen(emitter.position);
            auto   len           = 0.8f * g_cam_zoom;
            auto   thickness     = std::max(1.5f, 2.0f * g_cam_zoom / ZOOM_DEFAULT);
            auto   dx            = std::cos(emitter.angle) * len;
            auto   dy            = -std::sin(emitter.angle) * len;
            ImVec2 tip{screen_center.x + dx, screen_center.y + dy};

            auto   perp_x = -dy * 0.2f;
            auto   perp_y = dx * 0.2f;
            ImVec2 base{screen_center.x + dx * 0.65f, screen_center.y + dy * 0.65f};

            auto color = emitter.active ? IM_COL32(100, 255, 100, 255) : IM_COL32(255, 100, 100, 255);
            bg->AddLine(screen_center, base, color, thickness);
            bg->AddTriangleFilled(tip, {base.x + perp_x, base.y + perp_y}, {base.x - perp_x, base.y - perp_y}, color);
            bg->AddCircleFilled(screen_center, std::max(3.0f, 0.1f * g_cam_zoom), color);

            // Label on foreground so it's always visible.
            char label[8];
            std::snprintf(label, sizeof(label), "e%zu", i);
            auto label_pos = ImVec2{screen_center.x - 8, screen_center.y - 24};
            fg->AddText(label_pos, IM_COL32(255, 255, 255, 200), label);
        }
    }

    ImGui::Render();
    SDL_SetRenderScale(g_renderer, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);

    constexpr ImVec4 clear_color(0.10f, 0.10f, 0.10f, 1.00f);
    SDL_SetRenderDrawColorFloat(g_renderer, clear_color.x, clear_color.y, clear_color.z, clear_color.w);

    SDL_RenderClear(g_renderer);
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), g_renderer);
    SDL_RenderPresent(g_renderer);

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void *appstate, SDL_Event *event)
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

            // Ctrl+Left-click: Start drawing a line.
            if ((SDL_GetModState() & SDL_KMOD_CTRL) != 0)
            {
                g_current_stroke.clear();
                g_current_stroke.emplace_back(world_point);
            }
            // Just Left-click: Drag emitter or body.
            else
            {
                // Check emitters first (small grab radius in world units).
                constexpr float EMITTER_GRAB_RADIUS = 0.5f;
                for (std::int32_t i{}; i < (std::int32_t)g_emitters.size(); ++i)
                {
                    auto dx = world_point.x - g_emitters[i].position.x;
                    auto dy = world_point.y - g_emitters[i].position.y;
                    if (dx * dx + dy * dy <= EMITTER_GRAB_RADIUS * EMITTER_GRAB_RADIUS)
                    {
                        g_dragged_emitter = i;

                        break;
                    }
                }

                // If no emitter hit, try grabbing a body.
                if (g_dragged_emitter < 0)
                {
                    g_mouse_body = find_body_at(world_point);
                    if (B2_IS_NON_NULL(g_mouse_body))
                    {
                        b2Body_SetAwake(g_mouse_body, true);

                        auto def         = b2DefaultMouseJointDef();
                        def.bodyIdA      = g_ground_id;
                        def.bodyIdB      = g_mouse_body;
                        def.target       = world_point;
                        def.maxForce     = 10000.0f * b2Body_GetMass(g_mouse_body);
                        def.hertz        = 60.0f;
                        def.dampingRatio = 1.0f;

                        g_mouse_joint = b2CreateMouseJoint(g_world, &def);

                        b2Body_SetFixedRotation(g_mouse_body, true);
                        b2Body_EnableSleep(g_mouse_body, false);
                    }
                }
            }
        }
        // Right-click.
        else if (event->button.button == SDL_BUTTON_RIGHT)
        {
            if ((SDL_GetModState() & SDL_KMOD_CTRL) != 0)
            {
                g_erasing = true;
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
        // Middle click.
        if (event->button.button == SDL_BUTTON_MIDDLE)
        {
            g_cam_dragging = false;
        }
        // Left click.
        else if (event->button.button == SDL_BUTTON_LEFT)
        {
            if (!g_current_stroke.empty())
            {
                finish_stroke();
            }
            if (g_dragged_emitter >= 0)
            {
                g_dragged_emitter = -1;
            }
            else if (!g_current_stroke.empty())
            {
                finish_stroke();
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
        // Right click.
        else if (event->button.button == SDL_BUTTON_RIGHT)
        {
            g_box_selecting = false;

            ImVec2 end_pos{event->button.x, event->button.y};
            auto   dx = end_pos.x - g_box_select_start.x;
            auto   dy = end_pos.y - g_box_select_start.y;

            // Ctrl+Right was erase mode -- already handled in motion.
            if (g_erasing)
            {
                g_erasing = false;
            }
            // Click, not drag: Toggle selection on body under cursor.
            else if (dx * dx + dy * dy < 25.0f)
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
                    // Clicked empty space: clear selection.
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

                auto   w0 = screen_to_world(g_box_select_start.x, g_box_select_start.y);
                auto   w1 = screen_to_world(end_pos.x, end_pos.y);
                b2AABB aabb{
                    {std::min(w0.x, w1.x), std::min(w0.y, w1.y)},
                    {std::max(w0.x, w1.x), std::max(w0.y, w1.y)},
                };

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
        else if (g_dragged_emitter >= 0)
        {
            auto world_pt                          = screen_to_world(event->motion.x, event->motion.y);
            g_emitters[g_dragged_emitter].position = world_pt;
        }
        else if (B2_IS_NON_NULL(g_mouse_joint))
        {
            auto world_pt = screen_to_world(event->motion.x, event->motion.y);
            b2MouseJoint_SetTarget(g_mouse_joint, world_pt);
        }
        else if (g_erasing)
        {
            auto world_pt    = screen_to_world(event->motion.x, event->motion.y);
            auto hit_dist_sq = 0.15f * 0.15f;
            for (std::size_t i{}; i < g_drawn_lines.size(); ++i)
            {
                auto       &line = g_drawn_lines[i];
                std::size_t hit_seg{};
                bool        found_hit{};
                for (std::size_t j = 1; j < line.points.size(); ++j)
                {
                    if (point_to_segment_dist_sq(world_pt, line.points[j - 1], line.points[j]) <= hit_dist_sq)
                    {
                        hit_seg   = j;
                        found_hit = true;

                        break;
                    }
                }

                if (!found_hit)
                {
                    continue;
                }

                // Destroy old body.
                b2DestroyBody(line.body);

                // Split into left [0..hit_seg-1] and right [hit_seg..end].
                auto &pts   = line.points;
                auto  left  = std::vector(pts.begin(), pts.begin() + (std::ptrdiff_t)hit_seg);
                auto  right = std::vector(pts.begin() + (std::ptrdiff_t)hit_seg, pts.end());

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
        }
        else if (!g_current_stroke.empty())
        {
            auto  world_point = screen_to_world(event->motion.x, event->motion.y);
            auto &last        = g_current_stroke.back();
            auto  dx          = world_point.x - last.x;
            auto  dy          = world_point.y - last.y;
            if (dx * dx + dy * dy >= MIN_STROKE_DIST * MIN_STROKE_DIST)
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
            // Delete selected bodies.
            for (auto &&body : g_bodies)
            {
                if (body.selected)
                {
                    b2DestroyBody(body.body);
                }
            }

            std::erase_if(g_bodies, [](const PhysBody &body) { return body.selected; });
        }
        else if (event->key.key == SDLK_ESCAPE)
        {
            // Clear selection.
            for (auto &&body : g_bodies)
            {
                body.selected = false;
            }
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

void SDL_AppQuit(void *appstate, SDL_AppResult result)
{
    g_bodies.clear();
    g_emitters.clear();
    g_drawn_lines.clear();
    g_current_stroke.clear();

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
