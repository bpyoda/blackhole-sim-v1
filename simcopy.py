"""
3D Black Hole Gravity Field Visualizer
======================================
Physics References (all Newtonian / classical mechanics):

  Newton's Law of Universal Gravitation:   F = G·M1·M2 / r^2
  Gravitational Potential Energy:          U = -G·M1·M2 / r
  Conservation of Momentum:               Sum(mi*vi) = const  (inelastic mergers)
  Conservation of Energy:                 E = K + U = const
  Keplerian Orbital Motion:               v_circ = sqrt(G*M / r)
  Escape Velocity:                        v_esc  = sqrt(2*G*M / r)
  Hill Sphere Radius:                     r_H    = a * cbrt(m / 3M)
  Lagrange Regions:                       L1-L5 equilibrium points (3-body problem)
  Restricted Three-Body Problem:          massless test particles in field of two massive bodies
  Velocity Verlet / Leapfrog Integration: symplectic, preserves phase-space volume
"""

import taichi as ti
import numpy as np
import math
import time

# ── GPU init ──────────────────────────────────────────────────────────────────
try:
    ti.init(arch=ti.gpu, default_fp=ti.f32, fast_math=True)
except Exception:
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    print("GPU unavailable – running on CPU.")

# ── Window ───────────────────────────────────────────────────────────────────
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 800
ASPECT        = WINDOW_WIDTH / WINDOW_HEIGHT   # 1.6

# ── Simulation limits ─────────────────────────────────────────────────────────
MAX_BH        = 12
MAX_P         = 1000
TRAIL_LEN     = 30
STARFIELD_N   = 200

# ── Physics ───────────────────────────────────────────────────────────────────
# Newton: F = G*M1*M2/r^2
G             = 1.2e-4
BH_MASS       = 1.0
BH_RADIUS     = 0.024
SOFTENING     = 8e-4
PARTICLE_LIFE = 260.0
BOUNDARY      = 2.8

DT_BASE  = 0.018
DT_LOW   = 0.007
DT_HIGH  = 0.055
SUBSTEPS = 2

# ── Taichi Fields ─────────────────────────────────────────────────────────────
bh_active = ti.field(ti.i32,  shape=MAX_BH)
bh_pos    = ti.Vector.field(3, ti.f32, shape=MAX_BH)
bh_vel    = ti.Vector.field(3, ti.f32, shape=MAX_BH)
bh_acc    = ti.Vector.field(3, ti.f32, shape=MAX_BH)
bh_acc_n  = ti.Vector.field(3, ti.f32, shape=MAX_BH)
bh_mass   = ti.field(ti.f32,  shape=MAX_BH)
bh_radius = ti.field(ti.f32,  shape=MAX_BH)

p_pos     = ti.Vector.field(3, ti.f32, shape=MAX_P)
p_vel     = ti.Vector.field(3, ti.f32, shape=MAX_P)
p_acc     = ti.Vector.field(3, ti.f32, shape=MAX_P)
p_age     = ti.field(ti.f32,  shape=MAX_P)

trail_pos = ti.Vector.field(3, ti.f32, shape=(TRAIL_LEN, MAX_P))
trail_head = ti.field(ti.i32, shape=())

cam_yaw   = ti.field(ti.f32, shape=())
cam_pitch = ti.field(ti.f32, shape=())
cam_dist  = ti.field(ti.f32, shape=())

bh_sx = ti.field(ti.f32, shape=MAX_BH)
bh_sy = ti.field(ti.f32, shape=MAX_BH)
bh_sr = ti.field(ti.f32, shape=MAX_BH)
bh_sz = ti.field(ti.f32, shape=MAX_BH)

t_sx = ti.field(ti.f32, shape=(TRAIL_LEN, MAX_P))
t_sy = ti.field(ti.f32, shape=(TRAIL_LEN, MAX_P))

g_dark    = ti.field(ti.i32, shape=())
g_quality = ti.field(ti.i32, shape=())
g_speed   = ti.field(ti.i32, shape=())

# Init camera & settings
cam_yaw[None]    = 0.65
cam_pitch[None]  = 0.50
cam_dist[None]   = 1.05
g_dark[None]     = 0
g_quality[None]  = 1
g_speed[None]    = 1
trail_head[None] = 0

# ── Helper functions (inlined into kernels) ───────────────────────────────────

@ti.func
def bh_acc_at(pos: ti.template(), idx: ti.i32) -> ti.Vector:
    a = ti.Vector([0.0, 0.0, 0.0])
    for j in range(MAX_BH):
        if j != idx and bh_active[j]:
            rv   = bh_pos[j] - pos
            dist = ti.sqrt(rv.dot(rv) + SOFTENING * SOFTENING)
            a   += (G * bh_mass[j] / (dist * dist * dist)) * rv
    return a

@ti.func
def particle_acc(pos: ti.template()) -> ti.Vector:
    a = ti.Vector([0.0, 0.0, 0.0])
    for j in range(MAX_BH):
        if bh_active[j]:
            rv   = bh_pos[j] - pos
            dist = ti.sqrt(rv.dot(rv) + SOFTENING * SOFTENING)
            a   += (G * bh_mass[j] / (dist * dist * dist)) * rv
    return a

@ti.func
def respawn(i: ti.i32):
    n_bh = 0
    for k in range(MAX_BH):
        if bh_active[k]:
            n_bh += 1
    if n_bh > 0:
        target = int(ti.random() * n_bh)
        sel = 0
        cnt = 0
        for k in range(MAX_BH):
            if bh_active[k]:
                if cnt == target:
                    sel = k
                    break
                cnt += 1
        spread = 0.35
        r      = bh_radius[sel] * 2.1 + ti.random() * spread
        theta  = ti.random() * (2.0 * math.pi)
        phi    = ti.acos(2.0 * ti.random() - 1.0)
        sp     = ti.sin(phi)
        cp     = ti.cos(phi)
        st     = ti.sin(theta)
        ct     = ti.cos(theta)
        off    = ti.Vector([r * sp * ct, r * sp * st, r * cp * 0.25])
        pos_new = bh_pos[sel] + off
        # Keplerian circular velocity: v = sqrt(G*M/r)
        v_circ  = ti.sqrt(G * bh_mass[sel] / (r + SOFTENING))
        tangent = ti.Vector([-off.y, off.x, 0.0])
        t_len   = tangent.norm()
        if t_len > 1e-6:
            tangent = tangent / t_len
        p_pos[i] = pos_new
        p_vel[i] = bh_vel[sel] + tangent * v_circ * 0.5
        p_acc[i] = particle_acc(pos_new)
        p_age[i] = PARTICLE_LIFE
    else:
        p_pos[i] = ti.Vector([(ti.random()-0.5)*1.2, (ti.random()-0.5)*1.2, (ti.random()-0.5)*0.15])
        p_vel[i] = ti.Vector([0.0, 0.0, 0.0])
        p_acc[i] = ti.Vector([0.0, 0.0, 0.0])
        p_age[i] = PARTICLE_LIFE

@ti.func
def project_pt(wp: ti.template(), eye: ti.template(),
               xa: ti.template(), ya: ti.template(), za: ti.template()) -> ti.Vector:
    d   = wp - eye
    px  = d.dot(xa)
    py  = d.dot(ya)
    pz  = d.dot(za)
    fov = 0.85
    return ti.Vector([
        (px / (-pz + 1e-6)) * fov + 0.5 * ASPECT,
        (py / (-pz + 1e-6)) * fov + 0.5
    ])

# ── Init ──────────────────────────────────────────────────────────────────────

@ti.kernel
def init_simulation():
    for i in range(MAX_BH):
        bh_active[i] = 0
        bh_pos[i]    = ti.Vector([0.0, 0.0, 0.0])
        bh_vel[i]    = ti.Vector([0.0, 0.0, 0.0])
        bh_acc[i]    = ti.Vector([0.0, 0.0, 0.0])
        bh_acc_n[i]  = ti.Vector([0.0, 0.0, 0.0])
        bh_mass[i]   = BH_MASS
        bh_radius[i] = BH_RADIUS

    # Stable binary orbit: v_orb = sqrt(G*M/sep)
    sep   = 0.50
    half  = sep * 0.5
    v_orb = ti.sqrt(G * BH_MASS / sep)

    bh_active[0] = 1
    bh_pos[0]    = ti.Vector([-half, 0.0, 0.0])
    bh_vel[0]    = ti.Vector([0.0,  v_orb, 0.0])

    bh_active[1] = 1
    bh_pos[1]    = ti.Vector([ half, 0.0, 0.0])
    bh_vel[1]    = ti.Vector([0.0, -v_orb, 0.0])

    for i in range(MAX_BH):
        if bh_active[i]:
            bh_acc[i] = bh_acc_at(bh_pos[i], i)

    # Randomise particle ages → immediate colour diversity (red through blue)
    for i in range(MAX_P):
        p_age[i]  = ti.random() * PARTICLE_LIFE
        angle     = ti.random() * 2.0 * math.pi
        r_spawn   = 0.08 + ti.random() * 0.55
        p_pos[i]  = ti.Vector([r_spawn * ti.cos(angle), r_spawn * ti.sin(angle), (ti.random()-0.5)*0.18])
        v_tang    = ti.sqrt(ti.max(G * BH_MASS / (r_spawn + SOFTENING), 0.0)) * 0.55
        p_vel[i]  = ti.Vector([-ti.sin(angle) * v_tang, ti.cos(angle) * v_tang, 0.0])
        p_acc[i]  = particle_acc(p_pos[i])
        for s in range(TRAIL_LEN):
            trail_pos[s, i] = p_pos[i]

    trail_head[None] = 0

# ── Physics ───────────────────────────────────────────────────────────────────

@ti.kernel
def update_physics(dt: ti.f32):
    # --- Black holes: Velocity Verlet (symplectic, long-term stable) ---
    # x(t+dt) = x + v*dt + 0.5*a*dt^2
    for i in range(MAX_BH):
        if bh_active[i]:
            bh_pos[i] += bh_vel[i] * dt + 0.5 * bh_acc[i] * (dt * dt)

    # Compute new accelerations at new positions
    for i in range(MAX_BH):
        bh_acc_n[i] = ti.Vector([0.0, 0.0, 0.0])
        if bh_active[i]:
            bh_acc_n[i] = bh_acc_at(bh_pos[i], i)

    # v(t+dt) = v + 0.5*(a + a_new)*dt
    for i in range(MAX_BH):
        if bh_active[i]:
            bh_vel[i] += 0.5 * (bh_acc[i] + bh_acc_n[i]) * dt
            bh_acc[i]  = bh_acc_n[i]

    # Inelastic mergers (conservation of momentum: p = sum(m*v))
    for i in range(MAX_BH):
        if bh_active[i]:
            for j in range(i + 1, MAX_BH):
                if bh_active[j]:
                    rv   = bh_pos[j] - bh_pos[i]
                    dist = rv.norm()
                    if dist < (bh_radius[i] + bh_radius[j]) * 0.9:
                        m1 = bh_mass[i];  m2 = bh_mass[j];  mt = m1 + m2
                        bh_pos[i]    = (bh_pos[i]*m1 + bh_pos[j]*m2) / mt
                        bh_vel[i]    = (bh_vel[i]*m1 + bh_vel[j]*m2) / mt
                        bh_mass[i]   = mt
                        bh_radius[i] = BH_RADIUS * ti.pow(mt / BH_MASS, 0.35)
                        bh_active[j] = 0

    # --- Tracer particles: Leapfrog (kick-drift-kick, cheaper than Verlet) ---
    # v_half = v + 0.5*a*dt  |  x_new = x + v_half*dt  |  a_new = f(x_new)  |  v_new = v_half + 0.5*a_new*dt
    for i in range(MAX_P):
        if p_age[i] <= 0.0 or p_pos[i].norm() > BOUNDARY:
            respawn(i)
        else:
            inside = False
            for j in range(MAX_BH):
                if bh_active[j]:
                    if (p_pos[i] - bh_pos[j]).norm() < bh_radius[j] * 0.9:
                        inside = True
            if inside:
                respawn(i)
            else:
                v_half   = p_vel[i] + 0.5 * p_acc[i] * dt
                p_pos[i] += v_half * dt
                a_new    = particle_acc(p_pos[i])
                p_vel[i] = v_half + 0.5 * a_new * dt
                p_acc[i] = a_new
                p_age[i] -= dt

    # Write trail ring-buffer
    h = trail_head[None]
    for i in range(MAX_P):
        trail_pos[h, i] = p_pos[i]
    trail_head[None] = (h + 1) % TRAIL_LEN

# ── Projection kernel ─────────────────────────────────────────────────────────

@ti.kernel
def project_scene():
    pitch = cam_pitch[None]
    yaw   = cam_yaw[None]
    dist  = cam_dist[None] * 0.8
    eye   = dist * ti.Vector([ti.cos(pitch)*ti.cos(yaw),
                               ti.cos(pitch)*ti.sin(yaw),
                               ti.sin(pitch)])
    za    = eye.normalized()
    up    = ti.Vector([0.0, 0.0, 1.0])
    xa    = up.cross(za).normalized()
    ya    = za.cross(xa)

    for i in range(MAX_BH):
        if bh_active[i]:
            sc        = project_pt(bh_pos[i], eye, xa, ya, za)
            bh_sx[i]  = sc.x;  bh_sy[i] = sc.y
            d         = bh_pos[i] - eye
            pz        = d.dot(za)
            bh_sz[i]  = pz
            bh_sr[i]  = bh_radius[i] / (-pz + 1e-6) * 0.85

    for s in range(TRAIL_LEN):
        for i in range(MAX_P):
            sc         = project_pt(trail_pos[s, i], eye, xa, ya, za)
            t_sx[s, i] = sc.x
            t_sy[s, i] = sc.y

# ── Add BH kernel ─────────────────────────────────────────────────────────────

@ti.kernel
def kernel_add_bh(idx: ti.i32, wx: ti.f32, wy: ti.f32, wz: ti.f32):
    bh_active[idx] = 1
    bh_pos[idx]    = ti.Vector([wx, wy, wz])
    bh_vel[idx]    = ti.Vector([0.0, 0.0, 0.0])
    bh_mass[idx]   = BH_MASS
    bh_radius[idx] = BH_RADIUS
    bh_acc[idx]    = bh_acc_at(ti.Vector([wx, wy, wz]), idx)

# ── Pre-baked colour LUT ──────────────────────────────────────────────────────

def _build_lut(n=512):
    lut = np.zeros(n, dtype=np.uint32)
    for k in range(n):
        f   = k / (n - 1)      # 0 = blue / fading,  1 = red / fresh
        age = max(0.0, min(1.0, f))
        if   age > 0.833: t=(age-0.833)/0.167; r,g,b=1.0,0.5*(1-t),0.0
        elif age > 0.666: t=(age-0.666)/0.167; r,g,b=1.0,0.5+0.5*(1-t),0.0
        elif age > 0.500: t=(age-0.500)/0.166; r,g,b=t,1.0,0.0
        elif age > 0.333: t=(age-0.333)/0.167; r,g,b=0.0,1.0,1.0-t
        elif age > 0.166: t=(age-0.166)/0.167; r,g,b=0.0,t,1.0
        else:              t=age/0.166;          r,g,b=0.0,0.0,t*0.6
        lut[k] = (int(r*255)<<16)|(int(g*255)<<8)|int(b*255)
    return lut

COLOR_LUT = _build_lut()

# Pre-baked starfield
rng_sf   = np.random.default_rng(42)
SF_X     = rng_sf.random(STARFIELD_N).astype(np.float32)
SF_Y     = rng_sf.random(STARFIELD_N).astype(np.float32)
SF_BRIG  = (rng_sf.random(STARFIELD_N) * 0.28 + 0.05).astype(np.float32)
SF_RAD   = (rng_sf.random(STARFIELD_N) * 0.8  + 0.5 ).astype(np.float32)
SF_COL   = np.array([int(b*255)*0x010101 for b in SF_BRIG], dtype=np.uint32)

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("     BLACK HOLE GRAVITY FIELD VISUALIZER  (Taichi GPU)")
    print("=" * 66)
    print("  Left Click   : Spawn black hole")
    print("  Right Drag   : Rotate camera")
    print("  Scroll / Up/Down: Zoom")
    print("  SPACE        : Pause / Resume")
    print("  R            : Reset   |   M: Toggle Light/Dark")
    print("=" * 66)

    gui = ti.GUI("Black Hole Gravity Lab",
                 res=(WINDOW_WIDTH, WINDOW_HEIGHT),
                 background_color=0x060613)

    init_simulation()

    paused     = False
    prev_rmb   = False
    prev_mouse = np.zeros(2)
    click_lock = False
    last_t     = time.perf_counter()
    fps_s      = 60.0
    scroll_acc = 0.0

    while gui.running:
        now    = time.perf_counter()
        dt_fps = now - last_t;  last_t = now
        if dt_fps > 0:
            fps_s = 0.92 * fps_s + 0.08 / dt_fps

        spd = g_speed[None]
        dt_sim = DT_LOW if spd == 0 else (DT_HIGH if spd == 2 else DT_BASE)

        # Events
        for e in gui.get_events():
            if e.type == ti.GUI.PRESS:
                if e.key == ti.GUI.ESCAPE: gui.running = False
                elif e.key == ' ':         paused = not paused
                elif e.key in ('r','R'):   init_simulation()
                elif e.key in ('m','M'):   g_dark[None] = 1 - g_dark[None]
                elif e.key == ti.GUI.UP:   cam_dist[None] = max(0.25, cam_dist[None]-0.08)
                elif e.key == ti.GUI.DOWN: cam_dist[None] = min(5.0,  cam_dist[None]+0.08)
            elif e.type == ti.GUI.WHEEL:
                delta = getattr(e, 'delta', None)
                if delta is not None:
                    scroll_acc += delta[1] if hasattr(delta, '__len__') else float(delta)

        if abs(scroll_acc) > 0:
            cam_dist[None] = float(np.clip(cam_dist[None] - scroll_acc * 0.0015, 0.25, 5.0))
            scroll_acc = 0.0

        mp  = gui.get_cursor_pos()
        lmb = gui.is_pressed(ti.GUI.LMB)
        rmb = gui.is_pressed(ti.GUI.RMB)
        SIDEBAR_W = 0.21
        in_bar    = mp[0] < SIDEBAR_W

        if rmb:
            if prev_rmb:
                dx = mp[0] - prev_mouse[0];  dy = mp[1] - prev_mouse[1]
                cam_yaw[None]   -= dx * 3.5
                cam_pitch[None]  = float(np.clip(cam_pitch[None] + dy*2.0, -0.15, 1.45))
            prev_mouse[:] = mp
        prev_rmb = rmb

        if lmb and in_bar:
            y = mp[1]
            if   0.74 <= y < 0.80: g_dark[None] = 1 - g_dark[None]; click_lock=True
            elif 0.58 <= y < 0.62: g_quality[None]=0; click_lock=True
            elif 0.53 <= y < 0.57: g_quality[None]=1; click_lock=True
            elif 0.48 <= y < 0.52: g_quality[None]=2; click_lock=True
            elif 0.30 <= y < 0.34: g_speed[None]=0; click_lock=True
            elif 0.25 <= y < 0.29: g_speed[None]=1; click_lock=True
            elif 0.20 <= y < 0.24: g_speed[None]=2; click_lock=True

        if lmb and not in_bar and not click_lock:
            pitch = cam_pitch[None];  yaw = cam_yaw[None];  d = cam_dist[None]*0.8
            dx_sc = (mp[0]*ASPECT - 0.5*ASPECT) / 0.85
            dy_sc = (mp[1] - 0.5) / 0.85
            x_w   = -dx_sc*math.sin(yaw) - dy_sc*math.sin(pitch)*math.cos(yaw)
            y_w   =  dx_sc*math.cos(yaw) - dy_sc*math.sin(pitch)*math.sin(yaw)
            z_w   =  dy_sc*math.cos(pitch)
            act_np = bh_active.to_numpy()
            free   = np.argwhere(act_np == 0).flatten()
            if len(free) > 0:
                kernel_add_bh(int(free[0]), float(x_w*d), float(y_w*d), float(z_w*d))
            click_lock = True
        if not lmb:
            click_lock = False

        # Physics substeps
        if not paused:
            steps = 1 if g_quality[None]==0 else SUBSTEPS
            step_dt = dt_sim / steps
            for _ in range(steps):
                update_physics(step_dt)

        project_scene()

        # Single batch GPU read
        bh_act_np  = bh_active.to_numpy()
        bh_sx_np   = bh_sx.to_numpy()
        bh_sy_np   = bh_sy.to_numpy()
        bh_sr_np   = bh_sr.to_numpy()
        bh_sz_np   = bh_sz.to_numpy()
        t_sx_np    = t_sx.to_numpy()
        t_sy_np    = t_sy.to_numpy()
        p_age_np   = p_age.to_numpy()
        h_idx      = int(trail_head[None])
        dark       = int(g_dark[None])
        quality    = int(g_quality[None])

        trail_draw = 6 if quality==0 else (16 if quality==1 else TRAIL_LEN)

        gui.clear(0x060613)

        # Starfield
        for si in range(STARFIELD_N):
            gui.circle((float(SF_X[si]), float(SF_Y[si])),
                       color=int(SF_COL[si]), radius=float(SF_RAD[si]))

        # Vectorised colour lookup
        age_fracs = p_age_np / PARTICLE_LIFE
        lut_idx   = np.clip((age_fracs * 511).astype(np.int32), 0, 511)
        colors    = COLOR_LUT[lut_idx]

        # Gather visible BHs for lensing
        vis_bh = []
        if dark:
            for k in range(MAX_BH):
                if bh_act_np[k] and bh_sz_np[k] < 0.0:
                    vis_bh.append((bh_sx_np[k]/ASPECT, bh_sy_np[k],
                                   float(bh_sr_np[k])))

        # Particle trails
        for step in range(trail_draw - 1):
            s1 = (h_idx - 1 - step) % TRAIL_LEN
            s2 = (s1 - 1) % TRAIL_LEN
            x1 = t_sx_np[s1] / ASPECT;  y1 = t_sy_np[s1]
            x2 = t_sx_np[s2] / ASPECT;  y2 = t_sy_np[s2]
            opacity = (trail_draw - step) / trail_draw

            for i in range(MAX_P):
                px1 = float(x1[i]);  py1 = float(y1[i])
                px2 = float(x2[i]);  py2 = float(y2[i])
                if abs(px1-px2) > 0.12 or abs(py1-py2) > 0.12:
                    continue
                if vis_bh:
                    for bx,by,br in vis_bh:
                        dx_ = px1-bx;  dy_ = py1-by
                        d_  = math.sqrt(dx_*dx_ + dy_*dy_) + 1e-6
                        if d_ > br * 0.5:
                            defl = (br*br*1.2) / (d_ + 1e-5)
                            px1  = bx + (dx_/d_)*(d_+defl)
                            py1  = by + (dy_/d_)*(d_+defl)
                c   = int(colors[i])
                rc  = (c>>16)&0xFF;  gc = (c>>8)&0xFF;  bc = c&0xFF
                oc  = (int(rc*opacity)<<16)|(int(gc*opacity)<<8)|int(bc*opacity)
                gui.line((px1,py1),(px2,py2), color=oc, radius=1.0)

        # Black holes
        halo_layers = 1 if quality==0 else (2 if quality==1 else 4)
        rot_t       = time.perf_counter() * 2.3
        pitch_py    = cam_pitch[None];  yaw_py = cam_yaw[None];  dist_py = cam_dist[None]*0.8
        eye_np      = np.array([dist_py*math.cos(pitch_py)*math.cos(yaw_py),
                                 dist_py*math.cos(pitch_py)*math.sin(yaw_py),
                                 dist_py*math.sin(pitch_py)])
        za_np       = eye_np / (np.linalg.norm(eye_np)+1e-9)
        xa_np       = np.cross([0,0,1], za_np); xa_np /= (np.linalg.norm(xa_np)+1e-9)
        ya_np       = np.cross(za_np, xa_np)

        bh_pos_np   = bh_pos.to_numpy()

        for i in range(MAX_BH):
            if not bh_act_np[i] or bh_sz_np[i] >= 0.0:
                continue
            sx_i = bh_sx_np[i] / ASPECT
            sy_i = bh_sy_np[i]
            r_px = float(bh_sr_np[i]) * WINDOW_HEIGHT

            if dark:
                n_seg = 18 if quality<2 else 28
                for r_mult in (0.85, 1.0, 1.15):
                    disk_r = float(bh_radius[i]) * 2.6 * r_mult
                    pts    = []
                    for si2 in range(n_seg + 1):
                        ang  = rot_t + si2*(2.0*math.pi/n_seg)
                        off  = np.array([disk_r*math.cos(ang), disk_r*math.sin(ang), 0.0])
                        p3   = bh_pos_np[i] + off
                        dv   = p3 - eye_np
                        px_  = dv.dot(xa_np);  py_ = dv.dot(ya_np);  pz_ = dv.dot(za_np)
                        sc_x = (px_/(-pz_+1e-6))*0.85 + 0.5*ASPECT
                        sc_y = (py_/(-pz_+1e-6))*0.85 + 0.5
                        dx_  = sc_x/ASPECT - sx_i;  dy_ = sc_y - sy_i
                        dd   = math.sqrt(dx_*dx_+dy_*dy_)+1e-6
                        defl = (bh_sr_np[i]**2*1.5)/(dd+1e-5)
                        sc_x = sx_i + (dx_/dd)*(dd+defl)
                        sc_y = sy_i + (dy_/dd)*(dd+defl)
                        pts.append((sc_x, sc_y))
                    for si2 in range(len(pts)-1):
                        ax,ay=pts[si2];  bx_,by_=pts[si2+1]
                        if abs(ax-bx_)<0.2 and abs(ay-by_)<0.2:
                            gui.line((ax,ay),(bx_,by_), color=0xFFAA22, radius=1.0)

                for layer in range(halo_layers):
                    hr  = bh_sr_np[i]*(1.4+layer*0.55)
                    ho  = 0.22/(layer+1.0)
                    hcr = int(0xFF*ho);  hcb = int(0xBB*ho)
                    gui.circle((sx_i,sy_i), color=(hcr<<16)|hcb, radius=hr*WINDOW_HEIGHT)
                gui.circle((sx_i,sy_i), color=0x000000, radius=r_px)
            else:
                for layer in range(halo_layers):
                    hr  = r_px*(1.6+layer*0.9)
                    ho  = 0.18/(layer+1.0)
                    hc  = int(0xFF*ho)
                    gui.circle((sx_i,sy_i), color=(hc<<16)|(int(hc*0.8)<<8)|int(hc*0.4), radius=hr)
                gui.circle((sx_i,sy_i), color=0xFFFFFF, radius=r_px)

        # Sidebar
        gui.rect((0.0,0.0),(SIDEBAR_W,1.0), color=0x05050F)
        gui.line((SIDEBAR_W,0.0),(SIDEBAR_W,1.0), color=0x1E1E38, radius=1)
        gui.text("BLACK HOLE",  (0.015,0.935), font_size=17, color=0xFFAA00)
        gui.text("GRAVITY LAB", (0.015,0.905), font_size=12, color=0x7777AA)

        dark_now = int(g_dark[None])
        mc = 0x1E1E38 if dark_now else 0x0A0A1A
        gui.rect((0.01,0.73),(SIDEBAR_W-0.01,0.80), color=mc)
        gui.text("DARK MODE" if dark_now else "LIGHT MODE", (0.022,0.745), font_size=14, color=0xFFFFFF)

        gui.text("VISUAL QUALITY", (0.015,0.67), font_size=10, color=0x7777AA)
        q_now = int(g_quality[None])
        for qi,(label,qy) in enumerate([("LOW",0.61),("MEDIUM",0.56),("HIGH",0.51)]):
            bc = 0x1E1E38 if q_now==qi else 0x08081A
            gui.rect((0.01,qy-0.02),(SIDEBAR_W-0.01,qy+0.02), color=bc)
            gui.text(label,(0.022,qy-0.008),font_size=13,color=0xFFFFFF if q_now==qi else 0x666688)

        gui.text("SIM SPEED", (0.015,0.40), font_size=10, color=0x7777AA)
        s_now = int(g_speed[None])
        for si3,(label,sy) in enumerate([("LOW",0.33),("MEDIUM",0.27),("HIGH",0.21)]):
            bc = 0x1E1E38 if s_now==si3 else 0x08081A
            gui.rect((0.01,sy-0.02),(SIDEBAR_W-0.01,sy+0.02), color=bc)
            gui.text(label,(0.022,sy-0.008),font_size=13,color=0xFFFFFF if s_now==si3 else 0x666688)

        n_bh = int(np.sum(bh_act_np))
        gui.text(f"BLACK HOLES: {n_bh}", (0.015,0.06), font_size=12, color=0xFFAA00)
        gui.text(f"FPS: {fps_s:.0f}",    (0.015,0.02), font_size=12, color=0x7777AA)

        gui.show()

if __name__ == "__main__":
    main()
