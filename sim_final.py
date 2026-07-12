"""
ULTIMATE SPECTRAL CHAOS: N-BODY PROBLEM, BLACK HOLE MERGER
==========================================================
Implements Newtonian N-body gravity enriched with cinematic rendering inspired
by general relativistic astrophysics (Kerr metric, Schwarzschild geometry,
accretion disc magnetohydrodynamics, gravitational lensing).



IMPORTANT: FOR ALL MENTIONS OF THE PARTICLE AGE IN THIS FILE, IT REFERS TO THE PARTICLE TEMPERATURE AS VISUALLY SEEN BY THE COLOR IN THE SIMULATION. THE AGE VARIABLE IS HOW THE COLOR IS DICTATED IN THE SIMULATOR HOWEVER, IN THE THEORETICAL SIDE OF PHYSICS, THE PARTICLES WILL HAVE DIFFERING ENERGY STATES. SO THE "OLDER" THE PARTICLE IS, THE LESS ENERGY IT WILL HAVE BEFORE IT EVENTUALLY FADES. THIS CAN BE SEEN IN THE LIFE CYCLE OF STARS AS MAIN SEQUENCE STARS FADE INTO WHITE DWARFS. AND FOR THE SAKE OF THE SIMULATION, ALL THESE PARTICLES ARE TREATED AS NEAR-MASSLESS AND THE BLACK HOLE(S) GROW SLIGHTLY WHEN CONSUMING THE PARTICLES. THEIR INDIVIDUAL MEANINGS ARE ALSO VISIBLE IN THE LEGEND AS SEEN IN THE USER INTERFACE IN THE SIMULATION. THANK YOU FOR READING THIS. ENJOY CREATING BLACK HOLES MY SIMULATION!

Physics concepts modelled or visualised: (this was a copy-paste list that i added for reference here. i tried learning about these complex concepts and revisited your presentation as well)
  Newton's Law of Universal Gravitation : F = G·M1·M2 / r²
  Gravitational Potential               : Φ = −G·M / r
  Conservation of Momentum              : Σ mi·vi = const  (inelastic mergers)
  Conservation of Energy                : E = K + U = const
  Keplerian Circular Velocity           : v = √(G·M / r)
  Escape Velocity                       : ve = √(2·G·M / r)
  Schwarzschild Radius (analogue)       : rs = 2·G·M / c²
  Photon Sphere                         : r_ph ≈ 1.5·rs  (photons orbit here)
  Kerr Metric / Frame Dragging          : a = J/(M·c)  spin parameter
  Ergosphere                            : frame-dragging exceeds c
  Accretion Luminosity                  : L ∝ G·M·ṁ / r_inner
  Relativistic Doppler / Beaming        : I_obs = I_em · (1 + β·cos θ)^4
  Gravitational Lensing Deflection      : α = 4·G·M / (c²·b)
  Roche Limit / Tidal Disruption        : d = rm·(2·Mp/ms)^(1/3)
  Hill Sphere                           : rH = a·(m/3M)^(1/3)
  Orbital Inspiral / Gravitational Wave : dE/dt ∝ −G^4·M²·m²(M+m)/r^5
  Leapfrog / Velocity Verlet            : symplectic integrators  ΔE → 0
  Logarithmic Accretion Growth          : r ∝ log(1 + absorbed / κ)
  Magnetohydrodynamics (MHD)            : disc turbulence, Balbus-Hawley instability
  Mass-Energy Equivalence               : E = m·c²
  N-body Dynamics                       : all pairs interact via Newton
"""

import taichi as ti
import numpy as np
import math
import time

# ─── GPU init ──────────────────────────────────────────────────────────────────
try:
    ti.init(arch=ti.gpu, default_fp=ti.f32, fast_math=True)
    print("[GPU] Metal / CUDA backend initialised.")
except Exception:
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    print("[CPU] GPU unavailable — running on CPU.")

# ─── Window ────────────────────────────────────────────────────────────────────
W, H   = 1400, 900
ASPECT = W / H          # ≈ 1.555

# ─── Simulation limits ─────────────────────────────────────────────────────────
MAX_BH    = 12
MAX_P     = 1000
TRAIL_LEN = 30
SF_N      = 1400        # dense navy-sky star catalogue; lensing makes it feel spatial

# ─── Base physics constants ────────────────────────────────────────────────────
G_BASE   = 1.2e-4       # Gravitational constant (simulation units)
BH_M0    = 1.0          # Reference Schwarzschild mass
BH_R0    = 0.024        # Reference Schwarzschild radius analogue
SOFTEN   = 8e-4         # Plummer softening — prevents r→0 divergence
P_LIFE   = 260.0        # Tracer particle lifetime (sim-time units)
BOUNDARY = 2.8          # Outer boundary ≈ Hill sphere edge

# Integration (Velocity Verlet + Leapfrog)
DT_BASE  = 0.018
DT_LOW   = 0.007
DT_HIGH  = 0.055
SUBSTEPS = 2
ORBIT_SEPARATION = 0.24  # compact binary: visibly fast, still safely outside capture radius

# ─── Taichi fields — GPU-resident simulation state ─────────────────────────────
bh_act    = ti.field(ti.i32,  shape=MAX_BH)
bh_pos    = ti.Vector.field(3, ti.f32, shape=MAX_BH)
bh_vel    = ti.Vector.field(3, ti.f32, shape=MAX_BH)
bh_acc    = ti.Vector.field(3, ti.f32, shape=MAX_BH)
bh_acc_n  = ti.Vector.field(3, ti.f32, shape=MAX_BH)
bh_mass   = ti.field(ti.f32,  shape=MAX_BH)
bh_rad    = ti.field(ti.f32,  shape=MAX_BH)
bh_absorb = ti.field(ti.f32,  shape=MAX_BH)   # accreted mass (for log growth)

# Tracer particles — restricted three-body analogues (massless geodesics)
p_pos  = ti.Vector.field(3, ti.f32, shape=MAX_P)
p_vel  = ti.Vector.field(3, ti.f32, shape=MAX_P)
p_acc  = ti.Vector.field(3, ti.f32, shape=MAX_P)
p_age  = ti.field(ti.f32,  shape=MAX_P)

# Trail ring-buffer — records geodesic paths for spectral visualisation
trail_pos  = ti.Vector.field(3, ti.f32, shape=(TRAIL_LEN, MAX_P))
trail_head = ti.field(ti.i32, shape=())

# Camera
cam_yaw   = ti.field(ti.f32, shape=())
cam_pitch = ti.field(ti.f32, shape=())
cam_dist  = ti.field(ti.f32, shape=())

# Projected screen coords — transferred to CPU once per frame
bh_sx = ti.field(ti.f32, shape=MAX_BH)
bh_sy = ti.field(ti.f32, shape=MAX_BH)
bh_sr = ti.field(ti.f32, shape=MAX_BH)
bh_sz = ti.field(ti.f32, shape=MAX_BH)

t_sx  = ti.field(ti.f32, shape=(TRAIL_LEN, MAX_P))
t_sy  = ti.field(ti.f32, shape=(TRAIL_LEN, MAX_P))

# Rendering / UI settings
g_dark    = ti.field(ti.i32, shape=())   # 0 = Light  /  1 = Heavy
g_quality = ti.field(ti.i32, shape=())   # 0 Low / 1 Med / 2 High
g_speed   = ti.field(ti.i32, shape=())   # 0 Low / 1 Med / 2 High

# Initialise camera & mode
cam_yaw[None]    = 0.65
cam_pitch[None]  = 0.50
cam_dist[None]   = 1.05
g_dark[None]     = 0
g_quality[None]  = 1
g_speed[None]    = 1
trail_head[None] = 0

# ─── Physics control parameters (Python-side; used as kernel arguments) ─────────
# Named after astrophysical quantities for clarity in comments and labels.
phys = {
    "g_mult":         1.0,   # Gravitational constant multiplier (G_eff = G_BASE × g_mult)
    "bh_mass_mult":   1.0,   # BH mass factor (Schwarzschild mass scale)
    "spin":           0.65,  # Kerr spin parameter a ∈ [0,1]  (0=Schwarzschild, 1=extreme Kerr)
    "accretion_rate": 0.55,  # Disc surface density / particle spawn rate
    "disc_temp":      0.70,  # Disc colour temperature (Planck peak: blue-white → amber)
    "lensing_str":    0.85,  # Gravitational lensing deflection strength (Einstein angle scale)
    "orbital_vel":    1.00,  # Orbital velocity scale (Keplerian v_orb multiplier)
    "companion_mass": 1.00,  # Secondary BH mass ratio
    "tidal_str":      1.00,  # Tidal disruption force multiplier (Roche limit sensitivity)
    "time_dilation":  1.00,  # Relativistic time dilation visual factor (Lorentz γ)
    "turb_strength":  0.65,  # MHD turbulence amplitude (Balbus-Hawley instability scale)
    "merger_thresh":  0.92,  # Capture threshold as fraction of event-horizon sum
    "brightness":     1.00,  # Accretion disc emission brightness (luminosity multiplier)
    "growth_damp":    0.06,  # Logarithmic accretion growth damping κ (prevents inflation)
    "absorb_eff":     0.72,  # Absorption efficiency η (Novikov-Thorne accretion efficiency)
    "timescale":      1.00,  # Global simulation timescale multiplier
}

# ─── GPU helper functions (ti.func — inlined, no overhead) ─────────────────────

@ti.func
def bh_grav_acc(pos: ti.template(), skip: ti.i32, g_eff: ti.f32) -> ti.Vector:
    """
    Newton's Law of Gravitation on BH i from all other active BHs.
    a_i = Σ_{j≠i} G·M_j · (r_j − r_i) / |r_j − r_i|³
    Plummer softening: |r|² → |r|² + ε²  prevents divergence at r=0.
    """
    a = ti.Vector([0.0, 0.0, 0.0])
    for j in range(MAX_BH):
        if j != skip and bh_act[j]:
            rv   = bh_pos[j] - pos
            r2   = rv.dot(rv) + SOFTEN * SOFTEN
            dist = ti.sqrt(r2)
            a   += (g_eff * bh_mass[j] / (r2 * dist)) * rv
    return a

@ti.func
def particle_acc_fn(pos: ti.template(), g_eff: ti.f32) -> ti.Vector:
    """
    Gravitational acceleration on a massless test particle (geodesic proxy).
    Restricted Three-Body: particle has negligible mass → no back-reaction.
    Used in Leapfrog: a = ∇Φ  where  Φ = −Σ G·M_j / |r − r_j|
    """
    a = ti.Vector([0.0, 0.0, 0.0])
    for j in range(MAX_BH):
        if bh_act[j]:
            rv   = bh_pos[j] - pos
            r2   = rv.dot(rv) + SOFTEN * SOFTEN
            dist = ti.sqrt(r2)
            a   += (g_eff * bh_mass[j] / (r2 * dist)) * rv
    return a

@ti.func
def respawn_particle(i: ti.i32, g_eff: ti.f32):
    """
    Recycle tracer particle near a randomly chosen black hole.
    Spawn outside Schwarzschild radius in a spherical shell.
    Assign partial Keplerian orbital velocity: v_circ = √(G·M / r)
    — approximates bound Lagrangian orbit, not free-fall geodesic.
    Disc-biased z-compression gives disc-like spatial distribution.
    """
    n_bh = 0
    for k in range(MAX_BH):
        if bh_act[k]:
            n_bh += 1
    if n_bh > 0:
        target = int(ti.random() * n_bh)
        sel = 0
        cnt = 0
        for k in range(MAX_BH):
            if bh_act[k]:
                if cnt == target:
                    sel = k
                    break
                cnt += 1
        spread   = 0.40
        r        = bh_rad[sel] * 2.15 + ti.random() * spread
        theta    = ti.random() * (2.0 * math.pi)
        phi      = ti.acos(2.0 * ti.random() - 1.0)   # uniform on sphere
        sp = ti.sin(phi);  cp = ti.cos(phi)
        st = ti.sin(theta); ct = ti.cos(theta)
        off = ti.Vector([r * sp * ct, r * sp * st, r * cp * 0.22])
        pos_new = bh_pos[sel] + off
        v_circ  = ti.sqrt(g_eff * bh_mass[sel] / (r + SOFTEN))   # Keplerian v_circ
        tangent = ti.Vector([-off.y, off.x, 0.0])
        t_len   = tangent.norm()
        if t_len > 1e-6:
            tangent = tangent / t_len
        p_pos[i] = pos_new
        p_vel[i] = bh_vel[sel] + tangent * v_circ * 0.55
        p_acc[i] = particle_acc_fn(pos_new, g_eff)
        p_age[i] = P_LIFE
    else:
        p_pos[i] = ti.Vector([(ti.random()-0.5)*1.2, (ti.random()-0.5)*1.2, (ti.random()-0.5)*0.15])
        p_vel[i] = ti.Vector([0.0, 0.0, 0.0])
        p_acc[i] = ti.Vector([0.0, 0.0, 0.0])
        p_age[i] = P_LIFE

@ti.func
def project_pt(wp: ti.template(), eye: ti.template(),
               xa: ti.template(), ya: ti.template(), za: ti.template()) -> ti.Vector:
    """Perspective projection: 3-D world → normalised screen [0, ASPECT] × [0,1]."""
    d  = wp - eye
    px = d.dot(xa);  py = d.dot(ya);  pz = d.dot(za)
    f  = 0.85
    return ti.Vector([(px / (-pz + 1e-6)) * f + 0.5 * ASPECT,
                      (py / (-pz + 1e-6)) * f + 0.5])

# ─── Initialisation kernel ─────────────────────────────────────────────────────

@ti.kernel
def init_simulation(g_eff: ti.f32, v_scale: ti.f32):
    """
    Initialise binary black hole system in stable circular Keplerian orbit.
    Velocity Verlet requires acceleration at t=0 for first integration step.
    Tracer particles seeded with random ages → immediate full spectral diversity.
    """
    for i in range(MAX_BH):
        bh_act[i]    = 0
        bh_pos[i]    = ti.Vector([0.0, 0.0, 0.0])
        bh_vel[i]    = ti.Vector([0.0, 0.0, 0.0])
        bh_acc[i]    = ti.Vector([0.0, 0.0, 0.0])
        bh_acc_n[i]  = ti.Vector([0.0, 0.0, 0.0])
        bh_mass[i]   = BH_M0
        bh_rad[i]    = BH_R0
        bh_absorb[i] = 0.0

    # Circular orbit: each BH orbits CoM at distance sep/2
    # v_orb = √(G·M / sep)  — from centripetal force balance: G·M·m/sep² = m·v²/(sep/2)
    sep   = ORBIT_SEPARATION
    half  = sep * 0.5
    v_orb = ti.sqrt(g_eff * BH_M0 / sep) * v_scale

    bh_act[0] = 1;  bh_pos[0] = ti.Vector([-half, 0.0, 0.0]);  bh_vel[0] = ti.Vector([0.0,  v_orb, 0.0])
    bh_act[1] = 1;  bh_pos[1] = ti.Vector([ half, 0.0, 0.0]);  bh_vel[1] = ti.Vector([0.0, -v_orb, 0.0])

    for i in range(MAX_BH):
        if bh_act[i]:
            bh_acc[i] = bh_grav_acc(bh_pos[i], i, g_eff)

    # Seed particles across full age range → Red→Orange→Yellow→Green→Cyan→Blue all visible
    for i in range(MAX_P):
        p_age[i] = ti.random() * P_LIFE
        angle    = ti.random() * 2.0 * math.pi
        r_s      = 0.08 + ti.random() * 0.55
        p_pos[i] = ti.Vector([r_s * ti.cos(angle), r_s * ti.sin(angle), (ti.random()-0.5)*0.18])
        v_t      = ti.sqrt(ti.max(g_eff * BH_M0 / (r_s + SOFTEN), 0.0)) * 0.55
        p_vel[i] = ti.Vector([-ti.sin(angle)*v_t, ti.cos(angle)*v_t, 0.0])
        p_acc[i] = particle_acc_fn(p_pos[i], g_eff)
        for s in range(TRAIL_LEN):
            trail_pos[s, i] = p_pos[i]
    trail_head[None] = 0

# ─── Physics update ────────────────────────────────────────────────────────────

@ti.kernel
def update_physics(dt: ti.f32, g_eff: ti.f32, merge_r: ti.f32,
                   absorb_eff: ti.f32, growth_damp: ti.f32):
    """
    Velocity Verlet for BHs (symplectic — conserves phase-space volume):
      x(t+dt) = x + v·dt + ½·a·dt²
      a_new   = ∇Φ(x(t+dt))
      v(t+dt) = v + ½·(a + a_new)·dt
    Leapfrog (kick-drift-kick) for tracer particles — cheaper, equally symplectic.
    Logarithmic accretion growth: r(t) = r₀·(1 + log(1 + absorbed/κ))
    prevents runaway Schwarzschild radius inflation despite continuous absorption.
    """
    # Step 1 — Verlet position update
    for i in range(MAX_BH):
        if bh_act[i]:
            bh_pos[i] += bh_vel[i] * dt + 0.5 * bh_acc[i] * (dt * dt)

    # Step 2 — New accelerations (gravitational potential gradient)
    for i in range(MAX_BH):
        bh_acc_n[i] = ti.Vector([0.0, 0.0, 0.0])
        if bh_act[i]:
            bh_acc_n[i] = bh_grav_acc(bh_pos[i], i, g_eff)

    # Step 3 — Verlet velocity update
    for i in range(MAX_BH):
        if bh_act[i]:
            bh_vel[i] += 0.5 * (bh_acc[i] + bh_acc_n[i]) * dt
            bh_acc[i]  = bh_acc_n[i]

    # Inelastic merger — conservation of momentum: p_total = m1v1 + m2v2
    # Schwarzschild radius: logarithmic growth suppresses visual inflation
    for i in range(MAX_BH):
        if bh_act[i]:
            for j in range(i+1, MAX_BH):
                if bh_act[j]:
                    rv   = bh_pos[j] - bh_pos[i]
                    dist = rv.norm()
                    if dist < (bh_rad[i] + bh_rad[j]) * merge_r:
                        m1 = bh_mass[i]; m2 = bh_mass[j]; mt = m1 + m2
                        bh_pos[i]    = (bh_pos[i]*m1 + bh_pos[j]*m2) / mt
                        bh_vel[i]    = (bh_vel[i]*m1 + bh_vel[j]*m2) / mt
                        bh_mass[i]   = mt
                        bh_absorb[i] += m2 * absorb_eff
                        # log(1+x) growth: heavily damped — large x → small Δr
                        r_new = BH_R0 * (1.0 + ti.log(1.0 + bh_absorb[i] /
                                ti.max(growth_damp * 80.0, 0.001)))
                        bh_rad[i]    = r_new
                        bh_act[j]    = 0

    # Leapfrog for tracer particles (kick-drift-kick):
    #   v½ = v + ½·a·dt  |  x_new = x + v½·dt  |  a_new = f(x_new)  |  v_new = v½ + ½·a_new·dt
    for i in range(MAX_P):
        if p_age[i] <= 0.0 or p_pos[i].norm() > BOUNDARY:
            respawn_particle(i, g_eff)
        else:
            inside = False
            for j in range(MAX_BH):
                if bh_act[j]:
                    if (p_pos[i] - bh_pos[j]).norm() < bh_rad[j] * merge_r * 0.9:
                        inside = True
            if inside:
                # Tidal disruption — particle absorbed by event horizon
                for j in range(MAX_BH):
                    if bh_act[j]:
                        if (p_pos[i] - bh_pos[j]).norm() < bh_rad[j] * 2.0:
                            bh_absorb[j] += absorb_eff * 0.0008
                            r_n = BH_R0 * (1.0 + ti.log(1.0 + bh_absorb[j] /
                                  ti.max(growth_damp * 80.0, 0.001)))
                            bh_rad[j] = r_n
                respawn_particle(i, g_eff)
            else:
                v_half   = p_vel[i] + 0.5 * p_acc[i] * dt
                p_pos[i] += v_half * dt
                a_new    = particle_acc_fn(p_pos[i], g_eff)
                p_vel[i] = v_half + 0.5 * a_new * dt
                p_acc[i] = a_new
                p_age[i] -= dt

    # Trail ring-buffer — stores geodesic path for spectral trail rendering
    h = trail_head[None]
    for i in range(MAX_P):
        trail_pos[h, i] = p_pos[i]
    trail_head[None] = (h + 1) % TRAIL_LEN

# ─── Projection kernel ─────────────────────────────────────────────────────────

@ti.kernel
def project_scene():
    """CAD orbit camera: spherical coord → perspective projection basis."""
    pitch = cam_pitch[None];  yaw = cam_yaw[None];  dist = cam_dist[None] * 0.8
    eye   = dist * ti.Vector([ti.cos(pitch)*ti.cos(yaw),
                               ti.cos(pitch)*ti.sin(yaw),
                               ti.sin(pitch)])
    za  = eye.normalized()
    up  = ti.Vector([0.0, 0.0, 1.0])
    xa  = up.cross(za).normalized()
    ya  = za.cross(xa)
    for i in range(MAX_BH):
        if bh_act[i]:
            sc        = project_pt(bh_pos[i], eye, xa, ya, za)
            bh_sx[i]  = sc.x;  bh_sy[i] = sc.y
            dv        = bh_pos[i] - eye
            pz        = dv.dot(za)
            bh_sz[i]  = pz
            bh_sr[i]  = bh_rad[i] / (-pz + 1e-6) * 0.85
    for s in range(TRAIL_LEN):
        for i in range(MAX_P):
            sc          = project_pt(trail_pos[s, i], eye, xa, ya, za)
            t_sx[s, i]  = sc.x
            t_sy[s, i]  = sc.y

# ─── Add BH kernel ─────────────────────────────────────────────────────────────

@ti.kernel
def kernel_add_bh(idx: ti.i32, wx: ti.f32, wy: ti.f32, wz: ti.f32,
                  vx: ti.f32, vy: ti.f32, vz: ti.f32, g_eff: ti.f32):
    bh_act[idx]    = 1
    bh_pos[idx]    = ti.Vector([wx, wy, wz])
    bh_vel[idx]    = ti.Vector([vx, vy, vz])
    bh_mass[idx]   = BH_M0
    bh_rad[idx]    = BH_R0
    bh_absorb[idx] = 0.0
    bh_acc[idx]    = bh_grav_acc(ti.Vector([wx, wy, wz]), idx, g_eff)

@ti.kernel
def refresh_bh_accelerations(g_eff: ti.f32):
    """Refresh every BH after a user spawn changes the gravitational field."""
    for i in range(MAX_BH):
        if bh_act[i]:
            bh_acc[i] = bh_grav_acc(bh_pos[i], i, g_eff)

# ─── Pre-baked assets ──────────────────────────────────────────────────────────

def _build_lut(n=512):
    """
    Spectral colour LUT: particle age fraction [0=fading→blue, 1=fresh→red].
    Progression: Red → Orange → Yellow → Green → Cyan → Blue → fade
    Represents particle against time (red is older, blue is newer)
    """
    lut = np.zeros(n, dtype=np.uint32)
    for k in range(n):
        f   = k / (n-1)
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

# Deep cosmological starfield — power-law brightness (few bright, many dim)
_srng  = np.random.default_rng(42)
SF_X   = _srng.uniform(0, 1, SF_N).astype(np.float32)
SF_Y   = _srng.uniform(0, 1, SF_N).astype(np.float32)
SF_B   = (_srng.uniform(0,1,SF_N)**3.2*0.72+0.08).astype(np.float32)
SF_RAD = (_srng.uniform(0,1,SF_N)*1.25+0.35).astype(np.float32)
_stype = _srng.random(SF_N)
SF_COL = np.zeros(SF_N, dtype=np.uint32)
for _k in range(SF_N):
    _b = float(SF_B[_k])
    if   _stype[_k] > 0.85: SF_COL[_k]=(int(_b*180)<<16)|(int(_b*210)<<8)|int(_b*255)  # O/B blue
    elif _stype[_k] > 0.60: SF_COL[_k]=(int(_b*255)<<16)|(int(_b*255)<<8)|int(_b*220)  # A/F white
    else:                    SF_COL[_k]=(int(_b*255)<<16)|(int(_b*190)<<8)|int(_b*140)  # G/K/M warm

# ─── Accretion disc pre-computation ────────────────────────────────────────────
# These arrays define the structural "DNA" of the Keplerian disc.
# Rotation is applied each frame; turbulence = MHD magnetorotational instability.
N_DISC    = 4600        # textured emission filaments, not a sparse dot cloud
_drng     = np.random.default_rng(99)
_r_frac   = _drng.beta(1.8, 4.5, N_DISC).astype(np.float32)   # inner-edge heavy
DISC_IN   = 2.15   # inner truncation ≈ innermost stable circular orbit (ISCO)
DISC_OUT  = 7.2    # outer truncation (magnetic pressure / tidal truncation)
disc_r_loc = (BH_R0*(DISC_IN + _r_frac*(DISC_OUT - DISC_IN))).astype(np.float32)
disc_theta = _drng.uniform(0, 2*np.pi, N_DISC).astype(np.float32)
disc_z_loc = (_drng.normal(0, BH_R0*0.28, N_DISC)*(1.0-_r_frac*0.5)).astype(np.float32)
disc_bri   = _drng.uniform(0.42, 1.0, N_DISC).astype(np.float32)
disc_size  = _drng.uniform(1.5, 5.0, N_DISC).astype(np.float32)
# Turbulence: Balbus-Hawley MHD instability causes chaotic mass transport
disc_tph   = _drng.uniform(0, 2*np.pi, N_DISC).astype(np.float32)
disc_tr    = (_drng.uniform(0, 0.18, N_DISC)*(1.0-_r_frac*0.4)).astype(np.float32)
disc_tz    = _drng.uniform(0, BH_R0*0.14, N_DISC).astype(np.float32)
disc_filament = _drng.uniform(0.018, 0.105, N_DISC).astype(np.float32)
disc_lane = _drng.uniform(-1.0, 1.0, N_DISC).astype(np.float32)

# ─── Physics slider widget ─────────────────────────────────────────────────────

class PhysicsSlider:
    """Click-drag slider for ti.GUI physics controls."""
    def __init__(self, key, label, x, y, w, h, mn, mx, fmt="{:.2f}"):
        self.key=key; self.label=label
        self.x=x; self.y=y; self.w=w; self.h=h
        self.mn=mn; self.mx=mx; self.fmt=fmt
        self.active=False

    def draw(self, gui, val):
        t = (val-self.mn)/max(self.mx-self.mn,1e-6)
        gui.rect((self.x, self.y),(self.x+self.w, self.y+self.h),             color=0x080820)
        gui.rect((self.x, self.y),(self.x+self.w*t, self.y+self.h),           color=0x1E3A6A)
        gui.circle((self.x+self.w*t, self.y+self.h*0.5), color=0x77AAFF, radius=4)
        gui.text(self.label,(self.x, self.y+self.h+0.003),   font_size=9, color=0x7788BB)
        gui.text(self.fmt.format(val),(self.x+self.w+0.006,self.y+0.001), font_size=9, color=0xFFCC66)

    def try_update(self, mx, my, lmb):
        hit=(self.x-0.005<=mx<=self.x+self.w+0.005 and self.y-0.012<=my<=self.y+self.h+0.018)
        if lmb and (hit or self.active):
            self.active=True
            t=max(0.0,min(1.0,(mx-self.x)/max(self.w,1e-6)))
            return self.mn+t*(self.mx-self.mn)
        else:
            self.active=False
        return None

_PX=0.805; _PW=0.160; _SH=0.010; _GAP=0.048

def _sl(key, lbl, row, mn, mx, fmt="{:.2f}"):
    return PhysicsSlider(key, lbl, _PX, 0.935-row*_GAP, _PW, _SH, mn, mx, fmt)

SLIDERS = [
    _sl("g_mult",         "G  Multiplier",            0,  0.1,  5.0),
    _sl("bh_mass_mult",   "BH  Mass",                 1,  0.2,  5.0),
    _sl("spin",           "Kerr  Spin  a",             2,  0.0,  0.999),
    _sl("accretion_rate", "Accretion  Rate",           3,  0.0,  1.0),
    _sl("disc_temp",      "Disc  Temperature",         4,  0.0,  1.0),
    _sl("lensing_str",    "Lensing  Strength",         5,  0.0,  3.0),
    _sl("orbital_vel",    "Orbital  Velocity",         6,  0.3,  2.5),
    _sl("companion_mass", "Companion  Mass  Ratio",    7,  0.1,  3.0),
    _sl("tidal_str",      "Tidal  Force",              8,  0.0,  3.0),
    _sl("time_dilation",  "Time  Dilation  γ",         9,  0.0,  1.0),
    _sl("turb_strength",  "MHD  Turbulence",          10,  0.0,  2.0),
    _sl("merger_thresh",  "Merger  Threshold",        11,  0.3,  1.5),
    _sl("brightness",     "Disc  Brightness",         12,  0.1,  3.0),
    _sl("growth_damp",    "Growth  Damping  κ",       13,  0.01, 1.0),
    _sl("absorb_eff",     "Absorption  Eff.  η",      14,  0.0,  1.0),
    _sl("timescale",      "Simulation  Timescale",    15,  0.1,  3.0),
]

# ─── Legend entries ────────────────────────────────────────────────────────────
# Particle colour categories reflect physical state along the age spectrum
LEGEND = [
    (0xFF2200, "Fresh dust particles",       "age ≈ 1.0  (fresh)"),
    (0xFF8800, "Ionised plasma / shocked gas",       "age ≈ 0.8"),
    (0xFFDD00, "High-energy radiation / X-ray",     "age ≈ 0.6"),
    (0x33FF55, "Gravitationally bound orbit",        "age ≈ 0.4"),
    (0x00FFFF, "Cooling matter / inspiral stream",  "age ≈ 0.2"),
    (0x2255FF, "sd",       "age ≈ 0.0  (recycled)"),
]

# ─── Screen-space lensing (vectorised numpy) ────────────────────────────────────

def apply_lensing_np(sx, sy, bh_xs, bh_ys, bh_rs, strength):
    """
    Screen-space approximation of gravitational lensing.
    Einstein deflection angle: α = 4·G·M / (c²·b)  →  Δr ∝ rs² / b
    Operates on numpy arrays for batch efficiency.
    """
    ox = sx.copy().astype(np.float64)
    oy = sy.copy().astype(np.float64)
    for bx, by, br in zip(bh_xs, bh_ys, bh_rs):
        dx   = ox - bx;  dy = oy - by
        d    = np.sqrt(dx*dx + dy*dy) + 1e-8
        # Lensing radius in [0,1]-space: sr / ASPECT
        lens_r = br / ASPECT
        defl = strength * (lens_r * lens_r * 2.0) / (d + 1e-6)
        mask = d > lens_r * 0.6
        ox   = np.where(mask, bx + (dx/d)*(d+defl), ox)
        oy   = np.where(mask, by + (dy/d)*(d+defl), oy)
    return ox.astype(np.float32), oy.astype(np.float32)

# ─── Dark Mode BH rendering (Interstellar-inspired) ───────────────────────────

def render_dark_bh(gui, i, sx_i, sy_i, r_px, sr_i, bh_pos_i, bh_rad_i,
                   eye_np, xa_np, ya_np, za_np, yaw,
                   spin, disc_temp_v, turb_str, bright, quality, t_now):
    """
    Cinematic volumetric accretion disc:
    - Doppler-beamed chromatic gradient (blue-white approaching / amber receding)
    - MHD turbulence filaments (Balbus-Hawley instability)
    - Photon ring at r_ph ≈ (1.5 − a*0.3) × rs  (Kerr correction)
    - Frame-dragging ergosphere spiral (Kerr metric, a > 0)
    - Event horizon silhouette
    Uses gui.circles() batch API — single draw call for entire disc.
    """
    fov_f  = 0.85
    n_d    = N_DISC if quality==2 else (int(N_DISC*0.72) if quality==1 else int(N_DISC*0.45))

    # Angular velocity from Kerr spin: frame-dragging enhances ω
    omega  = 2.1 + spin * 3.8
    t_rot  = t_now * omega

    # Rotating disc with MHD turbulence (Magneto-Rotational Instability)
    theta_now = disc_theta[:n_d] + t_rot
    turb_r    = disc_r_loc[:n_d] * (1.0 + disc_tr[:n_d]*np.sin(disc_tph[:n_d]+omega*0.27*t_now)*turb_str)
    turb_r    = np.maximum(turb_r, bh_rad_i * 2.0)
    turb_z    = disc_z_loc[:n_d] + disc_tz[:n_d]*np.sin(disc_tph[:n_d]*1.7+omega*0.14*t_now)*turb_str

    # 3D disc positions in world space
    x3 = bh_pos_i[0] + turb_r * np.cos(theta_now)
    y3 = bh_pos_i[1] + turb_r * np.sin(theta_now)
    z3 = bh_pos_i[2] + turb_z

    # Batch perspective projection
    dv  = np.stack([x3-eye_np[0], y3-eye_np[1], z3-eye_np[2]], axis=1)
    px_ = dv @ xa_np;  py_ = dv @ ya_np;  pz_ = dv @ za_np
    ok  = pz_ < -0.01
    sc_x = (px_/(-pz_+1e-6))*fov_f + 0.5*ASPECT
    sc_y = (py_/(-pz_+1e-6))*fov_f + 0.5

    # Relativistic Doppler beaming:
    # Disc rotates CCW; tangential velocity: v = ω·r·(−sinθ, cosθ, 0)
    # Approaching when v · (−cos yaw, −sin yaw, 0) > 0 → doppler_raw = sin(θ − yaw)
    # Beaming: I_obs/I_em ≈ (1 + β·doppler)^2.5  (simplified from exponent 4)
    doppler_raw = np.sin(theta_now - yaw).astype(np.float32)
    beta        = np.float32(0.28 + spin * 0.28)   # effective v/c; spin increases beaming
    beam        = np.clip((1.0 + beta * doppler_raw)**2.5, 0.05, 8.0)

    # Filmic blackbody palette: electric blue/cyan on the approaching edge,
    # gold, rose and molten orange braided through the receding plasma lanes.
    # This deliberately avoids the old flat brown-to-white split.
    temp = np.float32(0.4 + disc_temp_v * 0.6)
    filament = np.clip(0.58 + 0.30*np.sin(5.0*theta_now - 8.0*_r_frac[:n_d] - 2.8*t_now)
                       + 0.16*np.sin(13.0*theta_now + disc_lane[:n_d]*4.0), 0.12, 1.0)
    cool = np.clip(0.50 + 0.50*doppler_raw, 0.0, 1.0)
    hot = np.clip(1.0 - cool, 0.0, 1.0)
    r_f = np.clip((0.98*hot + 0.35*cool + 0.30*filament) * beam * temp, 0, 1)
    g_f = np.clip((0.30*hot + 0.72*cool + 0.22*filament) * beam * temp, 0, 1)
    b_f = np.clip((0.10*hot + 1.00*cool + 0.34*filament) * beam * temp, 0, 1)

    # Radial brightness: inner disc hottest — accretion luminosity ∝ 1/r (Novikov-Thorne)
    radfall = np.clip((1.0 - _r_frac[:n_d])**1.05 * disc_bri[:n_d] * filament * bright, 0.0, 1.0)
    r_col   = np.clip(r_f * radfall, 0, 1)
    g_col   = np.clip(g_f * radfall, 0, 1)
    b_col   = np.clip(b_f * radfall, 0, 1)
    c_int   = ((r_col*255).astype(np.int32)<<16)|((g_col*255).astype(np.int32)<<8)|(b_col*255).astype(np.int32)

    # Screen-space lensing on disc points (Einstein deflection, fast approximation)
    sc_nx = sc_x / ASPECT;  sc_ny = sc_y
    sc_nx_l, sc_ny_l = apply_lensing_np(sc_nx, sc_ny, [sx_i], [sy_i],
                                         [sr_i], 0.55)

    # Cull: behind camera, off-screen, or inside event horizon silhouette
    eh_r = sr_i / ASPECT                      # event horizon normalised radius
    d_to_bh = np.sqrt((sc_nx_l-sx_i)**2 + (sc_ny_l-sy_i)**2)
    vis = ok & (np.abs(sc_nx_l-0.5)<0.85) & (np.abs(sc_ny_l-0.5)<0.85) & (d_to_bh > eh_r*0.85)

    if np.any(vis):
        # Each DNA sample becomes a sheared luminous filament.  Thousands of
        # curved, lensed strokes read as turbulent plasma rather than particles.
        theta_end = theta_now + disc_filament[:n_d] * (1.0 + 0.65*(1.0-_r_frac[:n_d]))
        r_end = turb_r * (1.0 + 0.018*np.sin(9.0*theta_now + t_now))
        end_3d = np.stack([bh_pos_i[0] + r_end*np.cos(theta_end),
                           bh_pos_i[1] + r_end*np.sin(theta_end),
                           bh_pos_i[2] + turb_z + disc_tz[:n_d]*0.35], axis=1)
        d_end = end_3d - eye_np[np.newaxis, :]
        ex = d_end @ xa_np; ey = d_end @ ya_np; ez = d_end @ za_np
        sx_end = (ex/(-ez+1e-6))*fov_f/ASPECT + 0.5
        sy_end = (ey/(-ez+1e-6))*fov_f + 0.5
        sx_end, sy_end = apply_lensing_np(sx_end, sy_end, [sx_i], [sy_i], [sr_i], 0.55)
        vis &= (ez < -0.01) & (np.abs(sx_end-0.5)<0.85) & (np.abs(sy_end-0.5)<0.85)
        begins = np.stack([sc_nx_l[vis], sc_ny_l[vis]], axis=1).astype(np.float32)
        ends = np.stack([sx_end[vis], sy_end[vis]], axis=1).astype(np.float32)
        cols = c_int[vis].astype(np.uint32)
        if len(begins):
            gui.lines(begins, ends, radius=1.0 + 1.8*float(np.mean(radfall[vis])), color=cols)

    # Photon ring — photons on marginally unstable circular orbits
    # Kerr: r_ph ∈ [rs, 3rs] depending on spin direction; simplified: 1.5 − a*0.3
    ph_r    = bh_rad_i * (1.5 - spin * 0.32)
    n_ph    = 52 if quality>0 else 26
    ph_ang  = np.linspace(0, 2*np.pi, n_ph, endpoint=False)
    ph_3d   = np.stack([bh_pos_i[0]+ph_r*np.cos(ph_ang),
                         bh_pos_i[1]+ph_r*np.sin(ph_ang),
                         np.full(n_ph, bh_pos_i[2])], axis=1)
    d_ph    = ph_3d - eye_np[np.newaxis,:]
    px_ph   = d_ph@xa_np; py_ph=d_ph@ya_np; pz_ph=d_ph@za_np
    v_ph    = pz_ph < -0.01
    if np.any(v_ph):
        sx_ph  = (px_ph/(-pz_ph+1e-6))*fov_f/ASPECT + 0.5
        sy_ph  = (py_ph/(-pz_ph+1e-6))*fov_f + 0.5
        gui.circles(np.stack([sx_ph[v_ph],sy_ph[v_ph]],axis=1).astype(np.float32),
                    radius=2.2, color=0xFFEEBB)

    # Frame-dragging ergosphere spiral — visible Kerr metric effect (a > 0)
    # Ergosphere: region where no static observer can exist; spacetime dragged
    if spin > 0.08 and quality > 0:
        n_erg  = 32 if quality>1 else 18
        erg_r  = bh_rad_i * (1.9 - spin*0.45)
        s_ang  = np.linspace(0, 2*np.pi*(1+spin), n_erg)
        s_r    = np.linspace(erg_r, erg_r*1.7, n_erg)
        erg_off= t_now * omega * 0.5
        ex = bh_pos_i[0]+s_r*np.cos(s_ang+erg_off)
        ey = bh_pos_i[1]+s_r*np.sin(s_ang+erg_off)
        ez = np.full(n_erg, bh_pos_i[2])
        d_e  = np.stack([ex-eye_np[0],ey-eye_np[1],ez-eye_np[2]],axis=1)
        px_e = d_e@xa_np; py_e=d_e@ya_np; pz_e=d_e@za_np
        ve   = pz_e < -0.01
        if np.any(ve):
            se_x = (px_e/(-pz_e+1e-6))*fov_f/ASPECT + 0.5
            se_y = (py_e/(-pz_e+1e-6))*fov_f + 0.5
            gui.circles(np.stack([se_x[ve],se_y[ve]],axis=1).astype(np.float32),
                        radius=1.4, color=0xAA44FF)

    # Event horizon — pure black silhouette (Schwarzschild / Kerr boundary)
    gui.circle((sx_i,sy_i), color=0x000000, radius=1.5*r_px)

# ─── Light Mode BH rendering (inherited dark-mode rings + 3D sphere depth) ────

def render_light_bh(gui, i, sx_i, sy_i, r_px, bh_pos_i, bh_rad_i,
                    eye_np, xa_np, ya_np, za_np, quality, t_now):
    """
    Light-mode : preserves original dark-mode geometric accretion ring look
    (rotating elliptical rings projected in 3D perspective) augmented with
    subtle spherical shading (Lambertian gradient) and a thin photon ring.
    Light-mode particle trails (rainbow spectral colour) are unchanged.
    """
    fov_f = 0.85
    rot_t = t_now * 2.3
    n_seg = 24 if quality>0 else 16

    # Geometric accretion rings (original dark-mode appearance, preserved here).
    # The segments from every ring are submitted in one batch, avoiding per-line
    # Python GUI calls while retaining the original rotating ring geometry.
    ring_begins = []
    ring_ends = []
    for r_mult in (0.85, 1.0, 1.15):
        disk_r = bh_rad_i * 2.6 * r_mult
        pts    = []
        for si2 in range(n_seg+1):
            ang = rot_t + si2*(2.0*np.pi/n_seg)
            off = np.array([disk_r*np.cos(ang), disk_r*np.sin(ang), 0.0])
            p3  = bh_pos_i + off
            dv  = p3 - eye_np
            px_ = dv.dot(xa_np); py_=dv.dot(ya_np); pz_=dv.dot(za_np)
            sx_ = (px_/(-pz_+1e-6))*fov_f + 0.5*ASPECT
            sy_ = (py_/(-pz_+1e-6))*fov_f + 0.5
            pts.append((sx_/ASPECT, sy_))
        ring_pts = np.asarray(pts, dtype=np.float32)
        starts, ends = ring_pts[:-1], ring_pts[1:]
        valid = (np.abs(starts[:, 0]-ends[:, 0]) < 0.15) & (np.abs(starts[:, 1]-ends[:, 1]) < 0.15)
        if np.any(valid):
            ring_begins.append(starts[valid])
            ring_ends.append(ends[valid])
    if ring_begins:
        gui.lines(np.concatenate(ring_begins), np.concatenate(ring_ends),
                  radius=1.0, color=0xFFAA22)

    # 3D spherical shading — radial gradient gives dimensionality (Lambertian shading)
    n_shade = 6 if quality>0 else 3
    for layer in range(n_shade):
        frac   = layer / n_shade
        sh     = int(25 + frac*90)
        gui.circle((sx_i - frac*r_px*0.25, sy_i + frac*r_px*0.12),
                   color=(sh<<16)|(sh*2//3<<8)|sh, radius=r_px*(1.0-frac*0.5))

    # Photon ring — thin luminous ring just outside event horizon
    gui.circle((sx_i,sy_i), color=0xFFCC77, radius=r_px*1.09)

    # Red-purple halo (synchrotron corona analogue)
    for layer in range(2 if quality>0 else 1):
        hr  = r_px*(1.45+layer*0.55)
        ho  = 0.24/(layer+1.0)
        hcr = int(min(255,0xFF*ho)); hcb=int(min(255,0xCC*ho))
        gui.circle((sx_i,sy_i), color=(hcr<<16)|hcb, radius=hr)

    gui.circle((sx_i,sy_i), color=0x000000, radius=r_px)

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("   ULTIMATE SPECTRAL CHAOS: N-BODY PROBLEM, BLACK HOLE MERGER")
    print("="*70)
    print("  Left Click     : Spawn black hole")
    print("  Right Drag     : Rotate camera")
    print("  Scroll / ↑↓   : Zoom")
    print("  SPACE          : Pause / Resume")
    print("  R              : Reset")
    print("  M              : Toggle Light / Heavy mode")
    print("  T              : Toggle particle trails")
    print("  P              : Toggle Physics Controls panel")
    print("="*70)

    gui = ti.GUI("ULTIMATE SPECTRAL CHAOS: N-BODY PROBLEM, BLACK HOLE MERGER", res=(W,H), background_color=0x060613)

    def _geff(): return float(G_BASE * phys["g_mult"])

    init_simulation(float(_geff()), float(phys["orbital_vel"]))

    def _launch_orbiting_bh(idx, wx, wy, wz, g_eff_v):
        """Place a user BH on a tangential, prograde orbit around the live system.

        A stationary spawn falls straight in and makes the scene feel inert.  The
        local circular-speed estimate instead gives each added BH angular motion
        immediately, while the real N-body solver remains responsible for its
        later slingshots, inspiral, and merger.
        """
        active = bh_act.to_numpy().astype(bool)
        masses = bh_mass.to_numpy()
        positions = bh_pos.to_numpy()
        if np.any(active):
            total_m = float(np.sum(masses[active]))
            center = np.sum(positions[active] * masses[active, None], axis=0) / max(total_m, 1e-6)
            com_vel = np.mean(bh_vel.to_numpy()[active], axis=0)
        else:
            total_m, center, com_vel = BH_M0, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        radial = np.array([wx-center[0], wy-center[1]], dtype=np.float32)
        radius = float(np.linalg.norm(radial))
        if radius < 0.055:
            # A click at the centre still receives a deterministic orbit plane.
            radial = np.array([0.055, 0.0], dtype=np.float32)
            radius = 0.055
            wx = float(center[0] + radial[0]); wy = float(center[1])
        tangent = np.array([-radial[1], radial[0]], dtype=np.float32) / radius
        # Slightly sub-circular: companions spiral and merge theatrically rather
        # than escaping the compact default binary.
        speed = math.sqrt(g_eff_v * total_m / max(radius, 0.055)) * 0.88 * float(phys["orbital_vel"])
        direction = 1.0 if (idx % 2 == 0) else -1.0
        vx = float(com_vel[0] + tangent[0] * speed * direction)
        vy = float(com_vel[1] + tangent[1] * speed * direction)
        vz = float(com_vel[2] - (wz-center[2]) * speed * 0.22 / max(radius, 0.055))
        kernel_add_bh(idx, float(wx), float(wy), float(wz), vx, vy, vz, g_eff_v)
        refresh_bh_accelerations(g_eff_v)

    paused       = False
    show_phys    = False
    show_particles = True
    prev_rmb     = False
    prev_mouse   = np.zeros(2)
    click_lock   = False
    last_t       = time.perf_counter()
    fps_s        = 60.0
    scroll_acc   = 0.0
    SIDEBAR_W    = 0.200
    PANEL_X0     = 0.802

    bh_rad_np_cache = bh_rad.to_numpy()   # cache for Python-side render funcs

    while gui.running:
        now    = time.perf_counter()
        dt_fps = now - last_t;  last_t = now
        if dt_fps > 0:
            fps_s = 0.93*fps_s + 0.07/dt_fps

        g_eff_v  = _geff()
        spd      = g_speed[None]
        dt_sim   = (DT_LOW if spd==0 else (DT_HIGH if spd==2 else DT_BASE)) * phys["timescale"]

        # Events
        for e in gui.get_events():
            if e.type == ti.GUI.PRESS:
                k = e.key
                if   k == ti.GUI.ESCAPE: gui.running=False
                elif k == " ":           paused=not paused
                elif k in ("r","R"):     init_simulation(g_eff_v, phys["orbital_vel"]); bh_rad_np_cache=bh_rad.to_numpy()
                elif k in ("m","M"):     g_dark[None]=1-g_dark[None]
                elif k in ("t","T"):     show_particles=not show_particles
                elif k in ("p","P"):     show_phys=not show_phys
                elif k == ti.GUI.UP:     cam_dist[None]=max(0.25, cam_dist[None]-0.08)
                elif k == ti.GUI.DOWN:   cam_dist[None]=min(5.0,  cam_dist[None]+0.08)
            elif e.type == ti.GUI.WHEEL:
                delta = getattr(e,"delta",None)
                if delta is not None:
                    scroll_acc += delta[1] if hasattr(delta,"__len__") else float(delta)
        if abs(scroll_acc)>0:
            cam_dist[None]=float(np.clip(cam_dist[None]-scroll_acc*0.0015,0.25,5.0))
            scroll_acc=0.0

        mp   = gui.get_cursor_pos()
        lmb  = gui.is_pressed(ti.GUI.LMB)
        rmb  = gui.is_pressed(ti.GUI.RMB)
        in_bar   = mp[0] < SIDEBAR_W
        in_panel = show_phys and mp[0] > PANEL_X0

        if rmb and not in_bar and not in_panel:
            if prev_rmb:
                dx=mp[0]-prev_mouse[0]; dy=mp[1]-prev_mouse[1]
                cam_yaw[None]  -= dx*3.5
                cam_pitch[None] = float(np.clip(cam_pitch[None]+dy*2.0,-0.15,1.45))
            prev_mouse[:]=mp
        prev_rmb=rmb

        if lmb and in_bar:
            y=mp[1]
            if   0.74<=y<0.80: g_dark[None]=1-g_dark[None];  click_lock=True
            elif 0.58<=y<0.62: g_quality[None]=0;             click_lock=True
            elif 0.53<=y<0.57: g_quality[None]=1;             click_lock=True
            elif 0.48<=y<0.52: g_quality[None]=2;             click_lock=True
            elif 0.40<=y<0.44: g_speed[None]=0;               click_lock=True
            elif 0.35<=y<0.39: g_speed[None]=1;               click_lock=True
            elif 0.30<=y<0.34: g_speed[None]=2;               click_lock=True
            elif 0.20<=y<0.265: show_particles=not show_particles; click_lock=True
            elif 0.10<=y<0.155: show_phys=not show_phys;      click_lock=True

        if show_phys and lmb and in_panel:
            click_lock=True
            for sl in SLIDERS:
                nv=sl.try_update(mp[0], mp[1], lmb)
                if nv is not None:
                    phys[sl.key]=nv
        elif not lmb:
            for sl in SLIDERS:
                sl.active=False

        if lmb and not in_bar and not in_panel and not click_lock:
            pitch=cam_pitch[None]; yaw=cam_yaw[None]; d=cam_dist[None]*0.8
            dx_sc=(mp[0]*ASPECT-0.5*ASPECT)/0.85
            dy_sc=(mp[1]-0.5)/0.85
            x_w=-dx_sc*math.sin(yaw)-dy_sc*math.sin(pitch)*math.cos(yaw)
            y_w= dx_sc*math.cos(yaw)-dy_sc*math.sin(pitch)*math.sin(yaw)
            z_w= dy_sc*math.cos(pitch)
            act_np=bh_act.to_numpy()
            free=np.argwhere(act_np==0).flatten()
            if len(free)>0:
                _launch_orbiting_bh(int(free[0]), float(x_w*d), float(y_w*d), float(z_w*d), g_eff_v)
                bh_rad_np_cache=bh_rad.to_numpy()
            click_lock=True
        if not lmb:
            click_lock=False

        # Physics substeps
        if not paused:
            steps = 1 if g_quality[None]==0 else SUBSTEPS
            sdp   = dt_sim/steps
            for _ in range(steps):
                update_physics(sdp, g_eff_v,
                               float(phys["merger_thresh"]),
                               float(phys["absorb_eff"]),
                               float(phys["growth_damp"]))

        project_scene()

        # Single bulk GPU→CPU transfer
        bh_act_np = bh_act.to_numpy()
        bh_sx_np  = bh_sx.to_numpy()
        bh_sy_np  = bh_sy.to_numpy()
        bh_sr_np  = bh_sr.to_numpy()
        bh_sz_np  = bh_sz.to_numpy()
        t_sx_np   = t_sx.to_numpy()
        t_sy_np   = t_sy.to_numpy()
        p_age_np  = p_age.to_numpy()
        bh_pos_np = bh_pos.to_numpy()
        bh_rad_np_cache = bh_rad.to_numpy()
        h_idx     = int(trail_head[None])
        dark      = int(g_dark[None])
        quality   = int(g_quality[None])
        trail_draw = 6 if quality==0 else (16 if quality==1 else TRAIL_LEN)
        lens_str  = float(phys["lensing_str"])

        # Camera basis (Python-side, for 3D rendering functions)
        pitch_py = float(cam_pitch[None])
        yaw_py   = float(cam_yaw[None])
        dist_py  = float(cam_dist[None])*0.8
        eye_np   = np.array([dist_py*math.cos(pitch_py)*math.cos(yaw_py),
                              dist_py*math.cos(pitch_py)*math.sin(yaw_py),
                              dist_py*math.sin(pitch_py)])
        za_np    = eye_np/(np.linalg.norm(eye_np)+1e-9)
        xa_np    = np.cross([0,0,1], za_np); xa_np/=(np.linalg.norm(xa_np)+1e-9)
        ya_np    = np.cross(za_np, xa_np)

        # Visible BH screen data for lensing
        vis_bh_x=[]; vis_bh_y=[]; vis_bh_r=[]
        for k in range(MAX_BH):
            if bh_act_np[k] and bh_sz_np[k]<0:
                vis_bh_x.append(bh_sx_np[k]/ASPECT)
                vis_bh_y.append(bh_sy_np[k])
                vis_bh_r.append(float(bh_sr_np[k]))

        # ── RENDER ──────────────────────────────────────────────────────────
        gui.clear(0x02091D)

        # Starfield — lensed in both modes (lens intensity may differ)
        sf_x = SF_X.copy();  sf_y = SF_Y.copy()
        star_lens = lens_str * (0.8 if dark else 0.5)
        if star_lens > 0 and vis_bh_x:
            sf_x, sf_y = apply_lensing_np(sf_x, sf_y, vis_bh_x, vis_bh_y, vis_bh_r, star_lens)
        gui.circles(np.stack([sf_x,sf_y],axis=1).astype(np.float32),
                    radius=SF_RAD.astype(float), color=SF_COL)

        # ── Particle trails — optional batch render ────────────────────────
        if show_particles:
            age_fracs = p_age_np / P_LIFE
            lut_idx   = np.clip((age_fracs*511).astype(np.int32),0,511)
            colors    = COLOR_LUT[lut_idx]

            seg_b=[]; seg_e=[]; seg_c=[]
            for step in range(trail_draw-1):
                s1=(h_idx-1-step)%TRAIL_LEN
                s2=(s1-1)%TRAIL_LEN
                x1=t_sx_np[s1]/ASPECT; y1=t_sy_np[s1]
                x2=t_sx_np[s2]/ASPECT; y2=t_sy_np[s2]
                opacity=(trail_draw-step)/trail_draw

            # Particle lensing (dark mode only — vectorised numpy)
                if dark and lens_str>0 and vis_bh_x:
                    x1l,y1l = apply_lensing_np(x1,y1,vis_bh_x,vis_bh_y,vis_bh_r,lens_str*0.7)
                else:
                    x1l,y1l = x1,y1

                valid = (np.abs(x1l-x2)<0.13) & (np.abs(y1l-y2)<0.13)
                if not np.any(valid): continue

                c_arr = colors.astype(np.int64)
                rc=(c_arr>>16)&0xFF; gc=(c_arr>>8)&0xFF; bc=c_arr&0xFF
                oc=((rc*opacity).astype(np.int32)<<16)|((gc*opacity).astype(np.int32)<<8)|(bc*opacity).astype(np.int32)

                seg_b.append(np.stack([x1l[valid],y1l[valid]],axis=1).astype(np.float32))
                seg_e.append(np.stack([x2[valid], y2[valid]], axis=1).astype(np.float32))
                seg_c.append(oc[valid].astype(np.uint32))

            if seg_b:
                gui.lines(np.concatenate(seg_b),np.concatenate(seg_e),
                          radius=1.0, color=np.concatenate(seg_c))

        # ── Black holes ────────────────────────────────────────────────────
        t_now = time.perf_counter()
        for i in range(MAX_BH):
            if not bh_act_np[i] or bh_sz_np[i]>=0:
                continue
            sx_i = bh_sx_np[i]/ASPECT
            sy_i = bh_sy_np[i]
            r_px = float(bh_sr_np[i])*H
            br_i = float(bh_rad_np_cache[i])

            if dark:
                render_dark_bh(gui, i, sx_i, sy_i, r_px, float(bh_sr_np[i]), bh_pos_np[i], br_i,
                               eye_np, xa_np, ya_np, za_np, yaw_py,
                               float(phys["spin"]), float(phys["disc_temp"]),
                               float(phys["turb_strength"]), float(phys["brightness"]),
                               quality, t_now)
            else:
                render_light_bh(gui, i, sx_i, sy_i, r_px, bh_pos_np[i], br_i,
                                eye_np, xa_np, ya_np, za_np, quality, t_now)

        # ── Sidebar ────────────────────────────────────────────────────────
        # Airy, low-cost UI: a few quiet floating surfaces and halo dots rather
        # than continuous decoration. This leaves Light Mode rendering budget to
        # the particle trails and geometric rings.
        gui.rect((0.008,0.014),(SIDEBAR_W-0.010,0.982),color=0x09091B)
        gui.line((SIDEBAR_W-0.010,0.030),(SIDEBAR_W-0.010,0.966),color=0x2A2448,radius=1)
        gui.circle((0.024,0.965), color=0x5E3E91, radius=11)
        gui.circle((0.031,0.958), color=0xD5A5FF, radius=4)
        gui.text("ULTIMATE SPECTRAL CHAOS",(0.045,0.944),font_size=17,color=0xE6C7FF)
        gui.text("N-BODY PROBLEM, BLACK HOLE MERGER",(0.045,0.916),font_size=12,color=0x9B91C2)

        dn=int(g_dark[None])
        gui.rect((0.018,0.73),(SIDEBAR_W-0.020,0.80),color=0x30224A if dn else 0x15243D)
        gui.line((0.024,0.733),(SIDEBAR_W-0.026,0.733),color=0x8962BA if dn else 0x67B6D1,radius=1)
        gui.text("HEAVY MODE" if dn else "LIGHT MODE",(0.031,0.747),font_size=14,color=0xF4ECFF)

        gui.text("VISUAL QUALITY",(0.012,0.67),font_size=10,color=0x7777AA)
        qn=int(g_quality[None])
        for qi,(lb,qy) in enumerate([("LOW",0.61),("MEDIUM",0.56),("HIGH",0.51)]):
            gui.rect((0.018,qy-0.017),(SIDEBAR_W-0.020,qy+0.017),color=0x24213C if qn==qi else 0x101025)
            gui.text(lb,(0.031,qy-0.008),font_size=13,color=0xF1E9FF if qn==qi else 0x77728E)

        gui.text("SIM SPEED",(0.012,0.47),font_size=10,color=0x7777AA)
        sn=int(g_speed[None])
        for si3,(lb,sy_) in enumerate([("LOW",0.43),("MEDIUM",0.38),("HIGH",0.33)]):
            gui.rect((0.018,sy_-0.017),(SIDEBAR_W-0.020,sy_+0.017),color=0x24213C if sn==si3 else 0x101025)
            gui.text(lb,(0.031,sy_-0.008),font_size=13,color=0xF1E9FF if sn==si3 else 0x77728E)

        dust_color = 0x243D59 if show_particles else 0x17152A
        gui.rect((0.018,0.205),(SIDEBAR_W-0.020,0.265),color=dust_color)
        gui.circle((0.034,0.235),color=0x63D9FF if show_particles else 0x5F5572,radius=5)
        gui.text("DUST  ON" if show_particles else "DUST  OFF",(0.048,0.238),font_size=12,
                 color=0xD4F4FF if show_particles else 0xA39BB5)
        gui.text("rainbow massless trails",(0.048,0.216),font_size=8,color=0x8D9AAF)

        pp_bc=0x1E3A5A if show_phys else 0x0A0A1A
        gui.rect((0.018,0.090),(SIDEBAR_W-0.020,0.155),color=pp_bc)
        gui.circle((0.032,0.124),color=0xC98BFF if show_phys else 0x695381,radius=4)
        gui.text("PHYSICS",(0.043,0.123),font_size=13,color=0xE9C4FF)
        gui.text("CONTROLS",(0.043,0.099),font_size=10,color=0xAA8CC6)

        n_bh_active=int(np.sum(bh_act_np))
        gui.text(f"BLACK HOLE: {n_bh_active}",(0.012,0.058),font_size=12,color=0xFFAA00)
        gui.text(f"FPS: {fps_s:.0f}",(0.012,0.030),font_size=12,color=0x7777AA)

        # ── Physics panel ──────────────────────────────────────────────────
        if show_phys:
            gui.rect((PANEL_X0+0.006,0.014),(0.994,0.982),color=0x0B0920)
            gui.line((PANEL_X0+0.006,0.030),(PANEL_X0+0.006,0.966),color=0x45305F,radius=1)
            gui.circle((PANEL_X0+0.025,0.963),color=0x8A63B9,radius=6)
            gui.text("PHYSICS CONTROLS",(PANEL_X0+0.040,0.955),font_size=12,color=0xE6C7FF)
            gui.text("Kerr metric  ·  N-body  ·  MHD",(PANEL_X0+0.040,0.938),font_size=8,color=0x9B82BA)
            for sl in SLIDERS:
                sl.draw(gui, phys[sl.key])

        # ── Particle legend ────────────────────────────────────────────────
        lx0=SIDEBAR_W+0.008; lx1=lx0+0.225
        ly_top=0.205
        gui.rect((lx0,0.008),(lx1,ly_top+0.018),color=0x04040E)
        gui.line((lx0,0.008),(lx0,ly_top+0.018),color=0x151530,radius=1)
        gui.line((lx0,ly_top+0.018),(lx1,ly_top+0.018),color=0x151530,radius=1)
        gui.text("PARTICLE  PROPERTIES",(lx0+0.005,ly_top+0.002),font_size=9,color=0x7788BB)
        for idx,(col,label,note) in enumerate(LEGEND):
            ey=ly_top-0.011-idx*0.031
            gui.rect((lx0+0.004,ey),(lx0+0.016,ey+0.016),color=col)
            gui.text(label,(lx0+0.019,ey+0.004),font_size=8,color=0xAABBCC)

        gui.show()

if __name__ == "__main__":
    main()
