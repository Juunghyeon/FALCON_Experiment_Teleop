"""
CMM-based balance descriptor computation for decoupled lower-body policy logging.

Computes the Support-Aware Balance Threat Descriptor:
  1. Non-self balance demand  h_dot_ns = h_dot - h_dot_L  (finite-difference)
  2. Disturbance-induced ZMP shift  delta_p_D
  3. Normalized disturbance magnitude  delta_bar_D = ||delta_p_D|| / d_ref
  4. Normalized directional support margin  m_bar_D = d_dir / d_ref

  d_ref is a phase-invariant reference length (robot constant, set from
  nominal double-support directional margin).

Reference: "Support-Normalized Balance Threat Descriptor for Decoupled
Lower-Body Control in Humanoid Loco-Manipulation" (FALCON extension)
"""

import numpy as np
import mujoco

# Root of the upper-body subtree (torso + arms).
# Non-self momentum = upper-body contribution = what the lower-body policy must compensate for.
TORSO_BODY_NAME = "torso_link"

# Lower-body body names for G1 29-DOF
# Complement of UPPER_BODY_NAMES in waist_wrench.py
LOWER_BODY_NAMES = [
    "pelvis",
    "waist_yaw_link",
    "waist_roll_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]

# Foot contact site names (defined in g1_29dof_old_freebase.xml).
# These 4 corner sites per foot are used for an accurate support polygon.
FOOT_CONTACT_SITE_NAMES = [
    "left_foot_contact_1",
    "left_foot_contact_2",
    "left_foot_contact_3",
    "left_foot_contact_4",
    "right_foot_contact_1",
    "right_foot_contact_2",
    "right_foot_contact_3",
    "right_foot_contact_4",
]

# Fallback: ankle body names + fixed offsets (used if sites are missing)
FOOT_BODY_NAMES   = ["left_ankle_roll_link", "right_ankle_roll_link"]
FOOT_FORWARD      = 0.12   # forward extent from ankle origin (m)
FOOT_BACKWARD     = 0.05   # backward extent from ankle origin (m)
FOOT_HALF_WID     = 0.03   # lateral half-width (m)

# Root bodies of each foot subtree for contact detection.
# All descendants (e.g. dummy_lf_* geom bodies) are included automatically.
FOOT_ROOT_LEFT  = "left_ankle_pitch_link"
FOOT_ROOT_RIGHT = "right_ankle_pitch_link"


def get_body_ids(model, names):
    """Return array of body IDs for the given names, skipping any not found."""
    ids = []
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            ids.append(bid)
    return np.array(ids, dtype=int)


# ---------------------------------------------------------------------------
# Non-self centroidal momentum via MuJoCo subtree quantities
# ---------------------------------------------------------------------------

def compute_nonself_momentum(model, data):
    """
    Compute non-self centroidal momentum h_ns = h_upper (6D) and whole-body CoM.

    "Non-self" = upper-body subtree (torso_link + arms), i.e. the part whose
    dynamics the lower-body policy must compensate for.

    Uses mj_subtreeVel (called internally) which fills:
        data.subtree_linvel[i]  -- CoM velocity of subtree i  (3,)
        data.subtree_angmom[i]  -- angular momentum of subtree i about the
                                   whole-body CoM (body-1 subtree CoM)  (3,)

    Returns:
        h_ns  (6,): [l_ns (3); k_ns (3)] non-self centroidal momentum
        com   (3,): whole-body CoM position (data.subtree_com[1])
    """
    mujoco.mj_subtreeVel(model, data)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME)
    if torso_id == -1:
        return np.zeros(6), data.subtree_com[1].copy()

    com       = data.subtree_com[1].copy()            # whole-body CoM
    com_upper = data.subtree_com[torso_id].copy()     # upper-body CoM

    m_upper = model.body_subtreemass[torso_id]
    l_ns    = m_upper * data.subtree_linvel[torso_id]

    # subtree_angmom[i] is about subtree_com[i], not about whole-body CoM.
    # Shift to whole-body CoM via parallel-axis: k_G = k_i + (c_i - c_G) x l_i
    k_ns = data.subtree_angmom[torso_id] + np.cross(com_upper - com, l_ns)

    return np.concatenate([l_ns, k_ns]), com


# ---------------------------------------------------------------------------
# ZMP from contacts
# ---------------------------------------------------------------------------

def compute_zmp_from_contacts(model, data):
    """
    Compute ZMP (2D) and total vertical contact force from active contacts.

    Returns (p_zmp, total_fz):
        p_zmp     np.ndarray (2,) or None if no active contact
        total_fz  float >= 0
    """
    num_px = 0.0
    num_py = 0.0
    total_fz = 0.0

    for i in range(data.ncon):
        fv = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, fv)
        fn = fv[0]   # normal force magnitude (>= 0 in MuJoCo)
        if fn < 1e-4:
            continue
        # normal direction in world frame: first row of contact frame matrix.
        # MuJoCo contact normal points from geom1 → geom2.
        # If foot is geom1 (common), normal points downward (normal_z < 0),
        # but the actual upward reaction magnitude is fn * |normal_z|.
        normal_z = data.contact[i].frame.reshape(3, 3)[0, 2]
        fz_world = fn * abs(normal_z)
        if fz_world < 1e-4:
            continue
        pos = data.contact[i].pos
        num_px += pos[0] * fz_world
        num_py += pos[1] * fz_world
        total_fz += fz_world

    if total_fz < 1e-3:
        return None, 0.0
    return np.array([num_px / total_fz, num_py / total_fz]), total_fz


# ---------------------------------------------------------------------------
# Foot contact state
# ---------------------------------------------------------------------------

def _subtree_body_ids(model, root_name):
    """Return set of all body IDs in the subtree rooted at root_name (inclusive)."""
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root_id == -1:
        return set()
    ids = set()
    # Walk all bodies; collect those whose ancestor chain includes root_id
    for bid in range(model.nbody):
        cur = bid
        while cur > 0:
            if cur == root_id:
                ids.add(bid)
                break
            cur = model.body_parentid[cur]
    return ids


def compute_foot_contacts(model, data, force_threshold=1.0,
                          left_ids=None, right_ids=None):
    """
    Return (l_contact, r_contact) as int 0 or 1.

    Traverses the full foot subtree (ankle_pitch_link and all descendants,
    including any dummy geom bodies) so contact is detected regardless of
    how the XML assigns geoms to child bodies.

    left_ids / right_ids: pre-built sets from _subtree_body_ids() for caching.
    If not provided, they are built on every call (slow for real-time use).

    Encoding:
        (1, 1) → dual support
        (1, 0) → left single support
        (0, 1) → right single support
        (0, 0) → flight / invalid
    """
    if left_ids is None:
        left_ids  = _subtree_body_ids(model, FOOT_ROOT_LEFT)
    if right_ids is None:
        right_ids = _subtree_body_ids(model, FOOT_ROOT_RIGHT)

    l_contact = 0
    r_contact = 0
    for i in range(data.ncon):
        fv = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, fv)
        if fv[0] < force_threshold:
            continue
        for geom_id in (data.contact[i].geom1, data.contact[i].geom2):
            body_id = int(model.geom_bodyid[geom_id])
            if body_id in left_ids:
                l_contact = 1
            if body_id in right_ids:
                r_contact = 1
        if l_contact and r_contact:
            break

    return l_contact, r_contact


# ---------------------------------------------------------------------------
# Support polygon
# ---------------------------------------------------------------------------

def compute_support_polygon(model, data, contact_site_names=None, foot_body_names=None):
    """
    Return (N, 2) array of support-polygon corner points.

    Primary: reads world-frame positions of foot-corner sites defined in the XML
             (left/right_foot_contact_1~4 from g1_29dof_old_freebase.xml).
    Fallback: ankle body position ± fixed offsets when sites are missing.

    Returns np.ndarray (N, 2) or None.
    """
    if contact_site_names is None:
        contact_site_names = FOOT_CONTACT_SITE_NAMES

    points = []
    for name in contact_site_names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid == -1:
            continue
        points.append(data.site_xpos[sid, :2].copy())

    if points:
        return np.array(points)

    # --- fallback: ankle body + fixed offsets ---
    if foot_body_names is None:
        foot_body_names = FOOT_BODY_NAMES
    for name in foot_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            continue
        p = data.xpos[bid, :2]
        points.extend([
            p + np.array([ FOOT_FORWARD,   FOOT_HALF_WID]),
            p + np.array([ FOOT_FORWARD,  -FOOT_HALF_WID]),
            p + np.array([-FOOT_BACKWARD,  FOOT_HALF_WID]),
            p + np.array([-FOOT_BACKWARD, -FOOT_HALF_WID]),
        ])
    return np.array(points) if points else None


# ---------------------------------------------------------------------------
# Directional margin
# ---------------------------------------------------------------------------

def compute_directional_margin(p_zmp, support_points, u_D, eps=1e-6):
    """
    Largest alpha >= 0 s.t. p_zmp + alpha*u_D is inside the convex hull
    of support_points.  Returns eps if degenerate.
    """
    if support_points is None or len(support_points) < 2:
        return eps

    # Convex-hull vertices (ordered)
    if len(support_points) >= 3:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(support_points)
            verts = support_points[hull.vertices]
        except Exception:
            verts = support_points
    else:
        verts = support_points

    n = len(verts)
    min_t = np.inf
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        d = b - a
        # Solve: p_zmp + t*u_D = a + s*d  => [u_D, -d] [t; s] = a - p_zmp
        denom = u_D[0] * d[1] - u_D[1] * d[0]
        if abs(denom) < 1e-9:
            continue
        diff = a - p_zmp
        t = (diff[0] * d[1] - diff[1] * d[0]) / denom
        s = (diff[0] * u_D[1] - diff[1] * u_D[0]) / denom
        if t > -eps and -eps <= s <= 1.0 + eps:
            min_t = min(min_t, max(t, 0.0))

    return float(max(min_t, eps)) if np.isfinite(min_t) else eps


# ---------------------------------------------------------------------------
# Full balance descriptor (stateless; caller supplies h_dot_ns)
# ---------------------------------------------------------------------------

def compute_zmp_shift(h_dot_ns, com_z, total_fz, eps=1e-3):
    """
    Disturbance-induced ZMP shift from non-self balance demand.

      delta_p_D = -c_z/F_z * [l_dot_ns_x; l_dot_ns_y]
                + 1/F_z   * [-k_dot_ns_y; k_dot_ns_x]

    Args:
        h_dot_ns  (6,): non-self centroidal momentum rate [l_dot; k_dot]
        com_z     float: CoM height (m)
        total_fz  float: total vertical contact force (N)
        eps       float: minimum denominator

    Returns:
        delta_p_D (2,)
    """
    denom = max(total_fz, eps)
    l_ns_x, l_ns_y = h_dot_ns[0], h_dot_ns[1]
    k_ns_x, k_ns_y = h_dot_ns[3], h_dot_ns[4]
    return np.array([
        -com_z / denom * l_ns_x + (-k_ns_y) / denom,
        -com_z / denom * l_ns_y + ( k_ns_x) / denom,
    ])


def compute_balance_descriptor(h_dot_ns, com_z, total_fz, p_zmp, support_points,
                               d_ref=0.1, eps=1e-6):
    """
    Full support-aware balance threat descriptor (revised formulation).

    Args:
        h_dot_ns        (6,): non-self momentum rate [l_dot_ns; k_dot_ns]
        com_z           float: CoM z-position
        total_fz        float: total vertical contact force
        p_zmp           (2,): current ZMP position
        support_points  (N,2) or None: support polygon vertices
        d_ref           float: phase-invariant reference length (m); set from
                               nominal DS representative directional margin
        eps             float

    Returns dict:
        h_dot_ns_bal   (4,): [l_dot_ns_x, l_dot_ns_y, k_dot_ns_x, k_dot_ns_y]
        delta_p_D      (2,): disturbance-induced ZMP shift
        u_D            (2,): disturbance direction unit vector
        d_dir          float: raw directional margin (m)
        delta_bar_D    float: normalized disturbance magnitude  ||delta_p_D|| / d_ref
        m_bar_D        float: normalized directional support margin  d_dir / d_ref
    """
    h_dot_ns_bal = np.array([h_dot_ns[0], h_dot_ns[1], h_dot_ns[3], h_dot_ns[4]])
    delta_p_D    = compute_zmp_shift(h_dot_ns, com_z, total_fz, eps)

    norm_d = np.linalg.norm(delta_p_D)
    u_D    = delta_p_D / (norm_d + eps)

    d_dir       = compute_directional_margin(p_zmp, support_points, u_D, eps)
    delta_bar_D = norm_d / max(d_ref, eps)
    m_bar_D     = float(d_dir) / max(d_ref, eps)

    return {
        'h_dot_ns_bal':  h_dot_ns_bal,
        'delta_p_D':     delta_p_D,
        'u_D':           u_D,
        'd_dir':         float(d_dir),
        'delta_bar_D':   float(delta_bar_D),
        'm_bar_D':       float(m_bar_D),
    }


__all__ = [
    'TORSO_BODY_NAME',
    'LOWER_BODY_NAMES',
    'FOOT_CONTACT_SITE_NAMES',
    'FOOT_BODY_NAMES',
    'get_body_ids',
    'compute_nonself_momentum',
    'compute_foot_contacts',
    'compute_zmp_from_contacts',
    'compute_support_polygon',
    'compute_directional_margin',
    'compute_zmp_shift',
    'compute_balance_descriptor',
]
