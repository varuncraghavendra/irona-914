"""Zero-shot grasp synthesis from one RGB-D frame, VLAD-Grasp derived.

What this is, precisely: the *geometric* half of VLAD-Grasp (arXiv 2511.05791,
ICRA 2026) -- its stage 2 object alignment and stage 3 grasp projection -- run
locally on the head camera's rendered depth. The paper's stage 1 asks GPT-5 or
Gemini 2.5 to *generate* a goal image containing a virtual cylindrical rod
through two antipodal contact points, then lifts that rod into 3D. No
image-generation model fits on this 12 GB GPU next to Isaac Sim, and the paper's
code is not released, so stage 1 is replaced here by an analytic antipodal
proxy: a circle fitted to the visible arc of the object's cross-section, which
recovers the same cylindrical rod the VLM was prompted to draw.

So: not the published method end to end. It keeps the part that makes the method
embodiment-agnostic -- a 6-DoF grasp pose in camera frame, which any arm's IK can
execute -- and drops the part that needs an external API.

Zero-shot in the sense that matters here: no training, no grasp dataset, no
object model, and no colour prior. perception.VaseFinder segments by the vase's
known magenta hue; nothing below knows what it is looking at. The only priors are
the robot's hand span and the fact that the target rests on a horizontal support.

Pipeline, all numpy:

    depth -> world point cloud -> support plane -> clusters -> pick target
          -> PCA axis -> slice at candidate heights -> circle fit per slice
          -> rod through the fitted axis, perpendicular to the approach
          -> 6-DoF grasp (centre, closing axis, approach, width, score)
"""
import numpy as np

# Working volume. The head camera sees the room, but only a tabletop-sized shell
# around the robot can hold something it could pick up.
MIN_RANGE = .20
MAX_RANGE = 3.00
FLOOR_CLEAR = .15          # below this is floor or its reflection, never a target
SUPPORT_BIN = .010         # m; z-histogram resolution used to find the support top
SUPPORT_SEARCH = (.20, 1.10)   # m; plausible height band for a table or pedestal top
SUPPORT_CLEAR = .012       # m above the support top before a point counts as object
OBJECT_CEILING = .45       # m above the support; taller than this is furniture
CLUSTER_CELL = .025        # m; XY grid cell for connected components
MIN_CLUSTER = 40           # points
# Cell coordinates are packed into one integer for the grid flood fill. The offset
# keeps negative world coordinates positive; the stride bounds the grid at
# +-800 cells, which is +-20 m at CLUSTER_CELL, far beyond the camera's range.
GRID_OFFSET = 800
GRID_STRIDE = 1600
# Clustering cost grows with the point count, and the physics callback cannot
# afford an unbounded segmentation pass. A hand-sized object still carries
# thousands of points after this cap; only room-sized surfaces get thinned.
MAX_SEGMENT_POINTS = 40000
MIN_OBJECT_HEIGHT = .030
MAX_FOOTPRINT = .30        # m; wider than this is not a hand-sized object
SLICE_HALF = .015          # m; half-height of the cross-section band that is fitted
SLICE_FRACTIONS = (.30, .38, .46, .54, .62)   # heights up the object that are tried
MIN_SLICE_POINTS = 12
HAND_SPAN = .105           # m; widest the Irona fingers usefully close around
MIN_WIDTH = .015
# Navigation-grade detection. A 58 mm object at 2 m covers about 5 x 12 px in a
# 320 x 180 render, so a cross-section slice holds barely a dozen points and no
# grasp can be fitted honestly. The robot still has to walk towards something, so
# an object-like cluster is reported as a provisional detection: good enough to
# steer by, explicitly not good enough to grasp from. The footprint limit is much
# tighter than MAX_FOOTPRINT so that a cushion or a chair back cannot be chased.
NAV_FOOTPRINT = .18
NAV_MIN_HEIGHT = .05
NAV_MAX_HEIGHT = .35
NAV_MIN_POINTS = 25
# Blind-walking guard: obstacle distance ahead, measured over the band the legs
# and torso sweep through.
CORRIDOR = .22             # m; half-width of the forward corridor that is checked
BODY_BAND = (.08, .45)     # m; world-z range that the body would collide with


def back_project(depth, intrinsics, camera_position, camera_rotation, stride=1):
    """Rendered distance-to-image-plane -> world points, plus their pixel indices.

    Same optical convention as perception.VaseFinder: +x right, +y up, -z forward
    in camera local axes, so the two modules agree about where a pixel lands.
    """
    fx, fy, cx, cy = intrinsics
    depth = np.asarray(depth, dtype=float)
    rows, cols = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    rows = rows.ravel(); cols = cols.ravel()
    ranges = depth[rows, cols]
    valid = np.isfinite(ranges) & (ranges > MIN_RANGE) & (ranges < MAX_RANGE)
    rows, cols, ranges = rows[valid], cols[valid], ranges[valid]
    local = np.stack([(cols + .5 - cx) * ranges / fx,
                      -(rows + .5 - cy) * ranges / fy,
                      -ranges], axis=1)
    points = local @ np.asarray(camera_rotation, dtype=float).T + np.asarray(camera_position, dtype=float)
    return points, rows, cols, ranges


def support_candidates(points, limit=3, separation=.05):
    """Candidate horizontal surface heights, most populated first.

    A z-histogram rather than a RANSAC plane fit: the camera looks across a table
    top, so it is the most populated height band, and a histogram cannot diverge
    the way a plane fit on a mostly-vertical scene can.

    Several candidates are returned because a furnished room has more than one
    horizontal surface -- this scene has a coffee table, a couch seat and
    shelving. Whichever surface happens to fill more pixels would otherwise
    decide the whole detection, and the object standing on the other one would be
    filtered away as if it were part of the furniture.
    """
    z = points[:, 2]
    window = z[(z > SUPPORT_SEARCH[0]) & (z < SUPPORT_SEARCH[1])]
    if window.size < MIN_CLUSTER:
        return []
    edges = np.arange(SUPPORT_SEARCH[0], SUPPORT_SEARCH[1] + SUPPORT_BIN, SUPPORT_BIN)
    counts, _ = np.histogram(window, bins=edges)
    heights = []
    for index in np.argsort(counts)[::-1]:
        if counts[index] < MIN_CLUSTER:
            break
        # Top of the bin: the surface is the upper face of that band.
        height = float(edges[index] + SUPPORT_BIN)
        if any(abs(height - kept) < separation for kept in heights):
            continue
        heights.append(height)
        if len(heights) >= limit:
            break
    return heights


def support_height(points):
    """Most populated horizontal surface height, or None."""
    heights = support_candidates(points, limit=1)
    return heights[0] if heights else None


def cluster_xy(points, cell=CLUSTER_CELL):
    """Connected components over an XY occupancy grid. Returns a list of index arrays.

    Grid flood fill rather than a KD-tree, because there is no scipy inside Isaac
    Sim's python. The flood fill runs over *cells* (a few thousand) and the label
    is then pushed back to the points in one vectorised step: at 640x360 the
    camera delivers 230k points per frame, and a per-point Python loop here cost
    240 ms, which does not fit in a physics callback.
    """
    if points.shape[0] == 0:
        return []
    keys = np.floor(points[:, :2] / cell).astype(np.int64)
    # One integer per cell instead of np.unique(axis=0): unique over a 2-D array
    # lexsorts structured rows and costs an order of magnitude more.
    packed = (keys[:, 0] + GRID_OFFSET) * GRID_STRIDE + (keys[:, 1] + GRID_OFFSET)
    unique, inverse = np.unique(packed, return_inverse=True)
    inverse = inverse.ravel()
    cells = np.stack([unique // GRID_STRIDE - GRID_OFFSET,
                      unique % GRID_STRIDE - GRID_OFFSET], axis=1)
    lookup = {int(value): index for index, value in enumerate(unique)}
    labels = np.full(cells.shape[0], -1, dtype=np.int64)
    label = 0
    for start in range(cells.shape[0]):
        if labels[start] >= 0:
            continue
        stack = [start]; labels[start] = label
        while stack:
            current = stack.pop()
            cx, cy = int(cells[current, 0]), int(cells[current, 1])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    key = (cx + dx + GRID_OFFSET) * GRID_STRIDE + (cy + dy + GRID_OFFSET)
                    neighbour = lookup.get(key)
                    if neighbour is not None and labels[neighbour] < 0:
                        labels[neighbour] = label; stack.append(neighbour)
        label += 1
    point_labels = labels[inverse]
    order = np.argsort(point_labels, kind='stable')
    sorted_labels = point_labels[order]
    boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
    return [group for group in np.split(order, boundaries) if group.size >= MIN_CLUSTER]


def fit_circle(xy):
    """Algebraic (Kasa) circle fit. Returns (centre_xy, radius, rms_residual).

    This is the step that stands in for VLAD-Grasp's generated goal image. The
    camera sees only the near arc of the object, so the visible points' centroid
    sits half a radius in front of the true axis; fitting a circle to the arc
    recovers the axis and the full width, which is what the virtual cylindrical
    rod in the paper encodes.
    """
    xy = np.asarray(xy, dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    design = np.stack([x, y, np.ones_like(x)], axis=1)
    rhs = x ** 2 + y ** 2
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    centre = np.array([solution[0] / 2, solution[1] / 2])
    radius_squared = solution[2] + centre @ centre
    if not np.isfinite(radius_squared) or radius_squared <= 0:
        return None
    radius = float(np.sqrt(radius_squared))
    residual = float(np.sqrt(np.mean((np.linalg.norm(xy - centre, axis=1) - radius) ** 2)))
    return centre, radius, residual


def describe_cluster(points, support_z):
    """Geometric summary of one candidate object."""
    low = float(np.percentile(points[:, 2], 2))
    high = float(np.percentile(points[:, 2], 98))
    centroid = points.mean(axis=0)
    spread = points[:, :2] - centroid[:2]
    footprint = float(2 * np.percentile(np.linalg.norm(spread, axis=1), 90))
    return dict(points=int(points.shape[0]), base_z=low, top_z=high,
                height=high - low, footprint=footprint,
                centroid=centroid, above_support=low - support_z)


def graspable(summary):
    return (summary['height'] >= MIN_OBJECT_HEIGHT
            and summary['footprint'] <= MAX_FOOTPRINT
            and summary['top_z'] - summary['base_z'] <= OBJECT_CEILING)


def synthesize(points, support_z, approach, hand_span=HAND_SPAN):
    """Best antipodal grasp on one cluster, or None.

    approach is the unit world vector the palm travels along. The rod (the line
    joining the two finger contacts) is placed perpendicular to it through the
    fitted object axis, which is the only choice that puts both contacts on the
    object's sides rather than on its front and back where the fingers cannot
    reach.
    """
    summary = describe_cluster(points, support_z)
    if not graspable(summary):
        return None
    # PCA is used only to check the object stands upright. A toppled or strongly
    # tilted object would need the rod placed across its long axis instead, and
    # this routine does not claim to handle that case. A few thousand points fix
    # the principal axis of a hand-sized object; the SVD does not need all of them.
    sample = points if points.shape[0] <= 4000 else points[::max(points.shape[0] // 4000, 1)]
    centred = sample - sample.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    principal = axes[0] / max(np.linalg.norm(axes[0]), 1e-9)
    upright = abs(float(principal[2]))

    approach = np.asarray(approach, dtype=float)
    approach = approach / max(np.linalg.norm(approach), 1e-9)
    best = None
    for fraction in SLICE_FRACTIONS:
        height = summary['base_z'] + fraction * summary['height']
        band = points[np.abs(points[:, 2] - height) < SLICE_HALF]
        if band.shape[0] < MIN_SLICE_POINTS:
            continue
        fit = fit_circle(band[:, :2])
        if fit is None:
            continue
        centre_xy, radius, residual = fit
        width = 2 * radius
        if not (MIN_WIDTH < width < hand_span):
            continue
        # Score: a clean circular cross-section the fingers can span, sampled by
        # plenty of points, near the middle of the object's body.
        fit_quality = 1.0 / (1.0 + residual / .004)
        span_quality = 1.0 - abs(width - .55 * hand_span) / (.55 * hand_span)
        support = min(1.0, band.shape[0] / 60.0)
        centrality = 1.0 - abs(fraction - .46) / .46
        score = float(.42 * fit_quality + .24 * span_quality + .20 * support + .14 * centrality)
        candidate = dict(centre=np.array([centre_xy[0], centre_xy[1], height]),
                         width=width, radius=radius, residual=residual,
                         fraction=fraction, slice_points=int(band.shape[0]), score=score)
        if best is None or candidate['score'] > best['score']:
            best = candidate
    if best is None:
        return None
    # The rod: horizontal, perpendicular to the approach. The palm normal follows
    # the approach, so this is the direction the fingers close along.
    closing = np.cross(np.array([0., 0., 1.]), approach)
    closing /= max(np.linalg.norm(closing), 1e-9)
    best.update(approach=approach, closing=closing, upright=upright,
                base_z=summary['base_z'], top_z=summary['top_z'],
                object_height=summary['height'], footprint=summary['footprint'],
                points=summary['points'],
                contacts=[(best['centre'] - best['radius'] * closing).tolist(),
                          (best['centre'] + best['radius'] * closing).tolist()])
    return best


def forward_clearance(points, camera_position, camera_rotation, exclude=None):
    """Distance to the nearest body-height obstacle straight ahead, or None.

    Used only to stop the robot advancing while it has nothing to walk towards.
    Once it has a target the tuned approach logic closes the last few centimetres
    itself, and it has to: the grasp stance puts the pelvis about 150 mm from the
    table edge, far closer than any general-purpose clearance rule would allow.
    """
    forward = np.asarray(camera_rotation, dtype=float) @ np.array([0., 0., -1.])
    forward[2] = 0.0
    length = float(np.linalg.norm(forward))
    if length < 1e-9:
        return None
    forward /= length
    lateral = np.cross(np.array([0., 0., 1.]), forward)
    relative = points - np.asarray(camera_position, dtype=float)
    along = relative @ forward
    across = relative @ lateral
    band = ((points[:, 2] > BODY_BAND[0]) & (points[:, 2] < BODY_BAND[1])
            & (np.abs(across) < CORRIDOR) & (along > .05))
    if exclude is not None:
        band &= ~exclude
    if band.sum() < 12:
        return None
    # 5th percentile rather than the minimum: one stray pixel on a reflection
    # should not stop the robot dead.
    return float(np.percentile(along[band], 5))


def navigation_candidate(objects):
    """Best object-like cluster to walk towards when no grasp can be fitted yet.

    objects is the list built during detect(); each entry carries its cluster
    points and geometric summary.
    """
    viable = [entry for entry in objects
              if entry['summary']['footprint'] <= NAV_FOOTPRINT
              and NAV_MIN_HEIGHT <= entry['summary']['height'] <= NAV_MAX_HEIGHT
              and entry['summary']['points'] >= NAV_MIN_POINTS]
    if not viable:
        return None
    # Prefer the most compact one, then the best supported: a vase on a table
    # beats a sliver of a chair leg.
    return min(viable, key=lambda entry: (entry['summary']['footprint'],
                                          -entry['summary']['points']))


def project(point, intrinsics, camera_position, camera_rotation, shape):
    """World point -> pixel, or None when it falls behind or outside the image."""
    fx, fy, cx, cy = intrinsics
    local = np.asarray(camera_rotation, dtype=float).T @ (np.asarray(point, dtype=float)
                                                          - np.asarray(camera_position, dtype=float))
    if local[2] >= -1e-6:
        return None
    distance = -local[2]
    u = local[0] * fx / distance + cx - .5
    v = -local[1] * fy / distance + cy - .5
    if not (0 <= u < shape[1] and 0 <= v < shape[0]):
        return None
    return float(u), float(v)


class ZeroShotGraspFinder:
    """Drop-in replacement for perception.VaseFinder with no colour prior.

    Exposes the same surface the mission and the recorder already use --
    detect(pose_at), status, frames, detections, last_rgb/last_depth/last_mask,
    last_detection -- so the walking, settling and reaching logic is untouched.
    The extra attribute is last_grasp: the full 6-DoF proposal, which the mission
    uses for the palm rotation and the dashboard draws.
    """

    def __init__(self, camera_path, resolution=(320, 180), name='depth', hand_span=HAND_SPAN):
        from isaacsim.sensors.camera import Camera
        self.camera = Camera(prim_path=camera_path, resolution=resolution, name='zeroshot_finder')
        self.camera_name = name
        self.hand_span = float(hand_span)
        self.frames = 0; self.detections = 0; self.last = None
        self.intrinsics = None
        self.image_mean = None; self.mask_pixels = 0; self.render_frame = None
        self.status = 'no frame'; self.stamp = 0.0
        self.last_rgb = None; self.last_depth = None; self.last_mask = None; self.last_detection = None
        self.last_grasp = None; self.last_clusters = []; self.support_z = None
        self.last_pose = None
        # Distance to the nearest body-height obstacle ahead, and whether the last
        # detection was only good enough to navigate by.
        self.clearance = None; self.provisional = False

    def initialize(self):
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()
        matrix = np.asarray(self.camera.get_intrinsics_matrix(), dtype=float)
        self.intrinsics = (matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2])

    def detect(self, pose_at, approach=None, max_range=MAX_RANGE):
        """Segment the scene geometrically and synthesize a grasp on the best object.

        approach is the world direction the palm will come from. Before the robot
        has stopped there is no arm plan yet, so the camera's own view direction is
        used, which is close enough to rank clusters; the rod is recomputed once the
        mission has a real approach.
        """
        frame = self.camera.get_current_frame()
        rgba = frame.get('rgb'); depth = frame.get('distance_to_image_plane')
        self.render_frame = frame.get('rendering_frame')
        self.stamp = float(frame.get('rendering_time') or 0.0)
        pose = pose_at(self.stamp)
        if pose is None:
            self.status = 'no camera pose yet'; return None
        camera_position, camera_rotation = pose
        self.last_pose = (np.asarray(camera_position, dtype=float), np.asarray(camera_rotation, dtype=float))
        if rgba is None or depth is None:
            self.status = 'no frame'; return None
        rgba = np.asarray(rgba); depth = np.asarray(depth, dtype=float)
        if rgba.ndim != 3 or rgba.size == 0 or depth.shape[:2] != rgba.shape[:2]:
            self.status = 'empty frame'; return None
        self.frames += 1
        rgb = (rgba[..., :3].astype(float) / 255.0 if rgba.dtype == np.uint8
               else np.asarray(rgba, dtype=float)[..., :3])
        self.image_mean = float(rgb.mean())
        self.last_rgb = rgb; self.last_depth = depth
        self.last_mask = None; self.last_detection = None; self.last_grasp = None
        self.last_clusters = []; self.provisional = False

        points, rows, cols, _ = back_project(depth, self.intrinsics, camera_position, camera_rotation)
        if points.shape[0] < MIN_CLUSTER:
            self.status = 'depth frame empty'; return None
        supports = support_candidates(points)
        if not supports:
            self.status = 'no support surface in view'; return None

        if approach is None:
            # Camera forward is -z of the optical frame, flattened into the ground plane.
            forward = np.asarray(camera_rotation, dtype=float) @ np.array([0., 0., -1.])
            forward[2] = 0.0
            approach = forward / max(np.linalg.norm(forward), 1e-9)

        proposals = []
        objects = []
        clusters_seen = 0
        for support_z in supports:
            above = (points[:, 2] > support_z + SUPPORT_CLEAR) \
                & (points[:, 2] < support_z + OBJECT_CEILING) & (points[:, 2] > FLOOR_CLEAR)
            index = np.flatnonzero(above)
            if index.size < MIN_CLUSTER:
                continue
            if index.size > MAX_SEGMENT_POINTS:
                index = index[::int(np.ceil(index.size / MAX_SEGMENT_POINTS))]
            candidates = points[index]
            candidate_rows = rows[index]; candidate_cols = cols[index]
            for members in cluster_xy(candidates):
                clusters_seen += 1
                cluster_points = candidates[members]
                summary = describe_cluster(cluster_points, support_z)
                record = dict(summary)
                record['centroid'] = summary['centroid'].tolist()
                record['support_z'] = round(float(support_z), 4)
                grasp = synthesize(cluster_points, support_z, approach, hand_span=self.hand_span)
                record['graspable'] = grasp is not None
                record['score'] = None if grasp is None else round(grasp['score'], 3)
                self.last_clusters.append(record)
                objects.append(dict(summary=summary, points=cluster_points, support_z=support_z,
                                    rows=candidate_rows[members], cols=candidate_cols[members]))
                if grasp is not None:
                    proposals.append((grasp, support_z, cluster_points,
                                      candidate_rows[members], candidate_cols[members]))

        self.clearance = forward_clearance(points, camera_position, camera_rotation)

        if not proposals:
            # Nothing graspable yet. Report an object-like cluster to walk towards
            # instead, flagged provisional so the mission does not try to grasp it.
            candidate = navigation_candidate(objects)
            if candidate is None:
                self.status = (f'{clusters_seen} clusters over {len(supports)} surfaces, none graspable'
                               if clusters_seen else f'{len(supports)} surfaces, nothing standing on them')
                return None
            return self._provisional(candidate, camera_position, camera_rotation, depth, clusters_seen,
                                     len(supports))
        grasp, support_z, cluster_points, member_rows, member_cols = max(
            proposals, key=lambda item: item[0]['score'])
        self.support_z = support_z

        # A mask over the chosen cluster's own pixels, so the recorder and the live
        # dashboard outline exactly the points the grasp was fitted to.
        mask = np.zeros(depth.shape, dtype=bool)
        mask[member_rows, member_cols] = True
        self.last_mask = mask
        self.mask_pixels = int(mask.sum())
        self.status = (f'grasp {grasp["width"]*1000:.0f} mm wide, score {grasp["score"]:.2f}, '
                       f'{clusters_seen} clusters on {len(supports)} surfaces')

        centre = grasp['centre']
        detection = dict(position=np.array([centre[0], centre[1], centre[2]]),
                         base_z=grasp['base_z'], top_z=grasp['top_z'],
                         pixels=int(member_rows.size),
                         range=float(np.linalg.norm(cluster_points.mean(axis=0) - camera_position)),
                         stamp=self.stamp, grasp_z=float(centre[2]), provisional=False,
                         width=float(grasp['width']), score=float(grasp['score']),
                         residual_mm=round(float(grasp['residual']) * 1000, 2),
                         upright=round(float(grasp['upright']), 3))
        pixel = project(centre, self.intrinsics, camera_position, camera_rotation, depth.shape)
        rows_masked, cols_masked = np.nonzero(mask)
        detection['uv'] = list(pixel) if pixel else [float(cols_masked.mean() + .5),
                                                     float(rows_masked.mean() + .5)]
        detection['contacts_uv'] = [project(np.asarray(contact), self.intrinsics, camera_position,
                                            camera_rotation, depth.shape)
                                    for contact in grasp['contacts']]
        self.detections += 1
        self.last = detection; self.last_detection = detection; self.last_grasp = grasp
        return detection

    def _provisional(self, candidate, camera_position, camera_rotation, depth,
                     clusters_seen, surfaces):
        """Navigation-grade detection: a position to walk to, with no grasp attached.

        The axis is still recovered by a circle fit, over the whole cluster rather
        than one slice, because even a coarse fit beats the visible centroid: the
        centroid of a front-facing arc sits about half a radius too near.
        """
        summary = candidate['summary']
        cluster_points = candidate['points']
        fit = fit_circle(cluster_points[:, :2])
        centre_xy = None
        if fit is not None and fit[1] < .5 * NAV_FOOTPRINT:
            centre_xy = fit[0]
        if centre_xy is None:
            # Fall back to pushing the visible centroid away from the camera by
            # half the measured footprint.
            centroid = summary['centroid']
            ray = centroid - np.asarray(camera_position, dtype=float)
            ray[2] = 0.0
            ray /= max(np.linalg.norm(ray), 1e-9)
            centre_xy = (centroid + .5 * summary['footprint'] * ray)[:2]
        height = summary['base_z'] + .46 * summary['height']
        centre = np.array([centre_xy[0], centre_xy[1], height])

        mask = np.zeros(depth.shape, dtype=bool)
        mask[candidate['rows'], candidate['cols']] = True
        self.last_mask = mask
        self.mask_pixels = int(mask.sum())
        self.support_z = candidate['support_z']
        self.provisional = True
        self.status = (f'provisional: {summary["footprint"]*1000:.0f} mm object at '
                       f'{np.linalg.norm(centre[:2]-np.asarray(camera_position)[:2]):.2f} m, '
                       f'too coarse to fit a grasp ({clusters_seen} clusters on {surfaces} surfaces)')
        detection = dict(position=centre, base_z=summary['base_z'], top_z=summary['top_z'],
                         pixels=int(summary['points']),
                         range=float(np.linalg.norm(cluster_points.mean(axis=0) - camera_position)),
                         stamp=self.stamp, grasp_z=float(height), provisional=True,
                         footprint=float(summary['footprint']))
        pixel = project(centre, self.intrinsics, camera_position, camera_rotation, depth.shape)
        rows_masked, cols_masked = np.nonzero(mask)
        detection['uv'] = list(pixel) if pixel else [float(cols_masked.mean() + .5),
                                                     float(rows_masked.mean() + .5)]
        self.detections += 1
        self.last = detection; self.last_detection = detection; self.last_grasp = None
        return detection

    def describe(self):
        """What went into the last proposal, for the mission report."""
        grasp = self.last_grasp
        return dict(method='VLAD-Grasp-derived geometry (local, training-free)',
                    reference='arXiv:2511.05791 stages 2-3; stage 1 goal-image replaced '
                              'by an analytic circle-fit antipodal proxy',
                    support_z=None if self.support_z is None else round(float(self.support_z), 4),
                    provisional=bool(self.provisional),
                    clearance_m=None if self.clearance is None else round(float(self.clearance), 3),
                    clusters=self.last_clusters,
                    grasp=None if grasp is None else dict(
                        centre=np.asarray(grasp['centre']).round(4).tolist(),
                        width_mm=round(float(grasp['width']) * 1000, 1),
                        fit_residual_mm=round(float(grasp['residual']) * 1000, 2),
                        score=round(float(grasp['score']), 3),
                        slice_fraction=grasp['fraction'],
                        slice_points=grasp['slice_points'],
                        upright=round(float(grasp['upright']), 3),
                        closing=np.asarray(grasp['closing']).round(4).tolist(),
                        approach=np.asarray(grasp['approach']).round(4).tolist(),
                        contacts=[np.asarray(c).round(4).tolist() for c in grasp['contacts']]))
