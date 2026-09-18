"""Publishes the robot's live view and its intended next actions to disk.

Deliberately a file drop rather than a socket: the physics callback must never
block on a client, and the dashboard process has to be able to start, stop and
reattach at any point in a run without the simulation noticing. Each update
writes the annotated colour frame, a colourised depth frame and one JSON
snapshot, each via a temporary file plus os.replace, so a reader never sees a
half-written frame.

scripts/live_dashboard.py serves the directory this writes.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from zeroshot_grasp import project


class LivePublisher:
    def __init__(self, directory, period=.25, jpeg_quality=85):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.period = float(period)
        self.quality = int(jpeg_quality)
        self.next_publish = 0.0
        self.updates = 0
        self.started = time.time()

    def due(self, t):
        return t >= self.next_publish

    # ------------------------------------------------------------------ writing
    def _replace(self, name, payload, binary=True):
        temporary = self.directory / (name + '.tmp')
        if binary:
            temporary.write_bytes(payload)
        else:
            temporary.write_text(payload)
        os.replace(temporary, self.directory / name)

    def _annotate(self, finder, mission, t, actions, place_uv):
        import cv2
        rgb = finder.last_rgb
        image = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
        scale = 3
        image = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale),
                           interpolation=cv2.INTER_NEAREST)
        mask = finder.last_mask
        if mask is not None and mask.any():
            edges = cv2.Canny(mask.astype(np.uint8) * 255, 40, 120)
            edges = cv2.resize(edges, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            image[edges > 0] = (60, 255, 255)
            rows, cols = np.nonzero(mask)
            cv2.rectangle(image, (int(cols.min()) * scale, int(rows.min()) * scale),
                          (int(cols.max()) * scale, int(rows.max()) * scale), (60, 220, 255), 1)
        detection = finder.last_detection
        if detection is not None:
            # The fitted rod: the two antipodal contacts and the axis between them.
            contacts = [c for c in (detection.get('contacts_uv') or []) if c]
            if len(contacts) == 2:
                a = (int(contacts[0][0] * scale), int(contacts[0][1] * scale))
                b = (int(contacts[1][0] * scale), int(contacts[1][1] * scale))
                cv2.line(image, a, b, (255, 120, 0), 2, cv2.LINE_AA)
                for point in (a, b):
                    cv2.circle(image, point, 5, (255, 120, 0), 2, cv2.LINE_AA)
            if detection.get('uv'):
                u, v = (int(round(value * scale)) for value in detection['uv'])
                cv2.drawMarker(image, (u, v), (0, 90, 255), cv2.MARKER_CROSS, 18, 2)
        if place_uv:
            u, v = (int(round(value * scale)) for value in place_uv)
            cv2.drawMarker(image, (u, v), (120, 255, 120), cv2.MARKER_TILTED_CROSS, 20, 2)
            cv2.putText(image, 'place', (u + 8, v - 6), cv2.FONT_HERSHEY_SIMPLEX, .42,
                        (120, 255, 120), 1, cv2.LINE_AA)

        lines = [f"t={t:6.2f}s   state={mission.state}", f"sees: {finder.status}"]
        if detection is not None:
            lines.append(f"grasp: {detection.get('width', 0)*1000:.0f} mm wide, "
                         f"score {detection.get('score', 0):.2f}, "
                         f"fit {detection.get('residual_mm', 0):.1f} mm")
        lines.append('next: ' + (actions[0] if actions else '-'))
        if len(actions) > 1:
            lines.append('then: ' + actions[1])
        for i, text in enumerate(lines):
            origin = (8, 20 + 18 * i)
            cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def publish(self, t, finder, mission, frames, vase_pose=None, extra=None):
        """Write one live snapshot. Returns True when something was written."""
        import cv2
        if finder.last_rgb is None:
            return False
        self.next_publish = t + self.period
        self.updates += 1
        actions = mission.next_actions()

        place_uv = None
        pose = getattr(finder, 'last_pose', None)
        if mission.place_target is not None and pose is not None and finder.intrinsics is not None:
            place_uv = project(mission.place_target, finder.intrinsics, pose[0], pose[1],
                               finder.last_rgb.shape[:2])

        image = self._annotate(finder, mission, t, actions, place_uv)
        ok, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            self._replace('frame.jpg', buffer.tobytes())
        if finder.last_depth is not None:
            depth = finder.last_depth
            finite = np.isfinite(depth) & (depth > 0)
            scaled = np.zeros(depth.shape, dtype=np.uint8)
            if finite.any():
                clipped = np.clip((depth[finite] - .2) / 3.8, 0, 1)
                scaled[finite] = (255 * (1 - clipped)).astype(np.uint8)
            coloured = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
            coloured = cv2.resize(coloured, (coloured.shape[1] * 3, coloured.shape[0] * 3),
                                  interpolation=cv2.INTER_NEAREST)
            ok, buffer = cv2.imencode('.jpg', coloured, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if ok:
                self._replace('depth.jpg', buffer.tobytes())

        detection = finder.last_detection or {}
        state = dict(
            t=round(float(t), 3), wall=round(time.time(), 3), updates=self.updates,
            state=mission.state, actions=actions,
            detector=finder.status,
            # Frame bookkeeping, so a stalled render is visible on the dashboard
            # instead of being inferred from a picture that stopped changing.
            frames=finder.frames, detections=finder.detections,
            frame_stamp=round(float(finder.stamp), 3),
            frame_age_s=round(float(t) - float(finder.stamp), 3),
            render_frame=finder.render_frame,
            support_z=getattr(finder, 'support_z', None),
            clusters=getattr(finder, 'last_clusters', []),
            grasp=dict(
                width_mm=None if detection.get('width') is None
                else round(float(detection['width']) * 1000, 1),
                score=detection.get('score'),
                fit_residual_mm=detection.get('residual_mm'),
                upright=detection.get('upright'),
                centre=None if detection.get('position') is None
                else np.asarray(detection['position']).round(4).tolist(),
            ) if detection else None,
            palm=np.round(mission.measured_grasp_point(frames), 4).tolist(),
            pelvis=np.round(frames['pelvis'][0], 4).tolist(),
            closure=round(float(mission.closure), 3),
            attached=mission.attached_at, released=mission.released_at,
            place_target=None if mission.place_target is None
            else np.asarray(mission.place_target).round(4).tolist(),
            place_residual_mm=None if mission.place_residual is None
            else round(mission.place_residual * 1000, 1),
            grasp_policy=mission.policy_note,
            events=mission.events[-9:],
            # Ground truth, shown for the operator's benefit only. Nothing in the
            # perception or control path reads it.
            object_truth=None if vase_pose is None else np.round(vase_pose[0], 4).tolist(),
        )
        if extra:
            state.update(extra)
        self._replace('state.json', json.dumps(state, indent=1), binary=False)
        return True

    def finish(self, summary=None):
        self._replace('summary.json', json.dumps(summary or {}, indent=1), binary=False)
