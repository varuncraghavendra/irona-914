"""Head-camera vase finder: hue segmentation over rendered RGB plus rendered depth.

No ground-truth pose is read from the stage. The only prior is the object model
itself (colour, body radius and height), which is what a task-specific detector
would be given; vase.MODEL carries it and the scene builder fills it in. Depth is the ideal rendered distance to the image plane, so this
is a clean-data detector, not a noise/clutter-robust perception stack.
"""
import collections

import numpy as np
from sensors import CAMERAS,OPTICAL_R
from vase import MODEL

# USD camera axes are +X right, +Y up, -Z forward; this maps them into the robot frame.
USD_CAMERA_R=OPTICAL_R@np.diag([1.,-1.,-1.])
MIN_SATURATION=.28
MIN_VALUE=.08
MIN_PIXELS=18
DEPTH_BAND=.20                # metres around the median depth kept as one object
HEIGHT_GAP=.06                # world-z gap that separates two clusters
MIN_HEIGHT=.02                # below the floor is a mirror image, not an object
MAX_EXTENT=.45                # a cluster taller than this is not one vase


class PoseHistory:
    """Short ring of timestamped head poses.

    The rendered frame that a detection comes from is a few frames older than the
    physics state, so back-projecting it with the live head pose puts the object
    wherever the head has moved to since. Keep a history and use the pose that
    belongs to the frame's own timestamp.
    """

    def __init__(self,span=1.5,period=.004):
        self.span=span;self.period=period;self.samples=collections.deque()

    def push(self,t,position,rotation):
        if self.samples and float(t)-self.samples[-1][0]<self.period:return
        self.samples.append((float(t),np.asarray(position,dtype=float),np.asarray(rotation,dtype=float)))
        while self.samples and self.samples[0][0]<float(t)-self.span:self.samples.popleft()

    def at(self,t):
        """Nearest recorded pose, or None if nothing has been recorded yet."""
        if not self.samples:return None
        best=min(self.samples,key=lambda sample:abs(sample[0]-float(t)))
        return best[1],best[2]

    def lag(self,t):
        if not self.samples:return None
        return float(self.samples[-1][0]-float(t))


def camera_pose(head_position,head_rotation,name='depth'):
    """World pose of a head camera from the measured head link pose."""
    head_rotation=np.asarray(head_rotation,dtype=float)
    return (np.asarray(head_position,dtype=float)+head_rotation@np.asarray(CAMERAS[name]['position']),
            head_rotation@USD_CAMERA_R)


def hue_mask(rgb):
    """Boolean mask of saturated magenta pixels. rgb is float (H,W,3) in 0..1."""
    high=rgb.max(-1);low=rgb.min(-1);chroma=high-low
    red,green,blue=rgb[...,0],rgb[...,1],rgb[...,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        hue=60*np.where(high==red,((green-blue)/chroma)%6,
                        np.where(high==green,(blue-red)/chroma+2,(red-green)/chroma+4))
        saturation=np.where(high>0,chroma/np.maximum(high,1e-9),0)
    low_hue,high_hue=MODEL.hue
    return (np.isfinite(hue)&(saturation>MIN_SATURATION)&(high>MIN_VALUE)
            &(hue>low_hue)&(hue<high_hue))


class VaseFinder:
    """Wraps one extra render product on a head camera and segments the vase."""

    def __init__(self,camera_path,resolution=(320,180),name='depth'):
        from isaacsim.sensors.camera import Camera
        self.camera=Camera(prim_path=camera_path,resolution=resolution,name='vase_finder')
        self.camera_name=name
        self.frames=0;self.detections=0;self.last=None
        self.intrinsics=None
        # Diagnostics, so a run that never sees the vase can say why.
        self.image_mean=None;self.mask_pixels=0;self.render_frame=None;self.status='no frame';self.stamp=0.0
        # Kept for the dashboard recorder: the frame the detector actually judged.
        self.last_rgb=None;self.last_depth=None;self.last_mask=None;self.last_detection=None

    def initialize(self):
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()
        matrix=np.asarray(self.camera.get_intrinsics_matrix(),dtype=float)
        self.intrinsics=(matrix[0,0],matrix[1,1],matrix[0,2],matrix[1,2])

    @staticmethod
    def largest_cluster(points):
        """Split the segmented points by height and keep the best vase candidate.

        A glossy floor mirrors the vase, and the mirrored pixels carry the floor's
        depth, so they land as a second blob well away from the real one. Splitting
        on world z separates them; the vase is the biggest cluster that is no taller
        than a vase.
        """
        order=np.argsort(points[:,2]);ordered=points[order]
        splits=np.nonzero(np.diff(ordered[:,2])>HEIGHT_GAP)[0]+1
        best=None
        for group in np.split(ordered,splits):
            if group.shape[0]<MIN_PIXELS:continue
            if group[-1,2]-group[0,2]>MAX_EXTENT:continue
            if best is None or group.shape[0]>best.shape[0]:best=group
        return best

    def detect(self,pose_at,approach=None,max_range=8.0):
        """Return a detection dict in world coordinates, or None.

        pose_at(stamp) supplies the camera pose for the frame's own render time.
        approach is accepted and ignored: this detector's grasp point comes from
        the object model, not from the direction the palm will arrive from. It is
        in the signature so the mission can call either finder the same way.

        position  segmented blob centre, pushed one body radius along the view ray
                  so it estimates the vase axis rather than its visible surface
        base_z    lowest segmented point, used with the known body height to pick
                  a grasp height that ignores the narrower neck
        """
        frame=self.camera.get_current_frame()
        # The frame dictionary is keyed by annotator name: 'rgb' holds RGBA.
        rgba=frame.get('rgb');depth=frame.get('distance_to_image_plane')
        self.render_frame=frame.get('rendering_frame')
        self.stamp=float(frame.get('rendering_time') or 0.0)
        pose=pose_at(self.stamp)
        if pose is None:
            self.status='no camera pose yet';return None
        camera_position,camera_rotation=pose
        if rgba is None or depth is None:
            self.status='no frame';return None
        rgba=np.asarray(rgba);depth=np.asarray(depth,dtype=float)
        if rgba.ndim!=3 or rgba.size==0 or depth.shape[:2]!=rgba.shape[:2]:
            self.status='empty frame';return None
        self.frames+=1
        rgb=rgba[...,:3].astype(float)/255.0 if rgba.dtype==np.uint8 else np.asarray(rgba,dtype=float)[...,:3]
        self.image_mean=float(rgb.mean())
        mask=hue_mask(rgb)&np.isfinite(depth)&(depth>.15)&(depth<max_range)
        self.last_rgb=rgb;self.last_depth=depth;self.last_mask=mask;self.last_detection=None
        self.mask_pixels=int(mask.sum())
        if self.mask_pixels<MIN_PIXELS:
            self.status=f'{self.mask_pixels} magenta px, image mean {self.image_mean:.2f}';return None
        self.status='tracking'
        rows,cols=np.nonzero(mask)
        ranges=depth[rows,cols]
        near=np.abs(ranges-np.median(ranges))<DEPTH_BAND
        rows,cols,ranges=rows[near],cols[near],ranges[near]
        if rows.size<MIN_PIXELS:
            self.status='blob too fragmented';return None
        fx,fy,cx,cy=self.intrinsics
        local=np.stack([(cols+.5-cx)*ranges/fx,-(rows+.5-cy)*ranges/fy,-ranges],axis=1)
        points=local@np.asarray(camera_rotation,dtype=float).T+np.asarray(camera_position,dtype=float)
        points=points[points[:,2]>MIN_HEIGHT]
        if points.shape[0]<MIN_PIXELS:
            self.status='only sub-floor reflections';return None
        points=self.largest_cluster(points)
        if points is None:
            self.status='no cluster matches the vase model';return None
        # The blob is the front half of a cylinder, so the median point sits about
        # half a radius in front of the axis and depends on the viewing geometry.
        # Take the nearest returns as the front surface and add the known radius.
        centre=np.median(points,axis=0)
        origin=np.asarray(camera_position,dtype=float)
        ray=centre-origin;ray/=max(np.linalg.norm(ray),1e-9)
        front=float(np.percentile(np.linalg.norm(points-origin,axis=1),10))
        detection=dict(position=origin+(front+MODEL.radius)*ray,base_z=float(np.percentile(points[:,2],2)),
                       top_z=float(np.percentile(points[:,2],98)),pixels=int(points.shape[0]),
                       range=float(np.linalg.norm(centre-np.asarray(camera_position,dtype=float))),stamp=self.stamp)
        detection['grasp_z']=MODEL.grasp_height(detection['base_z'])
        detection['uv']=[float(cols.mean()+.5),float(rows.mean()+.5)]
        self.detections+=1;self.last=detection;self.last_detection=detection
        return detection
