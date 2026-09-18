"""Fetch mission: walk continuously, find the vase with the head camera, grasp, lift.

Everything here is closed loop on measured link poses and on the camera
detection. The only object knowledge used is the vase model itself (colour,
body radius, body height), never its stage transform.
"""
import sys

import numpy as np
from control import HOME,smooth
from manipulate import ARM_DOF,ARM_JOINTS,arm_ik,close_hand,grasp_point,palm_rotation
from model import DT,INDEX
from perception import PoseHistory,camera_pose
from vase import MODEL

APPROACH_X=.28          # ask for the stop here: the gait needs up to a step to level up
APPROACH_Y=.15          # and this far to the grasping side
ALIGN_TOLERANCE=.040
STEER_DEADBAND=.018
BLIND_RANGE=.50         # closer than this the latched world estimate is enough
MARCH_LIMIT=25.0        # s of stepping *in place* before settling for the alignment we have
CROSS_BODY_Y=.105       # closer to the body centre line and the arm folds across its own torso
CROSS_BODY_SHIFT=.05    # and never aim more than this away from where the vase actually is
AIM_LIMIT=.07           # total distance the aim point may drift from the latched estimate
FILTER_TIME=1.2         # s; the pelvis sways +-70 mm every step, so filter the offsets
NECK_STEP=.06           # rad per perception tick, so the camera pose stays quiet
JUMP_REJECT=.30         # m; a detection this far from a settled estimate is noise
CORRECTION_LIMIT=.06    # m; past this the palm is blocked, not merely sagging
PREGRASP_BACK=.075      # palm waits this far behind the vase along the approach
CLOSURE=.85
LIFT_ANGLE=.45          # rad of shoulder pitch, wrist pitch compensated: about 0.10 m
SEARCH_LIMIT_X=1.7      # stop walking rather than march into the arena wall
DURATIONS=dict(settle=1.5,close=1.3,lift=2.5,hold=3.0,correct=.9)
GROOT_PERIOD=.4         # s between policy queries; a chunk covers more than this
GROOT_BLEND=.12         # s to slew onto each commanded arm pose
# Cartesian waypoints relative to the grasp point: over the vase, behind it at
# grasp height, then onto it. Interpolating arm joints straight from the rest
# pose to the grasp sweeps the hand through the vase and knocks it over.
WAYPOINTS=[(-.13,.11,1.8),(-PREGRASP_BACK,0.,1.4),(0.,0.,1.2)]


class FetchMission:
    def __init__(self,walker,finder,side='right',grasp_assist=True,perception_period=.20,
                 verbose=False,report=print,policy=None,policy_seconds=10.0):
        self.walker=walker;self.finder=finder;self.side=side
        self.sign=1 if side=='left' else -1
        self.grasp_assist=grasp_assist;self.perception_period=perception_period;self.verbose=verbose
        self.report=report
        self.state='search';self.entered=0.0;self.q=HOME.copy()
        self.estimate=None;self.last_seen=None;self.hits=0;self.next_perception=0.0
        self.target=None;self.pose_from=HOME.copy();self.pose_to=HOME.copy()
        self.closure=0.0;self.lift=0.0;self.corrections=0;self.attempts=0
        self.attach_request=False;self.attached_at=None
        self.events=[];self.reach_error=None;self.grasp_error=None;self.enclosed=None
        self.approach_direction=None;self.grasp_pose=HOME.copy()
        self.waypoints=[];self.waypoint=0;self.palm_rotation=None
        self.reported=-10.0;self.rejected=0;self.tracked=None;self.blind=False;self.marching_since=None
        self.best_estimate=None;self.best_quality=0.0;self.aim_offset=np.zeros(3)
        self.history=PoseHistory();self.neck=np.zeros(2)
        self.policy=policy;self.policy_seconds=float(policy_seconds)
        self.policy_note='groot' if policy is not None else 'scripted'
        self.policy_started=None;self.policy_chunk=None;self.policy_step=0
        self.policy_next=0.0;self.policy_from=None;self.policy_calls=0;self.policy_events=[]

    # ---------------------------------------------------------------- helpers
    def note(self,t,text):
        self.events.append(dict(t=round(float(t),3),state=self.state,text=text))
        self.report(f"[{t:7.2f}s] {self.state:8s} {text}")
        sys.stdout.flush()

    def enter(self,t,state):
        self.pose_from=self.q.copy();self.state=state;self.entered=float(t)
        self.note(t,'entered')

    def elapsed(self,t):return float(t)-self.entered

    def track(self,relative):
        """Low-pass the along-track and cross-track offsets to the grasp line."""
        sample=np.array([relative[0],relative[1]-self.sign*APPROACH_Y])
        alpha=DT/FILTER_TIME
        self.tracked=sample if self.tracked is None else (1-alpha)*self.tracked+alpha*sample

    def relative(self,frames,point):
        """A world point in the measured pelvis frame."""
        position,rotation=frames['pelvis']
        return rotation.T@(np.asarray(point,dtype=float)-position)

    def sense(self,t,frames):
        self.history.push(t,*frames['head'])
        if t<self.next_perception:return None
        self.next_perception=t+self.perception_period
        detection=self.finder.detect(lambda stamp:self.pose_at(stamp))
        if detection is None:return None
        seen=np.array([detection['position'][0],detection['position'][1],detection['grasp_z']])
        if self.estimate is not None and self.hits>=3 and np.linalg.norm(seen-self.estimate)>JUMP_REJECT:
            self.rejected+=1
            if self.rejected<6:return None
            # Six disagreeing frames in a row: trust the camera, not the filter.
            self.estimate=None;self.hits=0
        self.rejected=0
        # Remember the observation with the most evidence: by the time the robot is
        # close the vase is clipped at the frame edge and half hidden by its own arm,
        # and that late view is what drags the estimate sideways.
        quality=detection['pixels']/(1.0+abs(detection['range']-.75))
        if quality>self.best_quality:
            self.best_quality=quality
            self.best_estimate=seen.copy()
        if self.tracked is not None and self.tracked[0]<BLIND_RANGE+.05 and self.hits>=5:
            # This close the vase is clipped by the frame and partly hidden by the
            # robot's own arm. Keep the estimate measured from further back.
            return detection
        self.estimate=seen if self.estimate is None else .65*self.estimate+.35*seen
        self.last_seen=float(t);self.hits+=1
        if self.verbose:
            self.report(f"[{t:7.2f}s] detect stamp={detection['stamp']:7.3f} lag={t-detection['stamp']:+.3f}s "
                        f"px={detection['pixels']:4d} range={detection['range']:.2f} "
                        f"raw={np.round(seen,3).tolist()} filtered={np.round(self.estimate,3).tolist()}")
            sys.stdout.flush()
        return detection

    def pose_at(self,stamp):
        pose=self.history.at(stamp)
        return None if pose is None else camera_pose(*pose)

    def look_at(self,q,frames,point,slew=False):
        """Point the head at a world point, within the neck limits.

        Positive neck_pitch looks down. The camera moves with the neck, so the
        command is only stepped once per perception tick and by a bounded amount:
        a fast servo would smear the very frames it is aiming with.
        """
        if slew:
            origin,_=camera_pose(*frames['head'])
            relative=frames['pelvis'][1].T@(np.asarray(point,dtype=float)-origin)
            wanted=np.array([np.clip(np.arctan2(relative[1],relative[0]),-1.39,1.39),
                             np.clip(np.arctan2(-relative[2],max(np.hypot(relative[0],relative[1]),1e-3)),-.52,.61)])
            self.neck+=np.clip(wanted-self.neck,-NECK_STEP,NECK_STEP)
        q[INDEX['neck_yaw']]=self.neck[0];q[INDEX['neck_pitch']]=self.neck[1]
        return q

    def grasp_target(self,frames):
        """World aim point and palm rotation.

        The aim point is the latched estimate plus a bounded offset. Guards,
        retries and palm corrections all adjust that offset rather than the point
        itself, so repeated passes cannot walk the aim away from the vase.
        """
        position,rotation=frames['pelvis']
        point=np.asarray(self.target,dtype=float)+self.aim_offset
        heading=point-position;heading[2]=0
        yaw=np.arctan2(heading[1],heading[0])
        return point,palm_rotation(yaw)

    def nudge_aim(self,delta,limit=AIM_LIMIT):
        """Adjust the aim offset, clamped to a sphere around the latched estimate."""
        offset=self.aim_offset+np.asarray(delta,dtype=float)
        length=float(np.linalg.norm(offset))
        if length>limit:offset*=limit/length
        self.aim_offset=offset
        return self.aim_offset

    def solve(self,frames,point,rotation):
        solution,error=arm_ik(self.pose_from,point,self.side,frames['pelvis'][0],frames['pelvis'][1],
                              target_rotation=rotation)
        self.pose_to=solution
        return error

    def blend(self,t,duration,base):
        """Interpolate the arm joints from the pose at state entry to the solution."""
        alpha=smooth(self.elapsed(t)/max(duration,1e-6))
        q=base.copy()
        dof=ARM_DOF[self.side]
        q[dof]=(1-alpha)*self.pose_from[dof]+alpha*self.pose_to[dof]
        return q,alpha

    def diagnose(self,t,frames,label,target,measured):
        """Report why a commanded palm pose was not reached."""
        commanded,_=grasp_point(self.q,self.side,*frames['pelvis'])
        palm=self.measured_grasp_point(frames)
        text=(f"{label}: target {np.round(target,3).tolist()} commanded {np.round(commanded,3).tolist()} "
              f"measured {np.round(palm,3).tolist()} pelvis {np.round(frames['pelvis'][0],3).tolist()}")
        if measured is not None:
            worst=max(((abs(measured[i]-self.q[i]),ARM_JOINTS[k]) for k,i in enumerate(ARM_DOF[self.side])),
                      key=lambda pair:pair[0])
            text+=f"; worst joint tracking error {np.degrees(worst[0]):.1f} deg on {worst[1]}"
        self.note(t,text)

    def run_policy(self,t,base,frames):
        """Drive the arm from GR00T N1.5 action chunks.

        The policy returns sixteen future steps per query. We re-query a few times
        a second and slew onto the step that matches the elapsed time, so a slow
        inference never leaves the arm without a command.
        """
        if t>=self.policy_next and self.finder.last_rgb is not None:
            self.policy_next=t+GROOT_PERIOD
            chunk=self.policy.act(self.finder.last_rgb,self.q)
            if chunk is None:
                if self.policy.failures<=1:
                    self.note(t,f"GR00T request failed: {self.policy.last_error}")
            else:
                self.policy_chunk=chunk;self.policy_step=0;self.policy_calls+=1
                self.policy_from=self.q.copy()
                self.policy_anchor=float(t)
                if self.verbose:
                    arm=np.round(np.asarray(chunk[f'{self.side}_arm'])[0],3).tolist()
                    self.report(f"[{t:7.2f}s] groot    latency {self.policy.last_latency} ms arm[0]={arm}")
                    sys.stdout.flush()
        if self.policy_chunk is None:
            return base
        # Walk through the chunk in simulated time, holding the last step if the
        # next query is late.
        elapsed=t-getattr(self,'policy_anchor',t)
        step=int(elapsed/GROOT_BLEND)
        commanded,closure=self.policy.apply(self.q,self.policy_chunk,step=step)
        alpha=smooth(min(1.0,(elapsed-step*GROOT_BLEND)/GROOT_BLEND))
        blended=base.copy()
        dof=ARM_DOF[self.side]
        source=self.policy_from if self.policy_from is not None else self.q
        blended[dof]=(1-alpha)*source[dof]+alpha*commanded[dof]
        hand=close_hand(blended,self.side,closure)
        self.closure=closure
        self.policy_from=self.q.copy()
        return hand

    def measured_grasp_point(self,frames):
        position,rotation=frames[self.side+'_hand']
        return position+rotation@MODEL.grasp_local

    # ------------------------------------------------------------ state machine
    def update(self,t,frames,object_position=None,measured=None):
        detection=self.sense(t,frames)
        pelvis=frames['pelvis'][0]
        walking=self.state in ('search','approach')
        base=self.walker.target(t,frames['pelvis'][1])

        if walking:
            if self.estimate is not None:
                relative=self.relative(frames,self.estimate)
                if self.state=='search' and self.hits>=2:
                    self.enter(t,'approach')
                    self.note(t,f"vase at {np.round(self.estimate,3).tolist()}, "
                                f"{np.linalg.norm(relative[:2]):.2f} m ahead")
                if self.state=='approach':
                    # The pelvis sways a full step width every cycle, so steer and
                    # decide on filtered offsets, not on the instantaneous ones.
                    self.track(relative)
                    ahead,offset=self.tracked
                    self.walker.steer(0.0 if abs(offset)<STEER_DEADBAND else offset)
                    aligned=abs(offset)<ALIGN_TOLERANCE
                    marching=ahead<=APPROACH_X and not aligned
                    if marching and self.marching_since is None:self.marching_since=float(t)
                    patience=0.0 if self.marching_since is None else t-self.marching_since
                    self.walker.march(marching and patience<MARCH_LIMIT)
                    if ahead<=APPROACH_X and (aligned or patience>=MARCH_LIMIT):
                        self.walker.march(False);self.walker.request_stop()
                        latched=self.best_estimate if self.best_estimate is not None else self.estimate
                        self.target=np.asarray(latched,dtype=float).copy()
                        self.aim_offset=np.zeros(3)
                        self.note(t,f"in range at {ahead:.3f} m ahead, {offset:+.3f} m off the grasp line; "
                                    f"stopping the gait")
                        self.enter(t,'settle')
                    elif t-self.last_seen>4.0 and ahead>BLIND_RANGE:
                        # Far away and no longer visible: that detection was noise.
                        self.note(t,'lost the detection, searching again')
                        self.enter(t,'search');self.estimate=None;self.hits=0;self.tracked=None
                    elif t-self.last_seen>4.0 and not self.blind:
                        self.blind=True
                        self.note(t,f"vase below the camera at {ahead:.2f} m; closing in on the latched estimate")
                base=self.look_at(base,frames,self.estimate,slew=detection is not None)
            else:
                if t-self.reported>2.5:
                    self.reported=float(t)
                    self.note(t,f"scanning: {self.finder.status}")
                self.walker.steer(0.0)
                self.neck=np.array([.5*np.sin(2*np.pi*t/14),.30])
                base[INDEX['neck_yaw']]=self.neck[0];base[INDEX['neck_pitch']]=self.neck[1]
                # Only cover ground while the vase has never been found. Searching
                # forward after an attempt would walk into whatever it is standing at.
                self.walker.march(self.target is not None)
                if pelvis[0]>SEARCH_LIMIT_X:self.walker.request_stop()
            self.q=base
            return self.q

        if self.estimate is not None:base=self.look_at(base,frames,self.estimate,slew=detection is not None)

        if self.state=='settle':
            base=close_hand(base,self.side,0.0)
            if self.elapsed(t)>=DURATIONS['settle'] and self.walker.stopped:
                point,rotation=self.grasp_target(frames)
                # Reaching inboard of this line folds the upper arm onto the torso,
                # self-collision stalls the shoulder drives and the palm never arrives.
                # Aim at the outboard side of the vase instead of across the body.
                # Outboard distance of the target from the body centre line, on the
                # grasping side. Small values mean the arm has to fold across its
                # own torso, where self-collision stalls the shoulder drives.
                outboard=self.sign*(frames['pelvis'][1].T@(point-frames['pelvis'][0]))[1]
                if outboard<CROSS_BODY_Y:
                    shift=min(CROSS_BODY_Y-outboard,CROSS_BODY_SHIFT)*self.sign
                    self.aim_offset=np.zeros(3)
                    self.nudge_aim(frames['pelvis'][1]@np.array([0.,shift,0.]))
                    point,rotation=self.grasp_target(frames)
                    self.note(t,f"vase sits {outboard*1000:.0f} mm outboard, inside the cross-body limit; "
                                f"aiming {abs(shift)*1000:.0f} mm further out to keep the arm off the torso")
                approach=point-frames['pelvis'][0];approach[2]=0
                approach/=max(np.linalg.norm(approach),1e-9)
                self.approach_direction=approach
                self.palm_rotation=rotation
                self.waypoints=[point+back*approach+np.array([0,0,rise]) for back,rise,_ in WAYPOINTS]
                self.waypoint=0
                if self.policy is not None:
                    self.note(t,f"handing the grasp to GR00T N1.5: \"{self.policy.instruction}\"")
                    self.policy_started=float(t);self.policy_next=float(t)
                    self.policy_chunk=None;self.policy_step=0;self.policy_from=self.q.copy()
                    self.enter(t,'groot')
                else:
                    residual=self.solve(frames,self.waypoints[0],rotation)
                    self.note(t,f"reaching in {len(self.waypoints)} waypoints, first IK residual {residual*1000:.1f} mm")
                    self.enter(t,'reach')
            self.q=base;return self.q

        if self.state=='groot':
            self.q=self.run_policy(t,base,frames)
            palm=self.measured_grasp_point(frames)
            reference=np.asarray(self.target if object_position is None else object_position,dtype=float)
            distance=float(np.linalg.norm(reference-palm))
            if distance<MODEL.radius+.03 and self.closure>.5:
                self.note(t,f"GR00T closed the hand on the vase ({distance*1000:.0f} mm at the palm)")
                self.pose_from=self.q.copy();self.grasp_pose=self.q.copy()
                if self.grasp_assist and self.attached_at is None:self.attach_request=True
                self.enter(t,'lift')
            elif t-self.policy_started>=self.policy_seconds:
                self.note(t,f"GR00T ran for {self.policy_seconds:.0f}s and left the palm {distance*1000:.0f} mm "
                            f"from the vase; falling back to the scripted grasp")
                self.policy_events.append(dict(t=round(float(t),2),event='timeout',palm_error_mm=round(distance*1000,1)))
                self.policy_note='groot+scripted'
                self.pose_from=self.q.copy()
                point,rotation=self.grasp_target(frames)
                self.solve(frames,self.waypoints[0],rotation)
                self.waypoint=0
                self.enter(t,'reach')
            return self.q

        if self.state=='reach':
            self.q,alpha=self.blend(t,WAYPOINTS[self.waypoint][2],base)
            self.q=close_hand(self.q,self.side,0.0)
            if alpha>=1:
                reached=float(np.linalg.norm(self.measured_grasp_point(frames)-self.waypoints[self.waypoint]))
                self.note(t,f"waypoint {self.waypoint+1}/{len(self.waypoints)} reached, palm {reached*1000:.0f} mm off")
                if reached>.05:
                    self.diagnose(t,frames,f"waypoint {self.waypoint+1} miss",self.waypoints[self.waypoint],measured)
                self.waypoint+=1
                self.pose_from=self.q.copy()
                if self.waypoint<len(self.waypoints):
                    self.solve(frames,self.waypoints[self.waypoint],self.palm_rotation)
                    self.entered=float(t)
                else:
                    self.reach_error=reached
                    self.enter(t,'enclose')
            return self.q

        if self.state=='enclose':
            self.q,alpha=self.blend(t,DURATIONS['correct'],base)
            self.q=close_hand(self.q,self.side,0.0)
            if alpha>=1:
                point=self.grasp_target(frames)[0]
                residual=point-self.measured_grasp_point(frames)
                if np.linalg.norm(residual)>CORRECTION_LIMIT:
                    # The palm is nowhere near its command: the arm is blocked or the
                    # aim point is out of reach. Correcting again only winds the target
                    # further away, so report it and let the grasp check fail cleanly.
                    self.diagnose(t,frames,'palm not tracking',point,measured)
                    self.note(t,f"palm {np.linalg.norm(residual)*1000:.0f} mm off, beyond correction range")
                    self.enter(t,'close')
                elif np.linalg.norm(residual)>.008 and self.corrections<3:
                    # Close out the drive sag and model mismatch measured at the palm.
                    self.corrections+=1
                    self.nudge_aim(.8*residual)
                    self.pose_from=self.q.copy()
                    self.grasp_error=self.solve(frames,self.grasp_target(frames)[0],self.palm_rotation)
                    self.note(t,f"palm {np.linalg.norm(residual)*1000:.0f} mm off the vase, correcting")
                    self.entered=float(t)
                else:
                    self.note(t,f"palm on the vase within {np.linalg.norm(residual)*1000:.0f} mm")
                    self.enter(t,'close')
            return self.q

        if self.state=='close':
            self.closure=CLOSURE*smooth(self.elapsed(t)/DURATIONS['close'])
            self.q=close_hand(self.blend(t,DURATIONS['close'],base)[0],self.side,self.closure)
            if self.elapsed(t)>=DURATIONS['close']:
                palm=self.measured_grasp_point(frames)
                aim=self.grasp_target(frames)[0]
                # object_position stands in for a grip sensor: it only gates the
                # decision that something is actually between the fingers. The
                # target the arm reached for came from the camera alone.
                reference=np.asarray(aim if object_position is None else object_position,dtype=float)
                enclosed=float(np.linalg.norm(reference-palm))
                self.enclosed=enclosed
                if enclosed<MODEL.radius+.03:
                    if self.grasp_assist and self.attached_at is None:
                        self.attach_request=True
                    self.note(t,f"fingers closed around the vase ({enclosed*1000:.0f} mm at the palm)")
                    self.pose_from=self.q.copy();self.grasp_pose=self.q.copy()
                    self.enter(t,'lift')
                elif self.attempts<2:
                    # Retry from the same stance. Walking again here would drive the
                    # robot into the pedestal it is standing against, and the vase is
                    # below the camera at this range, so there is nothing to re-detect.
                    self.attempts+=1;self.corrections=0
                    self.nudge_aim(.012*self.approach_direction)
                    self.note(t,f"nothing enclosed ({enclosed*1000:.0f} mm), reaching again "
                                f"{self.attempts*12:.0f} mm deeper")
                    self.closure=0.0;self.pose_from=self.q.copy()
                    self.entered=float(t);self.state='settle'
                    self.note(t,'re-settling before the second attempt')
                else:
                    self.note(t,'giving up on the grasp')
                    self.enter(t,'failed')
            return self.q

        if self.state=='lift':
            alpha=smooth(self.elapsed(t)/DURATIONS['lift'])
            self.lift=LIFT_ANGLE*alpha
            q=self.grasp_pose.copy()
            # Raise the shoulder and pitch the wrist back by the same angle so the
            # palm stays upright while the hand rises.
            q[INDEX[self.side+'_shoulder_pitch']]-=self.lift
            q[INDEX[self.side+'_wrist_pitch']]+=self.lift
            self.q=close_hand(q,self.side,self.closure)
            if alpha>=1:self.enter(t,'hold')
            return self.q

        if self.state in ('hold','done','failed'):
            if self.state=='hold' and self.elapsed(t)>=DURATIONS['hold']:self.enter(t,'done')
            return self.q
        return self.q

    def confirm_attach(self,t):
        self.attach_request=False;self.attached_at=float(t)

    def summary(self):
        return dict(state=self.state,detections=self.finder.detections,frames=self.finder.frames,
                    estimate=None if self.estimate is None else np.asarray(self.estimate).tolist(),
                    grasp_target=None if self.target is None else np.asarray(self.target).tolist(),
                    reach_error_mm=None if self.reach_error is None else round(self.reach_error*1000,1),
                    grasp_ik_residual_mm=None if self.grasp_error is None else round(self.grasp_error*1000,1),
                    corrections=self.corrections,attempts=self.attempts,
                    grasp_policy=self.policy_note,
                    policy=None if self.policy is None else dict(self.policy.summary(),
                        calls=self.policy_calls,events=self.policy_events),
                    enclosed_mm=None if self.enclosed is None else round(self.enclosed*1000,1),
                    attached_at=self.attached_at,events=self.events)
