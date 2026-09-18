#!/usr/bin/env python3
"""Live ROS 2 acceptance test. Run AFTER the simulation starts publishing.

Checks data payloads, expected frames, 3D lidar spread, increasing stamps,
clock, all 62 joints, and TF connectivity. Returns nonzero on failure.
"""
import argparse,json,time,sys,math,struct
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image,CameraInfo,PointCloud2,JointState
from rosgraph_msgs.msg import Clock
from tf2_ros import Buffer,TransformListener

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--timeout',type=float,default=30);parser.add_argument('--report',default='sensor_runtime_report.json');args=parser.parse_args()
    rclpy.init();node=rclpy.create_node('irona_sensor_smoke_test');tf=Buffer();listener=TransformListener(tf,node)
    data={};stamps={};counts={};regressions=[];subs=[]
    expected={
      '/irona/camera/color/image_raw':(Image,'camera_color_optical_frame'),
      '/irona/camera/color/camera_info':(CameraInfo,'camera_color_optical_frame'),
      '/irona/camera/depth/image_rect_raw':(Image,'camera_depth_optical_frame'),
      '/irona/camera/depth/camera_info':(CameraInfo,'camera_depth_optical_frame'),
      '/irona/camera/depth/points':(PointCloud2,'camera_depth_optical_frame'),
      '/irona/lidar/points':(PointCloud2,'lidar_link'),'/joint_states':(JointState,None),'/clock':(Clock,None)}
    def callback(msg,key):
        data[key]=msg;counts[key]=counts.get(key,0)+1
        stamp=msg.clock if key=='/clock' else msg.header.stamp;t=stamp.sec+stamp.nanosec*1e-9
        if key in stamps and t<stamps[key][-1]-1e-9:regressions.append(key)
        stamps.setdefault(key,[]).append(t)
    for topic,(typ,frame) in expected.items():
        subs.append(node.create_subscription(typ,topic,lambda msg,key=topic:callback(msg,key),qos_profile_sensor_data))
    start=time.monotonic()
    while time.monotonic()-start<args.timeout:rclpy.spin_once(node,timeout_sec=.1)
    errors=[];details={}
    for topic,(typ,frame) in expected.items():
        msg=data.get(topic)
        if msg is None:errors.append('No messages: '+topic);continue
        if counts[topic]<2 or max(stamps[topic])-min(stamps[topic])<=0:errors.append('Non-advancing stream: '+topic)
        if frame and msg.header.frame_id!=frame:errors.append('Incorrect frame: '+topic)
        if frame:
            try:tf.lookup_transform('world',frame,Time())
            except Exception as ex:errors.append('Missing TF '+frame+': '+str(ex))
        details[topic]={'messages':counts[topic],'stamp_span_s':max(stamps[topic])-min(stamps[topic])}
        if typ==Image:
            if msg.width<=0 or msg.height<=0 or len(msg.data)!=msg.step*msg.height:errors.append('Invalid image payload: '+topic)
            details[topic].update(width=msg.width,height=msg.height,encoding=msg.encoding)
        if typ==CameraInfo:
            if not all(math.isfinite(x) for x in msg.k) or msg.k[0]<=0 or msg.k[4]<=0:errors.append('Invalid intrinsics: '+topic)
        if typ==JointState:
            if len(msg.name)!=62 or len(msg.position)!=62 or not all(math.isfinite(x) for x in msg.position):errors.append('Expected 62 finite joint positions')
        if typ==PointCloud2:
            if msg.width*msg.height==0 or len(msg.data)!=msg.row_step*msg.height:errors.append('Empty/invalid point cloud: '+topic)
            fields={f.name:f for f in msg.fields}
            if not all(k in fields for k in ['x','y','z']):errors.append('Point cloud lacks XYZ: '+topic);continue
            if frame=='lidar_link':
                pts=[];fmt='>f' if msg.is_bigendian else '<f'
                for row in range(msg.height):
                    for col in range(0,msg.width,max(1,msg.width//8000)):
                        offset=row*msg.row_step+col*msg.point_step
                        if all(fields[k].datatype==7 for k in ['x','y','z']):
                            pts.append([struct.unpack_from(fmt,msg.data,offset+fields[k].offset)[0] for k in ['x','y','z']])
                pts=[p for p in pts if all(math.isfinite(x) for x in p) and p[0]*p[0]+p[1]*p[1]>1e-4]
                if pts:
                    quadrants={int((math.atan2(p[1],p[0])+math.pi)/(math.pi/2))%4 for p in pts}
                    zs=[p[2] for p in pts];details[topic].update(sampled_points=len(pts),quadrants_seen=len(quadrants),z_span_m=max(zs)-min(zs))
                    if len(quadrants)<4:errors.append('Lidar did not see all four arena quadrants')
                    if max(zs)-min(zs)<.1:errors.append('Lidar returns do not show 3D vertical spread')
                else:errors.append('No finite lidar XYZ points')
    if regressions:errors.append('Timestamp regression: '+str(sorted(set(regressions))))
    report={'isaac_ros_runtime_tested':True,'passed':not errors,'errors':errors,'topics':details}
    open(args.report,'w').write(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    node.destroy_node();rclpy.shutdown();sys.exit(bool(errors))
if __name__=='__main__':main()
