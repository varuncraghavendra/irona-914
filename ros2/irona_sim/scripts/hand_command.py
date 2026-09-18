#!/usr/bin/env python3
"""Send a slow open/close target to the physical simulated finger joints."""
import argparse,sys,time
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
sys.path.insert(0,str(Path(get_package_share_directory('irona_sim'))/'project/source'))
import rclpy
from sensor_msgs.msg import JointState
from model import NAMES
from control import hand_pose,HOME

def main():
    p=argparse.ArgumentParser();p.add_argument('--closure',type=float,default=.6);p.add_argument('--side',choices=['both','left','right'],default='both')
    args=p.parse_args()
    if not 0<=args.closure<=1:p.error('--closure must be between 0 and 1')
    rclpy.init();node=rclpy.create_node('irona_hand_command');pub=node.create_publisher(JointState,'/irona/arm_hand_command',10)
    msg=JointState();q=hand_pose(HOME,args.closure)
    for i,n in enumerate(NAMES):
        if any('_'+f+'_' in n for f in ['index','middle','ring','little','thumb']) and (args.side=='both' or n.startswith(args.side+'_')):
            msg.name.append(n);msg.position.append(float(q[i]))
    end=time.monotonic()+3
    while time.monotonic()<end:
        pub.publish(msg);rclpy.spin_once(node,timeout_sec=.1)
    node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
