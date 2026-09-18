"""Isaac Sim 5.x native ROS 2 bridge. Call after SimulationApp + extensions.

Graphs use NVIDIA's camera/RTX helper nodes, simulation clock and explicit
world->pelvis TF. robot_state_publisher owns the rest of the transform tree.
"""
import numpy as np
from sensors import CAMERAS,TOPICS
from model import NAMES,INDEX,ROOT
from control import ARM_HAND_INDICES,LO,HI

ROOT_PATH='/World/Irona'
SENSOR_PATH=ROOT_PATH+'/Links/head/Sensors'


def graph_specs(camera=True,lidar=True):
    """Serializable specs used both for offline USD authoring and live OmniGraph."""
    common_nodes=[('Tick','omni.graph.action.OnPlaybackTick'),('Context','isaacsim.ros2.bridge.ROS2Context')]
    common_values=[('Context.inputs:useDomainIDEnvVar',True)]
    specs=[]
    nodes=common_nodes+[('Time','isaacsim.core.nodes.IsaacReadSimulationTime'),
        ('Clock','isaacsim.ros2.bridge.ROS2PublishClock'),
        ('JointState','isaacsim.ros2.bridge.ROS2PublishJointState'),
        ('BaseTF','isaacsim.ros2.bridge.ROS2PublishRawTransformTree'),
        ('VaseTF','isaacsim.ros2.bridge.ROS2PublishRawTransformTree'),
        ('Command','isaacsim.ros2.bridge.ROS2SubscribeJointState')]
    vals=common_values+[('Time.inputs:resetOnStop',False),('Clock.inputs:topicName',TOPICS['clock']),
        ('JointState.inputs:topicName',TOPICS['joint_states']),('JointState.inputs:targetPrim',[ROOT_PATH+'/Links/pelvis']),
        ('BaseTF.inputs:parentFrameId','world'),('BaseTF.inputs:childFrameId','pelvis'),
        ('BaseTF.inputs:topicName',TOPICS['tf']),('BaseTF.inputs:translation',[.015,0,.455]),
        ('BaseTF.inputs:rotation',[0.,0.,0.,1.]),
        # Where the head camera currently thinks the vase is, so RViz shows the
        # perception result next to the sensor data it came from.
        ('VaseTF.inputs:parentFrameId','world'),('VaseTF.inputs:childFrameId','vase_estimate'),
        ('VaseTF.inputs:topicName',TOPICS['tf']),('VaseTF.inputs:translation',[0.,0.,0.]),
        ('VaseTF.inputs:rotation',[0.,0.,0.,1.]),
        ('Command.inputs:topicName',TOPICS['arm_command'])]
    con=[]
    for node in ['Clock','JointState','BaseTF','VaseTF','Command']:
        con += [('Tick.outputs:tick',node+'.inputs:execIn'),('Context.outputs:context',node+'.inputs:context')]
        if node!='Command':con += [('Time.outputs:simulationTime',node+'.inputs:timeStamp')]
    specs.append(dict(path='/World/IronaROS/State',nodes=nodes,values=vals,connections=con))
    for name in ['color','depth','lidar']:
        if (name=='lidar' and not lidar) or (name!='lidar' and not camera):continue
        # Isaac Sim 5.1 registers this one under its Ogn* implementation name.
        nodes=common_nodes+[('Once','isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame'),
                            ('Product','isaacsim.core.nodes.IsaacCreateRenderProduct')]
        res=[1,1] if name=='lidar' else CAMERAS[name]['resolution']
        vals=common_values+[('Product.inputs:cameraPrim',[SENSOR_PATH+'/'+name]),
            ('Product.inputs:width',res[0]),('Product.inputs:height',res[1]),('Product.inputs:enabled',True)]
        con=[('Tick.outputs:tick','Once.inputs:execIn'),('Once.outputs:step','Product.inputs:execIn')]
        if name=='lidar':
            helpers=[('Points','isaacsim.ros2.bridge.ROS2RtxLidarHelper','point_cloud',TOPICS['lidar_points'])]
            frame='lidar_link'
        else:
            frame=CAMERAS[name]['frame']
            helpers=[('Image','isaacsim.ros2.bridge.ROS2CameraHelper','rgb' if name=='color' else 'depth',TOPICS['rgb' if name=='color' else 'depth']),
                ('Info','isaacsim.ros2.bridge.ROS2CameraInfoHelper',None,TOPICS['rgb_info' if name=='color' else 'depth_info'])]
            if name=='depth':helpers.append(('Points','isaacsim.ros2.bridge.ROS2CameraHelper','depth_pcl',TOPICS['depth_points']))
        for n,typ,mode,topic in helpers:
            nodes.append((n,typ));vals += [(n+'.inputs:frameId',frame),(n+'.inputs:topicName',topic),
                (n+'.inputs:useSystemTime',False),(n+'.inputs:resetSimulationTimeOnStop',False),
                (n+'.inputs:queueSize',2),(n+'.inputs:frameSkipCount',0 if name=='lidar' else 1)]
            if mode:vals.append((n+'.inputs:type',mode))
            if name=='lidar':vals.append((n+'.inputs:fullScan',True))
            con += [('Product.outputs:execOut',n+'.inputs:execIn'),('Product.outputs:renderProductPath',n+'.inputs:renderProductPath'),
                    ('Context.outputs:context',n+'.inputs:context')]
        specs.append(dict(path='/World/IronaROS/'+name.capitalize(),nodes=nodes,values=vals,connections=con))
    return specs


def author_graph_usd(path):
    """Portable authored graph layer. Isaac loads node schemas when extensions enable.

    The runtime reconstructs the same specs through og.Controller to validate node
    registration against the installed Isaac version, then exports its live stage.
    """
    from pxr import Usd,Sdf,UsdGeom,Gf
    stage=Usd.Stage.CreateNew(str(path));UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    def attribute(prim,port,value=None):
        if port in ['inputs:cameraPrim','inputs:targetPrim']:
            rel=prim.CreateRelationship(port,custom=False)
            if value is not None:rel.SetTargets(value)
            return rel
        key=port.split(':')[-1]
        if key in ['execIn','execOut','tick','step']:typ=Sdf.ValueTypeNames.UInt
        elif key=='context':typ=Sdf.ValueTypeNames.UInt64
        elif key in ['timeStamp','simulationTime']:typ=Sdf.ValueTypeNames.Double
        elif key=='rotation':typ=Sdf.ValueTypeNames.Quatd;value=Gf.Quatd(value[3],Gf.Vec3d(*value[:3])) if value is not None else None
        elif key=='translation':typ=Sdf.ValueTypeNames.Vector3d;value=Gf.Vec3d(*value) if value is not None else None
        elif isinstance(value,bool):typ=Sdf.ValueTypeNames.Bool
        elif isinstance(value,int):typ=Sdf.ValueTypeNames.UInt64 if key=='queueSize' else Sdf.ValueTypeNames.UInt
        else:typ=Sdf.ValueTypeNames.Token if key in ['renderProductPath','type'] else Sdf.ValueTypeNames.String
        a=prim.CreateAttribute(port,typ,custom=True)
        if value is not None:a.Set(value)
        return a
    for spec in graph_specs():
        g=stage.DefinePrim(spec['path'],'OmniGraph')
        g.CreateAttribute('evaluator:name',Sdf.ValueTypeNames.Token,custom=False).Set('execution')
        g.CreateAttribute('pipelineStage',Sdf.ValueTypeNames.Token,custom=False).Set('pipelineStageSimulation')
        nodes={}
        for name,typ in spec['nodes']:
            p=stage.DefinePrim(spec['path']+'/'+name,'OmniGraphNode');nodes[name]=p
            p.CreateAttribute('node:type',Sdf.ValueTypeNames.Token).Set(typ)
            p.CreateAttribute('node:typeVersion',Sdf.ValueTypeNames.Int).Set(2 if typ.endswith(('ROS2CameraHelper','ROS2SubscribeJointState')) else 1)
        for key,value in spec['values']:
            n,port=key.split('.',1);attribute(nodes[n],port,value)
        for src,dest in spec['connections']:
            sn,sp=src.split('.',1);dn,dp=dest.split('.',1)
            a=attribute(nodes[sn],sp);attribute(nodes[dn],dp).SetConnections([a.GetPath()])
    stage.GetRootLayer().Save()


class RosBridge:
    def __init__(self,stage,camera=True,lidar=True):
        import omni.graph.core as og
        from pxr import Sdf
        self.og=og;self.stage=stage;self.allowed=set(ARM_HAND_INDICES.tolist());self.last_bad=None
        for spec in graph_specs(camera,lidar):
            if stage.GetPrimAtPath(spec['path']):stage.RemovePrim(spec['path'])
            og.Controller.edit({'graph_path':spec['path'],'evaluator_name':'execution'}, {
                og.Controller.Keys.CREATE_NODES:spec['nodes'],og.Controller.Keys.SET_VALUES:[(k,[Sdf.Path(x) for x in v] if k.endswith(('cameraPrim','targetPrim')) else v) for k,v in spec['values']],
                og.Controller.Keys.CONNECT:spec['connections']})
        self.prefix='/World/IronaROS/State/'

    def update_base_tf(self,position,quat_wxyz):
        q=quat_wxyz
        self.og.Controller.attribute(self.prefix+'BaseTF.inputs:translation').set(np.asarray(position,dtype=float))
        self.og.Controller.attribute(self.prefix+'BaseTF.inputs:rotation').set([float(q[1]),float(q[2]),float(q[3]),float(q[0])])

    def update_vase_tf(self,position):
        """Publish the current camera estimate of the vase as world->vase_estimate."""
        self.og.Controller.attribute(self.prefix+'VaseTF.inputs:translation').set(np.asarray(position,dtype=float))

    def command_target(self,target):
        # Commands latch until replaced. Only arm/hand joints are accepted so a
        # perception/manipulation client cannot accidentally overwrite the gait.
        names=self.og.Controller.attribute(self.prefix+'Command.outputs:jointNames').get()
        values=self.og.Controller.attribute(self.prefix+'Command.outputs:positionCommand').get()
        if names is None or values is None or len(names)==0:return target
        if len(names)!=len(values) or not np.isfinite(values).all():return target
        indices=[INDEX.get(str(n),-1) for n in names]
        if any(i not in self.allowed for i in indices) or len(set(indices))!=len(indices):return target
        q=target.copy();q[indices]=np.clip(np.asarray(values),LO[indices],HI[indices]);return q
