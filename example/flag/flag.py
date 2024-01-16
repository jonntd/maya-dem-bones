
sys.path.append(r"D:\program\maya-dem-bones-master\build\maya2019_win64\plug-ins")

import math
from maya import cmds
from maya.api import OpenMaya, OpenMayaAnim

import dem_bones

db = dem_bones.DemBones()
db.nBones = 50
db.bindUpdate = 2
db.jointName = "test"

db.compute("flag_mesh_out", "flag_mesh_abc", start_frame=1, end_frame=250)
db.createJoints()
db.applySkin()


for frame in range(db.start_frame, db.end_frame + 1):
    for influence in db.influences:
        matrix = OpenMaya.MMatrix(db.anim_matrix(influence, frame))
        matrix = OpenMaya.MTransformationMatrix(matrix)
        translate = matrix.translation(OpenMaya.MSpace.kWorld)
        rotate = matrix.rotation().asVector()
        print("%5f " %rotate.x ,"%5f " %rotate.y ,"%5f " %rotate.z)

        cmds.setKeyframe("{}.translateX".format(influence), time=frame, value=translate.x)
        cmds.setKeyframe("{}.translateY".format(influence), time=frame, value=translate.y)
        cmds.setKeyframe("{}.translateZ".format(influence), time=frame, value=translate.z)
        cmds.setKeyframe("{}.rotateX".format(influence), time=frame, value=math.degrees(rotate.x))
        cmds.setKeyframe("{}.rotateY".format(influence), time=frame, value=math.degrees(rotate.y))
        cmds.setKeyframe("{}.rotateZ".format(influence), time=frame, value=math.degrees(rotate.z))


sel = OpenMaya.MSelectionList()
sel.add("skinCluster1")
skin_cluster_obj = sel.getDependNode(0)
skin_cluster_fn = OpenMayaAnim.MFnSkinCluster(skin_cluster_obj)

sel = OpenMaya.MSelectionList()
sel.add("flag_mesh_out")
mesh_dag = sel.getDagPath(0)
mesh_dag.extendToShape()

skin_cluster_fn.setWeights(
    mesh_dag,
    OpenMaya.MObject(),
    OpenMaya.MIntArray(range(len(db.influences))),
    OpenMaya.MDoubleArray(db.weights)
)