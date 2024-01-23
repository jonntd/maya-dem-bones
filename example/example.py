

import sys
sys.path.append(r"D:\program\maya-dem-bones-master\build\maya2019_win64\plug-ins")

import math
from maya import cmds
from maya.api import OpenMaya, OpenMayaAnim

import dem_bones

db = dem_bones.DemBones()
db.nBones = 50
db.bindUpdate = 0
db.jointName = "test"
db.colourSet = "colorSet1"
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


db = dem_bones.DemBones()
db.nBones = 50
db.bindUpdate = 0
db.jointName = "test"
db.colourSet = "colorSet1"
db.compute("flag_mesh_out", "flag_mesh_abc", start_frame=1, end_frame=250)
db.createJoints()
db.applySkin()


db = dem_bones.DemBones()
db.nBones = 50
db.bindUpdate = 0
db.jointName = "test"
db.colourSet = "colorSet1"
db.compute("Bone", "Bone_Anim_Bone", start_frame=1, end_frame=250)
db.createJoints()
db.applySkin()


db = dem_bones.DemBones()
db.num_transform_iterations = 0
db.compute("Bone", "Bone_Anim_Bone", start_frame=1, end_frame=250)
db.createJoints()
db.applySkin()


db = dem_bones.DemBones()
db.nBones = 50
db.bindUpdate = 2
db.jointName = "test"
db.colourSet = "colorSet1"
db.compute("flag_mesh_out", "flag_mesh_abc", start_frame=1, end_frame=250)
db.createJoints()
db.applySkin()


db = dem_bones.DemBones()
db.nBones = 50
db.bindUpdate = 0
db.jointName = "test"
db.colourSet = "colorSet1"
db.compute("flag_mesh_out", "flag_mesh_abc", start_frame=1, end_frame=250)
db.createJoints()
db.applySkin()


db = dem_bones.DemBones()
db.bindUpdate = 0
db.compute("face_skinned", "face_shapes", start_frame=1001, end_frame=1052)
db.createJoints()
db.applySkin()


db = dem_bones.DemBones()
db.nBones = 5
db.bindUpdate = 2
db.compute("pSphere2", "pSphere1", start_frame=1, end_frame=2)
db.createJoints()
db.applySkin()


db = dem_bones.DemBones()
db.nBones = 4
db.bindUpdate = 0
db.compute("pCylinder1", "pCylinder2", start_frame=1, end_frame=2)
db.createJoints()
db.applySkin()


db = dem_bones.DemBones()
db.nBones = 5
db.bindUpdate = 1
db.compute("Bone", "Bone_Anim_Bone", start_frame=1, end_frame=200)
db.createJoints()
db.applySkin()






# Only solve helper bones using demLock attribute of the joints
# -i="Bone_Helpers.fbx" -a="Bone_Anim.abc" --bindUpdate=1 
db = dem_bones.DemBones()
db.bindUpdate = 0
db.compute("Bone", "Bone_Anim_Bone", start_frame=1, end_frame=200)
db.createJoints()
db.applySkin()


# Partially solve skinning weights using per-vertex color attribute of the mesh
# -i="Bone_PartiallySkinned.fbx" -a="Bone_Anim.abc" --nTransIters=0 
db = dem_bones.DemBones()
db.bindUpdate = 0
db.num_transform_iterations = 0
db.colourSet = "colorSet1"
db.compute("Bone", "Bone_Anim_Bone", start_frame=1, end_frame=200)
db.createJoints()
db.applySkin()


# Optimize given bone transformations and skinning weights from input meshes sequence
# -i="Bone_All.fbx" -a="Bone_Anim.abc" --bindUpdate=1 
db = dem_bones.DemBones()
db.bindUpdate = 1
db.compute("Bone", "Bone_Anim_Bone", start_frame=1, end_frame=200)
db.createJoints()
db.applySkin()


# rem Solve bone transformations from input meshes sequence and input skinning weights
# -i="Bone_Skin.fbx" -a="Bone_Anim.abc" --nWeightsIters=0 
db = dem_bones.DemBones()
db.bindUpdate = 0
db.num_weight_iterations = 0
db.compute("Bone", "Bone_Anim_Bone", start_frame=1, end_frame=200)
db.createJoints()
db.applySkin()


# rem Joint grouping 
# -i="Bone_Geom.fbx" -a="Bone_Anim.abc" -b=20 --bindUpdate=2
db = dem_bones.DemBones()
db.nBones = 5
db.bindUpdate = 2
db.compute("Bone", "Bone_Anim_Bone", start_frame=1, end_frame=200)
db.createJoints()
db.applySkin()


# rem Solve skinning weights from input meshes sequence and input bone transformations
# -i="Bone_Trans.fbx" -a="Bone_Anim.abc" --nTransIters=0 
db = dem_bones.DemBones()
db.num_transform_iterations = 0
db.compute("Bone", "Bone_Anim_Bone", start_frame=1, end_frame=200)
db.createJoints()
db.applySkin()









