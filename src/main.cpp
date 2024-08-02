#pragma once
#include <maya/MVector.h>
#include <maya/MQuaternion.h>
#include <maya/MEulerRotation.h>
#include <maya/MFnDagNode.h>
#include <maya/MFnTransform.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MObject.h>
#include <maya/MObjectArray.h>
#include <maya/MSelectionList.h>
#include <maya/MItDag.h>
#include <maya/MFnSkinCluster.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnAnimCurve.h>

#include <maya/MTime.h>
#include <maya/MIntArray.h>
#include <maya/MFloatArray.h>

#include <maya/MSelectionList.h>
#include <maya/MAnimControl.h>
#include <maya/MDGModifier.h>
#include <maya/MDagModifier.h>
#include <maya/MGlobal.h>
#include <maya/MDagPath.h>
#include <maya/MFnSkinCluster.h>
#include <maya/MFnSingleIndexedComponent.h>
#include <maya/MItDependencyGraph.h>
#include <maya/MIntArray.h>
#include <maya/MDoubleArray.h>

#include <map>
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <DemBones/DemBonesExt.h>
#include <DemBones/MatBlocks.h>
#include <Python.h>
#include "utils.h"


namespace py = pybind11;
using namespace std;
using namespace Eigen;
using namespace Dem;


class DemBonesModel : public DemBonesExt<double, float> {
public:
	int sF = 1001;
	int eF = 1010;
	vector<string> bonesMaya;
	vector<double> weightsMaya;
	vector<int> parents;
	bool needCreateJoints = 0;
	map<string, MMatrix> bindMatricesMaya;
	map<string, map<int, MMatrix>> animMatricesMaya;
	map<string, MTransformationMatrix::RotationOrder> rotOrderMaya;
	string  source_name;
	string  target_name;


	MDagPathArray m_joints;
	double tolerance;
	int patience;
	int nBones = -1;
	string colourSet = "demLock";
	string jointName = "DB";
	MatrixXd lr, lt, gb, lbr, lbt;
	DemBonesModel() : tolerance(1e-3), patience(3) { bindUpdate = 0; nIters = 30; clear(); }

	void cbIterBegin() {
		LOG("  iteration #" << iter << endl);
	}

	bool cbIterEnd() {
		double err = rmse();
		LOG("    rmse = " << err << endl);
		if ((err < prevErr * (1 + weightEps)) && ((prevErr - err) < tolerance * prevErr)) {
			np--;
			if (np == 0) {
				LOG("  convergence is reached" << endl);
				return true;
			}
		}
		else np = patience;
		prevErr = err;
		return false;
	}

	void cbInitSplitBegin() {}

	void cbInitSplitEnd() {}

	void cbWeightsBegin() {}

	void cbWeightsEnd() { LOG("    updated weights..." << endl); }

	void cbTranformationsBegin() {}

	void cbTransformationsEnd() { LOG("    updated transforms..." << endl); }

	bool cbTransformationsIterEnd() { return false; }

	bool cbWeightsIterEnd() { return false; }

	void extractSource(MDagPath& dag, MFnMesh& mesh) {
		MatrixXd wd(0, 0);
		MIntArray indices;
		MDoubleArray weights;
		MDagPath boneParentMaya;
		MDagPathArray bonesMaya;


		map<string, MatrixXd, less<string>, aligned_allocator<pair<const string, MatrixXd>>> mT;
		map<string, VectorXd, less<string>, aligned_allocator<pair<const string, VectorXd>>> wT;
		map<string, Matrix4d, less<string>, aligned_allocator<pair<const string, Matrix4d>>> bindMatrices;
		bool hasKeyFrame = false;

		time.setValue(sF);
		anim.setCurrentTime(time);

		// update model: bind vertex positions
		status = mesh.getPoints(points, MSpace::kWorld);
		CHECK_MSTATUS_AND_THROW(status);

		LOG("    update model: bind vertex positions" << endl);

		u.resize(3, nV);

#pragma omp parallel for
		for (int i = 0; i < nV; i++) {
			MPoint point = points[i];
			u.col(i) << point.x, point.y, point.z;
		}

		// update model: face connections
		int nFV = mesh.numPolygons();
		fv.resize(nFV);

		for (int i = 0; i < nFV; i++) {
			mesh.getPolygonVertices(i, indices);
			int length = indices.length();
			for (int j = 0; j < length; j++)
				fv[i].push_back((int)indices[j]);
		}
		LOG("    update model: face connections" << endl);

		// update model: weights and skeleton
		MItDependencyGraph graphIter(dag.node(), MFn::kSkinClusterFilter, MItDependencyGraph::kUpstream);
		MObject rootNode = graphIter.currentItem(&status);
		nB = 0;

		LOG("  kSkinClusterFilter  " << status << endl);

		if (MS::kSuccess == status)
		{
			// query bones
			MFnSkinCluster skinCluster(rootNode);
			nB = skinCluster.influenceObjects(bonesMaya, &status);
			CHECK_MSTATUS_AND_THROW(status);
			LOG("    query bones" << endl);
			// get bones names
			boneName.resize(nB);
			for (int j = 0; j < nB; j++)
			{
				string name = bonesMaya[j].partialPathName().asUTF8();
				boneName[j] = name;
				boneIndex[name] = j;
			}
			LOG("    get bones names" << endl);
			// get bone weights
			for (int j = 0; j < nB; j++)
			{
				status = skinCluster.getWeights(dag, MObject::kNullObj, j, weights);
				CHECK_MSTATUS_AND_THROW(status);

				wT[boneName[j]] = VectorXd::Zero(nV);
				for (int k = 0; k < nV; k++)
					wT[boneName[j]](k) = weights[k];
			}
			LOG("    get bone weights" << endl);
		}
		parent.resize(nB);
		bind.resize(nS * 4, nB * 4);
		preMulInv.resize(nS * 4, nB * 4);
		rotOrder.resize(nS * 3, nB);
		orient.resize(nS * 3, nB);
		lockM.resize(nB);
		lockW = VectorXd::Zero(nV);

		// update model: weights locked
		MString colour_set = MString(colourSet.c_str());
		if (mesh.hasColorChannels(colour_set)) {
			MColorArray colours;
			status = mesh.getVertexColors(colours, &colour_set);
			CHECK_MSTATUS_AND_THROW(status);
			LOG("    hasColorChannels: " << colourSet << endl);
			for (int c = 0; c < (int)colours.length(); c++)
				lockW(c) = Conversion::toGrayScale(colours[c]);
			LOG("    update model: weights locked" << endl);
		}


		// update model: weights
		if (wT.size() != 0)
		{
			wd = MatrixXd::Zero(nB, nV);
			for (int j = 0; j < nB; j++)
				wd.row(j) = wT[boneName[j]].transpose();
		}
		LOG("    update model: weights" << endl);
		// update model: skeleton
		for (int j = 0; j < nB; j++)
		{
			// get name
			string name = boneName[j];

			// get parent
			MObject boneObj = bonesMaya[j].node();
			MFnDagNode boneDagFn(boneObj, &status);
			CHECK_MSTATUS_AND_THROW(status);
			MObject boneParentObj = boneDagFn.parent(0);

			if (!boneParentObj.isNull() && boneParentObj.hasFn(MFn::kJoint)) {
				status = MDagPath::getAPathTo(boneParentObj, boneParentMaya);
				CHECK_MSTATUS_AND_THROW(status);

				string parentName = boneParentMaya.partialPathName().asUTF8();
				if (boneIndex.find(parentName) == boneIndex.end())
					parent(j) = -1;
				else
					parent(j) = boneIndex[parentName];
			}
			else {
				parent(j) = -1;
			}

			// get bind matrix
			//mT[name].resize(nF * 4, 4);
			mT[name].resize(nF * 4, nB * 4);
			Matrix4d bindMatrix = Conversion::toMatrix4D(bonesMaya[j].inclusiveMatrix());
			bind.blk4(0, j) = bindMatrix;
			bindMatrices[name] = bindMatrix;

			// get rotation order
			MPlug rotateOrderPlug = boneDagFn.findPlug("rotateOrder", &status);
			CHECK_MSTATUS_AND_THROW(status);

			int rotateOrder = rotateOrderPlug.asInt();
			switch (rotateOrder) {
			case 0: {
				rotOrder.vec3(0, j) = Vector3i(0, 1, 2);
				rotOrderMaya[name] = MTransformationMatrix::RotationOrder::kXYZ;
				break;
			}
			case 1: {
				rotOrder.vec3(0, j) = Vector3i(1, 2, 0);
				rotOrderMaya[name] = MTransformationMatrix::RotationOrder::kYZX;
				break;
			}
			case 2: {
				rotOrder.vec3(0, j) = Vector3i(2, 0, 1);
				rotOrderMaya[name] = MTransformationMatrix::RotationOrder::kZXY;
				break;
			}
			case 3: {
				rotOrder.vec3(0, j) = Vector3i(0, 2, 1);
				rotOrderMaya[name] = MTransformationMatrix::RotationOrder::kXZY;
				break;
			}
			case 4: {
				rotOrder.vec3(0, j) = Vector3i(1, 0, 2);
				rotOrderMaya[name] = MTransformationMatrix::RotationOrder::kYXZ;
				break;
			}
			case 5: {
				rotOrder.vec3(0, j) = Vector3i(2, 1, 0);
				rotOrderMaya[name] = MTransformationMatrix::RotationOrder::kZYX;
				break;
			}
			}

			// get joint orient
			MPlug jointOrientPlug = boneDagFn.findPlug("jointOrient", &status);
			CHECK_MSTATUS_AND_THROW(status);

			double jointOrientX = jointOrientPlug.child(0).asMAngle().asDegrees();
			double jointOrientY = jointOrientPlug.child(1).asMAngle().asDegrees();
			double jointOrientZ = jointOrientPlug.child(2).asMAngle().asDegrees();
			orient.vec3(0, j) = Vector3d(jointOrientX, jointOrientY, jointOrientZ);

			// get pre multiply inverse
			if (!boneParentObj.isNull() && parent(j) == -1) {
				status = MDagPath::getAPathTo(boneParentObj, boneParentMaya);
				CHECK_MSTATUS_AND_THROW(status);

				Matrix4d gp = Conversion::toMatrix4D(boneParentMaya.inclusiveMatrix());
				preMulInv.blk4(0, j) = gp.inverse();
			}
			else {
				preMulInv.blk4(0, j) = Matrix4d::Identity();
			}

			// get dem lock
			MPlug demLockPlug = boneDagFn.findPlug("demLock", &status);
			if (MS::kSuccess == status) lockM(j) = demLockPlug.asInt(); else lockM(j) = 0;
		}

		// update model: skeleton animation
		m.resize(nF * 4, nB * 4);

		if (nB != 0)
		{
			for (int k = sF; k < eF + 1; k++)
			{
				time.setValue(k);
				anim.setCurrentTime(time);
				int num = k - sF;
				for (int j = 0; j < nB; j++)
				{
					// get name
					string name = boneName[j];

					// set matrix
					Matrix4d matrix = Conversion::toMatrix4D(bonesMaya[j].inclusiveMatrix());
					mT[name].blk4(num, 0) = matrix * bindMatrices[name].inverse();
				}
			}
		}
		for (int j = 0; j < nB; j++)
		{
			m.block(0, j * 4, nF * 4, 4) = mT[boneName[j]];
		}

		// update model: animation state
		MString transformAttributes[9] = { "tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz" };
		for (int j = 0; j < nB; j++)
		{
			string nj = boneName[j];
			for (int k = 0; k < 9; k++)
			{
				MFnDependencyNode node(bonesMaya[j].node(), &status);
				CHECK_MSTATUS_AND_THROW(status);
				MPlug plug = node.findPlug(transformAttributes[k], &status);
				CHECK_MSTATUS_AND_THROW(status);
				if (plug.isDestination())
				{
					hasKeyFrame = true;
					break;
				}
			}
		}
		w = (wd / nS).sparseView(1, 1e-20);
		lockW = lockW /= (double)nS;
		if (!hasKeyFrame) m.resize(0, 0);

		// report
		LOG("extracted source" << endl);
		LOG("===== vertices =====¡·¡·  " << nV  << endl);
		//if (nB != 0) LOG("  " << nB << " joints" << endl);
		//if (hasKeyFrame) LOG("  keyframes found" << endl);
		//if (w.size() != 0) LOG("  skinning found" << endl);
		LOG("===== joints =====¡·¡· " << nB  << endl);
		LOG("===== keyframes found =====¡·¡· " << hasKeyFrame << endl);
		LOG("===== skinning found =====¡·¡· " << w.size() << endl);

	}

	void extractTarget(MFnMesh& mesh) 
	{
		// update model: animated vertex positions
		for (int k = sF; k < eF + 1; k++) 
		{
			time.setValue(k);
			anim.setCurrentTime(time);

			int num = k - sF;
			fTime(num) = time.asUnits(MTime::kSeconds);
			status = mesh.getPoints(points, MSpace::kWorld);
			CHECK_MSTATUS_AND_THROW(status);

			#pragma omp parallel for
			for (int i = 0; i < nV; i++) {
				MPoint point = points[i];
				v.col(i).segment<3>(num * 3) << (float)point.x, (float)point.y, (float)point.z;
			}
		}

		// update model: subject ids
		subjectID.resize(nF);
		for (int s = 0; s < nS; s++)
			for (int k = fStart(s); k < fStart(s + 1); k++)
				subjectID(k) = s;

		LOG("extracted target" << endl);
	}

	void compute(string& source, string& target, int& startFrame, int& endFrame) {
		// log parameters
		LOG("parameters" << endl);
		LOG("  source                   = " << source << endl);
		LOG("  target                   = " << target << endl);
		LOG("  start_frame              = " << startFrame << endl);
		LOG("  end_frame                = " << endFrame << endl);
		LOG("  num_iterations           = " << nIters << endl);
		LOG("  patience                 = " << patience << endl);
		LOG("  tolerance                = " << tolerance << endl);
		LOG("  num_transform_iterations = " << nTransIters << endl);
		LOG("  num_weight_iterations    = " << nWeightsIters << endl);
		LOG("  translation_affine       = " << transAffine << endl);
		LOG("  translation_affine_norm  = " << transAffineNorm << endl);
		LOG("  max_influences           = " << nnz << endl);
		LOG("  weights_smooth           = " << weightsSmooth << endl);
		LOG("  weights_smooth_step      = " << weightsSmoothStep << endl);
		LOG("  weights_epsilon          = " << weightEps << endl);
		LOG("  nBones                   = " << nBones << endl);

		source_name = source;
		target_name = target;

		// variables
		prevErr = -1;
		np = patience;

		// get geometry
		MDagPath sourcePath = Conversion::toMDagPath(source, true);
		MDagPath targetPath = Conversion::toMDagPath(target, true);

		MFnMesh sourceMeshFn(sourcePath, &status);
		CHECK_MSTATUS_AND_THROW(status);
		MFnMesh targetMeshFn(targetPath, &status);
		CHECK_MSTATUS_AND_THROW(status);

		// update model
		nS = 1;
		sF = startFrame;
		eF = endFrame;
		nF = endFrame - startFrame + 1;
		nV = sourceMeshFn.numVertices();
		int cF = (int)anim.currentTime().value();

		if (sF >= eF)
			throw std::exception("Start frame is not allowed to be equal or larger than the end frame.");
		if (nV != sourceMeshFn.numVertices())
			throw std::exception("Vertex count between source and target do not match.");

		v.resize(3 * nF, nV);
		fTime.resize(nF);
		fStart.resize(nS + 1);
		fStart(0) = 0;
		fStart(1) = nF;
		// update model: source + target
		extractTarget(targetMeshFn);
		extractSource(sourcePath, sourceMeshFn);
		LOG("nB " << nB << endl);
		// initialize model
		if (nB == 0)
		{
			nB = nBones;
			init();
			LOG("===== init =====" << endl);

		}
		// compute model
		LOG("computing" << endl);
		DemBonesExt<double, float>::compute();

		// compute transformations + weights
		VectorXd tVal, rVal;

		if (nB == 0)computeRTB(0, lr, lt, gb, lbr, lbt);
		else; computeRTB(0, lr, lt, gb, lbr, lbt, false);

		needCreateJoints = (boneName.size() == 0);
		if (needCreateJoints)
		{
			LOG("needCreateJoints " << endl);
			boneName.resize(nB);
			for (int j = 0; j < nB; j++)
			{
				ostringstream s;
				s << jointName + "_joint" << j;
				boneName[j] = s.str();
				LOG("boneName " << boneName[j] << endl);
			}
			for (int j = 0; j < nB; j++)
			{
				string name = boneName[j];
				bonesMaya.push_back(name);
				MVector translate = MVector(lbt(0, j), lbt(1, j), lbt(2, j));
				MVector rotate = MVector(lbr(0, j), lbr(1, j), lbr(2, j));
				bindMatricesMaya[name] = Conversion::toMMatrix(translate, rotate, MTransformationMatrix::RotationOrder::kXYZ);

				tVal = lt.col(j);
				rVal = lr.col(j);

				for (int k = sF; k < eF + 1; k++)
				{
					int num = k - sF;
					MVector translate = MVector(tVal(num * 3), tVal((num * 3) + 1), tVal((num * 3) + 2));
					MVector rotate = MVector(rVal(num * 3), rVal((num * 3) + 1), rVal((num * 3) + 2));
					animMatricesMaya[name][k] = Conversion::toMMatrix(translate, rotate, MTransformationMatrix::RotationOrder::kXYZ);
				}

			}
			weightsMaya.resize(nB * nV);
			Eigen::SparseMatrix<double> wT = w.transpose();
			for (int j = 0; j < nB; j++)
			{
				parents.push_back(parent[j]);
				for (Eigen::SparseMatrix<double>::InnerIterator it(wT, j); it; ++it)
				{
					weightsMaya[((int)it.row() * nB) + j] = it.value();
				}
			}

		}
		else {
			for (int j = 0; j < nB; j++)
			{
				string name = boneName[j];
				bonesMaya.push_back(name);
				MVector translate = MVector(lbt(0, j), lbt(1, j), lbt(2, j));
				MVector rotate = MVector(lbr(0, j), lbr(1, j), lbr(2, j));
				bindMatricesMaya[name] = Conversion::toMMatrix(translate, rotate, rotOrderMaya[name]);

				tVal = lt.col(j);
				rVal = lr.col(j);

				for (int k = sF; k < eF + 1; k++)
				{
					int num = k - sF;
					MVector translate = MVector(tVal(num * 3), tVal((num * 3) + 1), tVal((num * 3) + 2));
					MVector rotate = MVector(rVal(num * 3), rVal((num * 3) + 1), rVal((num * 3) + 2));
					animMatricesMaya[name][k] = Conversion::toMMatrix(translate, rotate, rotOrderMaya[name]);
				}
			}

			weightsMaya.resize(nB * nV);
			Eigen::SparseMatrix<double> wT = w.transpose();
			for (int j = 0; j < nB; j++)
			{

				for (Eigen::SparseMatrix<double>::InnerIterator it(wT, j); it; ++it)
				{
					weightsMaya[((int)it.row() * nB) + j] = it.value();
				}
			}

		}
		time.setValue(cF);
		anim.setCurrentTime(time);
	}

	array<double, 16> bindMatrix(string& bone) {
		if (bindMatricesMaya.find(bone) == bindMatricesMaya.end())
			throw std::exception("Provided influence is not valid.");

		return Conversion::toMatrixArray(bindMatricesMaya[bone]);
	}

	array<double, 16> animMatrix(string& bone, int& frame) {
		if (animMatricesMaya.find(bone) == animMatricesMaya.end())
			throw std::exception("Provided bone is not valid.");
		if (animMatricesMaya[bone].find(frame) == animMatricesMaya[bone].end())
			throw std::exception("Provided frame is not valid.");

		return Conversion::toMatrixArray(animMatricesMaya[bone][frame]);
	}

	void clear() {
		DemBonesExt<double, float>::clear();
		points.clear();
		boneIndex.clear();
	}


	MStatus createJoint(const MString& name, MObject& joint)
	{
		MStatus status;
		MDagModifier dagMod;
		joint = dagMod.createNode("joint");
		CHECK_MSTATUS_AND_RETURN_IT(status);
		status = dagMod.renameNode(joint, name);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		status = dagMod.doIt();
		CHECK_MSTATUS_AND_RETURN_IT(status);
		return status;
	}

	MStatus createJoint(const MString& name, const MObject& parent, MObject& joint)
	{
		MStatus status;
		MDagModifier dagMod;
		joint = dagMod.createNode("joint", parent, &status);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		status = dagMod.renameNode(joint, name);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		status = dagMod.doIt();
		CHECK_MSTATUS_AND_RETURN_IT(status);
		return status;
	}

	void createJoints()
	{
		if (needCreateJoints)
		{
			unsigned int numBones = boneName.size();
			MDagPath rootJointDagPath;
			MObject rootJoint;
			createJoint(MString(jointName.c_str()) + "_joint_root", rootJoint);
			MDagPath::getAPathTo(rootJoint, rootJointDagPath);
			m_joints.clear();
			MObject root;
			/// Create root bones
			m_joints.setLength(numBones);
			for (unsigned int b = 0; b < numBones; ++b)
			{
				if (parent[b] == -1)
				{
					MObject newJoint;
					if (rootJointDagPath.isValid())
						createJoint(MString(boneName[b].c_str()), rootJointDagPath.node(), newJoint);
					else
						createJoint(MString(boneName[b].c_str()), newJoint);

					MDagPath::getAPathTo(newJoint, m_joints[b]);
					root = newJoint;
				}
			}

			for (unsigned int b = 0; b < numBones; ++b)
			{
				if (parent[b] != -1)
				{
					MObject newJoint;
					createJoint(MString(boneName[b].c_str()), newJoint);
					MDagPath::getAPathTo(newJoint, m_joints[b]);
					MDagPath& p = m_joints[parent[b]];
					if (p.isValid())
						setParent(m_joints[b], p);
				}
			}

		}
		else
		{
			LOG("m_joints " << endl);
			m_joints.clear();
			/// Create root bones
			int numBones = boneName.size();
			m_joints.setLength(numBones);
			for (unsigned int b = 0; b < numBones; ++b)
			{
				MDagPath newJoint = Conversion::toMDagPath(boneName[b], false);
				m_joints[b] = newJoint;
			}
		}

		int numBones = boneName.size();
		for (unsigned int b = 0; b < numBones; ++b)
		{
			MDagPath joint = m_joints[b];
			setPosition(joint.node(), MPoint(lbt(0, b), lbt(1, b), lbt(2, b)), MSpace::kTransform);
			MEulerRotation rot = MEulerRotation(lbr(0, b), lbr(1, b), lbr(2, b));
			setRotation(joint.node(), rot, MSpace::kTransform);
		}

		for (unsigned int b = 0; b < numBones; ++b)
		{
			MObject joint = m_joints[b].node();
			MFnDependencyNode fnDep(joint, &status);
			MPlug txPlug = fnDep.findPlug("tx");
			MPlug tyPlug = fnDep.findPlug("ty");
			MPlug tzPlug = fnDep.findPlug("tz");
			MPlug rxPlug = fnDep.findPlug("rx");
			MPlug ryPlug = fnDep.findPlug("ry");
			MPlug rzPlug = fnDep.findPlug("rz");

			MFnAnimCurve fnAnimTx, fnAnimTy, fnAnimTz, fnAnimRx, fnAnimRy, fnAnimRz;
			MObject txCurve = fnAnimTx.create(txPlug, MFnAnimCurve::AnimCurveType::kAnimCurveTL);
			MObject tyCurve = fnAnimTy.create(tyPlug, MFnAnimCurve::AnimCurveType::kAnimCurveTL);
			MObject tzCurve = fnAnimTz.create(tzPlug, MFnAnimCurve::AnimCurveType::kAnimCurveTL);
			MObject rxCurve = fnAnimRx.create(rxPlug, MFnAnimCurve::AnimCurveType::kAnimCurveTA);
			MObject ryCurve = fnAnimRy.create(ryPlug, MFnAnimCurve::AnimCurveType::kAnimCurveTA);
			MObject rzCurve = fnAnimRz.create(rzPlug, MFnAnimCurve::AnimCurveType::kAnimCurveTA);

			Eigen::VectorXd translation = lt.col(b);
			Eigen::VectorXd rotation = lr.col(b);
			Eigen::VectorXd tx = Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>>(translation.data(), translation.size() / 3);
			Eigen::VectorXd ty = Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>>(translation.data() + 1, translation.size() / 3);
			Eigen::VectorXd tz = Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>>(translation.data() + 2, translation.size() / 3);

			Eigen::VectorXd rx = Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>>(rotation.data(), rotation.size() / 3);
			Eigen::VectorXd ry = Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>>(rotation.data() + 1, rotation.size() / 3);
			Eigen::VectorXd rz = Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>>(rotation.data() + 2, rotation.size() / 3);
			unsigned int idx = 0;
			for (int f = sF; f <= eF; ++f)
			{
				time.setValue(f);
				fnAnimTx.addKeyframe(time, tx(idx));
				fnAnimTy.addKeyframe(time, ty(idx));
				fnAnimTz.addKeyframe(time, tz(idx));
				fnAnimRx.addKeyframe(time, rx(idx));
				fnAnimRy.addKeyframe(time, ry(idx));
				fnAnimRz.addKeyframe(time, rz(idx));
				++idx;
			}
		}
	}

	void applySkin()
	{
		LOG("===== applySkin =====  " << endl);
		LOG("===== source_name =====¡·¡· " << source_name << endl);

		MDagPath m_targetMesh = Conversion::toMDagPath(source_name, false);

		// If there is no skinCluster we are going to generate one
		// TODO We should add all the joints as influences
		MDagPath targetMeshShape = m_targetMesh;
		extendToShapeWithType(targetMeshShape, MFn::kMesh);

		MObject skinCluster;
		getSkinCluster(targetMeshShape, skinCluster);

		// If there is no skinCluster we are going to generate one
		// TODO We should add all the joints as influences
		if (skinCluster.isNull())
		{
			MDGModifier dgMod;
			// Set startFrame in order to have bind matrices correct when applying skinCluster
			time.setValue(sF);
			MAnimControl::setCurrentTime(time);

			std::stringstream cmd;
			cmd << "skinCluster -toSelectedBones ";
			for (unsigned int ii = 0; ii < m_joints.length(); ++ii)
			{
				cmd << m_joints[ii].partialPathName().asUTF8() << " ";
			}
			cmd << m_targetMesh.partialPathName().asUTF8();

			status = dgMod.commandToExecute(cmd.str().c_str());
			CHECK_MSTATUS_AND_THROW(status);

			status = dgMod.doIt();
			CHECK_MSTATUS_AND_THROW(status);
		}

		getSkinCluster(targetMeshShape, skinCluster);
		MFnSkinCluster fnSkin(skinCluster, &status);
		CHECK_MSTATUS_AND_THROW(status);

		MDagPathArray influenceDagPaths;
		fnSkin.influenceObjects(influenceDagPaths);

		for (unsigned int ii = 0; ii < influenceDagPaths.length(); ++ii)
		{
			std::string influenceName = influenceDagPaths[ii].partialPathName().asUTF8();
		}

		int numVertices = w.outerSize();
		int numBones = w.innerSize();

		// Loop through vertices
		for (int c = 0; c < numVertices; ++c)
		{
			// Loop through bones
			MIntArray influenceIndices;
			MDoubleArray weights;
			//std::string message = "Vertex " + std::to_string(c) + ": ";
			for (Eigen::SparseMatrix<double>::InnerIterator it(w, c); it; ++it)
			{
				int influenceIdx = it.row();
				double weight = it.value();

				influenceIndices.append(influenceIdx);
				weights.append(weight);
				//message += std::to_string(influenceIdx) + ": " + std::to_string(weight) + ", ";
			}
			//std::cout << message << std::endl;
			setVertexInfluenceWeights(fnSkin, targetMeshShape, c, influenceIndices, weights, false);
		}

	}

	MStatus getSkinCluster(const MDagPath& dagPath, MObject& skinCluster)
	{
		MObject node = dagPath.node();
		MItDependencyGraph it = MItDependencyGraph(node, MFn::kSkinClusterFilter, MItDependencyGraph::kUpstream);
		if (!it.isDone())
		{
			skinCluster = it.currentItem();
			return MS::kSuccess;
		}
		skinCluster = MObject::kNullObj;
		return MS::kFailure;
	}

	MStatus getInfluenceObjects(const MObject& skinCluster, MDagPathArray& dagPathArray)
	{
		MStatus status;
		MFnSkinCluster fnSkin(skinCluster, &status);
		fnSkin.influenceObjects(dagPathArray, &status);
		return status;
	}

	MStatus setVertexInfluenceWeights(MFnSkinCluster& fnSkin, const MDagPath& dagPath, int vertexIndex, MIntArray& influences, MDoubleArray& weights, bool normalize)
	{
		MStatus status;
		MFnSingleIndexedComponent fnComp;
		MObject component = fnComp.create(MFn::kMeshVertComponent, &status);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		status = fnComp.addElement(vertexIndex);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		status = fnSkin.setWeights(dagPath, component, influences, weights, normalize);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		return status;
	}


	MStatus getDagPathFromName(const MString& name, MDagPath& dagPath)
	{
		MSelectionList list;
		MStatus status = list.add(name);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		status = list.getDagPath(0, dagPath);
		return status;
	}

	MStatus extendToShapeWithType(MDagPath& path, MFn::Type type)
	{
		MStatus status;
		if (path.apiType() == type)
		{
			return MS::kSuccess;
		}
		unsigned int shapeCount;
		status = path.numberOfShapesDirectlyBelow(shapeCount);
		for (unsigned int ii = 0; ii < shapeCount; ++ii)
		{
			status = path.extendToShapeDirectlyBelow(ii);
			CHECK_MSTATUS_AND_RETURN_IT(status);
			if (path.hasFn(type))
			{
				MFnDagNode fnNode(path, &status);
				CHECK_MSTATUS_AND_RETURN_IT(status);
				if (!fnNode.isIntermediateObject())
				{
					return MS::kSuccess;
				}
			}
			path.pop();
		}
		return MS::kFailure;

	}



	MStatus setParent(MDagPath& child, MDagPath& parent)
	{
		MStatus status;
		MFnDagNode fnDag(parent, &status);
		CHECK_MSTATUS_AND_RETURN_IT(status);

		return fnDag.addChild(child.node());
	}

	MStatus setPosition(const MObject& ob, const MPoint& pos, MSpace::Space space)
	{
		MStatus status;
		MFnTransform fnTransform(ob, &status);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		status = fnTransform.setTranslation(MVector(pos), space);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		return MS::kSuccess;
	}

	MStatus setRotation(const MObject& ob, const MQuaternion& rot, MSpace::Space space)
	{
		MStatus status;
		MFnTransform fnTransform(ob, &status);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		status = fnTransform.setRotation(rot, space);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		return MS::kSuccess;
	}
	MStatus setRotation(const MObject& ob, const MEulerRotation& rot, MSpace::Space space)
	{
		return setRotation(ob, rot.asQuaternion(), space);
	}



private:
	double prevErr;
	int np;

	MStatus status;
	MTime time;
	MAnimControl anim;
	MPointArray points;
	map<string, int> boneIndex;
} model;


PYBIND11_MODULE(_core, m) {
	py::class_<DemBonesModel>(m, "DemBones")
		.def(py::init<>())
		.def_readwrite("num_iterations", &DemBonesModel::nIters, "Number of global iterations, default = 30")
		.def_readwrite("num_transform_iterations", &DemBonesModel::nTransIters, "Number of bone transformations update iterations per global iteration, default = 5")
		.def_readwrite("translation_affine", &DemBonesModel::transAffine, "Translations affinity soft constraint, default = 10.0")
		.def_readwrite("translation_affine_norm", &DemBonesModel::transAffineNorm, "p-norm for bone translations affinity soft constraint, default = 4.0")
		.def_readwrite("num_weight_iterations", &DemBonesModel::nWeightsIters, "Number of weights update iterations per global iteration, default = 3")
		.def_readwrite("max_influences", &DemBonesModel::nnz, "Number of non-zero weights per vertex, default = 8")
		.def_readwrite("weights_smooth", &DemBonesModel::weightsSmooth, "Weights smoothness soft constraint, default = 1e-4")
		.def_readwrite("weights_smooth_step", &DemBonesModel::weightsSmoothStep, "Step size for the weights smoothness soft constraint, default = 1.0")
		.def_readwrite("weights_epsilon", &DemBonesModel::weightEps, "Epsilon for weights solver, default = 1e-15")
		.def_readwrite("nBones", &DemBonesModel::nBones, "number of bones")
		.def_readwrite("bindUpdate", &DemBonesModel::bindUpdate, "update bind pose (0=no update, 1=update joint positions, 2=regroup joints under one root)")
		.def_readwrite("colourSet", &DemBonesModel::colourSet, "weights locked")
		.def_readwrite("jointName", &DemBonesModel::jointName, "create joint name")


		.def_readonly("parent", &DemBonesModel::parents, "parents id")
		.def_readonly("start_frame", &DemBonesModel::sF, "Start frame of solver")
		.def_readonly("end_frame", &DemBonesModel::eF, "End frame of solver")
		.def_readonly("influences", &DemBonesModel::bonesMaya, "List of all influences")
		.def_readonly("weights", &DemBonesModel::weightsMaya, "List of weights for all influences and vertices")

		.def("createJoints", &DemBonesModel::createJoints, "Root mean squared reconstruction error")
		.def("applySkin", &DemBonesModel::applySkin, "Root mean squared reconstruction error")
		.def("rmse", &DemBonesModel::rmse, "Root mean squared reconstruction error")
		.def("compute", &DemBonesModel::compute, "Skinning decomposition of alternative updating weights and bone transformations", py::arg("source"), py::arg("target"), py::arg("start_frame"), py::arg("end_frame"))
		.def("bind_matrix", &DemBonesModel::bindMatrix, "Get the bind matrix for the provided influence", py::arg("influence"))
		.def("anim_matrix", &DemBonesModel::animMatrix, "Get the animation matrix for the provided influence at the provided frame", py::arg("influence"), py::arg("frame"));
}
