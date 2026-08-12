#pragma once
#include <TopoDS_Shape.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Edge.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <vector>
#include "FeatureInfo.h"

class FeatureExtractor
{
public:
    FeatureExtractor(const TopoDS_Shape &shape);
    void Extract();
    const std::vector<FaceFeature> &GetResults() const { return myResults; }

private:
    void ComputeGeometricAttributes(const TopTools_IndexedMapOfShape &faceMap);
    int IdentifyEdgeType(const TopoDS_Face& f1, const TopoDS_Face& f2, const TopoDS_Edge& e);
    bool ComputeSampledDihedralStats(
        const TopoDS_Face& f1,
        const TopoDS_Face& f2,
        const TopoDS_Edge& e,
        double& meanAngle,
        double& stdAngle,
        double* maxAngle = nullptr,
        double* maxRightAngleDeviation = nullptr
    );
    std::string BuildFaceKey(const FaceFeature& feature, const std::vector<FaceFeature>& allFeatures) const;

    TopoDS_Shape myShape;
    TopTools_IndexedDataMapOfShapeListOfShape myEdgeFaceMap;
    std::vector<FaceFeature> myResults;
};
