#ifndef FEATURE_INFO_H
#define FEATURE_INFO_H

#pragma once

#include <string>
#include <vector>

#include <GeomAbs_SurfaceType.hxx>

enum EdgeType
{
    CONVEX = 1,
    CONCAVE = 2,
    SMOOTH = 3,
    OTHER = 4
};

struct FaceFeature
{
    int id = 0;
    std::string faceKey;
    double area = 0.0;
    double relativeArea = 0.0;
    double perimeter = 0.0;
    double compactness = 0.0;
    int surfaceType = 0;
    double normalX = 0.0;
    double normalY = 0.0;
    double normalZ = 0.0;
    double centerX = 0.0;
    double centerY = 0.0;
    double centerZ = 0.0;
    double normalizationCenterX = 0.0;
    double normalizationCenterY = 0.0;
    double normalizationCenterZ = 0.0;
    double normalizationScale = 1.0;
    double meanCurvature = 0.0;
    double uvInsideFraction = 0.0;
    double normalVariation = 0.0;
    double curvatureVariation = 0.0;
    double radius = 0.0;
    int numWires = 0;
    int innerWireCount = 0;
    double minInnerWireLength = 0.0;
    double maxInnerWireLength = 0.0;
    std::vector<double> innerWireLengths;
    double minInnerLoopBoundaryDihedralMax = -1.0;
    double minInnerLoopBoundaryRightAngleDeviation = -1.0;
    int hasValidInnerLoopBoundaryDihedral = 0;
    int innerLoopAllDihedralBelowThreshold = 0;
    int hasInnerLoopInteriorBfsDepthAtMost2 = 0;
    int hasSmallFlatInnerLoop = 0;
    int hasSmallRightAngleInnerLoop = 0;
    int numEdges = 0;
    double neighborAreaMean = 0.0;
    double neighborAreaMax = 0.0;
    double areaToNeighborMean = 0.0;
    double areaToNeighborMax = 0.0;
    double normalNeighborDotMean = 0.0;
    double normalNeighborDotMin = 0.0;
    double normalNeighborDotMax = 0.0;
    int neighborPlaneCount = 0;
    int neighborCylinderCount = 0;
    int neighborCurvedCount = 0;
    int convexEdgeCount = 0;
    int concaveEdgeCount = 0;
    int smoothEdgeCount = 0;
    double convexEdgeRatio = 0.0;
    double concaveEdgeRatio = 0.0;
    std::vector<int> neighborIds;
    std::vector<int> neighborEdgeTypes;
    std::vector<double> neighborAreaRatios;
    std::vector<int> neighborSurfaceTypes;
    std::vector<double> sharedEdgeLengths;
    std::vector<double> neighborDihedralMeans;
    std::vector<double> neighborDihedralStds;
    int semanticTag = 0;
};

#endif
