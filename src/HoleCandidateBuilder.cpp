#include "HoleCandidateBuilder.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kSmallHoleRadiusThreshold = 2.0;
constexpr double kPlanarHoleRadiusThreshold = 6.5;
constexpr double kCenterTolerance = 1e-3;
constexpr double kRadiusTolerance = 1e-3;

const FaceFeature* FindFaceById(const std::vector<FaceFeature>& faces, int id) {
    for (const auto& face : faces) {
        if (face.id == id) {
            return &face;
        }
    }
    return nullptr;
}

bool IsDuplicateCandidate(
    const HoleCandidateFeature& lhs,
    const HoleCandidateFeature& rhs
) {
    const bool closeRadius = std::abs(lhs.estimatedHoleRadius - rhs.estimatedHoleRadius) <= kRadiusTolerance;
    const bool closeX = std::abs(lhs.wireCenterX - rhs.wireCenterX) <= kCenterTolerance;
    const bool closeY = std::abs(lhs.wireCenterY - rhs.wireCenterY) <= kCenterTolerance;
    return closeRadius && closeX && closeY;
}

std::vector<HoleCandidateFeature> DeduplicateCandidates(const std::vector<HoleCandidateFeature>& input) {
    std::vector<HoleCandidateFeature> deduplicated;
    deduplicated.reserve(input.size());

    for (const auto& candidate : input) {
        bool duplicate = false;
        for (auto& existing : deduplicated) {
            if (!IsDuplicateCandidate(existing, candidate)) {
                continue;
            }

            duplicate = true;
            existing.label = std::max(existing.label, candidate.label);
            break;
        }

        if (!duplicate) {
            deduplicated.push_back(candidate);
        }
    }

    for (size_t idx = 0; idx < deduplicated.size(); ++idx) {
        deduplicated[idx].candidateId = static_cast<int>(idx);
    }

    return deduplicated;
}
}

std::vector<HoleCandidateFeature> BuildPlanarHoleCandidates(
    const std::vector<FaceFeature>& faces,
    int graphId,
    const std::string& modelName
) {
    std::vector<HoleCandidateFeature> candidates;
    int nextCandidateId = 0;

    for (const auto& face : faces) {
        if (face.surfaceType != GeomAbs_Plane) {
            continue;
        }

        if (face.innerWireLengths.empty()) {
            continue;
        }

        int adjacentCylinderCount = 0;
        int adjacentSmallCylinderCount = 0;
        double minAdjacentCylinderRadius = std::numeric_limits<double>::max();
        double maxAdjacentCylinderRadius = 0.0;
        int typedEdgeCount = 0;
        int concaveEdgeCount = 0;

        for (size_t neighborIdx = 0; neighborIdx < face.neighborIds.size(); ++neighborIdx) {
            const FaceFeature* neighbor = FindFaceById(faces, face.neighborIds[neighborIdx]);
            if (neighbor != nullptr && neighbor->surfaceType == GeomAbs_Cylinder) {
                adjacentCylinderCount++;
                if (neighbor->radius > 0.0) {
                    minAdjacentCylinderRadius = std::min(minAdjacentCylinderRadius, neighbor->radius);
                    maxAdjacentCylinderRadius = std::max(maxAdjacentCylinderRadius, neighbor->radius);
                    if (neighbor->radius <= kSmallHoleRadiusThreshold) {
                        adjacentSmallCylinderCount++;
                    }
                }
            }

            if (neighborIdx < face.neighborEdgeTypes.size()) {
                typedEdgeCount++;
                if (face.neighborEdgeTypes[neighborIdx] < 0) {
                    concaveEdgeCount++;
                }
            }
        }

        if (minAdjacentCylinderRadius == std::numeric_limits<double>::max()) {
            minAdjacentCylinderRadius = 0.0;
        }

        const double concaveEdgeRatio =
            typedEdgeCount > 0 ? static_cast<double>(concaveEdgeCount) / typedEdgeCount : 0.0;

        for (size_t wireIdx = 0; wireIdx < face.innerWireLengths.size(); ++wireIdx) {
            HoleCandidateFeature candidate;
            candidate.candidateId = nextCandidateId++;
            candidate.graphId = graphId;
            candidate.modelName = modelName;
            candidate.hostFaceId = face.id;
            candidate.localWireIndex = static_cast<int>(wireIdx);
            candidate.hostFaceArea = face.area;
            candidate.hostFaceRelativeArea = face.relativeArea;
            candidate.hostFacePerimeter = face.perimeter;
            candidate.hostInnerWireCount = face.innerWireCount;
            candidate.hostNeighborFaceCount = static_cast<int>(face.neighborIds.size());
            candidate.wireLength = face.innerWireLengths[wireIdx];
            candidate.wireLengthRatio = face.perimeter > 1e-6 ? candidate.wireLength / face.perimeter : 0.0;
            candidate.estimatedHoleRadius = candidate.wireLength / (2.0 * kPi);
            candidate.estimatedHoleArea = kPi * candidate.estimatedHoleRadius * candidate.estimatedHoleRadius;
            if (wireIdx < face.innerWireCenterXs.size()) {
                candidate.wireCenterX = face.innerWireCenterXs[wireIdx];
                candidate.wireCenterY = face.innerWireCenterYs[wireIdx];
                candidate.wireCenterZ = face.innerWireCenterZs[wireIdx];
            }
            candidate.adjacentFaceCount = static_cast<int>(face.neighborIds.size());
            candidate.adjacentCylinderCount = adjacentCylinderCount;
            candidate.adjacentSmallCylinderCount = adjacentSmallCylinderCount;
            candidate.minAdjacentCylinderRadius = minAdjacentCylinderRadius;
            candidate.maxAdjacentCylinderRadius = maxAdjacentCylinderRadius;
            candidate.concaveEdgeRatio = concaveEdgeRatio;

            const bool isSmallInnerWire =
                candidate.estimatedHoleRadius > 0.0 &&
                candidate.estimatedHoleRadius <= kSmallHoleRadiusThreshold;
            const bool isContextRichMediumHole =
                candidate.estimatedHoleRadius > 0.0 &&
                candidate.estimatedHoleRadius <= 3.0 &&
                candidate.adjacentSmallCylinderCount >= 4;
            const bool isConcaveMediumHole =
                candidate.estimatedHoleRadius > 0.0 &&
                candidate.estimatedHoleRadius <= 2.3 &&
                candidate.concaveEdgeRatio >= 0.5;
            const bool isPlanarArrayHole =
                candidate.estimatedHoleRadius > 0.0 &&
                candidate.estimatedHoleRadius <= kPlanarHoleRadiusThreshold &&
                candidate.hostFaceArea >= 1000.0 &&
                candidate.hostInnerWireCount >= 2;
            const bool hasHoleLikeContext =
                candidate.adjacentSmallCylinderCount > 0 ||
                candidate.concaveEdgeRatio >= 0.3;
            candidate.label =
                ((isSmallInnerWire && hasHoleLikeContext) || isContextRichMediumHole || isConcaveMediumHole || isPlanarArrayHole)
                    ? 1
                    : 0;

            candidates.push_back(candidate);
        }
    }

    return DeduplicateCandidates(candidates);
}
