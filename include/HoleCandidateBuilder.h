#pragma once

#include <string>
#include <vector>

#include "FeatureInfo.h"
#include "HoleCandidateFeature.h"

std::vector<HoleCandidateFeature> BuildPlanarHoleCandidates(
    const std::vector<FaceFeature>& faces,
    int graphId,
    const std::string& modelName
);
