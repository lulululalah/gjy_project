#pragma once

#include <string>
#include <vector>

int RunWingRivetInjection(
    const std::string& inputFile,
    const std::vector<int>& hostFaceIds = {}
);
int RunBooleanHostFaceExport(const std::string& inputFile);
int RunStarDecalInjection(
    const std::string& inputFile,
    int hostFaceId = -1,
    double maxStarRadiusScale = 0.440,
    int textStyle = 0,
    bool rotateText180 = false
);
int RunBatchWingRivetInjection(const std::string& inputDir);
int RunWingRivetDatasetValidation(const std::string& inputDir);
