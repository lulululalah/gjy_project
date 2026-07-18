#pragma once

#include <string>

int RunWingRivetInjection(const std::string& inputFile);
int RunStarDecalInjection(const std::string& inputFile, int hostFaceId = -1);
int RunBatchWingRivetInjection(const std::string& inputDir);
int RunWingRivetDatasetValidation(const std::string& inputDir);
