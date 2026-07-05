#pragma once

#include <string>

// Face-level exports for the graph-based pipeline.
int RunWingRivetTrainingExport(const std::string& inputDir, const std::string& outputCsv);
void RunSingleInferenceExport(const std::string& inputFile, const std::string& outputCsv);
void RunSingleFaceDump(const std::string& inputFile, const std::string& outputJson);
int RunFaceIdConsistencyCheck(const std::string& inputFile);
