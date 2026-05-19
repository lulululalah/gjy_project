#pragma once

#include <string>

// Face-level exports kept for the original graph pipeline.
void RunBatchTrainingExport(const std::string& inputDir, const std::string& outputCsv);
void RunSingleInferenceExport(const std::string& inputFile, const std::string& outputCsv);

// Hole-candidate exports are the current path for micro-hole dirty-geometry experiments.
void RunHoleCandidateTrainingExport(const std::string& inputDir, const std::string& outputCsv);
void RunSingleHoleCandidateInferenceExport(const std::string& inputFile, const std::string& outputCsv);
