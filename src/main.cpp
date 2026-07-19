#include "FeatureInjector.h"
#include "Workflow.h"

#include <filesystem>
#include <exception>
#include <iostream>
#include <string>

namespace
{
    struct DetectorPaths
    {
        std::filesystem::path wingRivetDatasetDir;
        std::filesystem::path wingRivetTrainingCsv;
        std::filesystem::path faceInferenceCsv;
        std::filesystem::path faceDumpJson;
    };

    DetectorPaths ResolvePaths(const std::filesystem::path &exePath)
    {
        const std::filesystem::path projectRoot = exePath.parent_path().parent_path().parent_path();
        return {
            projectRoot / "data" / "plane_model",
            projectRoot / "data" / "wing_rivet_training_set.csv",
            projectRoot / "data" / "current_inference.csv",
            projectRoot / "data" / "current_faces.json",
        };
    }

    void PrintUsage()
    {
        std::cout
            << "Usage:\n"
            << "  Detector.exe --train\n"
            << "  Detector.exe --predict <file>\n"
            << "  Detector.exe --dump-faces <file>\n"
            << "  Detector.exe --check-face-id <file>\n"
            << "  Detector.exe --inject-wing-rivets <file>\n"
            << "  Detector.exe --inject-star-decals <after-rivet-file> [--host-face <id>] [--max-radius-scale <0..0.440>]\n"
            << "  Detector.exe --inject-v13-decal <after-rivet-file> --host-face <id>\n"
            << "  Detector.exe --inject-v2-decal <after-rivet-file> --host-face <id>\n"
            << "  Detector.exe --inject-v3-decal <after-rivet-file> --host-face <id> [--rotate-180]\n"
            << "  Detector.exe --inject-wing-rivets-batch <dir>\n"
            << "  Detector.exe --validate-wing-rivet-dataset <dir>\n"
            << "  Detector.exe --export-wing-rivet-training <dir>\n";
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        PrintUsage();
        return 1;
    }

    const std::string mode = argv[1];
    const auto paths = ResolvePaths(std::filesystem::absolute(argv[0]));

    if (mode == "--train")
    {
        return RunWingRivetTrainingExport(paths.wingRivetDatasetDir.string(), paths.wingRivetTrainingCsv.string());
    }
    else if (mode == "--predict" && argc >= 3)
    {
        const std::string filePath = argv[2];
        RunSingleInferenceExport(filePath, paths.faceInferenceCsv.string());
    }
    else if (mode == "--dump-faces" && argc >= 3)
    {
        const std::string filePath = argv[2];
        RunSingleFaceDump(filePath, paths.faceDumpJson.string());
    }
    else if (mode == "--check-face-id" && argc >= 3)
    {
        const std::string filePath = argv[2];
        return RunFaceIdConsistencyCheck(filePath);
    }
    else if (mode == "--inject-wing-rivets" && argc >= 3)
    {
        const std::string filePath = argv[2];
        return RunWingRivetInjection(filePath);
    }
    else if (mode == "--inject-star-decals" && argc >= 3)
    {
        const std::string filePath = argv[2];
        int hostFaceId = -1;
        double maxRadiusScale = 0.440;
        for (int argumentIndex = 3; argumentIndex < argc; argumentIndex += 2) {
            if (argumentIndex + 1 >= argc) {
                std::cout << "Expected a value after: " << argv[argumentIndex] << std::endl;
                return 1;
            }
            try {
                const std::string option = argv[argumentIndex];
                if (option == "--host-face") {
                    hostFaceId = std::stoi(argv[argumentIndex + 1]);
                } else if (option == "--max-radius-scale") {
                    maxRadiusScale = std::stod(argv[argumentIndex + 1]);
                } else {
                    std::cout << "Unknown star decal option: " << option << std::endl;
                    return 1;
                }
            } catch (const std::exception&) {
                std::cout << "Invalid value for star decal option: " << argv[argumentIndex] << std::endl;
                return 1;
            }
        }
        return RunStarDecalInjection(filePath, hostFaceId, maxRadiusScale);
    }
    else if (mode == "--inject-v13-decal" && argc == 5 && std::string(argv[3]) == "--host-face")
    {
        try {
            return RunStarDecalInjection(argv[2], std::stoi(argv[4]), 0.440, 1);
        } catch (const std::exception&) {
            std::cout << "Invalid host face ID: " << argv[4] << std::endl;
            return 1;
        }
    }
    else if (mode == "--inject-v2-decal" && argc == 5 && std::string(argv[3]) == "--host-face")
    {
        try {
            return RunStarDecalInjection(argv[2], std::stoi(argv[4]), 0.440, 2);
        } catch (const std::exception&) {
            std::cout << "Invalid host face ID: " << argv[4] << std::endl;
            return 1;
        }
    }
    else if (mode == "--inject-v3-decal" && (argc == 5 || argc == 6) && std::string(argv[3]) == "--host-face")
    {
        try {
            const bool rotateText180 = argc == 6 && std::string(argv[5]) == "--rotate-180";
            if (argc == 6 && !rotateText180) {
                std::cout << "Expected optional argument: --rotate-180" << std::endl;
                return 1;
            }
            return RunStarDecalInjection(argv[2], std::stoi(argv[4]), 0.440, 3, rotateText180);
        } catch (const std::exception&) {
            std::cout << "Invalid host face ID: " << argv[4] << std::endl;
            return 1;
        }
    }
    else if (mode == "--inject-wing-rivets-batch" && argc >= 3)
    {
        const std::string dirPath = argv[2];
        return RunBatchWingRivetInjection(dirPath);
    }
    else if (mode == "--validate-wing-rivet-dataset" && argc >= 3)
    {
        const std::string dirPath = argv[2];
        return RunWingRivetDatasetValidation(dirPath);
    }
    else if (mode == "--export-wing-rivet-training" && argc >= 3)
    {
        const std::string dirPath = argv[2];
        return RunWingRivetTrainingExport(dirPath, paths.wingRivetTrainingCsv.string());
    }
    else
    {
        std::cout << "Unknown command or missing argument." << std::endl;
        PrintUsage();
        return 1;
    }

    return 0;
}
