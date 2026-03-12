#include "csv_reader.h"
#include "kmeans.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>

void normalize(std::vector<StudentData>& data) {
    if (data.empty()) return;
    size_t num_features = data[0].features.size();
    
    for (size_t j = 0; j < num_features; ++j) {
        double min_val = data[0].features[j];
        double max_val = data[0].features[j];
        
        for (const auto& student : data) {
            min_val = std::min(min_val, student.features[j]);
            max_val = std::max(max_val, student.features[j]);
        }
        
        double range = max_val - min_val;
        if (range < 1e-9) continue;
        
        for (auto& student : data) {
            student.features[j] = (student.features[j] - min_val) / range;
        }
    }
}

int main() {
    // Try multiple paths
    std::vector<std::string> paths = {
        "data/student-mat.csv",
        "../data/student-mat.csv",
        "../../data/student-mat.csv",
        "n:/cpp_project/Graduation_project/data/student-mat.csv"
    };
    
    std::string filepath;
    std::vector<StudentData> data;
    
    for (const auto& p : paths) {
        std::cout << "Trying to load data from: " << p << std::endl;
        data = CsvReader::load_numeric_data(p);
        if (!data.empty()) {
            filepath = p;
            break;
        }
    }
    
    if (data.empty()) {
        std::cerr << "Failed to load data. Please check if the file exists." << std::endl;
        return 1;
    }
    
    std::cout << "Successfully loaded " << data.size() << " students from " << filepath << std::endl;
    if (!data.empty()) {
        std::cout << "Features per student: " << data[0].features.size() << std::endl;
    }
    
    std::cout << "Normalizing data..." << std::endl;
    normalize(data);
    
    int k = 3;
    std::cout << "Running K-Means with k=" << k << "..." << std::endl;
    
    KMeans kmeans(k);
    kmeans.fit(data);
    
    const auto& clusters = kmeans.get_clusters();
    
    std::cout << "\nClustering Results:" << std::endl;
    for (size_t i = 0; i < clusters.size(); ++i) {
        std::cout << "Cluster " << i << ": " << clusters[i].point_indices.size() << " students" << std::endl;
        std::cout << "Centroid (first 5 features): [";
        for (size_t j = 0; j < std::min((size_t)5, clusters[i].centroid.size()); ++j) {
            std::cout << std::fixed << std::setprecision(2) << clusters[i].centroid[j] << (j < std::min((size_t)5, clusters[i].centroid.size()) - 1 ? ", " : "");
        }
        if (clusters[i].centroid.size() > 5) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }

    // Save results to CSV
    std::string output_path = "output/clustering_results.csv";
    // Check if output dir exists, create if not (platform dependent, using system command for simplicity or assuming it exists)
    // The 'output' directory exists based on LS.
    
    std::ofstream out_file(output_path);
    if (out_file.is_open()) {
        out_file << "StudentIndex,ClusterID\n";
        
        // Create a mapping from student index to cluster ID
        std::vector<int> assignments(data.size(), -1);
        for (size_t i = 0; i < clusters.size(); ++i) {
            for (int idx : clusters[i].point_indices) {
                assignments[idx] = i;
            }
        }
        
        for (size_t i = 0; i < assignments.size(); ++i) {
            out_file << i << "," << assignments[i] << "\n";
        }
        std::cout << "\nResults saved to " << output_path << std::endl;
    } else {
        std::cerr << "Failed to save results to " << output_path << std::endl;
    }
    
    return 0;
}
