#pragma once
#include <vector>
#include "csv_reader.h"

struct Cluster {
    std::vector<double> centroid;
    std::vector<int> point_indices;
};

class KMeans {
public:
    KMeans(int k, int max_iters = 100);
    void fit(const std::vector<StudentData>& data);
    const std::vector<Cluster>& get_clusters() const;

private:
    int k;
    int max_iters;
    std::vector<Cluster> clusters;
    
    double distance(const std::vector<double>& a, const std::vector<double>& b);
    void initialize_centroids(const std::vector<StudentData>& data);
    void assign_clusters(const std::vector<StudentData>& data);
    void update_centroids(const std::vector<StudentData>& data);
};
