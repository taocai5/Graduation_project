#include "kmeans.h"
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iostream>

KMeans::KMeans(int k, int max_iters) : k(k), max_iters(max_iters) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
}

double KMeans::distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void KMeans::initialize_centroids(const std::vector<StudentData>& data) {
    clusters.clear();
    clusters.resize(k);
    if (data.empty()) return;

    // K-Means++ Initialization
    // 1. Choose first center uniformly at random
    int idx = std::rand() % data.size();
    clusters[0].centroid = data[idx].features;

    // 2. Choose remaining k-1 centers
    for (int i = 1; i < k; ++i) {
        std::vector<double> distances(data.size());
        double sum_sq_dist = 0.0;

        for (size_t j = 0; j < data.size(); ++j) {
            double min_dist_sq = std::numeric_limits<double>::max();
            for (int c = 0; c < i; ++c) {
                double d = distance(data[j].features, clusters[c].centroid);
                double d_sq = d * d;
                if (d_sq < min_dist_sq) {
                    min_dist_sq = d_sq;
                }
            }
            distances[j] = min_dist_sq;
            sum_sq_dist += min_dist_sq;
        }

        // Weighted random selection
        double r = ((double)std::rand() / RAND_MAX) * sum_sq_dist;
        double current_sum = 0.0;
        int selected_idx = -1;
        for (size_t j = 0; j < data.size(); ++j) {
            current_sum += distances[j];
            if (current_sum >= r) {
                selected_idx = j;
                break;
            }
        }
        if (selected_idx == -1) selected_idx = data.size() - 1; // Fallback
        
        clusters[i].centroid = data[selected_idx].features;
    }
}

void KMeans::assign_clusters(const std::vector<StudentData>& data) {
    for (auto& cluster : clusters) {
        cluster.point_indices.clear();
    }

    for (size_t i = 0; i < data.size(); ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        
        for (int j = 0; j < k; ++j) {
            double d = distance(data[i].features, clusters[j].centroid);
            if (d < min_dist) {
                min_dist = d;
                best_cluster = j;
            }
        }
        clusters[best_cluster].point_indices.push_back(i);
    }
}

void KMeans::update_centroids(const std::vector<StudentData>& data) {
    for (auto& cluster : clusters) {
        if (cluster.point_indices.empty()) continue;
        
        std::vector<double> new_centroid(cluster.centroid.size(), 0.0);
        for (int idx : cluster.point_indices) {
            const auto& features = data[idx].features;
            for (size_t d = 0; d < features.size(); ++d) {
                new_centroid[d] += features[d];
            }
        }
        
        for (size_t d = 0; d < new_centroid.size(); ++d) {
            new_centroid[d] /= cluster.point_indices.size();
        }
        cluster.centroid = new_centroid;
    }
}

void KMeans::fit(const std::vector<StudentData>& data) {
    if (data.empty()) return;
    
    initialize_centroids(data);
    
    for (int iter = 0; iter < max_iters; ++iter) {
        std::vector<std::vector<double>> old_centroids;
        for (const auto& c : clusters) old_centroids.push_back(c.centroid);
        
        assign_clusters(data);
        update_centroids(data);
        
        bool converged = true;
        for (int i = 0; i < k; ++i) {
            if (distance(old_centroids[i], clusters[i].centroid) > 1e-4) {
                converged = false;
                break;
            }
        }
        
        if (converged) {
            std::cout << "Converged at iteration " << iter + 1 << std::endl;
            break;
        }
    }
}

const std::vector<Cluster>& KMeans::get_clusters() const {
    return clusters;
}
