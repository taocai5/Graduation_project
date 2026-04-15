#include "kmeans.h"
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <numeric>

KMeans::KMeans(int k, int max_iters) : k(k), max_iters(max_iters) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
}

double KMeans::distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

void KMeans::initialize_centroids(const std::vector<DataRow>& data) {
    clusters.clear();
    clusters.resize(k);
    if (data.empty()) return;

    // K-Means++ initialization
    int idx = std::rand() % (int)data.size();
    clusters[0].centroid = data[idx].features;

    for (int i = 1; i < k; ++i) {
        std::vector<double> dist_sq(data.size());
        double total = 0.0;
        for (size_t j = 0; j < data.size(); ++j) {
            double min_d2 = std::numeric_limits<double>::max();
            for (int c = 0; c < i; ++c) {
                double d = distance(data[j].features, clusters[c].centroid);
                min_d2 = std::min(min_d2, d * d);
            }
            dist_sq[j] = min_d2;
            total += min_d2;
        }

        double r = ((double)std::rand() / RAND_MAX) * total;
        double cum = 0.0;
        int sel = (int)data.size() - 1;
        for (size_t j = 0; j < data.size(); ++j) {
            cum += dist_sq[j];
            if (cum >= r) { sel = (int)j; break; }
        }
        clusters[i].centroid = data[sel].features;
    }
}

void KMeans::assign_clusters(const std::vector<DataRow>& data) {
    for (auto& c : clusters) c.point_indices.clear();
    for (size_t i = 0; i < data.size(); ++i) {
        double min_d = std::numeric_limits<double>::max();
        int best = 0;
        for (int j = 0; j < k; ++j) {
            double d = distance(data[i].features, clusters[j].centroid);
            if (d < min_d) { min_d = d; best = j; }
        }
        clusters[best].point_indices.push_back((int)i);
    }
}

void KMeans::update_centroids(const std::vector<DataRow>& data) {
    for (auto& cluster : clusters) {
        if (cluster.point_indices.empty()) continue;
        size_t dim = cluster.centroid.size();
        std::vector<double> nc(dim, 0.0);
        for (int idx : cluster.point_indices)
            for (size_t d = 0; d < dim; ++d)
                nc[d] += data[idx].features[d];
        for (size_t d = 0; d < dim; ++d)
            nc[d] /= (double)cluster.point_indices.size();
        cluster.centroid = nc;
    }
}

void KMeans::fit(const std::vector<DataRow>& data) {
    if (data.empty()) return;
    initialize_centroids(data);
    for (int iter = 0; iter < max_iters; ++iter) {
        std::vector<std::vector<double>> prev;
        for (const auto& c : clusters) prev.push_back(c.centroid);
        assign_clusters(data);
        update_centroids(data);
        bool converged = true;
        for (int i = 0; i < k; ++i)
            if (distance(prev[i], clusters[i].centroid) > 1e-6) { converged = false; break; }
        if (converged) {
            std::cout << "  Converged at iteration " << iter + 1 << "\n";
            break;
        }
    }
}

const std::vector<Cluster>& KMeans::get_clusters() const { return clusters; }

double KMeans::inertia() const {
    // SSE: sum of squared distances from each point to its centroid
    // We don't store the original data here, so inertia is computed externally in main
    // This is a placeholder — actual SSE computed in main after fit()
    return 0.0;
}
