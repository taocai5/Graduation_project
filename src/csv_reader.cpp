#include "csv_reader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// Helper to strip quotes
static std::string strip_quotes(std::string s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

std::vector<StudentData> CsvReader::load_numeric_data(const std::string& filepath) {
    std::vector<StudentData> data;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filepath << std::endl;
        return data;
    }

    std::string line;
    // Read header
    if (!std::getline(file, line)) return data;

    // Map column names to indices
    std::vector<std::string> headers;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ';')) {
        headers.push_back(strip_quotes(cell));
    }

    // Identify numerical columns we want to use
    std::vector<std::string> target_cols = {
        "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
        "G1", "G2", "G3"
    };
    
    std::vector<int> target_indices;
    for (const auto& col : target_cols) {
        auto it = std::find(headers.begin(), headers.end(), col);
        if (it != headers.end()) {
            target_indices.push_back(std::distance(headers.begin(), it));
        }
    }

    while (std::getline(file, line)) {
        std::stringstream line_ss(line);
        std::vector<std::string> row_values;
        while (std::getline(line_ss, cell, ';')) {
            row_values.push_back(strip_quotes(cell));
        }

        if (row_values.size() != headers.size()) continue;

        StudentData student;
        for (int idx : target_indices) {
            try {
                student.features.push_back(std::stod(row_values[idx]));
            } catch (...) {
                student.features.push_back(0.0); // Fallback
            }
        }
        data.push_back(student);
    }

    return data;
}
