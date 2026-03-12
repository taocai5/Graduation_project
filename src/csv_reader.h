#pragma once
#include <vector>
#include <string>

struct StudentData {
    std::vector<double> features;
};

class CsvReader {
public:
    static std::vector<StudentData> load_numeric_data(const std::string& filepath);
};
