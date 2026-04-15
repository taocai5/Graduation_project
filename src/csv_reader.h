#pragma once
#include <vector>
#include <string>
#include <unordered_map>

// 一行数据：特征向量 + 原始列名映射（可选，用于调试）
struct DataRow {
    std::vector<double> features;
};

// 加载结果：数据 + 实际使用的列名（按顺序）
struct Dataset {
    std::vector<DataRow> rows;
    std::vector<std::string> feature_names;  // 实际加载的列名
};

class CsvReader {
public:
    // 自动检测分隔符（逗号 or 分号）
    // feature_cols 为空时：自动选取所有数值列（排除 skip_cols 中的列）
    // feature_cols 非空时：按指定列名加载
    static Dataset load(
        const std::string& filepath,
        const std::vector<std::string>& feature_cols = {},
        const std::vector<std::string>& skip_cols = {}
    );

private:
    static char detect_delimiter(const std::string& header_line);
    static std::vector<std::string> split(const std::string& line, char delim);
    static std::string strip(std::string s);
    static bool is_numeric_col(
        const std::vector<std::vector<std::string>>& rows, int col_idx, int sample_n = 20
    );
};
