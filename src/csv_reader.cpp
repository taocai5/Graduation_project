#include "csv_reader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

static std::string strip_impl(std::string s) {
    // trim whitespace and surrounding quotes
    auto not_space = [](char c){ return !std::isspace((unsigned char)c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        s = s.substr(1, s.size() - 2);
    return s;
}

std::string CsvReader::strip(std::string s) {
    return strip_impl(std::move(s));
}

char CsvReader::detect_delimiter(const std::string& header_line) {
    int commas    = std::count(header_line.begin(), header_line.end(), ',');
    int semicolons = std::count(header_line.begin(), header_line.end(), ';');
    return (semicolons > commas) ? ';' : ',';
}

std::vector<std::string> CsvReader::split(const std::string& line, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, delim))
        tokens.push_back(strip_impl(token));
    return tokens;
}

// 对某列采样前 sample_n 个非空行，判断是否为数值列
bool CsvReader::is_numeric_col(
    const std::vector<std::vector<std::string>>& rows, int col_idx, int sample_n)
{
    int checked = 0;
    for (const auto& row : rows) {
        if (col_idx >= (int)row.size()) continue;
        const std::string& v = row[col_idx];
        if (v.empty()) continue;
        try {
            std::size_t pos;
            std::stod(v, &pos);
            if (pos != v.size()) return false;  // trailing non-numeric chars
        } catch (...) {
            return false;
        }
        if (++checked >= sample_n) break;
    }
    return checked > 0;
}

Dataset CsvReader::load(
    const std::string& filepath,
    const std::vector<std::string>& feature_cols,
    const std::vector<std::string>& skip_cols)
{
    Dataset result;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[CsvReader] Cannot open: " << filepath << "\n";
        return result;
    }

    std::string header_line;
    if (!std::getline(file, header_line)) return result;

    char delim = detect_delimiter(header_line);
    std::vector<std::string> headers = split(header_line, delim);

    // 读取所有数据行（用于自动列检测时的采样）
    std::vector<std::vector<std::string>> raw_rows;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto row = split(line, delim);
        raw_rows.push_back(std::move(row));
    }

    // 确定要加载的列索引
    std::vector<int> col_indices;

    if (!feature_cols.empty()) {
        // 用户显式指定列名
        for (const auto& col : feature_cols) {
            auto it = std::find(headers.begin(), headers.end(), col);
            if (it == headers.end()) {
                std::cerr << "[CsvReader] Column not found: " << col << "\n";
                continue;
            }
            col_indices.push_back((int)std::distance(headers.begin(), it));
            result.feature_names.push_back(col);
        }
    } else {
        // 自动检测：选所有数值列，跳过 skip_cols
        for (int i = 0; i < (int)headers.size(); ++i) {
            const std::string& h = headers[i];
            // 是否在跳过列表中
            bool skip = std::find(skip_cols.begin(), skip_cols.end(), h) != skip_cols.end();
            if (skip) continue;
            if (is_numeric_col(raw_rows, i)) {
                col_indices.push_back(i);
                result.feature_names.push_back(h);
            }
        }
    }

    if (col_indices.empty()) {
        std::cerr << "[CsvReader] No usable columns found in: " << filepath << "\n";
        return result;
    }

    // 将原始字符串行转换为 DataRow
    result.rows.reserve(raw_rows.size());
    for (const auto& row : raw_rows) {
        if (row.size() < headers.size()) continue;

        DataRow dr;
        dr.features.reserve(col_indices.size());
        bool valid = true;
        for (int idx : col_indices) {
            const std::string& v = row[idx];
            if (v.empty()) {
                dr.features.push_back(0.0);  // 缺失值填 0
                continue;
            }
            try {
                dr.features.push_back(std::stod(v));
            } catch (...) {
                valid = false;
                break;  // 无法解析的行直接跳过
            }
        }
        if (valid && dr.features.size() == col_indices.size())
            result.rows.push_back(std::move(dr));
    }

    return result;
}
