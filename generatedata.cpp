#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <string>

// --- Cấu hình IFM ---
const int IFM_H = 32;
const int IFM_W = 32;
const int IFM_C = 64;

// --- Cấu hình Weight ---
const int W_H = 3;
const int W_W = 3;
const int W_C = 64;
const int W_F = 128;

int main() {
    std::string ifm_path = "c:\\Code c\\Learning\\ifm.txt";
    std::string weights_path = "c:\\Code c\\Learning\\weights.txt";
    std::ofstream ifm_file(ifm_path);
    std::ofstream weights_file(weights_path);

    if (!ifm_file.is_open()) {
        std::cerr << "Loi: Khong the mo file " << ifm_path << " de ghi." << std::endl;
        return 1;
    }
    if (!weights_file.is_open()) {
        std::cerr << "Loi: Khong the mo file " << weights_path << " de ghi." << std::endl;
        return 1;
    }

    // --- Khởi tạo bộ tạo số ngẫu nhiên ---
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_int_distribution<int> dist(-128, 127);

    // --- Ghi dữ liệu IFM theo thứ tự HWC ---
    std::cout << "Dang ghi du lieu IFM (" << IFM_H << "x" << IFM_W << "x" << IFM_C << ") theo HWC..." << std::endl;
    for (int h = 0; h < IFM_H; ++h) {
        for (int w = 0; w < IFM_W; ++w) {
            for (int c = 0; c < IFM_C; ++c) {
                ifm_file << dist(rng) << "\n";
            }
        }
    }

    // --- Ghi dữ liệu Weights theo thứ tự OIHW ---
    std::cout << "Dang ghi du lieu Weights (" << W_F << "x" << W_H << "x" << W_W << "x" << W_C << ") theo OIHW..." << std::endl;
    for (int f = 0; f < W_F; ++f) {
        for (int kh = 0; kh < W_H; ++kh) {
            for (int kw = 0; kw < W_W; ++kw) {
                for (int c = 0; c < W_C; ++c) {
                    weights_file << dist(rng) << "\n";
                }
            }
        }
    }

    ifm_file.close();
    weights_file.close();
    std::cout << "Da tao thanh cong file " << ifm_path << " va " << weights_path << "." << std::endl;

    return 0;
}