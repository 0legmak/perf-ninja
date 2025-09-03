#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <list>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <smmintrin.h>

using Char = unsigned char;
using Code = unsigned short;

constexpr auto kAlphaSz = std::numeric_limits<Char>::max() + 1;
constexpr auto kCodeCnt = std::numeric_limits<Code>::max() + 1;

struct TrieNode {
    TrieNode* l = nullptr;
    TrieNode* m = nullptr;
    TrieNode* r = nullptr;
    Char ch;
    Code code;
};

class Dictionary {
public:
    Dictionary() : nodes(kCodeCnt), roots(kAlphaSz), next_code(kAlphaSz) {
        for (auto i = 0; i < kAlphaSz; ++i) {
            nodes[i].ch = i;
            nodes[i].code = i;
            roots[i] = &nodes[i];
        }
    }

    ~Dictionary() {
        std::cout << "next_code=" << next_code << '\n';
    }

    std::optional<Code> consume(Char ch) {
        if (curr == nullptr) {
            curr = &roots[ch];
            const auto code = (*curr)->code;
            curr = &(*curr)->m;
            return code;
        }
        while (*curr) {
            if (ch < (*curr)->ch) {
                curr = &(*curr)->l;
            } else if (ch > (*curr)->ch) {
                curr = &(*curr)->r;
            } else {
                const auto code = (*curr)->code;
                curr = &(*curr)->m;
                return code;
            }
        }
        if (next_code < kCodeCnt) {
            nodes[next_code].ch = ch;
            nodes[next_code].code = next_code;
            *curr = &nodes[next_code];
            ++next_code;
        }
        curr = nullptr;
        return std::nullopt;
    }

private:
    std::vector<TrieNode> nodes;
    std::vector<TrieNode*> roots;
    TrieNode** curr = nullptr;
    size_t next_code;
};

void encode(const std::vector<Char>& decompressed, std::vector<Code>& compressed) {
    Dictionary dict;
    std::optional<Code> last_code;
    for (int i = 0; i < decompressed.size(); ++i) {
        const auto code = dict.consume(decompressed[i]);
        if (code) {
            last_code = code;
        } else {
            compressed.push_back(*last_code);
            --i;
        }
    }
    compressed.push_back(*last_code);
}

// #define CACHE_BYPASS

void decode(const std::vector<Code>& input, std::vector<Char>& output) {
    if (input.empty()) {
        return;
    }
#if defined(CACHE_BYPASS)
    alignas(__m128i) std::array<Char, 4 * sizeof(__m128i)> out_buf;
    size_t out_buf_idx = 0;
#endif
    std::array<std::vector<Char>, kCodeCnt> dict;
    for (auto i = 0; i < kAlphaSz; ++i) {
        dict[i].push_back(static_cast<Char>(i));
    }
    size_t input_idx = 0;
    size_t output_idx = 0;
    int code_cnt = kAlphaSz;
    std::vector<Char> val = dict[input[input_idx++]];
    while (true) {
        if (output_idx + val.size() > output.size()) {
            throw std::runtime_error("Corrupt data");
        }
#if defined(CACHE_BYPASS)
        int val_idx = 0;
        do {
            const size_t cp_sz = std::min(out_buf.size() - out_buf_idx, val.size() - val_idx);
            std::copy(val.begin() + val_idx, val.begin() + val_idx + cp_sz, out_buf.begin() + out_buf_idx);
            val_idx += cp_sz;
            out_buf_idx += cp_sz;
            if (out_buf.size() == out_buf_idx) {
                _mm_stream_si128((__m128i*)&output[output_idx] + 0, _mm_load_si128((const __m128i*)out_buf.data() + 0));
                _mm_stream_si128((__m128i*)&output[output_idx] + 1, _mm_load_si128((const __m128i*)out_buf.data() + 1));
                _mm_stream_si128((__m128i*)&output[output_idx] + 2, _mm_load_si128((const __m128i*)out_buf.data() + 2));
                _mm_stream_si128((__m128i*)&output[output_idx] + 3, _mm_load_si128((const __m128i*)out_buf.data() + 3));
                out_buf_idx = 0;
                output_idx += out_buf.size();
            }
        } while (val_idx != val.size());
#else
        std::copy(val.begin(), val.end(), output.begin() + output_idx);
        output_idx += val.size();
#endif
        if (input_idx == input.size()) {
            break;
        }
        const int code = input[input_idx++];
        if (code_cnt != kCodeCnt) {
            if (code < code_cnt) {
                val.push_back(dict[code].front());
            } else if (code == code_cnt) {
                val.push_back(val.front());
            } else {
                throw std::runtime_error("Corrupt data");
            }
            dict[code_cnt++] = val;
        }
        val = dict[code];
    }
#if defined(CACHE_BYPASS)
    std::copy(out_buf.begin(), out_buf.begin() + out_buf_idx, output.begin() + output_idx);
    output_idx += out_buf_idx;
#endif
    if (output_idx != output.size()) {
        throw std::runtime_error("Corrupt data");
    }
}

std::vector<Char> generate_input(size_t size) {
    std::default_random_engine re;
    std::uniform_int_distribution<int> dist('a', 'z');
    std::vector<Char> res(size);
    for (int i = 0; i < size; ++i) {
        res[i] = dist(re);
    }
    return res;
}

int main() {
    //const std::string test = "ABABABA";
    //std::vector<Char> input;
    //for (const auto c : test) {
    //    input.push_back(c);
    //}
    const auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<Char> input = generate_input(100 << 20);
    std::cout << "Input generation " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1) << '\n';
    //for (const auto c : input) {
    //    std::cout << c << ' ';
    //}
    //std::cout << '\n';
    std::vector<Code> compressed;
    const auto t2 = std::chrono::high_resolution_clock::now();
    encode(input, compressed);
    std::cout << "Compression " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t2) << '\n';
    std::cout << "Compressed size " << compressed.size() << '\n';
    //for (const auto c : compressed) {
    //    std::cout << c << ' ';
    //}
    //std::cout << '\n';

    std::vector<Char> decompressed(input.size());
    const auto t3 = std::chrono::high_resolution_clock::now();
    decode(compressed, decompressed);
    std::cout << "Decompression " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t3) << '\n';
    //for (const auto c : decompressed) {
    //    std::cout << c << ' ';
    //}
    //std::cout << '\n';

    if (decompressed != input) {
        std::cout << "Fail!\n";
    } else {
        std::cout << "Success!\n";
    }
}
