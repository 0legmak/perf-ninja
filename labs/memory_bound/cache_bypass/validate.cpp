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
using Code = uint32_t;

constexpr auto kAlphaSz = std::numeric_limits<Char>::max() + 1;
constexpr auto kCodeBits = 18;
constexpr auto kCodeCnt = 1 << kCodeBits;
constexpr size_t kBitsInByte = 8;

struct TrieNode {
    TrieNode* l = nullptr;
    TrieNode* m = nullptr;
    TrieNode* r = nullptr;
    Char ch;
    Code code;
};

class Dictionary {
public:
    Dictionary() : nodes(kCodeCnt), roots(kAlphaSz), next_code(kAlphaSz), total_code_word_size(kAlphaSz) {
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
            code_word_len = 1;
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
                ++code_word_len;
                return code;
            }
        }
        if (next_code < kCodeCnt) {
            nodes[next_code].ch = ch;
            nodes[next_code].code = next_code;
            *curr = &nodes[next_code];
            ++next_code;
            ++code_word_len;
            total_code_word_size += code_word_len;
        }
        curr = nullptr;
        return std::nullopt;
    }

    size_t get_total_code_word_size() const {
        return total_code_word_size;
    }

private:
    std::vector<TrieNode> nodes;
    std::vector<TrieNode*> roots;
    TrieNode** curr = nullptr;
    size_t next_code;
    size_t code_word_len;
    size_t total_code_word_size;
};

void encode(const std::vector<Char>& input, std::vector<unsigned char>& output, size_t& code_count, size_t& total_code_word_size) {
    Dictionary dict;
    std::optional<Code> last_code;
    size_t out_idx = 0;
    size_t filled_bits_in_last_byte = 0;
    code_count = 0;
    auto output_code = [&](Code code) {
        output.resize(out_idx + sizeof(code));
        code = (code << filled_bits_in_last_byte) | output[out_idx];
        memcpy(&output[out_idx], &code, sizeof(code));
        out_idx += (filled_bits_in_last_byte + kCodeBits) / kBitsInByte;
        filled_bits_in_last_byte = (filled_bits_in_last_byte + kCodeBits) % kBitsInByte;
        ++code_count;
    };
    for (int i = 0; i < input.size(); ++i) {
        const auto code = dict.consume(input[i]);
        if (code) {
            last_code = code;
        } else {
            output_code(*last_code);
            --i;
        }
    }
    output_code(*last_code);
    total_code_word_size = dict.get_total_code_word_size();
}

//#define CACHE_BYPASS

void decode(const std::vector<unsigned char>& input, std::vector<Char>& output, size_t code_count, size_t total_code_word_size) {
    if (code_count == 0) {
        return;
    }
#if defined(CACHE_BYPASS)
    alignas(__m128i) std::array<Char, 4 * sizeof(__m128i)> out_buf;
    size_t out_buf_idx = 0;
#endif
    std::vector<Char> code_words(total_code_word_size);
    size_t codes_idx = 0;
    std::array<std::span<Char>, kCodeCnt> dict;
    for (auto i = 0; i < kAlphaSz; ++i) {
        code_words[codes_idx] = i;
        dict[i] = std::span(code_words).subspan(codes_idx, 1);
        ++codes_idx;
    }
    size_t input_idx = 0;
    size_t consumed_bits_in_last_byte = 0;
    size_t output_idx = 0;
    int code_cnt = kAlphaSz;
    std::vector<Char> val;
    auto input_code = [&]() -> Code {
        Code code;
        memcpy(&code, &input[input_idx], sizeof(code));
        code = (code >> consumed_bits_in_last_byte) & ((1u << kCodeBits) - 1);
        input_idx += (consumed_bits_in_last_byte + kCodeBits) / kBitsInByte;
        consumed_bits_in_last_byte = (consumed_bits_in_last_byte + kCodeBits) % kBitsInByte;
        return code;
    };
    const auto code_word = dict[input_code()];
    val.assign(code_word.begin(), code_word.end());
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
        if (--code_count == 0) {
            break;
        }
        const int code = input_code();
        if (code_cnt != kCodeCnt) {
            if (code < code_cnt) {
                val.push_back(dict[code].front());
            } else if (code == code_cnt) {
                val.push_back(val.front());
            } else {
                throw std::runtime_error("Corrupt data");
            }
            std::copy(val.begin(), val.end(), code_words.begin() + codes_idx);
            dict[code_cnt++] = std::span(code_words).subspan(codes_idx, val.size());
            codes_idx += val.size();
        }
        val.assign(dict[code].begin(), dict[code].end());
    }
#if defined(CACHE_BYPASS)
    std::copy(out_buf.begin(), out_buf.begin() + out_buf_idx, output.begin() + output_idx);
    output_idx += out_buf_idx;
#endif
    if (output_idx != output.size()) {
        throw std::runtime_error("Corrupt data");
    }
    std::cout << "code_words.size() = " << code_words.size() << '\n';
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
    std::vector<Char> input = generate_input(50 << 20);
    std::cout << "Input generation " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1) << '\n';
    //for (const auto c : input) {
    //    std::cout << c << ' ';
    //}
    //std::cout << '\n';
    std::vector<unsigned char> compressed;
    const auto t2 = std::chrono::high_resolution_clock::now();
    size_t code_count;
    size_t total_code_word_size;
    encode(input, compressed, code_count, total_code_word_size);
    std::cout << "Compression " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t2) << '\n';
    std::cout << "Compressed size " << compressed.size() << '\n';
    std::cout << "total_code_word_size = " << total_code_word_size << '\n';
    //for (const auto c : compressed) {
    //    std::cout << c << ' ';
    //}
    //std::cout << '\n';

    std::vector<Char> decompressed(input.size());
    const auto t3 = std::chrono::high_resolution_clock::now();
    decode(compressed, decompressed, code_count, total_code_word_size);
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
