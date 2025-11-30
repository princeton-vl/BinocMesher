#ifndef UTILS_H
#define UTILS_H

// headers
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <array>
#include <deque>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>
#include <random>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <malloc.h>
#include <unistd.h>
#include <omp.h>

// DEBUG flag - enables bounds checking and assertions for debugging
#define DEBUG

// Common type and function shortcuts
#define ll long long
#define mp std::make_pair
#define pair std::pair
typedef pair<int, int> II;
#define array std::array
#define map std::map
#define priority_queue std::priority_queue
#define queue std::queue
#define set std::set
#define size_t std::size_t
#define sdeque std::deque
#define sort std::sort
#define smax std::max
#define smin std::min
#define sswap std::swap
#define unique std::unique
#define unordered_map std::unordered_map
#ifdef DEBUG
#define INDEX(a, i) (assert((i) >= 0 && (i) < (a).size()), (a)[i])
#else
#define INDEX(a, i) (a)[i]
#endif
#define INT_MAX 2147483647
#define cubex(x) (x) * (x) * (x)
#define cube_index(x, y, z, s) (x)*(s)*(s)+(y)*(s)+(z)
inline int sqr(int x) { return x * x; }
template<typename T>
inline int int_log(T x) { return int(smax(T(0), ceil(log2(x)))); }
inline int lowbit(int x) { return x&(-x); }
inline bool non_neg(int a[3][3]) {
    int det = 0;
    det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
        a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
        a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    return det >= 0;
}


template<typename T>
array<T, 3> make_array3(T value1, T value2, T value3) {
    return array<T, 3>{value1, value2, value3};
}

// Primary thread count for parallel operations
#define N_THREAD 32

// Clear containers and free unused memory
#define CLS(v) {v.clear(); v.shrink_to_fit(); malloc_trim(0);}  // Clear + shrink + trim heap
#define CL(v) {v.clear(); malloc_trim(0);}                     // Clear + trim heap

// Number of cached node groups - currently disabled
#define CACHECNT 1

// Macro for timing code blocks
#define MEASURE_TIME(description, timing, code_block) { \
    auto start = std::chrono::high_resolution_clock::now(); \
    code_block \
    auto end = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double> duration = end - start; \
    FILE *log = fopen(params::log_path.c_str(), "a"); \
    if (timing) fprintf(log, "%s  - Time taken: %lf seconds\n", description, (double)duration.count()); \
    fclose(log); \
}

// Macro for logging RSS (Resident Set Size) memory usage
// These macros are put at the locations where memory usage are likely to peak
#ifdef DEBUG
#define PRINT_RSS_MEMORY(description) { \
    std::ifstream file("/proc/self/statm"); \
    long rss; \
    file >> rss >> rss; \
    file.close(); \
    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; \
    rss *= page_size_kb; \
    FILE *log = fopen(params::log_path.c_str(), "a"); \
    fprintf(log, "%s, RSS Memory Usage: %lf GB\n", description, (double)rss * 1.0 / (1<<20)); \
    fclose(log); \
}
#else
#define PRINT_RSS_MEMORY(description) // No-op in release mode
#endif

// OpenMP pragma generation - #pragma can't expand inside MEASURE_TIME macro arguments
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define OMP_PRAGMA(x) _Pragma(TOSTRING(x))

// Custom vector wrapper optimized for frequent resizing and memory reuse
template <typename SizeType, typename T>
struct vec {
    static_assert(std::is_same<SizeType, int32_t>::value || std::is_same<SizeType, ll>::value, "SizeType must be int32_t or ll");
    std::vector<T> a;
    SizeType _size;
    
    vec() {
        _size = 0;
    }

    template<typename Type>
    vec(Type size) {
        static_assert(std::is_same<Type, SizeType>::value, "Argument must be of the corresponding type");
        resize(size);
    }
    
    void push_back(const T& v) {
        #ifdef DEBUG
        if (_size >= std::numeric_limits<SizeType>::max()) {
            throw std::overflow_error("Exceeded maximum size limit");
        }
        #endif
        if (_size < a.size()) {
            a[_size] = v;
            _size++;
        }
        else {
            a.push_back(v);
            _size++;
        }
        #ifdef DEBUG
        assert(_size <= INT_MAX);
        #endif
    }

    SizeType size()  const {
        return _size;
    }

    template<typename Type>
    T& operator[](Type index) {
        static_assert(std::is_same<Type, SizeType>::value, "Index must be of the corresponding type");
        #ifdef DEBUG
            assert(index >= 0 && index < _size);
        #endif
        return a[index];
    }

    
    bool operator==(vec<SizeType, T> &b){
        if (_size != b._size) return 0;
        for (int i = 0; i < _size; i++)
            if (a[i] != b.a[i]) return 0;
        return 1;
    }

    void clear(){
        _size = 0;
    }

    void shrink_to_fit(){
        a.resize(_size);
        a.shrink_to_fit();
    }

    template<typename Type>
    void resize(Type s) {
        static_assert(std::is_same<Type, SizeType>::value, "Argument must be of the corresponding type");
        if (a.size() < s) {
            a.resize(s);
        }
        _size = s;
    }

    template<typename Type>
    void resizefill(Type s, const T& v) {
        static_assert(std::is_same<Type, SizeType>::value, "Argument must be of the corresponding type");
        if (a.size() < s) {
            a.resize(s);
        }
        if (s > _size) std::fill(a.begin() + _size, a.begin() + s, v);
        _size = s;
    }

    void operator=(vec<SizeType, T> &b) {
        resize(b.size());
        for (int i = 0; i < _size; i++) {
            a[i] = b.a[i];
        }
    }

    bool empty() const {
        return _size == 0;
    }

    void insert_end(vec<SizeType, T> &b) {
        auto tmp = _size;
        if ((ll)_size + b._size >= std::numeric_limits<SizeType>::max()) {
            throw std::overflow_error("Exceeded maximum size limit");
        }
        resize(_size + b._size);
        std::copy(b.begin(), b.begin() + b._size, a.begin() + tmp);
    }

    template<typename Type>
    void erase(Type st, Type ed)  {
        static_assert(std::is_same<Type, SizeType>::value, "Argument must be of the corresponding type");
        a.erase(a.begin() + st, a.begin() + ed);
        _size -= ed - st;
    }

    void fill(T v){
        std::fill(a.begin(), a.end(), v);
    }

    typename std::vector<T>::iterator begin() {
        return a.begin();
    }
    typename std::vector<T>::iterator end() {
        return a.begin() + _size;
    }
};

// I/O functions for containers
template <typename SizeType, typename T>
void write_vec(FILE *outfile, vec<SizeType, T> &v) {
    SizeType n = v.size();
    fwrite(&n, sizeof(SizeType), 1, outfile);
    if (n != 0) fwrite(&v[(SizeType)0], sizeof(T), n, outfile);
}

template <typename SizeType, typename T>
void read_vec(FILE *infile, vec<SizeType, T> &v) {
    SizeType n;
    fread(&n, sizeof(SizeType), 1, infile);
    #ifdef DEBUG
    assert(n >= 0);
    #endif
    v.resize(n);
    if (n != 0) fread(&v[(SizeType)0], sizeof(T), n, infile);
}

template <typename SizeType, typename T>
void write_vec_headless(FILE *outfile, vec<SizeType, T> &v) {
    SizeType n = v.size();
    if (n != 0) fwrite(&v[(SizeType)0], sizeof(T), n, outfile);
}

template <typename SizeType, typename T>
void read_vec_headless(FILE *infile, vec<SizeType, T> &v) {
    v.resize(0);
    T buffer;
    while (1) {
        if (fread(&buffer, sizeof(T), 1, infile) == 0) break;
        v.push_back(buffer);
    }
}

template <typename T>
void write_deq(FILE *outfile, sdeque<T> &v) {
    ll n = v.size();
    fwrite(&n, sizeof(ll), 1, outfile);
    for (ll i = 0; i < n; i++) {
        fwrite(&v[i], sizeof(T), 1, outfile);
    }
}

template <typename T>
void read_deq(FILE *infile, sdeque<T> &v) {
    ll n;
    fread(&n, sizeof(ll), 1, infile);
    v.clear();
    for (ll i = 0; i < n; i++) {
        T buffer;
        fread(&buffer, sizeof(T), 1, infile);
        v.push_back(buffer);
    }
}

// Parallel sort implementation for different container types and comparison methods
template <typename Container, typename Compare>
void _sort_impl(Container &a, Compare cmp) {
    using SizeType = decltype(a.size());
    
    OMP_PRAGMA(omp parallel for)
    for (int g = 0; g < N_THREAD; g++) {
        SizeType start_g = ((ll)a.size() + N_THREAD - 1) / N_THREAD * g;
        SizeType end_g = smin((ll)a.size(), ((ll)a.size() + N_THREAD - 1) / N_THREAD * (g+1));
        if (start_g < end_g) {
            sort(a.begin() + start_g, a.begin() + end_g, cmp);
        }
    }
    
    for (int g = 0; g < N_THREAD-1; g++) {
        SizeType middle_g = ((ll)a.size() + N_THREAD - 1) / N_THREAD * (g+1);
        SizeType end_g = smin((ll)a.size(), ((ll)a.size() + N_THREAD - 1) / N_THREAD * (g+2));
        if (middle_g <= end_g) {
            std::inplace_merge(a.begin(), a.begin() + middle_g, a.begin() + end_g, cmp);
        }
    }
}

template <typename T>
struct function_pointer_wrapper {
    bool (*func)(const T&, const T&);
    function_pointer_wrapper(bool (*f)(const T&, const T&)) : func(f) {}
    bool operator()(const T& a, const T& b) const { return func(a, b); }
};

template <typename SizeType, typename T>
void _sort(vec<SizeType, T> &a, bool (*cmp)(const T&, const T&)) {
    _sort_impl(a, function_pointer_wrapper<T>(cmp));
}

template <typename SizeType, typename T>
void _sort(vec<SizeType, T> &a) {
    _sort_impl(a, std::less<T>());
}

template <typename T>
void _sort(sdeque<T> &a, bool (*cmp)(const T&, const T&)) {
    _sort_impl(a, function_pointer_wrapper<T>(cmp));
}

// sort-and-unique operations for different container types
template <typename SizeType, typename T, typename Compare, typename Equal>
void _make_unique_vec_impl(vec<SizeType, T> &a, Compare cmp, Equal eq) {
    _sort_impl(a, cmp);
    auto tmp = unique(a.begin(), a.begin() + a.size(), eq);
    SizeType ds = a.begin() + a.size() - tmp;
    a._size -= ds;
}

template <typename T>
struct equality_function_wrapper {
    bool (*func)(const T&, const T&);
    equality_function_wrapper(bool (*f)(const T&, const T&)) : func(f) {}
    bool operator()(const T& a, const T& b) const { return func(a, b); }
};

template <typename SizeType, typename T>
void make_unique(vec<SizeType, T> &a, bool (*cmp)(const T&, const T&), bool (*eq)(const T&, const T&)) {
    _make_unique_vec_impl(a, function_pointer_wrapper<T>(cmp), equality_function_wrapper<T>(eq));
}

template <typename SizeType, typename T>
void make_unique(vec<SizeType, T> &a) {
    _make_unique_vec_impl(a, std::less<T>(), std::equal_to<T>());
}

template <typename T>
void make_unique(sdeque<T> &a, bool (*cmp)(const T&, const T&), bool (*eq)(const T&, const T&)) {
    _sort(a, cmp);
    auto tmp = unique(a.begin(), a.end(), eq);
    a.erase(tmp, a.end());
}

#endif // UTILS_H

