#pragma once

#include <algorithm>
#include <utility>
#include <array>
#include <cstring>
#include <iomanip>
#include <vector>
#include <cassert>
#include <cinttypes>
#include <iostream>
#include <numeric>
#include <concepts>


#define THIS static_cast<std::conditional_t<std::is_void_v<Derived>, \
    std::remove_reference_t<decltype(*this)>, \
    ndspan::detail::copy_const_t<std::remove_reference_t<decltype(*this)>, Derived>>*>(this)

#define NDSPAN_INLINE __attribute__((always_inline)) inline

#define NDSPAN_LAMBDA_INLINE __attribute__((always_inline, flatten))

#define DEFAULT_RULE_OF_FOUR(CLASSNAME)                  \
    CLASSNAME(const CLASSNAME& other) = default;      \
    CLASSNAME(CLASSNAME&& other) = default;           \
    CLASSNAME& operator=(const CLASSNAME& other) = default; \
    CLASSNAME& operator=(CLASSNAME&& other) = default;

#define NDSPAN_BOUNDS_ASSERT(i, n) assert((i>=0 && size_t(i)<size_t(n)) && "Index out of bounds")

#define NDSPAN_FOR_LOOP(I, N, ...) \
ndspan::ForEach<N>([&]<size_t I>() __attribute__((always_inline, flatten)) { \
    __VA_ARGS__ \
})

#define NDSPAN_EXPAND(N, I, ...) \
ndspan::Expand<N>([&]<size_t... I>() __attribute__((always_inline, flatten)) { \
    __VA_ARGS__ \
})

namespace ndspan{

namespace detail{

template<typename From, typename To>
using copy_const_t = std::conditional_t<std::is_const_v<From>, const To, To>;


template<size_t I, std::size_t N, typename F, typename... Args>
NDSPAN_INLINE constexpr void for_each_impl(F& f, Args&... args){
    if constexpr (I < N) {
        f.template operator()<I>(args...);
        for_each_impl<I + 1, N>(f, args...);
    }
}

template<std::size_t I, typename FirstType, typename... ArgType>
NDSPAN_INLINE constexpr decltype(auto) helper_pack_elem(FirstType&& x0, ArgType&&... x) {
    if constexpr (I == 0) {
        return std::forward<FirstType>(x0);
    } else {
        static_assert(sizeof...(x) > 0, "Index out of bounds");
        return helper_pack_elem<I - 1>(std::forward<ArgType>(x)...);
    }
}


template<size_t... Args>
constexpr size_t validate_size(size_t size){
    assert(size == (Args * ...) && "Invalid initializer list size");
    return size;
}

} // namespace detail

template<size_t N, typename F, typename... Args>
NDSPAN_INLINE constexpr void ForEach(F&& f, Args&&... args){
    detail::for_each_impl<0, N>(f, args...);
}


template<typename F, size_t... I>
NDSPAN_INLINE constexpr decltype(auto) Expand_impl(F&& f, std::index_sequence<I...>){
    return f.template operator()<I...>();
}


template<size_t N, typename F>
NDSPAN_INLINE constexpr decltype(auto) Expand(F&& f){
    return Expand_impl(std::forward<F>(f), std::make_index_sequence<N>{});
}



template<std::size_t I, typename... Args>
NDSPAN_INLINE constexpr decltype(auto) pack_elem(Args&&... args) {
    return detail::helper_pack_elem<I>(std::forward<Args>(args)...);
}

template<typename Iterable>
NDSPAN_INLINE constexpr size_t prod(const Iterable& array){
    if (array.size() == 0){
        return 0;
    }
    size_t res = 1;
    for (size_t i=0; i<array.size(); i++){
        res *= array[i];
    }
    return res;
}

template<std::integral Int>
NDSPAN_INLINE constexpr size_t prod(const Int* array, size_t size){
    if (size == 0){
        return 0;
    }
    size_t res = 1;
    for (size_t i=0; i<size; i++){
        res *= Int(array[i]);
    }
    return res;
}


template<typename T>
NDSPAN_INLINE bool equal_arrays(const T* a, const T* b, size_t size){
    for (size_t i=0; i<size; i++){
        if (a[i]!=b[i]) {return false;}
    }
    return true;
}


template<typename T>
NDSPAN_INLINE T abs(const T& x){
    return x >= 0 ? x : -x;
}

template<typename T>
inline constexpr T max(const T& a, const T& b) {
    return std::max<T>(a, b);
}

template<typename T>
inline constexpr T min(const T& a, const T& b) {
    return std::min<T>(a, b);
}

template<typename T>
inline constexpr T min_of_pack(const auto&... args) {
    return (std::min<T>)({args...});
}


template<typename... Args>
inline constexpr auto max_of_pack(Args... args) {
    return (max)({args...});
}

template<typename Array>
void array_repr(std::ostream& out, const Array& array) {
    if (array.size() == 0){
        return;
    } else {
        out << array[0];
        for (std::size_t i = 1; i < array.size(); ++i) {
            out << ' ' << array[i];
        }
    }
}

template<typename T, size_t size>
NDSPAN_INLINE bool equal_arrays(const T* a, const T* b){
    for (size_t i=0; i<size; i++){
        if (a[i]!=b[i]) {return false;}
    }
    return true;
}

template<typename T>
NDSPAN_INLINE bool isStrictlyAscending(const T* array, size_t size){
    for (size_t i=1; i<size; i++){
        if (array[i] <= array[i-1]){
            return false;
        }
    }
    return true;
}


template<size_t...>
struct tail_product {
    static constexpr size_t value = 1;
};

// Recursive case
template<size_t Head, size_t... Tail>
struct tail_product<Head, Tail...> {
    static constexpr size_t value = Head * tail_product<Tail...>::value;
};


constexpr size_t comb(size_t n, size_t k) {
    assert(n >= k);

    k = ndspan::min(k, n - k);

    size_t res = 1;
    for (size_t i = 1; i <= k; ++i) {
        size_t num = n - i + 1;
        size_t den = i;

        size_t g = std::gcd(num, den);
        num /= g;
        den /= g;

        g = std::gcd(res, den);
        res /= g;
        den /= g;

        res *= num;   // den is now 1
    }
    return res;
}

} // namespace ndspan