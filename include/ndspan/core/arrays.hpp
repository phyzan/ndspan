#pragma once

#include <sstream>
#include <string>
#include "ndview.hpp"


namespace ndspan{

namespace detail{

template<typename Derived, typename T, Layout LY, size_t... DIMS>
class AbstractArray : public AbstractView<Derived, LY, T, DIMS...>{

    using CLS = AbstractArray<Derived, T, LY, DIMS...>;
    using Base = AbstractView<Derived, LY, T, DIMS...>;

protected:

    inline static constexpr size_t N = (sizeof...(DIMS) == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr size_t RANK = sizeof...(DIMS);

    using Base::Base;

    AbstractArray() = default;

    DEFAULT_RULE_OF_FOUR(AbstractArray)

    template<std::integral Int>
    explicit AbstractArray(const Int* shape, size_t ndim) : Base(shape, ndim) {}

    ~AbstractArray() = default;

    void _copy_from(const T* data){
        if (this->size() > 0){
            std::copy(data, data + this->size(), this->data());
        }
    }

};

} // namespace detail


template<typename T, Layout LY, size_t... DIMS>
class DynamicArray : public detail::AbstractArray<DynamicArray<T, LY, DIMS...>, T, LY, DIMS...>{

    using CLS = DynamicArray<T, LY, DIMS...>;
    using Base = detail::AbstractArray<DynamicArray<T, LY, DIMS...>, T, LY, DIMS...>;

public:

    inline static constexpr size_t N = (sizeof...(DIMS) == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr size_t RANK = sizeof...(DIMS);

    // ============= Default constructors =============
    DynamicArray() requires (N > 0) : Base(), data_(std::make_unique<T[]>(N)) {}

    DynamicArray() requires (N == 0) : Base(), data_(nullptr) {}

    // ============= Custom constructors =============
    explicit DynamicArray(const T* data) requires (N > 0) : DynamicArray(){
        this->_copy_from(data);
    }

    template<std::integral... Args>
    explicit DynamicArray(Args... shape) : Base(shape...) {
        this->make_resize(0, this->size());
    }

    template<std::integral Int>
    explicit DynamicArray(const T* data, const Int* shape, size_t ndim) : Base(shape, ndim){
        this->make_resize(0, this->size());
        if (data != nullptr){
            std::copy(data, data + this->size(), this->data());
        }
    }

    template<std::integral Int>
    explicit DynamicArray(T* data, const Int* shape, size_t ndim, bool own_it) : Base(shape, ndim){
        if (own_it){
            //takes ownership of data, which must have been allocated with new T[]
            data_.reset(data);
        } else {
            this->make_resize(0, this->size());
            std::copy(data, data + this->size(), this->data());
        }
    }

    template<std::integral... Args>
    explicit DynamicArray(const T* data, Args... shape) : DynamicArray(shape...){
        this->_copy_from(data);
    }

    DynamicArray(std::initializer_list<T> array) requires (RANK<2 && (N == 0)) : DynamicArray(array.begin(), array.size()) {}

    DynamicArray(std::initializer_list<T> array) requires (N>0) : DynamicArray(array.begin(), (detail::validate_size<DIMS...>(array.size()), DIMS)...) {}

    // ============= Copy semantics =============
    //COPY CONSTRUCTOR
    DynamicArray(const DynamicArray& other) : Base(other) {
        this->make_resize(0, this->size());
        std::copy(other.data(), other.data() + this->size(), this->data());
    }

    //ASSIGNMENT OPERATOR
    DynamicArray& operator=(const DynamicArray& other) {
        if (&other != this){
            this->make_resize(this->size(), other.size());
            Base::operator=(other);
            std::copy(other.data(), other.data() + this->size(), this->data());
        }
        return *this;
    }

    // ============= Move semantics =============
    //MOVE CONSTRUCTOR
    DynamicArray(DynamicArray&& other) noexcept = default;
    //MOVE-ASSIGNMENT OPERATOR
    DynamicArray& operator=(DynamicArray&& other) noexcept = default;

    // ============= Destructor =============
    ~DynamicArray() = default;

    NDSPAN_INLINE const T* data() const{
        return data_.get();
    }

    NDSPAN_INLINE T* data() {
        return data_.get();
    }

    template<std::integral Int>
    void resize(const Int* newsize, size_t ndim){
        size_t current_size = this->size();
        Base::resize(newsize, ndim); //if the new shape is invalid, this throws before data_ is touched below.
        size_t new_size = this->size();
        this->make_resize(current_size, new_size);
    }

    template<std::integral... Size>
    void resize(Size... newsize){
        size_t current_size = this->size();
        Base::resize(newsize...);
        size_t new_size = this->size();
        this->make_resize(current_size, new_size);
    }

    T* release(){
        reset_base_to_zero(std::make_index_sequence<Base::RANK>());
        return data_.release();
    }

private:

    void make_resize(size_t old_size, size_t new_size){
        if (new_size != old_size){
            //release first, so that a failed allocation leaves the array empty
            //instead of holding a buffer that no longer matches the shape.
            data_ = nullptr;
            if (new_size > static_cast<size_t>(PTRDIFF_MAX)){
                throw std::bad_array_new_length();
            } else if (new_size > 0){
                data_ = std::make_unique<T[]>(new_size);
            }
        }
    }

    template<size_t... I>
    NDSPAN_INLINE void reset_base_to_zero(std::index_sequence<I...>){
        if constexpr (Base::N == 0) {
            Base::resize(Base::SHAPE[I]...);
        }
    }

    std::unique_ptr<T[]> data_;

};


template<typename T, Layout LY, size_t... DIMS>
class StackArray : public detail::AbstractArray<StackArray<T, LY, DIMS...>, T, LY, DIMS...>{

    using CLS = StackArray<T, LY, DIMS...>;
    using Base = detail::AbstractArray<StackArray<T, LY, DIMS...>, T, LY, DIMS...>;

public:

    inline static constexpr size_t N = (sizeof...(DIMS) == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr size_t RANK = sizeof...(DIMS);

    static_assert(N>0, "StackArray requires only positive template dimensions");

    using Base::Base;

    StackArray() = default;

    constexpr explicit StackArray(const T* data) : Base(){
        std::copy(data, data + this->size(), this->data());
    }

    template<std::integral... Args>
    constexpr explicit StackArray(const T* data, Args... shape) : StackArray(shape...){
        std::copy(data, data + this->size(), this->data());
    }

    template<std::integral Int>
    constexpr explicit StackArray(const T* data, const Int* shape, size_t ndim) : StackArray(shape, ndim){
        std::copy(data, data + this->size(), this->data());
    }

    constexpr StackArray(std::initializer_list<T> array) : StackArray(array.begin(), (detail::validate_size<DIMS...>(array.size()), DIMS)...) {}

    StackArray(const StackArray&) = default;
    StackArray(StackArray&&) noexcept = default;
    StackArray& operator=(const StackArray& other) = default;
    StackArray& operator=(StackArray&& other) noexcept = default;
    ~StackArray() = default;

    NDSPAN_INLINE constexpr const T* data() const{
        return _data.data();
    }

    NDSPAN_INLINE constexpr T* data() {
        return _data.data();
    }

private:

    std::array<T, N> _data;

};


enum class Allocation : std::uint8_t {Heap, Stack, Auto};

namespace detail{

template <Allocation Alloc, Layout LY, typename T, size_t... DIMS>
struct ArrayAllocMap;

template <Layout LY, typename T, size_t... DIMS>
struct ArrayAllocMap<Allocation::Heap, LY, T, DIMS...> { using type = DynamicArray<T, LY, DIMS...>; };

template <Layout LY, typename T, size_t... DIMS>
struct ArrayAllocMap<Allocation::Stack, LY, T, DIMS...> { using type = StackArray<T, LY, DIMS...>; };

template <Layout LY, typename T>
struct ArrayAllocMap<Allocation::Auto, LY, T> { using type = DynamicArray<T, LY>; };

template <Layout LY, typename T, size_t... DIMS>
struct ArrayAllocMap<Allocation::Auto, LY, T, DIMS...> {
    using type = std::conditional_t<((DIMS * ...) == 0), DynamicArray<T, LY, DIMS...>, StackArray<T, LY, DIMS...>>; 
};

} // namespace detail



template <typename T, Allocation Alloc = Allocation::Auto, Layout LY = Layout::C, size_t... DIMS>
class Array : public detail::ArrayAllocMap<Alloc, LY, T, DIMS...>::type{

    using Base = detail::ArrayAllocMap<Alloc, LY, T, DIMS...>::type;

public:

    using value_type = Base::value_type;
    using iterator = Base::iterator;
    using const_iterator = Base::const_iterator;

    using Base::Base;

    //=============================== ACCESSORS ===================================

    // NdSpan interface
    NDSPAN_INLINE constexpr size_t size() const{
        return Base::size();
    }

    NDSPAN_INLINE constexpr size_t ndim() const {
        return Base::ndim();
    }

    NDSPAN_INLINE const size_t* shape() const {
        return Base::shape();
    }

    template<std::integral IDX_T>
    NDSPAN_INLINE constexpr size_t shape(IDX_T i) const {
        return Base::shape(i);
    }

    template<std::integral... Idx>
    NDSPAN_INLINE constexpr size_t offset(Idx... idx) const noexcept {
        return Base::offset(idx...);
    }

    template<std::integral Int>
    NDSPAN_INLINE constexpr size_t getOffset(const Int* idx_ptr) const noexcept{
        return Base::getOffset(idx_ptr);
    }

    NDSPAN_INLINE const T& getElem(const size_t* idx_ptr) const{
        return Base::getElem(idx_ptr);
    }

    template<size_t Nd>
    NDSPAN_INLINE constexpr size_t offset(const std::array<size_t, Nd>& idx) const noexcept {
        return Base::offset(idx);
    }

    template<std::integral... Idx>
    NDSPAN_INLINE void unpack_idx(size_t offset, Idx&... idx) const noexcept{
        Base::unpack_idx(offset, idx...);
    }

    template<std::integral INT, size_t Nd>
    NDSPAN_INLINE void unpack_idx(size_t offset, std::array<INT, Nd>& idx) const noexcept{
        Base::unpack_idx(offset, idx);
    }

    // View interface
    const_iterator begin() const { return Base::begin(); }
    const_iterator end() const { return Base::end(); }

    NDSPAN_INLINE const T* data() const{
        return Base::data();
    }

    template<std::integral... Int>
    NDSPAN_INLINE const T* ptr(Int... idx) const{
        return Base::ptr(idx...);
    }

    NDSPAN_INLINE View<const T, LY, DIMS...> view() const{
        if constexpr (Base::N > 0){
            return View<const T, LY, DIMS...>(this->data());
        } else if (Base::RANK > 0) {
            const size_t* s = this->shape();
            return NDSPAN_EXPAND(Base::RANK, I,
                return View<const T, LY, DIMS...>(this->data(), s[I]...);
            );
        } else {
            return View<const T, LY, DIMS...>(this->data(), this->shape(), this->ndim());
        }
    }

    NDSPAN_INLINE const T& back() const{
        return Base::back();
    }

    template<std::integral... Idx>
    NDSPAN_INLINE constexpr const T& operator()(Idx... idx) const {
        return Base::operator()(idx...);
    }

    template<typename... Idx>
    NDSPAN_INLINE auto operator()(Idx... i) const{
        return Base::operator()(i...);
    }

    template<std::integral IDX_T>
    NDSPAN_INLINE constexpr const T& operator[](IDX_T i) const{
        return Base::operator[](i);
    }


    //=============================== MODIFIERS ===================================

    NDSPAN_INLINE View<T, LY, DIMS...> view(){
        if constexpr (Base::N > 0){
            return View<T, LY, DIMS...>(this->data());
        } else if (Base::RANK > 0) {
            const size_t* s = this->shape();
            return NDSPAN_EXPAND(Base::RANK, I,
                return View<T, LY, DIMS...>(this->data(), s[I]...);
            );
        } else {
            return View<T, LY, DIMS...>(this->data(), this->shape(), this->ndim());
        }
    }

    // NdSpan interface
    template<std::integral... Args>
    NDSPAN_INLINE void reshape(Args... shape){
        Base::reshape(shape...);
    }

    template<std::integral... Args>
    NDSPAN_INLINE void constexpr resize(Args... shape){
        Base::resize(shape...);
    }

    template<std::integral Int>
    NDSPAN_INLINE void reshape(const Int* shape, size_t ndim){
        Base::reshape(shape, ndim);
    }

    template<std::integral Int>
    NDSPAN_INLINE void resize(const Int* shape, size_t ndim){
        Base::resize(shape, ndim);
    }


    // View interface

    iterator begin() { return Base::data(); }
    iterator end()   { return Base::end(); }

    NDSPAN_INLINE T* data(){
        return Base::data();
    }

    template<std::integral... Idx>
    NDSPAN_INLINE constexpr T& operator()(Idx... idx) {
        return Base::operator()(idx...);
    }

    template<typename... Idx>
    NDSPAN_INLINE auto operator()(Idx... i){
        return Base::operator()(i...);
    }

    template<std::integral IDX_T>
    NDSPAN_INLINE constexpr T& operator[](IDX_T i){
        return Base::operator[](i);
    }

    template<std::integral Int>
    NDSPAN_INLINE constexpr T& getElem(const Int* idx_ptr) noexcept{
        return Base::getElem(idx_ptr);
    }

    template<std::integral... Int>
    NDSPAN_INLINE constexpr T* ptr(Int... idx){
        return Base::ptr(idx...);
    }

    NDSPAN_INLINE Array& fill(const T& value){
        Base::fill(value);
        return *this;
    }

};


namespace detail{

template <typename T, size_t Rank, Layout LY = Layout::C>
struct HelperNdArray
{
    template <Allocation Alloc, std::size_t... Is>
    static Array<T, Alloc, LY, (static_cast<void>(Is), 0)...> make(std::index_sequence<Is...>);

    template <std::size_t... Is>
    static View<T, LY, (static_cast<void>(Is), 0)...> make_view(std::index_sequence<Is...>);

    template<Allocation Alloc>
    using type = decltype(make<Alloc>(std::make_index_sequence<Rank>{}));

    using ViewType = decltype(make_view(std::make_index_sequence<Rank>{}));
};

// Writes the block spanned by the axes [axis, ndim), consuming its formatted elements from "element".
inline void print_block(std::ostream& out, const std::string*& element, const size_t* widths, const size_t* shape, size_t ndim, size_t axis){
    const bool is_row = (axis + 1 == ndim);

    out << '{';
    for (size_t i = 0; i < shape[axis]; ++i){
        if (is_row){
            if (i > 0){
                out << ' ';
            }
            out << std::setw(static_cast<int>(widths[i])) << *element++;
        } else {
            if (i > 0){
                // Rows go on consecutive lines, larger blocks are separated by a blank line.
                out << (ndim - axis > 2 ? "\n\n" : "\n") << std::string(axis + 1, ' ');
            }
            print_block(out, element, widths, shape, ndim, axis + 1);
        }
    }
    out << '}';
}

} // namespace detail

// Writes an array of any rank as nested braces, in logical index order regardless of its layout.
// Each column of the last axis is aligned across the whole array, and elements are formatted
// using the settings of "out" (e.g. its precision). No trailing newline is written.
template<typename ArrayType>
std::ostream& print_array(std::ostream& out, const ArrayType& array){
    if (array.size() == 0){
        return out << "{}";
    }

    const size_t ndim = array.ndim();
    const size_t* shape = array.shape();

    std::vector<std::string> elements(array.size());
    std::vector<size_t> column_widths(shape[ndim-1], 0);
    std::vector<size_t> idx(ndim, 0);

    std::ostringstream element_stream;
    element_stream.copyfmt(out);
    element_stream.width(0);

    // Format each element once and determine column widths.
    for (std::string& element : elements){
        element_stream.str("");
        element_stream << array.data()[array.getOffset(idx.data())];
        element = element_stream.str();

        size_t& width = column_widths[idx[ndim-1]];
        width = ndspan::max(width, element.size());

        // Advance to the next element, with the last axis changing fastest.
        for (size_t axis = ndim; axis > 0; --axis){
            if (++idx[axis-1] < shape[axis-1]){
                break;
            }
            idx[axis-1] = 0;
        }
    }

    const std::string* first = elements.data();
    detail::print_block(out, first, column_widths.data(), shape, ndim, 0);
    return out;
}


template<typename T, size_t Rank, Layout LY, Allocation Alloc = Allocation::Auto>
using ArrayND = detail::HelperNdArray<T, Rank, LY>::template type<Alloc>;

template <typename T, size_t SIZE, Allocation Alloc = Allocation::Auto>
using Array1D = Array<T, Alloc, Layout::C, SIZE>; // For 1D, layout is unimportant

template <typename T, size_t M, size_t N, Layout LY, Allocation Alloc = Allocation::Auto>
using Array2D = Array<T, Alloc, LY, M, N>;

template <typename T, size_t M, size_t N, size_t K, Layout LY, Allocation Alloc = Allocation::Auto>
using Array3D = Array<T, Alloc, LY, M, N, K>;

template <typename T, size_t M, size_t N, size_t K, size_t L, Layout LY, Allocation Alloc = Allocation::Auto>
using Array4D = Array<T, Alloc, LY, M, N, K, L>;

template <typename T, size_t M, size_t N, size_t K, size_t L, size_t P, Layout LY, Allocation Alloc = Allocation::Auto>
using Array5D = Array<T, Alloc, LY, M, N, K, L, P>;

// ======== Row-major aliases =========
template<typename T, size_t Rank, Allocation Alloc = Allocation::Auto>
using CArrayND = ArrayND<T, Rank, Layout::C, Alloc>;

template<typename T, size_t M, size_t N, Allocation Alloc = Allocation::Auto>
using CArray2D = Array2D<T, M, N, Layout::C, Alloc>;

template<typename T, size_t M, size_t N, size_t K, Allocation Alloc = Allocation::Auto>
using CArray3D = Array3D<T, M, N, K, Layout::C, Alloc>;

template<typename T, size_t M, size_t N, size_t K, size_t L, Allocation Alloc = Allocation::Auto>
using CArray4D = Array4D<T, M, N, K, L, Layout::C, Alloc>;

template<typename T, size_t M, size_t N, size_t K, size_t L, size_t P, Allocation Alloc = Allocation::Auto>
using CArray5D = Array5D<T, M, N, K, L, P, Layout::C, Alloc>;

// ======== Column-major aliases =========
template<typename T, size_t Rank, Allocation Alloc = Allocation::Auto>
using FArrayND = ArrayND<T, Rank, Layout::F, Alloc>;

template<typename T, size_t M, size_t N, Allocation Alloc = Allocation::Auto>
using FArray2D = Array2D<T, M, N, Layout::F, Alloc>;

template<typename T, size_t M, size_t N, size_t K, Allocation Alloc = Allocation::Auto>
using FArray3D = Array3D<T, M, N, K, Layout::F, Alloc>;

template<typename T, size_t M, size_t N, size_t K, size_t L, Allocation Alloc = Allocation::Auto>
using FArray4D = Array4D<T, M, N, K, L, Layout::F, Alloc>;

template<typename T, size_t M, size_t N, size_t K, size_t L, size_t P, Allocation Alloc = Allocation::Auto>
using FArray5D = Array5D<T, M, N, K, L, P, Layout::F, Alloc>;


} // namespace ndspan