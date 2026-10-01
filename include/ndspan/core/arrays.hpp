#pragma once

#include "ndview.hpp"


namespace ndspan{

namespace detail{

template<typename Derived, typename T, Layout L, size_t... DIMS>
class AbstractArray : public AbstractMutView<Derived, L, T, DIMS...>{

    using CLS = AbstractArray<Derived, T, L, DIMS...>;
    using Base = AbstractMutView<Derived, L, T, DIMS...>;

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


template<typename T, Layout L, size_t... DIMS>
class DynamicArray : public detail::AbstractArray<DynamicArray<T, L, DIMS...>, T, L, DIMS...>{

    using CLS = DynamicArray<T, L, DIMS...>;
    using Base = detail::AbstractArray<DynamicArray<T, L, DIMS...>, T, L, DIMS...>;

public:

    inline static constexpr size_t N = (sizeof...(DIMS) == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr size_t RANK = sizeof...(DIMS);

    DynamicArray() : Base() {
        _data = (N > 0 ? new T[N] : nullptr);
    }

    explicit DynamicArray(const T* data) requires (N>0) : DynamicArray(){
        this->_copy_from(data);
    }

    template<std::integral... Args>
    explicit DynamicArray(Args... shape) : Base(shape...) {
        if (this->size() > 0 && this->size() <= static_cast<size_t>(PTRDIFF_MAX)){
            _data = new T[this->size()];
        }
    }

    template<std::integral Int>
    explicit DynamicArray(const T* data, const Int* shape, size_t ndim) : Base(shape, ndim){
        if (this->size() > 0 && this->size() <= static_cast<size_t>(PTRDIFF_MAX)){
            _data = new T[this->size()];
        }
        if (data != nullptr){
            std::copy(data, data + this->size(), this->data());
        }
    }

    template<std::integral Int>
    explicit DynamicArray(T* data, const Int* shape, size_t ndim, bool own_it) : Base(shape, ndim){
        if (own_it){
            _data = data;
        }else if (this->size() > 0 && this->size() <= static_cast<size_t>(PTRDIFF_MAX)){
            _data = new T[this->size()];
            std::copy(data, data + this->size(), this->data());
        }
    }

    template<std::integral... Args>
    explicit DynamicArray(const T* data, Args... shape) : DynamicArray(shape...){
        this->_copy_from(data);
    }

    DynamicArray(std::initializer_list<T> array) requires (RANK<2 && (N == 0)) : DynamicArray(array.begin(), array.size()) {}

    DynamicArray(std::initializer_list<T> array) requires (N>0) : DynamicArray(array.begin(), (detail::validate_size<DIMS...>(array.size()), DIMS)...) {}

    //COPY CONSTRUCTOR
    DynamicArray(const DynamicArray& other) : Base(static_cast<const Base&>(other)), _data((other.size() > 0 && other.size() <= static_cast<size_t>(PTRDIFF_MAX)) ? new T[other.size()] : nullptr) {
        std::copy(other.data(), other.data() + this->size(), this->data());
    }

    //MOVE CONSTRUCTOR
    DynamicArray(DynamicArray&& other) noexcept : Base(static_cast<Base&&>(std::move(other))), _data(other.release()) {}

    //ASSIGNMENT OPERATOR
    DynamicArray& operator=(const DynamicArray& other) {
        if (&other != this){
            if (this->size() != other.size()){
                delete[] _data;
                _data = (other.size() > 0 && other.size() <= static_cast<size_t>(PTRDIFF_MAX)) ? new T[other.size()] : nullptr;
            }
            Base::operator=(other);
            std::copy(other.data(), other.data() + this->size(), this->data());
        }
        return *this;
    }

    //MOVE-ASSIGNMENT OPERATOR
    DynamicArray& operator=(DynamicArray&& other) noexcept {
        if (&other != this){
            Base::operator=(std::move(other));
            delete[] _data;
            _data = other.release();
        }
        return *this;
    }

    ~DynamicArray() {
        delete[] _data;
        _data = nullptr;
    }

    NDSPAN_INLINE const T* data() const{
        return _data;
    }

    NDSPAN_INLINE T* data() {
        return _data;
    }

    template<std::integral Int>
    void resize(const Int* newsize, size_t ndim){
        size_t current_size = this->size();
        Base::resize(newsize, ndim);//if the new size is invalid, this will throw an error before the execution moves to resizing the _data below.
        size_t total_size = prod(newsize, ndim);
        if (total_size != current_size){
            delete[] _data;
            if (total_size == 0 || total_size > static_cast<size_t>(PTRDIFF_MAX)){
                _data = nullptr;
            }
            else{
                _data = new T[this->size()];
            }
        }
    }

    template<std::integral... Size>
    void resize(Size... newsize){
        size_t current_size = this->size();
        Base::resize(newsize...);
        size_t total_size = (newsize * ... * 1);
        if (total_size != current_size){
            delete[] _data;
            if (total_size == 0 || total_size > static_cast<size_t>(PTRDIFF_MAX)){
                _data = nullptr;
            }
            else{
                _data = new T[this->size()];
            }
        }
    }

    T* release(){
        T* res = _data;
        _data = nullptr;
        _reset_base_to_zero(std::make_index_sequence<Base::RANK>());
        return res;
    }

private:

    template<size_t... I>
    NDSPAN_INLINE void _reset_base_to_zero(std::index_sequence<I...>){
        if constexpr (Base::N == 0) {
            Base::resize(Base::SHAPE[I]...);
        }
    }

    T* _data = nullptr;

};


template<typename T, Layout L, size_t... DIMS>
class StackArray : public detail::AbstractArray<StackArray<T, L, DIMS...>, T, L, DIMS...>{

    using CLS = StackArray<T, L, DIMS...>;
    using Base = detail::AbstractArray<StackArray<T, L, DIMS...>, T, L, DIMS...>;

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

template <Allocation Alloc, Layout L, typename T, size_t... DIMS>
struct ArrayAllocMap;

template <Layout L, typename T, size_t... DIMS>
struct ArrayAllocMap<Allocation::Heap, L, T, DIMS...> { using type = DynamicArray<T, L, DIMS...>; };

template <Layout L, typename T, size_t... DIMS>
struct ArrayAllocMap<Allocation::Stack, L, T, DIMS...> { using type = StackArray<T, L, DIMS...>; };

template <Layout L, typename T>
struct ArrayAllocMap<Allocation::Auto, L, T> { using type = DynamicArray<T, L>; };

template <Layout L, typename T, size_t... DIMS>
struct ArrayAllocMap<Allocation::Auto, L, T, DIMS...> {
    using type = std::conditional_t<(((DIMS * ...) == 0)), DynamicArray<T, L, DIMS...>, StackArray<T, L, DIMS...>>; 
};

} // namespace detail



template <typename T, Allocation Alloc = Allocation::Auto, Layout L = Layout::C, size_t... DIMS>
class Array : public detail::ArrayAllocMap<Alloc, L, T, DIMS...>::type{

    using Base = detail::ArrayAllocMap<Alloc, L, T, DIMS...>::type;

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

    NDSPAN_INLINE View<T, L, DIMS...> view() const{
        if constexpr (Base::N > 0){
            return View<T, L, DIMS...>(this->data());
        }else if (Base::RANK > 0) {
            const size_t* s = this->shape();
            return NDSPAN_EXPAND(Base::RANK, I,
                View<T, L, DIMS...>(this->data(), s[I]...);
            );
        }else {
            return View<T, L, DIMS...>(this->data(), this->shape(), this->ndim());
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


    // MutView interface

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

template <typename T, size_t Rank, Layout L = Layout::C>
struct HelperNdArray
{
    template <Allocation Alloc, std::size_t... Is>
    static Array<T, Alloc, L, (static_cast<void>(Is), 0)...> make(std::index_sequence<Is...>);

    template <std::size_t... Is>
    static View<T, L, (static_cast<void>(Is), 0)...> make_view(std::index_sequence<Is...>);

    template<Allocation Alloc>
    using type = decltype(make<Alloc>(std::make_index_sequence<Rank>{}));

    using ViewType = decltype(make_view(std::make_index_sequence<Rank>{}));
};


} // namespace detail



template <typename T, size_t Rank, Allocation Alloc = Allocation::Auto, Layout L = Layout::C>
using NdArray = detail::HelperNdArray<T, Rank, L>::template type<Alloc>;

template <typename T, size_t Rank, Layout L = Layout::C>
using NdView = detail::HelperNdArray<T, Rank, L>::ViewType;


template <typename T, size_t SIZE=0, Allocation Alloc = Allocation::Auto>
using Array1D = Array<T, Alloc, Layout::C, SIZE>;

template<typename T, size_t Nr = 0, size_t Nc = 0, Allocation Alloc = Allocation::Auto, Layout L = Layout::C>
class Array2D : public Array<T, Alloc, L, Nr, Nc>{

    using Base = Array<T, Alloc, L, Nr, Nc>;

public:

    using Base::Base;

    DEFAULT_RULE_OF_FOUR(Array2D)

    NDSPAN_INLINE size_t Nrows() const {return this->shape(0);}

    NDSPAN_INLINE size_t Ncols() const {return this->shape(1);}

    void repr(std::ostream& out, int digits = 8) const {
        if (this->size() == 0) {
            return;
        }

        const std::size_t rows = this->Nrows();
        const std::size_t cols = this->Ncols();

        std::vector<std::string> elements(rows * cols);
        std::vector<std::size_t> column_widths(cols, 0);

        // Format each element once and determine column widths.
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                std::ostringstream element_stream;
                element_stream << std::setprecision(digits) << (*this)(i, j);
                auto& element = elements[i * cols + j];
                element = element_stream.str();
                column_widths[j] = ndspan::max(column_widths[j], element.size());
            }
        }

        // Write the aligned representation.
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                out << std::setw(static_cast<int>(column_widths[j]))
                    << elements[i * cols + j];

                if (j + 1 < cols) {
                    out << ' ';
                }
            }

            if (i + 1 < rows) {
                out << '\n';
            }
        }
    }
    
};


template <typename T, size_t M=0, size_t N=0, size_t K=0, Allocation Alloc = Allocation::Auto>
using Array3D = Array<T, Alloc, Layout::C, M, N, K>;


} // namespace ndspan