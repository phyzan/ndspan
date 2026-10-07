#pragma once


#include <memory>
#include "ndtools.hpp"

namespace ndspan{


namespace detail{

template<typename Derived, size_t... DIMS>
class AbstractNdSpan{

protected:
    inline static constexpr size_t RANK = sizeof...(DIMS);
    inline static constexpr size_t N = (RANK == 0 ? 0 : (DIMS * ... * 1));
    inline static constexpr std::array<size_t, RANK> SHAPE = {DIMS...};

    using CLS = AbstractNdSpan<Derived, DIMS...>;

public:
    //ACCESSORS

    NDSPAN_INLINE constexpr size_t size() const{
        //override
        return THIS->size();
    }

    NDSPAN_INLINE constexpr size_t ndim() const {
        //override
        return THIS->ndim();
    }

    NDSPAN_INLINE constexpr const size_t* shape() const {
        //override
        return THIS->shape();
    }

    template<std::integral... Idx>
    NDSPAN_INLINE constexpr size_t offset_impl(Idx... idx) const noexcept{
        //override
        return THIS->offset_impl(idx...);
    }

    template<std::integral Idx>
    NDSPAN_INLINE constexpr size_t getOffset_impl(const Idx* idx_ptr) const noexcept{
        //override
        return THIS->getOffset_impl(idx_ptr);
    }

    template<std::integral... Idx>
    NDSPAN_INLINE void unpack_idx_impl(size_t offset, Idx&... idx) const noexcept{
        //override
        return THIS->unpack_idx_impl(offset, idx...);
    }

    template<std::integral IDX_T>
    NDSPAN_INLINE constexpr size_t shape(IDX_T i) const {
        return shape()[i];
    }

    template<std::integral... Idx>
    NDSPAN_INLINE constexpr size_t offset(Idx... idx) const noexcept {
        //dimension and range check in debug mode. They will not be compiled when -DNDEBUG is enabled
        _dim_check(this->ndim(), idx...);
        NDSPAN_EXPAND(sizeof...(idx), I,
            assert(((idx >= 0 && size_t(idx) < this->shape(I)) && ...) && "Out of bounds");
        );
        return offset_impl(idx...);
    }

    template<std::integral Int>
    NDSPAN_INLINE constexpr size_t getOffset(const Int* idx_ptr) const noexcept{
        assert(idx_ptr != nullptr && "Null pointer in getOffset");
        return getOffset_impl(idx_ptr);
    }

    template<size_t Nd>
    NDSPAN_INLINE constexpr size_t offset(const std::array<size_t, Nd>& idx) const noexcept {
        return NDSPAN_EXPAND(Nd, I,
            return this->offset(idx[I]...);
        );
    }

    template<std::integral... Idx>
    NDSPAN_INLINE void unpack_idx(size_t offset, Idx&... idx) const noexcept{
        //dimension and offset check
        _dim_check(this->ndim(), idx...);
        _offset_check(offset);
        this->unpack_idx_impl(offset, idx...);
    }

    template<std::integral INT, size_t Nd>
    NDSPAN_INLINE void unpack_idx(size_t offset, std::array<INT, Nd>& idx) const noexcept{
        NDSPAN_EXPAND(Nd, I,
            this->unpack_idx(offset, idx[I]...);
        );
    }

    //MODIFIERS

    template<std::integral... Args>
    NDSPAN_INLINE void reshape(Args... shape){
        //override
        THIS->reshape(shape...);
    }

    template<std::integral... Args>
    NDSPAN_INLINE void constexpr resize(Args... shape){
        //override
        THIS->resize(shape...);
    }

    template<std::integral Int>
    NDSPAN_INLINE void reshape(const Int* shape, size_t ndim){
        //override
        THIS->reshape(shape, ndim);
    }

    template<std::integral Int>
    NDSPAN_INLINE void resize(const Int* shape, size_t ndim){
        //override
        THIS->resize(shape, ndim);
    }

protected:

    AbstractNdSpan() = default;

    DEFAULT_RULE_OF_FOUR(AbstractNdSpan)

    template<std::integral Int>
    NDSPAN_INLINE bool _conserves_shape(const Int* shape, size_t ndim) const{
        if (this->ndim() == ndim && this->ndim() > 0){
            int I = 0;
            return (( shape[I++]==DIMS) && ...);
        }else {
            return false;
        }
    }

    template<std::integral... IntType>
    NDSPAN_INLINE static void _shape_check(IntType... dim) {
        if constexpr ((std::is_signed_v<IntType> || ...)) {
            NDSPAN_EXPAND(sizeof...(dim), I,
                assert(((dim >= 0 ) && ...) && "Invalid dims");
            );
        }
    }

    template<std::integral Int>
    NDSPAN_INLINE static bool _is_valid_shape(const Int* dims, size_t ndim) {
        for (size_t i=0; i<ndim; i++){
            if (dims[i] < 0){
                return false;
            }
        }
        return true;
    }

    template<std::integral... Idx>
    NDSPAN_INLINE constexpr static void _dim_check(size_t ndim, Idx... idx) {
        //dimension check
        if constexpr (RANK > 0){
            static_assert(sizeof...(idx) == RANK, "Incorrect number of indices");
        }
        else if constexpr (RANK == 0){
            assert(sizeof...(idx) == ndim && "Incorrect number of indices");
        }
    }

    template<std::integral INT>
    NDSPAN_INLINE void _offset_check(INT offset) const{
        assert((offset < this->size() && offset >= 0) && "offset is off bounds");
    }

    template<std::integral INT>
    NDSPAN_INLINE void _range_check(INT i) const{
        assert((i < this->size() && i >= 0) && "Out of range");
    }

};

} // namespace detail


template<typename Derived, size_t... DIMS>
class StaticNdSpan : public detail::AbstractNdSpan<Derived, DIMS...>{

    using Base = detail::AbstractNdSpan<Derived, DIMS...>;

protected:
    StaticNdSpan() = default;

    DEFAULT_RULE_OF_FOUR(StaticNdSpan)

    template<std::integral... Args>
    constexpr explicit StaticNdSpan(Args... args){
        this->reshape(args...);
    }

    template<std::integral Int>
    constexpr explicit StaticNdSpan(const Int* shape, size_t ndim){
        this->reshape(shape, ndim);
    }

public:

    inline constexpr size_t size() const {
        return Base::N;
    }

    inline constexpr size_t ndim() const {
        return Base::RANK;
    }

    inline constexpr const size_t* shape() const {
        return Base::SHAPE.data();
    }

    using Base::shape;

    template<std::integral... Args>
    NDSPAN_INLINE constexpr void reshape(Args... shape){
        assert(((shape == DIMS) && ...) && "Runtime dims do not match template dims in reshape");
    }

    template<std::integral... Args>
    NDSPAN_INLINE constexpr void resize(Args... shape){
        //can only resize with the exact same shape
        reshape(shape...);
    }

    template<std::integral Int>
    NDSPAN_INLINE constexpr void reshape(const Int* shape, size_t ndim){
        assert(Base::_is_valid_shape(shape, ndim) && "Invalid dims");
        assert((ndim == this->ndim()) && "Invalid shape in StaticNdSpan::reshape");
        assert(this->_conserves_shape(shape, ndim) && "Runtime dims do not match template dims in reshape");
    }

    template<std::integral Int>
    NDSPAN_INLINE constexpr void resize(const Int* shape, size_t ndim){
        reshape(shape, ndim);
    }

};

template<typename Derived, size_t... DIMS>
class SemiStaticNdSpan : public detail::AbstractNdSpan<Derived, DIMS...>{

    using Base = detail::AbstractNdSpan<Derived, DIMS...>;
    using CLS = SemiStaticNdSpan<Derived, DIMS...>;

protected:

    inline static constexpr size_t RANK = sizeof...(DIMS);
    inline static constexpr size_t N = (DIMS * ...);

    static_assert(N==0 && RANK>0, "Use SemiStaticNdSpan for a static number of dims of dynamic length");

    SemiStaticNdSpan() = default;

    template<std::integral... Args>
    explicit constexpr SemiStaticNdSpan(Args... shape) {
        this->resize(shape...);
    }

    template<std::integral Int>
    explicit SemiStaticNdSpan(const Int* shape, size_t ndim){
        this->resize(shape, ndim);
    }

    DEFAULT_RULE_OF_FOUR(SemiStaticNdSpan)

public:

    template<std::integral... Args>
    void constexpr resize(Args... shape){
        //RANK > 0, but some template dims are zero
        static_assert((sizeof...(shape) == RANK), "Constructor must be called with as many dims as the number of template dims");

        _data[0] = size_t((shape*...));
        NDSPAN_EXPAND(RANK, I,
            assert(((shape >= 0 && (Base::RANK==0 || ((Base::SHAPE[I] > 0 ? shape == Base::SHAPE[I] : true)))) && ...) && "Runtime dims do not match template dims");
            ((_data[I+1] = size_t(shape)), ...);
        );
    }

    template<std::integral Int>
    void resize(const Int* shape, size_t ndim){
        assert(Base::_is_valid_shape(shape, ndim) && "Invalid dims");
        //TODO make sure non of these are zero at runtime, and they are equal to the template parameter
        assert(ndim == Base::RANK && "SemiStaticNdSpan::resize invalid shape");

        _data[0] = prod(shape, ndim);
        std::copy(shape, shape + ndim, _data+1);
    }

    template<std::integral... Args>
    void constexpr reshape(Args... shape){
        assert((size_t((shape*...)) == this->size()) && "Invalid new shape. The total size of the array is not conserved");
        //THIS-> so that the derived layout rebuilds its strides for the new shape
        THIS->resize(shape...);
    }

    template<std::integral Int>
    void reshape(const Int* shape, size_t ndim){
        assert((prod(shape, ndim) == this->size()) && "Invalid new shape. The total size of the array is not conserved");
        THIS->resize(shape, ndim);
    }

    NDSPAN_INLINE size_t size() const{
        return _data[0];
    }

    NDSPAN_INLINE constexpr size_t ndim() const{
        return RANK;
    }

    NDSPAN_INLINE const size_t* shape() const {
        return _data+1;
    }

    using Base::shape;

private:

    size_t _data[RANK+1] = {N, DIMS...};


};

class SemiStaticSpan1D : public detail::AbstractNdSpan<SemiStaticSpan1D, 0>{

    using Base = detail::AbstractNdSpan<SemiStaticSpan1D, 0>;
    using CLS = SemiStaticSpan1D;

public:

    inline static constexpr size_t RANK = 1;
    inline static constexpr size_t N = 0;

    SemiStaticSpan1D() = default;

    template<std::integral Int>
    explicit constexpr SemiStaticSpan1D(Int size) {
        this->resize(size);
    }

    template<std::integral Int>
    explicit SemiStaticSpan1D(const Int* shape, size_t ndim){
        this->resize(shape, ndim);
    }

    DEFAULT_RULE_OF_FOUR(SemiStaticSpan1D)

    template<std::integral Int>
    inline void constexpr resize(Int newsize){
        _size = newsize;
    }

    template<std::integral Int>
    void resize(const Int* shape, size_t ndim){
        //TODO make sure they are equal to the non zero template dims
        assert(Base::_is_valid_shape(shape, ndim) && "Invalid dims");      
        assert(ndim == 1 && "SemiStaticSpan1D::resize invalid shape");
        _size = size_t(shape[0]);
    }

    template<std::integral... Args>
    void constexpr reshape(Args... shape){
        assert((size_t((shape*...)) == this->size()) && "Invalid new shape. The total size of the array is not conserved");
        this->resize(shape...);
    }

    template<std::integral Int>
    void reshape(const Int* shape, size_t ndim){
        assert((prod(shape, ndim) == this->size()) && "Invalid new shape. The total size of the array is not conserved");
        this->resize(shape, ndim);
    }

    NDSPAN_INLINE size_t size() const{
        return _size;
    }

    NDSPAN_INLINE constexpr size_t ndim() const{
        return RANK;
    }

    NDSPAN_INLINE const size_t* shape() const {
        return &_size;
    }


    template<std::integral Int>
    NDSPAN_INLINE constexpr size_t offset_impl(Int idx) const noexcept{
        return idx;
    }

    template<std::integral Int>
    NDSPAN_INLINE constexpr size_t getOffset_impl(const Int* idx_ptr) const noexcept{
        return idx_ptr[0];
    }

    template<std::integral Int>
    NDSPAN_INLINE void unpack_idx_impl(size_t offset, Int& idx) const noexcept{
        idx=offset;
    }

    using Base::shape;

private:

    size_t _size = 0;
};



template<typename Derived>
class DynamicNdSpan : public detail::AbstractNdSpan<Derived>{

    using Base = detail::AbstractNdSpan<Derived>;
    using CLS = DynamicNdSpan<Derived>;

protected:

    static constexpr size_t RANK = 0;
    static constexpr size_t N = 0;
    

    static_assert(N==0, "For fixed compile time dims, use StaticNdSpan");

    DynamicNdSpan() = default;

    template<std::integral... Args>
    explicit constexpr DynamicNdSpan(Args... shape){
        Base::_shape_check(shape...);
        this->resize(shape...);
    }

    template<std::integral Int>
    explicit DynamicNdSpan(const Int* shape, size_t ndim) {
        assert(Base::_is_valid_shape(shape, ndim) && "Invalid dims");
        this->resize(shape, ndim);
    }

    //COPY CONSTRUCTOR
    DynamicNdSpan(const DynamicNdSpan& other) : Base(other){
        if (other.ndim() > 0){
            this->allocate(other.shape(), other.ndim());
        }
    }

    //MOVE CONSTRUCTOR
    DynamicNdSpan(DynamicNdSpan&& other) noexcept : Base(std::move(other)), ndim_(other.ndim_), size_(other.size_), shape_(std::move(other.shape_)) {
        other.ndim_ = 0;
        other.size_ = 0;
    }

    //ASSIGNMENT OPERATOR
    DynamicNdSpan& operator=(const DynamicNdSpan& other) {
        if (&other != this){
            Base::operator=(other);
            this->resize(other.shape(), other.ndim());
        }
        return *this;
    }

    //MOVE-ASSIGNMENT OPERATOR
    DynamicNdSpan& operator=(DynamicNdSpan&& other) noexcept {
        if (&other != this){
            Base::operator=(std::move(other));
            ndim_ = other.ndim_;
            size_ = other.size_;
            shape_ = std::move(other.shape_);
            other.ndim_ = 0;
            other.size_ = 0;
        }
        return *this;
    }

public:

    template<std::integral... Args>
    void constexpr resize(Args... shape){
        if constexpr (sizeof...(shape) == 0){
            this->deallocate();
        } else {
            Base::_shape_check(shape...);
            const std::array<size_t, sizeof...(shape)> new_shape = {static_cast<size_t>(shape)...};
            this->resize(new_shape.data(), new_shape.size());
        }
    }

    template<std::integral Int>
    constexpr void resize(const Int* shape, size_t ndim){
        assert((ndim == 0 || shape != nullptr) && "Null shape pointer");
        assert(Base::_is_valid_shape(shape, ndim) && "Negative dims not allowed");
        if (ndim == 0){
            this->deallocate();
        } else {
            this->allocate(shape, ndim);
        }
    }

    template<std::integral... Args>
    void constexpr reshape(Args... shape){
        assert((size_t((shape*...)) == this->size()) && "Invalid new shape. The total size of the array is not conserved");
        //THIS-> so that the derived layout rebuilds its strides for the new shape
        THIS->resize(shape...);
    }

    template<std::integral Int>
    constexpr void reshape(const Int* shape, size_t ndim){
        if (prod(shape, ndim) != this->size()){
            throw std::runtime_error("Invalid new shape. The total size of the array is not conserved");
        }
        THIS->resize(shape, ndim);
    }

    NDSPAN_INLINE size_t ndim() const{
        return ndim_;
    }
    
    NDSPAN_INLINE size_t size() const{
        return size_;
    }

    NDSPAN_INLINE const size_t* shape() const {
        return shape_.get();
    }

    using Base::shape;

protected:

    ~DynamicNdSpan() = default;

private:

    template<std::integral Int>
    void allocate(const Int* new_shape, size_t new_ndim){
        assert((new_shape != nullptr && new_ndim > 0) && "Invalid new shape");
        size_ = prod(new_shape, new_ndim);
        if (new_ndim != ndim_){
            ndim_ = new_ndim;
            shape_ = std::make_unique<size_t[]>(new_ndim);
        }
        std::copy(new_shape, new_shape + new_ndim, shape_.get());
    }

    void deallocate(){
        ndim_ = 0;
        size_ = 0;
        shape_ = nullptr;
    }

    size_t ndim_ = 0;
    size_t size_ = 0;
    std::unique_ptr<size_t[]> shape_;


};


// Derived Layouts inherit from this
template<typename Derived, size_t... DIMS>
using DerivedNdSpan = std::conditional_t<(sizeof...(DIMS) > 0 && (DIMS*...*1)>0), StaticNdSpan<Derived, DIMS...>, std::conditional_t<(sizeof...(DIMS)>0), SemiStaticNdSpan<Derived, DIMS...>, DynamicNdSpan<Derived>>>;

} // namespace ndspan