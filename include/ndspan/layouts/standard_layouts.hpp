#pragma once

#include "../ndspan.hpp"

namespace ndspan{

namespace detail{

template<typename Derived, size_t... DIMS>
class StridedDerivedNdSpan : public DerivedNdSpan<Derived, DIMS...>{

    using Base = DerivedNdSpan<Derived, DIMS...>;

public:

    using Base::Base;

    DEFAULT_RULE_OF_FOUR(StridedDerivedNdSpan)

    template<typename StrideType, typename ShapeType>
    static constexpr void set_strides(StrideType& s, const ShapeType& shape, size_t nd) {
        Derived::set_strides(s, shape, nd);
    }

    template<typename... Idx>
    NDSPAN_INLINE void unpack_idx_impl(size_t offset, Idx&... idx) const noexcept {
        if constexpr (Base::N > 0) {
            return Derived::strided_unpack(offset, STRIDES, Base::SHAPE, idx...);
        } else {
            return Derived::strided_unpack(offset, THIS->strides(), this->shape(), idx...);
        }
    }

protected:

    inline static constexpr std::array<size_t, Base::RANK> STRIDES = [](){
        std::array<size_t, Base::RANK> s{};
        set_strides(s, Base::SHAPE, Base::RANK);
        return s;
        }();

    template<size_t... I, std::integral... Idx>
    NDSPAN_INLINE static constexpr size_t _static_offset_impl(std::index_sequence<I...>, Idx... idx) noexcept {
        return ((static_cast<size_t>(idx) * STRIDES[I]) + ...);
    }

};


template<typename Derived, size_t... DIMS>
class StridedStaticNdSpan : public StridedDerivedNdSpan<Derived, DIMS...>{

    using Base = StridedDerivedNdSpan<Derived, DIMS...>;

public:

    using Base::N, Base::RANK;
    using Base::Base;

    DEFAULT_RULE_OF_FOUR(StridedStaticNdSpan)
    
    template<std::integral... Idx>
    NDSPAN_INLINE constexpr size_t offset_impl(Idx... idx) const noexcept{
        return Base::_static_offset_impl(std::make_index_sequence<Base::RANK>(), idx...);
    }

    template<std::integral Idx>
    NDSPAN_INLINE constexpr size_t getOffset_impl(const Idx* idx_ptr) const noexcept{
        size_t res = 0;
        for (size_t i = 0; i < this->ndim(); i++){
            res += static_cast<size_t>(idx_ptr[i]) * Base::STRIDES[i];
        }
        return res;
    }

    const size_t* strides() const{
        return Base::STRIDES.data();
    }

};


template<typename Derived, size_t... DIMS>
class StridedSemiStaticNdSpan : public StridedDerivedNdSpan<Derived, DIMS...>{

    using Base = StridedDerivedNdSpan<Derived, DIMS...>;

protected:

    inline static constexpr size_t RANK = Base::RANK;
    inline static constexpr size_t N = Base::N;

    static_assert(N==0 && RANK>0, "StridedSemiStaticNdSpan is for static number of dims of dynamic size");

    StridedSemiStaticNdSpan() = default;

    template<std::integral... Args>
    explicit constexpr StridedSemiStaticNdSpan(Args... shape) : Base(shape...) {
        this->_remake_strides();
    }

    template<std::integral Int>
    explicit constexpr StridedSemiStaticNdSpan(const Int* shape, size_t ndim) : Base(shape, ndim) {
        this->_remake_strides();
    }

    DEFAULT_RULE_OF_FOUR(StridedSemiStaticNdSpan)


public:

    const size_t* strides() const{
        return _fixed_strides;
    }

    template<std::integral... Args>
    void constexpr resize(Args... shape){
        Base::resize(shape...);
        this->_remake_strides();
    }

    template<std::integral Int>
    void constexpr resize(const Int* shape, size_t ndim){
        Base::resize(shape, ndim);
        this->_remake_strides();
    }

    template<std::integral... Idx>
    NDSPAN_INLINE constexpr size_t offset_impl(Idx... idx) const noexcept{
        return NDSPAN_EXPAND(RANK, I,
            return ((idx * _fixed_strides[I]) + ...);
        );
    }

    template<std::integral Idx>
    NDSPAN_INLINE constexpr size_t getOffset_impl(const Idx* idx_ptr) const noexcept{
        size_t res = 0;
        for (size_t i = 0; i < this->ndim(); i++){
            res += static_cast<size_t>(idx_ptr[i]) * _fixed_strides[i];
        }
        return res;
    }

private:

    inline void _remake_strides(){
        Derived::set_strides(_fixed_strides, this->shape(), this->ndim());
    }
    
    size_t _fixed_strides[RANK];

};


template<typename Derived, size_t N>
class StridedSemiStaticNdSpan<Derived, N> : public StridedDerivedNdSpan<Derived, N>{

    //Specialization for the 1D case.
    //Its only  benefit is that it does not store the strides array.

    using Base = StridedDerivedNdSpan<Derived, N>;

protected:

    inline static constexpr size_t RANK = 1;

    static_assert(N==0, "StridedSemiStaticNdSpan is for static number of dims of dynamic size");

    StridedSemiStaticNdSpan() = default;

    template<std::integral... Args>
    explicit constexpr StridedSemiStaticNdSpan(Args... shape) : Base(shape...) {}

    template<std::integral Int>
    explicit constexpr StridedSemiStaticNdSpan(const Int* shape, size_t ndim) : Base(shape, ndim) {}

};

template<typename Derived, size_t R, size_t C>
class StridedSemiStaticNdSpan<Derived, R, C> : public StridedDerivedNdSpan<Derived, R, C>{

    //Specialization for the 2D case

    using Base = StridedDerivedNdSpan<Derived, R, C>;

protected:

    inline static constexpr size_t RANK = 2;
    inline static constexpr size_t N = R*C;

    static_assert(N==0, "StridedSemiStaticNdSpan is for static number of dims of dynamic size");

    StridedSemiStaticNdSpan() = default;

    template<std::integral... Args>
    explicit constexpr StridedSemiStaticNdSpan(Args... shape) : Base(shape...) {}

    template<std::integral Int>
    explicit constexpr StridedSemiStaticNdSpan(const Int* shape, size_t ndim) : Base(shape, ndim) {}

    DEFAULT_RULE_OF_FOUR(StridedSemiStaticNdSpan)

};

template<typename Derived>
class StridedDynamicNdSpan : public StridedDerivedNdSpan<Derived>{

    using Base = StridedDerivedNdSpan<Derived>;
public:
    inline static constexpr size_t RANK = 0;
    inline static constexpr size_t N = 0;

protected:

    StridedDynamicNdSpan() = default;

    template<std::integral... Args>
    explicit constexpr StridedDynamicNdSpan(Args... shape) : Base(shape...) {
        _dyn_strides = new size_t[this->ndim()];
        Derived::set_strides(_dyn_strides, this->shape(), this->ndim());
    }

    template<std::integral Int>
    explicit constexpr StridedDynamicNdSpan(const Int* shape, size_t ndim) : Base(shape, ndim) {
        _dyn_strides = ndim > 0 ? new size_t[ndim] : nullptr;
        Derived::set_strides(_dyn_strides, shape, ndim);
    }

    //COPY CONSTRUCTOR
    StridedDynamicNdSpan(const StridedDynamicNdSpan& other) : Base(static_cast<const Base&>(other)){
        _dyn_strides = other.ndim() > 0 ? new size_t[other.ndim()] : nullptr;
        ndspan::copy_array(_dyn_strides, other._dyn_strides, this->ndim());
    }

    //MOVE CONSTRUCTOR
    StridedDynamicNdSpan(StridedDynamicNdSpan&& other) noexcept : Base(static_cast<Base&&>(other)), _dyn_strides(other._dyn_strides) {
        other._dyn_strides = nullptr;
    }

    //ASSIGNMENT OPERATOR
    StridedDynamicNdSpan& operator=(const StridedDynamicNdSpan& other){
        if (&other != this){
            size_t nd_old = this->ndim();
            Base::operator=(other);
            if (nd_old != other.ndim()){
                delete[] _dyn_strides;
                if (this->ndim() > 0){
                    _dyn_strides = new size_t[this->ndim()];
                }
                else{
                    _dyn_strides = nullptr;
                }
            }
            ndspan::copy_array(_dyn_strides, other._dyn_strides, this->ndim());
        }
        return *this;
    }

    //MOVE-ASSIGNMENT OPERATOR
    StridedDynamicNdSpan& operator=(StridedDynamicNdSpan&& other) noexcept {
        if (&other != this){
            Base::operator=(std::move(other));
            delete[] _dyn_strides;
            _dyn_strides = other._dyn_strides;
            other._dyn_strides = nullptr;
        }
        return *this;
    }

    ~StridedDynamicNdSpan() {
        delete[] _dyn_strides;
        _dyn_strides = nullptr;
    }

public:

    const size_t* strides() const{
        return _dyn_strides;
    }

    template<std::integral... Args>
    void constexpr resize(Args... shape){
        size_t nd_old = this->ndim();
        Base::resize(shape...);
        this->_realloc_strides(nd_old);
    }

    template<std::integral Int>
    void constexpr resize(const Int* shape, size_t ndim){
        size_t nd_old = this->ndim();
        Base::resize(shape, ndim);
        this->_realloc_strides(nd_old);
    }



    template<std::integral... Idx>
    NDSPAN_INLINE constexpr size_t offset_impl(Idx... idx) const noexcept{
        return NDSPAN_EXPAND(sizeof...(idx), I,
            return ((idx * _dyn_strides[I]) + ...);
        );
    }

    template<std::integral Idx>
    NDSPAN_INLINE constexpr size_t getOffset_impl(const Idx* idx_ptr) const noexcept{
        size_t res = 0;
        for (size_t i = 0; i < this->ndim(); i++){
            res += static_cast<size_t>(idx_ptr[i]) * _dyn_strides[i];
        }
        return res;
    }

private:

    void _realloc_strides(size_t nd_old){
        if (this->ndim() > nd_old){
            //only reallocate in this case
            delete[] _dyn_strides;
            _dyn_strides = new size_t[this->ndim()];
        }
        Derived::set_strides(_dyn_strides, this->shape(), this->ndim());
    }

    size_t* _dyn_strides = nullptr;
};


template<typename Derived, size_t... DIMS>
using StridedNdSpan = std::conditional_t<(sizeof...(DIMS) > 0 && (DIMS*...*1)>0), StridedStaticNdSpan<Derived, DIMS...>, std::conditional_t<(sizeof...(DIMS) > 0), StridedSemiStaticNdSpan<Derived, DIMS...>, StridedDynamicNdSpan<Derived>>>;

} // namespace detail



template<size_t... DIMS>
class RowMajorSpan : public detail::StridedNdSpan<RowMajorSpan<DIMS...>, DIMS...>{

    using Base = detail::StridedNdSpan<RowMajorSpan<DIMS...>, DIMS...>;

public:

    DEFAULT_RULE_OF_FOUR(RowMajorSpan)

    template<std::integral Int>
    constexpr explicit RowMajorSpan(const Int* shape, size_t ndim) : Base(shape, ndim) {}

    template<std::integral... Args>
    constexpr explicit RowMajorSpan(Args... shape) : Base(shape...){}

    template<typename StrideType, typename ShapeType>
    static constexpr void set_strides(StrideType& s, const ShapeType& shape, size_t nd) {
        size_t stride = 1;
        for (size_t i=nd; i-- > 0;){
            s[i] = stride;
            stride *= shape[i];
        }
    }

    template<typename STRIDE_T, typename  SHAPE_T, typename... Idx>
    NDSPAN_INLINE static void strided_unpack(size_t offset, const STRIDE_T& strides, const SHAPE_T& shape, Idx&... idx) noexcept {
        NDSPAN_EXPAND(sizeof...(idx), I,
            ((idx = offset / strides[I],
            offset %= strides[I]), ...);
        );
    }

};


template<size_t R, size_t C>
class RowMajorSpan<R, C> : public detail::StridedNdSpan<RowMajorSpan<R, C>, R, C>{

    using Base = detail::StridedNdSpan<RowMajorSpan<R, C>, R, C>;

public:

    DEFAULT_RULE_OF_FOUR(RowMajorSpan)

    template<std::integral Int>
    explicit RowMajorSpan(const Int* shape, size_t ndim) : Base(shape, ndim) {}

    template<std::integral... Args>
    explicit constexpr RowMajorSpan(Args... shape) : Base(shape...){}

    template<std::integral Int1, std::integral Int2>
    NDSPAN_INLINE constexpr size_t offset_impl(Int1 i, Int2 j) const {
        if constexpr (C > 0){
            return i*C + j;
        }else {
            return i*this->shape(1) + j;
        }
    }

    template<std::integral Idx>
    NDSPAN_INLINE constexpr size_t getOffset_impl(const Idx* idx_ptr) const noexcept{
        return offset_impl(idx_ptr[0], idx_ptr[1]);
    }

    template<std::integral Int1, std::integral Int2>
    NDSPAN_INLINE void unpack_idx_impl(size_t offset, Int1& i, Int2& j) const {
        if constexpr (C > 0) {
            i = offset/C;
            j = offset % C;
        }else {
            i = offset/this->shape(1);
            j = offset % this->shape(1);
        }
    }

};


template<size_t... DIMS>
class ColumnMajorSpan : public detail::StridedNdSpan<ColumnMajorSpan<DIMS...>, DIMS...>{

    using Base = detail::StridedNdSpan<ColumnMajorSpan<DIMS...>, DIMS...>;

public:

    using Base::Base;

    DEFAULT_RULE_OF_FOUR(ColumnMajorSpan)
    
    template<std::integral Int>
    explicit ColumnMajorSpan(const Int* shape, size_t ndim) : Base(shape, ndim) {}

    template<std::integral... Args>
    explicit constexpr ColumnMajorSpan(Args... shape) : Base(shape...){}

    template<typename StrideType, typename ShapeType>
    static constexpr void set_strides(StrideType& s, const ShapeType& shape, size_t nd) {
        size_t stride = 1;
        for (size_t i = 0; i < nd; ++i) {
            s[i] = stride;
            stride *= shape[i];
        }
    }

    template<typename STRIDE_T, typename  SHAPE_T, typename... Idx>
    NDSPAN_INLINE static void strided_unpack(size_t offset, const STRIDE_T& strides, const SHAPE_T& shape, Idx&... idx) noexcept {
        NDSPAN_EXPAND(sizeof...(idx), I,
            ((idx = offset % shape[I],
            offset /= shape[I]), ...);
        );
    }
    
};


template<size_t R, size_t C>
class ColumnMajorSpan<R, C> : public detail::StridedNdSpan<ColumnMajorSpan<R, C>, R, C>{

    using Base = detail::StridedNdSpan<ColumnMajorSpan<R, C>, R, C>;

public:

    using Base::Base;

    DEFAULT_RULE_OF_FOUR(ColumnMajorSpan)
    
    template<std::integral Int>
    explicit ColumnMajorSpan(const Int* shape, size_t ndim) : Base(shape, ndim) {}

    template<std::integral... Args>
    explicit constexpr ColumnMajorSpan(Args... shape) : Base(shape...){}

    template<std::integral Int1, std::integral Int2>
    NDSPAN_INLINE constexpr size_t offset_impl(Int1 i, Int2 j) const {
        if constexpr (R > 0){
            return j*R + i;
        }else {
            return j*this->shape(0) + i;
        }
    }

    template<std::integral Idx>
    NDSPAN_INLINE constexpr size_t getOffset_impl(const Idx* idx_ptr) const noexcept{
        return offset_impl(idx_ptr[0], idx_ptr[1]);
    }

    template<std::integral Int1, std::integral Int2>
    NDSPAN_INLINE void unpack_idx_impl(size_t offset, Int1& i, Int2& j) const {
        if constexpr (R > 0) {
            i = offset % R;
            j = offset / R;
        }else {
            i = offset % this->shape(0);
            j = offset / this->shape(0);
        }
    }

};

} // namespace ndspan