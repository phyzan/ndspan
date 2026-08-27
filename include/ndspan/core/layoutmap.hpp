#pragma once

#include "layouts/standard_layouts.hpp"
#include "layouts/morton.hpp"


namespace ndspan{
/*
All classes in the LayoutMap (e.g. AbstractRowMajorSpan) must have this private method:

template<std::integral... Idx>
NDSPAN_INLINE constexpr size_t _offset_impl(Idx... idx) const noexcept
*/
enum class Layout : std::uint8_t { C, F, Z};

namespace detail{

template <Layout L, size_t... DIMS>
struct LayoutMap;

template <size_t... DIMS>
struct LayoutMap<Layout::C, DIMS...> { using type = RowMajorSpan<DIMS...>; };

template <size_t... DIMS>
struct LayoutMap<Layout::F, DIMS...> { using type = ColumnMajorSpan<DIMS...>; };

template <size_t... DIMS>
struct LayoutMap<Layout::Z, DIMS...> { using type = ZorderNdSpan<DIMS...>; };

} // namespace detail

template <Layout L, size_t... DIMS>
class NdSpan : public std::conditional_t<(sizeof...(DIMS)==1 && (DIMS*...*1)==0), SemiStaticSpan1D, typename detail::LayoutMap<L, DIMS...>::type>{

    using Base = std::conditional_t<(sizeof...(DIMS)==1 && (DIMS*...*1)==0), SemiStaticSpan1D, typename detail::LayoutMap<L, DIMS...>::type>;

protected:

    inline static constexpr size_t RANK = Base::RANK;
    inline static constexpr size_t N = Base::N;
    inline static constexpr std::array<size_t, RANK> SHAPE = Base::SHAPE;

public:

    template<std::integral Int>
    explicit NdSpan(const Int* shape, size_t ndim) : Base(shape, ndim) {}

    template<std::integral... Args>
    explicit constexpr NdSpan(Args... shape) : Base(shape...){}

    //ACCESSORS
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

    //MODIFIERS
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
};


} // namespace ndspan