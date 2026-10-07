#pragma once

#include <unordered_map>
#include <array>
#include "standard_layouts.hpp"

namespace ndspan{

namespace detail{

template<std::integral INT>
constexpr INT next_pow_of_2(INT x) noexcept{
    INT y = 1;
    while (y < x){
        y *= 2;
    }
    return y;
}

} // namespace detail

template<size_t... DIMS>
class ZorderNdSpan : public DerivedNdSpan<ZorderNdSpan<DIMS...>, DIMS...>{

    using Base = DerivedNdSpan<ZorderNdSpan<DIMS...>, DIMS...>;

public:

    using Base::Base;

    ZorderNdSpan(){
        if constexpr (Base::N > 0){
            //the shape is a compile time constant, so the lookup tables can be built right away
            this->_rebuild_tables(Base::SHAPE.data(), Base::RANK);
        }
    }

    template<std::integral... Args>
    explicit constexpr ZorderNdSpan(Args... shape) : Base(shape...){
        this->resize(shape...);
    }

    template<std::integral Int>
    explicit ZorderNdSpan(const Int* shape, size_t ndim) : Base(shape, ndim){
        this->resize(shape, ndim);
    }

    DEFAULT_RULE_OF_FOUR(ZorderNdSpan)

    NDSPAN_INLINE size_t padded_size() const{
        return _padded_size;
    }

    template<std::integral... Args>
    void constexpr resize(Args... shape){
        Base::resize(shape...); //now the shape and size are updated
        _strided_span.resize(shape...);
        if constexpr (sizeof...(shape) == 0){
            this->_rebuild_tables(static_cast<const size_t*>(nullptr), 0);
        } else {
            const size_t new_shape[sizeof...(shape)] = {static_cast<size_t>(shape)...};
            this->_rebuild_tables(new_shape, sizeof...(shape));
        }
    }

    template<std::integral Int>
    void resize(const Int* shape, size_t ndim){
        Base::resize(shape, ndim); //now the shape and size are updated
        _strided_span.resize(shape, ndim);
        this->_rebuild_tables(shape, ndim);
    }

    template<std::integral... Idx>
    NDSPAN_INLINE constexpr size_t offset_impl(Idx... idx) const noexcept{
        size_t id = _strided_span.offset(idx...);
        return _offset_at_id[id];
    }

    template<std::integral Idx>
    NDSPAN_INLINE constexpr size_t getOffset_impl(const Idx* idx_ptr) const noexcept{
        size_t id = _strided_span.getOffset(idx_ptr);
        return _offset_at_id[id];
    }

    template<std::integral... Idx>
    NDSPAN_INLINE void unpack_idx_impl(size_t offset, Idx&... idx) const noexcept{
        size_t id = _id_at_offset[offset];
        _strided_span.unpack_idx(id, idx...);
    }

    template<typename ShapeType>
    static constexpr size_t morton_bits(const ShapeType& shape, size_t nd){
        size_t res = 0;
        for (size_t i = 0; i < nd; ++i) {
            if (shape[i] == 0){
                continue; //an empty dim would underflow below, and makes the grid empty anyway
            }
            size_t n = size_t(shape[i]) - 1;
            size_t bits = 0;
            while (n >> bits) {++bits;}
            res = ndspan::max(bits, res);
        }
        return res;
    }

    template<std::integral... Args>
    NDSPAN_INLINE static constexpr size_t padded_size(Args... shape){
        static_assert(sizeof...(shape) > 0, "The padded_size function requires at least one argument");
        const size_t shape_arr[sizeof...(shape)] = {size_t(shape)...};
        return _padded_size_of(shape_arr, sizeof...(shape));
    }

    template<std::integral INT, size_t Nd>
    NDSPAN_INLINE constexpr size_t morton_offset(const std::array<INT, Nd>& idx) const {
        return _morton_offset_aux(idx, std::make_index_sequence<Nd>());
    }

    template<std::integral... Coords>
    inline constexpr size_t morton_index(Coords... idx) const {
        this->_dim_check(this->ndim(), idx...);

        NDSPAN_EXPAND(sizeof...(idx), I,
            assert(((idx >= 0 && size_t(idx) < this->shape(I)) && ...) && "Out of bounds");
        );
        size_t result = 0;
        for (size_t bit = 0; bit < _bits; ++bit) {
            result |= _dyn_morton_dims_loop(bit, std::make_index_sequence<sizeof...(idx)>{}, idx...);
        }
        return result;
    }

    template<std::integral... Args>
    inline constexpr void morton_unpack_idx(size_t offset, Args&... idx) const {
        //same as unpack_idx, but directly computes the indices instead of performing memory lookups.
        // offset is the morton index
        ((idx = 0), ...);
        for (size_t bit = 0; bit < _bits; ++bit) {
            _morton_axis_aux<sizeof...(idx)>(offset, bit, std::make_index_sequence<sizeof...(idx)>(), idx...);
        }
    }

    template<size_t Nd, std::integral... Args>
    inline constexpr void morton_unpack_idx_from_array(size_t offset, std::array<size_t, Nd>& idx) const {
        return _morton_unpack_idx_from_array_aux<Nd>(offset, idx, std::make_index_sequence<Nd>());
    }

private:

    template<size_t Axis, size_t Nd>
    NDSPAN_INLINE static constexpr size_t
    _morton_bit(size_t morton_index, size_t bit) {
        return (( (morton_index >> (bit * Nd + Axis)) & 1) << bit );
    }

    template<size_t Nd, std::integral... Idx, size_t... I>
    NDSPAN_INLINE static constexpr void _morton_axis_aux(size_t offset, size_t bit, std::index_sequence<I...>, Idx&... idx) {
        ((idx |= _morton_bit<I, Nd>(offset, bit)),...);
    }

    template<size_t... Axis, std::integral... Coords>
    NDSPAN_INLINE static constexpr size_t _dyn_morton_dims_loop(size_t bit, std::index_sequence<Axis...>, Coords... coords) {
        return size_t(( (((coords >> bit) & 1) << (bit*sizeof...(Axis) + Axis)) | ... ));
    }

    template<std::integral INT, size_t Nd, size_t... I>
    inline constexpr size_t _morton_offset_aux(const std::array<INT, Nd>& idx, std::index_sequence<I...>) const {
        return morton_index(idx[I]...);
    }

    template<size_t Nd, size_t... I, std::integral... Args>
    inline constexpr void _morton_unpack_idx_from_array_aux(size_t offset, std::array<size_t, Nd>& idx, std::index_sequence<I...>) const {
        return this->morton_unpack_idx(offset, idx[I]...);
    }

    //the padded grid is a cube whose side is the next power of two of the largest dim
    template<std::integral Int>
    static constexpr size_t _padded_size_of(const Int* shape, size_t nd){
        if (nd == 0){
            return 0;
        }
        size_t p = 1;
        for (size_t i = 0; i < nd; i++){
            if (shape[i] == 0){
                return 0;
            }
            p = ndspan::max(p, detail::next_pow_of_2(size_t(shape[i])));
        }
        size_t total = 1;
        for (size_t i = 0; i < nd; i++){
            total *= p;
        }
        return total;
    }

    //runtime counterpart of morton_unpack_idx, for a rank that is only known at runtime
    NDSPAN_INLINE static constexpr void _morton_decode(size_t morton, size_t* idx, size_t nd, size_t bits) noexcept{
        for (size_t a = 0; a < nd; ++a){
            idx[a] = 0;
        }
        for (size_t bit = 0; bit < bits; ++bit){
            for (size_t a = 0; a < nd; ++a){
                idx[a] |= ((morton >> (bit*nd + a)) & size_t(1)) << bit;
            }
        }
    }

    //walks the padded z-order grid once and records the offset<->id mapping for the cells
    //that fall inside the true grid. Every entry point funnels through here.
    template<std::integral Int>
    void _rebuild_tables(const Int* shape, size_t nd){
        _bits = morton_bits(shape, nd);
        _padded_size = _padded_size_of(shape, nd);

        _id_at_offset.clear();
        _id_at_offset.shrink_to_fit();

        _offset_at_id.clear();
        _offset_at_id.shrink_to_fit();

        const size_t n = this->size();
        if (n == 0 || nd == 0){
            return;
        }
        assert(_bits*nd <= sizeof(size_t)*8 && "Z-order grid too large to index with a size_t");

        _id_at_offset.resize(n);
        _offset_at_id.resize(n);

        std::vector<size_t> idx(nd);
        size_t true_idx = 0;
        for (size_t padded_offset = 0; padded_offset < _padded_size; padded_offset++){
            //retrieve the (i, j, k) index of the padded grid
            _morton_decode(padded_offset, idx.data(), nd, _bits);

            //skip the cells of the padded grid that fall outside the true grid
            bool inside = true;
            for (size_t i = 0; i < nd; i++){
                if (idx[i] >= size_t(shape[i])){
                    inside = false;
                    break;
                }
            }
            if (inside){
                const size_t id = _strided_span.getOffset(idx.data());
                assert(id < n && "Z-order id out of range");
                _id_at_offset[true_idx] = id;
                _offset_at_id[id] = true_idx;
                true_idx++;
            }
        }
        assert(true_idx == n && "Z-order tables do not cover the whole grid");
    }

    size_t _padded_size = 0;
    size_t _bits = 0;
    RowMajorSpan<DIMS...> _strided_span; //converts the index (i, j, ...) to a unique id (integer). The unique id continuously ranges from 0 up to (dim1*dim2*...*dimNd - 1)
    std::vector<size_t> _id_at_offset; 
    std::vector<size_t> _offset_at_id;
};

} // namespace ndspan
