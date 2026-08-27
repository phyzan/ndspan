#pragma once

#include "layoutmap.hpp"

namespace ndspan{

namespace detail{


template<typename Derived, Layout L, typename T, size_t... DIMS>
class AbstractView : public NdSpan<L, DIMS...>{

    using CLS = AbstractView<Derived, L, T, DIMS...>;
    using Base = NdSpan<L, DIMS...>;

protected:

    using Base::Base;

    // place resize in protected, so that derived classes resize their internal data before resizing in NdSpan
    template<std::integral... Args>
    NDSPAN_INLINE void constexpr resize(Args... shape){
        Base::resize(shape...);
    }

    template<std::integral Int>
    NDSPAN_INLINE void resize(const Int* shape, size_t ndim){
        Base::resize(shape, ndim);
    }

public:

    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    //ACCESSORS

    const_iterator begin() const { return this->data(); }
    const_iterator end() const { return this->data() + this->size(); }
    
    NDSPAN_INLINE constexpr const T* data() const{
        //override
        return THIS->data();
    }

    NDSPAN_INLINE const T& back() const{
        return this->data()[this->size()-1];
    }

    template<std::integral... Int>
    NDSPAN_INLINE const T* ptr(Int... idx) const{
        return data()+this->offset(idx...);
    }

    template<std::integral Int>
    NDSPAN_INLINE const T& getElem(const Int* idx_ptr) const{
        return data()[this->getOffset(idx_ptr)];
    }

    template<std::integral... Idx>
    NDSPAN_INLINE constexpr const T& operator()(Idx... idx) const {
        return data()[this->offset(idx...)];
    }

    template<std::integral IDX_T>
    NDSPAN_INLINE constexpr const T& operator[](IDX_T i) const{
        NDSPAN_BOUNDS_ASSERT(i, this->size());
        return data()[i];
    }

    template<typename... Idx>
    NDSPAN_INLINE auto operator()(Idx... i) const{
        return tensor_call(THIS, i...);
    }

};

template<typename Derived, Layout L, typename T, size_t... DIMS>
class AbstractMutView : public AbstractView<Derived, L, T, DIMS...>{

    using CLS = AbstractMutView<Derived, L, T, DIMS...>;
    using Base = AbstractView<Derived, L, T, DIMS...>;

protected:

    using Base::Base;

public:

    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    //MODIFIERS

    iterator begin() { return this->data(); }
    iterator end()   { return this->data() + this->size(); }

    NDSPAN_INLINE T* data(){
        //override
        return THIS->data();
    }

    template<std::integral IDX_T>
    NDSPAN_INLINE constexpr T& operator[](IDX_T i){
        NDSPAN_BOUNDS_ASSERT(i, this->size());
        return data()[i];
    }

    template<std::integral... Int>
    NDSPAN_INLINE T* ptr(Int... idx){
        return data()+this->offset(idx...);
    }

    template<std::integral Int>
    NDSPAN_INLINE T& getElem(const Int* idx_ptr){
        return data()[this->getOffset(idx_ptr)];
    }

    template<std::integral... Idx>
    NDSPAN_INLINE constexpr T& operator()(Idx... idx) noexcept {
        return data()[this->offset(idx...)];
    }

    template<typename... Idx>
    NDSPAN_INLINE auto operator()(Idx... i){
        return tensor_call(THIS, i...);
    }

    Derived& fill(const T& value){
        std::fill(this->begin(), this->end(), value);
        return static_cast<Derived&>(*this);
    }

    using Base::data;
    using Base::operator();
    using Base::operator[];
    using Base::ptr;

};


} // namespace detail


template<typename T, Layout L = Layout::C, size_t... DIMS>
class View : public detail::AbstractView<View<T, L, DIMS...>, L, T, DIMS...>{

    using Base = detail::AbstractView<View<T, L, DIMS...>, L, T, DIMS...>;

public:

    explicit View(const T* data) requires (Base::N > 0) : Base(), _data(data) {}

    template<std::integral... Args>
    explicit View(const T* data, Args... shape) : Base(shape...), _data(data) {}

    template<std::integral Int>
    explicit View(const T* data, const Int* shape, size_t ndim) : Base(shape, ndim), _data(data) {}


    NDSPAN_INLINE const T* data() const{
        return _data;
    }

    void set_data(const T* data) {
        _data = data;
    }

private:

    const T* _data;
};


template<typename T, Layout L = Layout::C, size_t... DIMS>
class MutView : public detail::AbstractMutView<MutView<T, L, DIMS...>, L, T, DIMS...>{

    using Base = detail::AbstractMutView<MutView<T, L, DIMS...>, L, T, DIMS...>;

public:

    using Base::Base;

    explicit MutView(T* data) requires (Base::N > 0) : Base(), _data(data) {}

    template<std::integral... Args>
    explicit MutView(T* data, Args... shape) : Base(shape...), _data(data) {}

    template<std::integral Int>
    explicit MutView(T* data, const Int* shape, size_t ndim) : Base(shape, ndim), _data(data) {}

    NDSPAN_INLINE const T* data() const{
        return _data;
    }

    NDSPAN_INLINE T* data() {
        return _data;
    }

    void set_data(T* data) {
        _data = data;
    }

private:

    T* _data;
};


template<typename T, size_t Size=0>
using MutView1D = MutView<T, Layout::C, Size>;

template<typename T, size_t Size=0>
using View1D = View<T, Layout::C, Size>;

template<typename T, size_t M=0, size_t N=0>
using View2D = View<T, Layout::C, M, N>;

template<typename T, size_t M=0, size_t N=0, size_t K=0>
using View3D = View<T, Layout::C, M, N, K>;

template<typename Derived, std::integral... Int>
inline Derived::value_type& tensor_call(Derived& x, Int... idx){
    return x(idx...);
}

template<typename Derived, std::integral... Int>
inline const Derived::value_type& tensor_call(const Derived& x, Int... idx){
    return x(idx...);
}

} // namespace ndspan