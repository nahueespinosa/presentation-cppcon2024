#include <algorithm>
#include <functional>
#include <generator>
#include <iostream>
#include <random>
#include <ranges>

#include "concepts.h"

namespace demo {

template <typename V, typename RNG, typename P = std::ranges::range_value_t<V>>
requires std::ranges::view<V> &&
         std::ranges::random_access_range<V> &&
         std::uniform_random_bit_generator<RNG> &&
         ParticleLike<P>
std::generator<const P&> sample_coroutine_impl(V particles, RNG& gen) {
    auto dist = std::ranges::ref_view(particles) |
                std::views::transform([](const P& p) { return p.weight; }) |
                std::ranges::to<std::discrete_distribution<std::size_t>>();

    while (true) {
        co_yield particles[dist(gen)];
    }
}

template <typename R, typename RNG>
requires std::ranges::viewable_range<R>
auto sample_coroutine(R&& particles, RNG& gen) {
    return sample_coroutine_impl(std::views::all(std::forward<R>(particles)), gen);
}

template <typename V, typename RNG, typename P = std::ranges::range_value_t<V>>
requires std::ranges::view<V> &&
         std::ranges::sized_range<V> &&
         std::ranges::random_access_range<V> &&
         std::uniform_random_bit_generator<RNG> &&
         ParticleLike<P>
class sample_view : public std::ranges::view_interface<sample_view<V, RNG, P>> {
public:
    sample_view(V base, RNG& gen)
      : base_{std::move(base)},
        first_{std::ranges::begin(base_)},
        gen_{std::addressof(gen)},
        dist_{
            std::ranges::ref_view(base_) |
            std::views::transform([](const P& p) { return p.weight; }) |
            std::ranges::to<std::discrete_distribution<std::size_t>>()
        } {}

    sample_view(const sample_view&) = delete;
    sample_view(sample_view&&) = default;
    sample_view& operator=(const sample_view&) = delete;
    sample_view& operator=(sample_view&&) = default;

    auto begin() { return iterator{this}; }
    auto end() const noexcept { return std::unreachable_sentinel; }

private:
    V base_;
    std::ranges::const_iterator_t<V> first_;
    RNG* gen_;
    std::discrete_distribution<std::size_t> dist_;

    auto next() { return first_ + dist_(*gen_); }

    class iterator {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = P;

        explicit iterator(sample_view* v)
          : v_{v},
            cur_{v_->next()} {}

        auto& operator++() { 
            cur_ = v_->next();
            return *this;
        }

        auto operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        decltype(auto) operator*() const { return *cur_; }

    private:
        sample_view* v_;
        std::ranges::const_iterator_t<V> cur_;
    };

    static_assert(std::input_iterator<iterator>);
};

template <typename R, typename RNG, typename P = std::ranges::range_value_t<R>>
sample_view(R&&, RNG&) -> sample_view<std::views::all_t<R>, RNG, P>;

template <typename RNG>
requires std::uniform_random_bit_generator<RNG>
class sample_closure : public std::ranges::range_adaptor_closure<sample_closure<RNG>> {
public:
    explicit sample_closure(RNG& gen) : gen_{gen} {}

    template <typename R>
    requires std::ranges::viewable_range<R>
    std::ranges::view auto operator()(R&& range) const {
        return sample_view{std::forward<R>(range), gen_};
    }

private:
    RNG& gen_;
};

template <typename R, typename RNG>
auto sample(R&& range, RNG& gen) {
    return sample_view{std::forward<R>(range), gen};
}

template <typename RNG>
auto sample(RNG& gen) {
    return sample_closure{gen};
}

}  // namespace demo
