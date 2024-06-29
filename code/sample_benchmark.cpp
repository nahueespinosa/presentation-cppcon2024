#include <algorithm>
#include <functional>
#include <generator>
#include <iostream>
#include <random>
#include <ranges>

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <sample.h>

namespace demo {

namespace {

class mt19937 : public std::mt19937 {
    using std::mt19937::mt19937;
};

struct Particle {
    double state;
    double weight;
};

template <typename R, typename P = std::ranges::range_value_t<R>>
requires std::ranges::viewable_range<R> && std::ranges::sized_range<R> && ParticleLike<P>
auto compute_total_weight(const R& particles) {
    auto weights = particles | std::views::transform(&P::weight);
    return std::ranges::fold_left(weights, 0, std::plus<>{});
}

template <typename R, typename P = std::ranges::range_value_t<R>>
requires std::ranges::viewable_range<R> && std::ranges::sized_range<R> && ParticleLike<P>
void verify_particle_set(const R& particles, double n) {
    std::unordered_map<double, std::size_t> buckets;
    for (const auto& particle : particles) {
        ++buckets[particle.state];
    }

    ASSERT_NEAR(static_cast<double>(buckets[1]) / n, 1. / 9., 0.03);
    ASSERT_NEAR(static_cast<double>(buckets[2]) / n, 0., 0.01);
    ASSERT_NEAR(static_cast<double>(buckets[3]) / n, 3. / 9., 0.03);
    ASSERT_NEAR(static_cast<double>(buckets[4]) / n, 0., 0.01);
    ASSERT_NEAR(static_cast<double>(buckets[5]) / n, 5. / 9., 0.03);
}

static auto make_particle_set() {
    return std::vector<Particle>{{1, 1}, {2, 0}, {3, 3}, {4, 0}, {5, 5}};
}

static void CustomView(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));

    auto gen = mt19937{std::random_device()()};
    auto sample_generator = sample(make_particle_set(), gen) | std::views::take(n);

    auto new_particles = std::vector<Particle>{};
    new_particles.reserve(n);

    for (auto _ : state) {
        for (const auto& p : sample_generator) {
            new_particles.push_back(p);
        }

        state.PauseTiming();
        verify_particle_set(new_particles, static_cast<double>(n));
        new_particles.clear();
        state.ResumeTiming();
    }
}

BENCHMARK(CustomView)->DenseRange(10000, 60000, 10000)->MinWarmUpTime(1);

static void Coroutine(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));

    auto gen = mt19937{std::random_device()()};
    auto sample_generator = sample_coroutine(make_particle_set(), gen) | std::views::take(n);

    auto new_particles = std::vector<Particle>{};
    new_particles.reserve(n);

    for (auto _ : state) {
        for (const auto& p : sample_generator) {
            new_particles.push_back(p);
        }

        state.PauseTiming();
        verify_particle_set(new_particles, static_cast<double>(n));
        new_particles.clear();
        state.ResumeTiming();
    }
}

BENCHMARK(Coroutine)->DenseRange(10000, 60000, 10000)->MinWarmUpTime(1);

static void Baseline(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));

    auto particles = make_particle_set();
    auto gen = mt19937{std::random_device()()};
    auto dist = particles |
                std::views::transform([](const Particle& p) { return p.weight; }) |
                std::ranges::to<std::discrete_distribution<std::size_t>>();

    const auto first = std::begin(particles);

    auto new_particles = std::vector<Particle>{};
    new_particles.reserve(n);

    for (auto _ : state) {
        for (std::size_t i = 0; i < n; ++i) {
            new_particles.push_back(*(first + dist(gen)));
        }

        state.PauseTiming();
        verify_particle_set(new_particles, static_cast<double>(n));
        new_particles.clear();
        state.ResumeTiming();
    }
}

BENCHMARK(Baseline)->DenseRange(10000, 60000, 10000)->MinWarmUpTime(1);

using LargeState = std::array<double, 50>;

struct LargeParticle {
    LargeState state;
    double weight;
};

static void KeepStatesBaseline(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));

    const auto states = std::vector<LargeState>(4);
    const auto weights = std::vector<double>{1.0, 3.0, 2.0, 4.0};

    auto particles = make_particle_set();
    auto gen = mt19937{std::random_device()()};
    auto dist = weights | std::ranges::to<std::discrete_distribution<std::size_t>>();

    auto new_states = std::vector<LargeState>{};
    new_states.reserve(n);

    for (auto _ : state) {
        for (std::size_t i = 0; i < n; ++i) {
            new_states.push_back(states[dist(gen)]);
        }

        state.PauseTiming();
        benchmark::DoNotOptimize(std::ranges::fold_left(new_states, 0.0, [](double accum, const auto& s) { return accum + s[0]; }));
        new_states.clear();
        state.ResumeTiming();
    }
}

BENCHMARK(KeepStatesBaseline)->DenseRange(10000, 60000, 10000)->MinWarmUpTime(1);

template <typename S, typename W>
struct ParticleRef {
    const S& state;
    const W& weight;

    static_assert(ParticleLike<ParticleRef>);
};

static void KeepStates(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));

    auto gen = mt19937{std::random_device()()};

    auto states = std::vector<LargeState>(4);
    auto weights = std::vector<double>{1.0, 3.0, 2.0, 4.0};

    using namespace std::views;

    auto to_particle = [](const auto& state, const auto& weight) {
        return ParticleRef{state, weight};
    };

    auto to_state = [](const auto& p) -> decltype(auto) {
        return p.state;
    };

    auto sample_generator = zip_transform(to_particle, std::move(states), std::move(weights)) |
                            sample(gen) |
                            take(n) |
                            transform(to_state);

    auto new_states = std::vector<LargeState>{};
    new_states.reserve(n);

    for (auto _ : state) {
        for (const auto& p : sample_generator) {
            new_states.push_back(p);
        }

        state.PauseTiming();
        benchmark::DoNotOptimize(std::ranges::fold_left(new_states, 0.0, [](double accum, const auto& s) { return accum + s[0]; }));
        new_states.clear();
        state.ResumeTiming();
    }
}

BENCHMARK(KeepStates)->DenseRange(10000, 60000, 10000)->MinWarmUpTime(1);

}  // namespace

}  // namespace demo
