#include <gtest/gtest.h>
#include <sample.h>

namespace demo {

namespace {

class mt19937 : public std::mt19937 {
    using std::mt19937::mt19937;
};

thread_local auto gen = mt19937{std::random_device()()};

struct Particle {
    int state;
    double weight;
};

TEST(Sample, SingleElement) {
    const auto input = std::array{Particle{5, 1}};
    auto output = input | sample(gen) | std::views::take(20);
    ASSERT_EQ(std::ranges::count(output, 5, &Particle::state), 20);
}

TEST(Sample, DoubleDereference) {
    const auto input = std::array{Particle{1, 1}, Particle{3, 1}, Particle{5, 1}};
    auto output = sample(input, gen);
    auto it = std::ranges::begin(output);
    ++it;
    auto value = (*it).state;
    ASSERT_EQ(value, (*it).state);
    ASSERT_EQ(value, (*it).state);
}

TEST(Sample, NoBorrow) {
    const auto input = std::array{Particle{42, 1}};
    const auto create_view = [&]() { return input | sample(gen); };
    auto it = std::ranges::find(create_view(), 42, &Particle::state);
    static_assert(std::is_same_v<decltype(it), std::ranges::dangling>);
}

TEST(Sample, InitializeFromTemporary) {
    auto input = std::array{Particle{42, 1}};

    auto make_temp = [](const auto& p) -> Particle { return p; };

    [[maybe_unused]] auto output = input |
                                   std::views::transform(make_temp) |
                                   sample(gen) |
                                   std::views::take(10);
}

}  // namespace

}  // namespace demo
