#include <type_traits>

namespace demo {

template <typename T>
concept ParticleLike =
    std::is_object_v<T> &&
    requires(T a)
    {
        { a.state };
        { a.weight } -> std::convertible_to<double>;
    };

template <typename F, typename T>
concept StateUpdateFn =
    ParticleLike<T> &&
    requires(F f, T t) { { t.state = f(t.state) }; };

template <typename F, typename T>
concept ReweightFn =
    ParticleLike<T> &&
    requires(F f, T t) { { t.weight = f(t.state) }; };

}  // namespace demo
