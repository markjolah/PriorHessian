/** @file AnyRNG.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief A type-erased random number generator interface to std library generators.
 *
 * The standard library provides a generic concept of a random number generator that at
 * a minimum requires min, max, and operator().  We use reference capture to make a temporary
 * type-erased container for use in passing a generic random number generator in code that cannot be
 * templated (e.g., virtual function calls, or in combination with other type-erasure methods).
 * 
 */
#ifndef _ANY_RNG_ANYRNG_H
#define _ANY_RNG_ANYRNG_H

namespace any_rng
{

/** Generic, type-erased container for a random number generator.
 * 
 * Stores a reference to rng.  This is by design.  This will become invalid 
 * if this object outlives the RNG it refers too.  RNGs are expected to be global or
 * at least have lifetimes that span the entire lifetime of the code.  Therefore it
 * is normally safe to use this type erased object as if it were.
 * 
 * In and of itself this class does not check for reference validity or threaded useage.
 * This is intended as a single-threaded data structure.
 * 
 */
template<typename ResultT>
class AnyRng
{
public:
    template<typename RNG>
    explicit AnyRng(RNG &rng) 
        : min{RNG::min()},
          max{RNG::max()},
          _generate{ [&rng](){return rng();} }
    { }
    
    using result_type = ResultT;
    ResultT operator()() { return _generate(); }

    const ResultT min;
    const ResultT max;
private:
    std::function<ResultT()> _generate;
};

} /* namespace any_rng */

#endif /* _ANY_RNG_ANYRNG_H */
