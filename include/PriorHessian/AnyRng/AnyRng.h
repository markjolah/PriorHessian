/** @file AnyRNG.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018-2019
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
#include <typeinfo>
#include <memory>
#include <type_traits>

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
template<class ResultT>
class AnyRng
{
public:
    template<typename RNG, typename=std::enable_if_t<std::is_same<ResultT,typename RNG::result_type>::value>>
    explicit AnyRng(RNG &rng_) : handle{new RngWrapper<RNG>{rng_}} { } 
    using result_type = ResultT;
    const std::type_info& type_info() { return handle->type_info(); }
    ResultT min() const { return handle->min(); }
    ResultT max() const { return handle->max(); }
    void seed(ResultT seed=0) { handle->seed(seed); }
    ResultT operator()() { return handle->generate(); }
    void discard( ResultT z) { return handle->discard(z); }
    
private:
    class RngHandle
    {
    public:
        virtual ~RngHandle() = default;
        virtual const std::type_info& type_info() const = 0;
        virtual ResultT min() const = 0;
        virtual ResultT max() const = 0;        
        virtual void seed(ResultT seed) = 0;
        virtual ResultT generate() = 0;
        virtual void discard( ResultT z) = 0;
    };
    
    template<class RngT>
    class RngWrapper : public RngHandle {
    public:
        RngWrapper(RngT &rng_) : rng{rng_} { }
        const std::type_info& type_info() const override { return typeid(RngT); }
        ResultT min() const override { return rng.min(); }
        ResultT max() const override { return rng.max(); }        
        void seed(ResultT seed) override { rng.seed(seed); }
        ResultT generate() override { return rng(); }
        void discard( ResultT z) override { rng.discard(z); }
    private:
        RngT &rng;
    };
    
    std::unique_ptr<RngHandle> handle;
};

} /* namespace any_rng */

#endif /* _ANY_RNG_ANYRNG_H */
