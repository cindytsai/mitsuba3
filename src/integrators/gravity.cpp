#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class GravityIntegrator final : public SamplingIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SamplingIntegrator, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    GravityIntegrator(const Properties &props) : Base(props) {}

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium *medium, Float *aovs,
                                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        // Loop state
        Ray3f ray           = Ray3f(ray_);
        Spectrum throughput = 1.0f;
        Spectrum result     = 0.0f;
        Float eta           = 1.0f;
        UInt32 depth        = 0;

        Mask valid_ray = !m_hide_emitters && (scene->environment() != nullptr);

        // TODO: we probably don't need this or we deal with this at the very
        // end
        Interaction3f prev_si = dr::zeros<Interaction3f>();
        Float prev_bsdf_pdf   = 1.0f;
        Bool prev_bsdf_delta  = true;
        BSDFContext bsdf_ctx;

        struct LoopState {
            Ray3f ray;
            Spectrum throughput;
            Spectrum result;
            Float eta;
            UInt32 depth;
            Mask valid_ray;
            Interaction3f prev_si;
            Float prev_bsdf_pdf;
            Bool prev_bsdf_delta;
            Bool active;
            Sampler *sampler;

            DRJIT_STRUCT(LoopState, ray, throughput, result, eta, depth,
                         valid_ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                         active, sampler)
        } ls = { ray,     throughput,    result,
                 eta,     depth,         valid_ray,
                 prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                 active,  sampler };

        // TODO: Get the right sample for the bending of rays
        dr::tie(ls) = dr::while_loop(
            dr::make_tuple(ls), [](const LoopState &ls) { return ls.active; },
            [this, scene, bsdf_ctx](LoopState &ls) {
                // Map incoming ray from sensor to ray after the effect of gravity
                Float d = shortest_distance_point_to_ray(ls.ray);
                Ray3f bended_ray = Ray3f(ls.ray); // TODO: (START HERE)

                // Use the calculated ray to get the emitter mapping
                SurfaceInteraction3f si = scene->ray_intersect(
                    bended_ray, +RayFlags::All, ls.depth == 0u);

                // Sample the background emitter using the calculated si
                if (dr::any_or<true>(si.emitter(scene) != nullptr)) {
                    DirectionSample3f ds(scene, si, ls.prev_si);
                    Float em_pdf = 0.0f;

                    if (dr::any_or<true>(!ls.prev_bsdf_delta)) {
                        em_pdf = scene->pdf_emitter_direction(
                            ls.prev_si, ds, !ls.prev_bsdf_delta);
                    }

                    // Compute weight for emitter sample from previous bounce
                    Float mis_bsdf = 1;

                    // Sample the emitter and accumulate the results
                    ls.result = dr::fmadd(
                        ls.throughput,
                        ds.emitter->eval(si, ls.prev_bsdf_pdf > 0.0f) *
                            mis_bsdf,
                        ls.result);
                } else {
                    // If there is an intersection with the sphere, set the
                    // value manually
                    ls.result = 0;
                }

                ls.active = false;
            });

        return { dr::select(ls.valid_ray, ls.result, 0.0f), ls.valid_ray };
    }

    MI_DECLARE_CLASS()
protected:
    /// Important: declare a protected virtual destructor
    // virtual ~GravityIntegrator();
private:
    static std::array<Float, 3> cross_product(const std::array<Float, 3> &a,
                                              const std::array<Float, 3> &b) {
        return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                 a[0] * b[1] - a[1] * b[0] };
    }

    static Float norm(const std::array<Float, 3> &a) {
        return dr::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    }

    static Float shortest_distance_point_to_ray(
        const Ray3f &ray, const std::array<Float, 3> &p0 = { 0.0, 0.0, 0.0 }) {

        // Ray info
        std::array<Float, 3> p1 = { ray.o[0], ray.o[1], ray.o[2] };
        std::array<Float, 3> p2 = { ray.o[0] + ray.d[0], ray.o[1] + ray.d[1],
                                    ray.o[2] + ray.d[2] };

        // Calculate the shortest distance from the ray to point p0
        Float d = norm(cross_product(
                      { p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2] },
                      { p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2] })) /
                  norm({ p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] });

        return d;
    }
};

/// Implement RTTI data structures
MI_IMPLEMENT_CLASS_VARIANT(GravityIntegrator, SamplingIntegrator)
MI_EXPORT_PLUGIN(GravityIntegrator, "Integrator with ray bended by gravity")
NAMESPACE_END(mitsuba)
